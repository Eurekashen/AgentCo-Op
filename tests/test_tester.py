"""Site testers on synthetic site tables built in memory (no files, no agents, no network)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from agentcoop.coop.claims import Claim
from agentcoop.coop.tester import (TesterParams, TestResult, apply_bh, export_gene_stats,
                                   test_claim)
from agentcoop.coop.view import SiteTable

P = TesterParams()
POP = "CD14 Mono"
GENES = [f"G{i}" for i in range(60)]
PROGRAM = GENES[:6]


class _Table(SiteTable):
    """The tester reads CPM only through population_mean_cpm(); pinning that accessor keeps
    these tests independent of how view.py lays out ``mean_cpm`` across populations."""

    def population_mean_cpm(self, population: str) -> np.ndarray:
        return self.mean_cpm


def scrna_table(shift: dict[str, dict[str, float]] | None = None, *,
                groups: tuple[str, ...] = ("control", "treated"), donors: int = 4,
                n_cells: dict[str, int] | None = None, genes: list[str] = GENES,
                low_cpm: tuple[str, ...] = (), seed: int = 0) -> _Table:
    """Donor pseudobulk of one population: log2 CPM = gene baseline + donor offset + noise,
    plus ``shift[group][gene]`` in that group's donors."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(5, 8, len(genes))
    rows, values = [], []
    for g in groups:
        for k in range(donors):
            unit = f"{g}{k}"
            rows.append({"unit": unit, "donor": unit, "group": g, "population": POP,
                         "n_cells": (n_cells or {}).get(unit, 100)})
            v = base + rng.normal(0, 0.05) + rng.normal(0, 0.05, len(genes))
            for gene, s in (shift or {}).get(g, {}).items():
                v[genes.index(gene)] += s
            values.append(v)
    cpm = np.where(np.isin(genes, low_cpm), 5.0, 100.0)
    return _Table(kind="scrna", genes=list(genes), units=pd.DataFrame(rows),
                  values=np.array(values), mean_cpm=cpm, composition=None)


def bulk_table(shift: float, *, n: int = 12, seed: int = 0) -> _Table:
    """``n`` TB and ``n`` HC samples, per-gene rank-to-normal within the site. On PROGRAM the TB
    samples copy the HC samples' raw values plus ``shift``, so shift 0 is an exact null."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0, 1, (2 * n, len(GENES)))
    raw[:n, :len(PROGRAM)] = raw[n:, :len(PROGRAM)] + shift
    values = stats.norm.ppf((stats.rankdata(raw, axis=0) - 0.5) / (2 * n))
    units = pd.DataFrame({"unit": [f"s{i}" for i in range(2 * n)],
                          "donor": [f"s{i}" for i in range(2 * n)],
                          "group": ["TB"] * n + ["HC"] * n, "population": "all", "n_cells": 0})
    return _Table(kind="bulk", genes=list(GENES), units=units, values=values, mean_cpm=None,
                  composition=None)


def claim(up=PROGRAM, down=(), *, kind="de", population=POP, contrast=("treated", "control"),
          effect=None, cid="c1") -> Claim:
    return Claim(id=cid, site="A", kind=kind, population=population, target=contrast[0],
                 reference=contrast[1], up_genes=tuple(up), down_genes=tuple(down),
                 effect=effect, n_units=None, rank=1)


def planted(s: float, groups=("treated",), genes=PROGRAM) -> dict[str, dict[str, float]]:
    return {g: {gene: s for gene in genes} for g in groups}


# ---------------------------------------------------------------------------
# scRNA de
# ---------------------------------------------------------------------------

def test_planted_program_is_supported():
    r = test_claim(scrna_table(planted(1.0)), claim(), P, site="B")
    assert (r.status, r.reason, r.test, r.site, r.claim_id) == (
        "supported", "module", "replication", "B", "c1")
    assert r.competitive_ok is True and r.ci95[0] > 0 and r.p <= P.q
    assert abs(r.effect - 1.0) < 0.15 and r.n_per_group == (4, 4) and 0 < r.mde < 1.0


def test_a_two_direction_claim_holds_only_if_each_direction_holds():
    """Intersection-union: each half is its own module; the claim reports its weaker half."""
    up, down = PROGRAM[:3], PROGRAM[3:]
    shift = {"treated": {**{g: 1.0 for g in up}, **{g: -0.5 for g in down}}}
    table = scrna_table(shift)
    r = test_claim(table, claim(up=up, down=down), P, site="B")
    halves = [test_claim(table, claim(up=up), P, site="B"),
              test_claim(table, claim(up=(), down=down), P, site="B")]
    assert r.status == "supported" and all(h.status == "supported" for h in halves)
    assert r.p == max(h.p for h in halves) and abs(r.effect - 0.5) < 0.15
    # genes that do not move cannot ride on a half that does (v4 smoke, fsdev2-810)
    hitchhike = test_claim(scrna_table(planted(1.0, genes=up)), claim(up=up, down=down), P,
                           site="B")
    assert hitchhike.status in ("absent", "inconclusive")
    assert test_claim(scrna_table(planted(1.0, genes=up) | {"treated": {
        **{g: 1.0 for g in up}, **{g: 1.0 for g in down}}}), claim(up=up, down=down), P,
        site="B").status == "contradicted"          # the down half moves up


def test_a_two_direction_claim_needs_enough_genes_in_each_direction():
    r = test_claim(scrna_table(planted(1.0)), claim(up=PROGRAM[:5], down=PROGRAM[5:]), P,
                   site="B")
    assert (r.status, r.reason) == ("untestable", "genes")


def test_reversed_program_is_contradicted():
    r = test_claim(scrna_table(planted(-1.0)), claim(), P, site="B")
    assert r.status == "contradicted" and r.ci95[1] < 0 and r.competitive_ok is False


def test_null_program_is_not_called():
    r = test_claim(scrna_table(), claim(), P, site="B")
    assert r.status in ("absent", "inconclusive")


def test_global_shift_is_not_supported_thanks_to_competitive_null():
    """A compositional artefact moves every gene (here by 0.5..1.5); the claim's genes move by
    1.0, clearly non-zero, but no more than random gene sets do."""
    shift = dict(zip(GENES, np.linspace(0.5, 1.5, len(GENES))))
    shift.update({g: 1.0 for g in PROGRAM})
    r = test_claim(scrna_table({"treated": shift}), claim(), P, site="B")
    assert r.ci95[0] > 0 and r.p <= P.q          # a self-contained test would call it
    assert r.competitive_ok is False and r.status == "inconclusive"


def test_too_few_units_is_inconclusive_even_with_a_planted_effect():
    """Review focus 2: donors below min_cells are dropped; < min_units per group -> inconclusive."""
    small = {"treated0": 10, "treated1": 19}
    r = test_claim(scrna_table(planted(1.0), n_cells=small), claim(), P, site="B")
    assert (r.status, r.reason, r.n_per_group, r.effect, r.p) == (
        "inconclusive", "units", (2, 4), None, None)
    r = test_claim(scrna_table(planted(1.0), donors=2), claim(), P, site="B")
    assert (r.status, r.reason, r.n_per_group) == ("inconclusive", "units", (2, 2))


def test_partial_gene_coverage_is_untestable():
    """Review focus 2: 2 of 6 claim genes measurable (2 low-CPM, 2 absent) -> untestable(genes);
    3 of 6 meets max(min_genes, 50%) and is tested."""
    table = scrna_table(planted(1.0), genes=GENES + ["LOWA", "LOWB"], low_cpm=("LOWA", "LOWB"))
    r = test_claim(table, claim(up=PROGRAM[:2] + ["LOWA", "LOWB", "MISS1", "MISS2"]), P,
                   site="B")
    assert (r.status, r.reason, r.effect, r.p) == ("untestable", "genes", None, None)
    r = test_claim(table, claim(up=PROGRAM[:3] + ["LOWA", "LOWB", "MISS1"]), P, site="B")
    assert r.status == "supported"


def test_missing_groups_or_population_is_untestable():
    table = scrna_table(planted(1.0))
    r = test_claim(table, claim(contrast=("mimic", "control")), P, site="B")
    assert (r.status, r.reason) == ("untestable", "groups")
    r = test_claim(table, claim(population="NK"), P, site="B")
    assert (r.status, r.reason) == ("untestable", "population")


def test_scope_contrast_overrides_the_claims_contrast():
    groups = ("control", "treated", "mimic")
    specific = scrna_table(planted(1.0), groups=groups)
    r = test_claim(specific, claim(), P, site="M", test="scope", contrast=("mimic", "control"))
    assert (r.test, r.status) == ("scope", "absent")
    shared = scrna_table(planted(1.0, groups=("treated", "mimic")), groups=groups)
    r = test_claim(shared, claim(), P, site="M", test="scope", contrast=("mimic", "control"))
    assert (r.test, r.status) == ("scope", "supported")


# ---------------------------------------------------------------------------
# scRNA composition
# ---------------------------------------------------------------------------

def with_composition(fractions: dict[str, float], *, seed: int = 0) -> _Table:
    """Four donors per group; ``POP`` fraction per group plus small noise, NK is the rest."""
    rng = np.random.default_rng(seed)
    table = scrna_table(groups=tuple(fractions))
    rows = []
    for g, f in fractions.items():
        for k in range(4):
            x = f + rng.normal(0, 0.01)
            rows.append({"donor": f"{g}{k}", POP: x, "NK": 1 - x, "pDC": 0.0, "group": g})
    return replace(table, composition=pd.DataFrame(rows).set_index("donor"))


def test_composition_shift_is_signed_by_the_claimed_direction():
    table = with_composition({"control": 0.15, "treated": 0.30})
    up = test_claim(table, claim((), kind="composition", effect=1.0), P, site="B")
    assert (up.status, up.reason, up.competitive_ok) == ("supported", "logit", None)
    assert up.ci95[0] > 0 and up.n_per_group == (4, 4)
    down = test_claim(table, claim((), kind="composition", effect=-0.5), P, site="B")
    assert down.status == "contradicted" and down.effect == pytest.approx(-up.effect)


def test_composition_without_change_is_absent():
    table = with_composition({"control": 0.2, "treated": 0.2})
    r = test_claim(table, claim((), kind="composition", effect=1.0), P, site="B")
    assert r.status == "absent"


def test_composition_untestable_or_degenerate():
    table = with_composition({"control": 0.15, "treated": 0.30})
    r = test_claim(table, claim((), kind="composition", population="B", effect=1.0), P, site="B")
    assert (r.status, r.reason) == ("untestable", "population")
    r = test_claim(table, claim((), kind="composition", population="pDC", effect=1.0), P,
                   site="B")
    assert (r.status, r.reason, r.p) == ("inconclusive", "variance", None)
    r = test_claim(bulk_table(0.0), claim((), kind="composition", contrast=("TB", "HC"),
                                          effect=1.0), P, site="B")
    assert (r.status, r.reason) == ("untestable", "population")


# ---------------------------------------------------------------------------
# bulk de
# ---------------------------------------------------------------------------

def bulk_claim(up=PROGRAM) -> Claim:
    return claim(up, population="whole blood", contrast=("TB", "HC"))


def test_bulk_auc_supported_contradicted_and_null():
    r = test_claim(bulk_table(0.8), bulk_claim(), P, site="S2")
    assert (r.status, r.reason, r.competitive_ok) == ("supported", "auc", None)
    assert r.effect > 0.3 and r.ci95[0] > 0 and r.p <= P.q and r.n_per_group == (12, 12)
    assert test_claim(bulk_table(-0.8), bulk_claim(), P, site="S2").status == "contradicted"
    null = test_claim(bulk_table(0.0), bulk_claim(), P, site="S2")
    assert (null.status, null.effect) == ("inconclusive", 0.0)   # 12 v 12 cannot show absence


def test_bulk_partial_gene_coverage_is_untestable():
    r = test_claim(bulk_table(0.8), bulk_claim(PROGRAM[:2] + ["X1", "X2", "X3", "X4"]), P,
                   site="S2")
    assert (r.status, r.reason) == ("untestable", "genes")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_inputs_give_identical_results():
    for make, c in ((lambda: scrna_table({"treated": dict.fromkeys(GENES, 0.5)}), claim()),
                    (lambda: bulk_table(0.3), bulk_claim())):
        first, second = test_claim(make(), c, P, site="B"), test_claim(make(), c, P, site="B")
        assert first == second
    # the bootstrap is seeded per (claim, site): another site draws other resamples
    other = test_claim(bulk_table(0.3), bulk_claim(), P, site="C")
    assert other.ci95 != test_claim(bulk_table(0.3), bulk_claim(), P, site="B").ci95


# ---------------------------------------------------------------------------
# BH family and per-gene export (MA-genome input)
# ---------------------------------------------------------------------------

def result(status: str, p: float | None, ci: tuple[float, float] | None,
           reason: str = "module", cid: str = "c") -> TestResult:
    return TestResult(cid, "B", "replication", status, None if ci is None else sum(ci) / 2,
                      0.1, ci, (4, 4), 0.3, p, None, reason)


def test_apply_bh_demotes_marginal_results_and_keeps_raw_p():
    results = [result("supported", 0.001, (0.5, 1.5), cid="strong"),
               result("supported", 0.03, (0.05, 0.95), cid="wide"),
               result("supported", 0.04, (0.02, 0.20), cid="small"),
               result("contradicted", 0.02, (-0.90, -0.05), cid="against"),
               result("inconclusive", 0.5, (-0.6, 0.9)),
               result("inconclusive", 0.8, (-0.7, 0.8)),
               result("untestable", None, None, reason="genes", cid="none"),
               result("supported", 0.03, (0.01, 0.10), reason="", cid="no-band")]
    out = apply_bh(results, P)          # adjusted p: 0.007, then 0.0525..0.056 > q, 0.58, 0.8
    assert [r.status for r in out] == ["supported", "inconclusive", "absent", "inconclusive",
                                       "inconclusive", "inconclusive", "untestable",
                                       "inconclusive"]
    assert [r.p for r in out] == [r.p for r in results]
    assert apply_bh(results[:1], P) == results[:1]


def test_export_gene_stats_uses_the_testers_filters():
    table = scrna_table(planted(1.0), genes=GENES + ["LOWA"], low_cpm=("LOWA",),
                        n_cells={"treated0": 5})
    df = export_gene_stats(table, POP, ("treated", "control"), P)
    assert list(df.columns) == ["gene", "effect", "se", "n1", "n2"]
    assert "LOWA" not in set(df["gene"]) and len(df) == len(GENES)
    g0 = df.set_index("gene").loc["G0"]
    assert abs(g0["effect"] - 1.0) < 0.2 and g0["se"] > 0 and (g0["n1"], g0["n2"]) == (3, 4)
    assert export_gene_stats(table, POP, ("mimic", "control"), P).empty
    assert export_gene_stats(table, "NK", ("treated", "control"), P).empty
    bulk = export_gene_stats(bulk_table(0.8), "anything", ("TB", "HC"), P)
    assert len(bulk) == len(GENES) and set(bulk["n1"]) == {12}
