"""Deterministic site testers: does claim c hold at site s? (spec §2.4)

Testers run inside the site boundary on the site table and return summary statistics only. A
status is read off a 95% confidence interval and an equivalence band, so a site too small to
measure an effect says ``inconclusive``, not ``absent``. ``inconclusive`` and ``untestable``
are never support and never contradiction.

``TestResult.reason`` names the statistic of a computed result (``module``, ``logit``, ``auc``)
and the obstacle otherwise (``groups``, ``population``, ``genes``, ``units``, ``variance``);
``apply_bh`` reads it to pick the equivalence band. Random draws (competitive null, bootstrap)
are seeded by (claim id, site), so testing the same claim at the same site again is an exact
replay.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

from agentcoop.coop.claims import Claim
from agentcoop.coop.view import SiteTable


@dataclass(frozen=True)
class TesterParams:
    __test__ = False  # not a pytest class (name starts with "Test")
    delta_de: float = 0.25          # equivalence band, log2 module-score units (frozen on fsdev)
    delta_comp: float = 0.5         # logit units
    delta_auc: float = 0.05
    q: float = 0.05
    min_cells: int = 20
    min_units: int = 3              # per group
    min_genes: int = 3
    min_gene_frac: float = 0.5
    min_cpm: float = 20.0
    null_sets: int = 200
    null_quantile: float = 0.95
    n_boot: int = 1000


@dataclass(frozen=True)
class TestResult:
    __test__ = False  # not a pytest class (name starts with "Test")
    claim_id: str
    site: str
    test: str                       # "replication" | "scope"
    status: str                     # "supported" | "contradicted" | "absent" | "inconclusive" | "untestable"
    effect: float | None
    se: float | None
    ci95: tuple[float, float] | None
    n_per_group: tuple[int, int] | None
    mde: float | None
    p: float | None                 # raw two-sided p; BH applied by apply_bh
    competitive_ok: bool | None
    reason: str


def _seed(claim_id: str, site: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{claim_id}\x00{site}".encode()).digest()[:8], "big")


def _blocked(claim: Claim, site: str, test: str, status: str, reason: str,
             n: tuple[int, int] | None = None) -> TestResult:
    """A result without a statistic: the site could not test the claim, or not reliably."""
    return TestResult(claim.id, site, test, status, None, None, None, n, None, None, None, reason)


def _within(ci: tuple[float, float], band: float) -> bool:
    return -band < ci[0] and ci[1] < band


def _status(ci: tuple[float, float], p: float, q: float, band: float, *,
            up_ok: bool = True, down_ok: bool = True) -> str:
    """First match: supported, contradicted, absent (CI inside the band), inconclusive."""
    if ci[0] > 0 and p <= q and up_ok:
        return "supported"
    if ci[1] < 0 and p <= q and down_ok:
        return "contradicted"
    if _within(ci, band):
        return "absent"
    return "inconclusive"


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Difference of means, its standard error and the Welch–Satterthwaite df."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    diff = float(a.mean() - b.mean())
    if va + vb == 0:
        return diff, 0.0, math.nan
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return diff, float(math.sqrt(va + vb)), float(df)


def _t_inference(diff: float, se: float, df: float) -> tuple[tuple[float, float], float, float]:
    """95% CI, two-sided p, and the effect detectable with 80% power at alpha 0.05."""
    t975 = float(stats.t.ppf(0.975, df))
    p = float(2 * stats.t.sf(abs(diff) / se, df))
    mde = (t975 + float(stats.t.ppf(0.8, df))) * se
    return (diff - t975 * se, diff + t975 * se), p, mde


def _unit_rows(table: SiteTable, population: str, target: str, reference: str,
               params: TesterParams) -> tuple[np.ndarray, np.ndarray]:
    """Rows of the target and reference units: in scRNA the population's donors with at least
    ``min_cells`` cells (smaller pseudobulks are noise); in bulk every sample."""
    u = table.units
    keep = np.ones(len(u), bool) if table.kind == "bulk" else (
        u["population"].eq(population) & u["n_cells"].ge(params.min_cells)).to_numpy()
    group = u["group"].to_numpy()
    return np.flatnonzero(keep & (group == target)), np.flatnonzero(keep & (group == reference))


def _measurable(table: SiteTable, population: str, params: TesterParams) -> np.ndarray:
    """Genes the site can measure: in scRNA mean CPM >= ``min_cpm`` in the population (sparse
    genes make noisy module scores); in bulk every gene (values are within-site ranks)."""
    if table.kind == "bulk":
        return np.ones(len(table.genes), bool)
    return table.population_mean_cpm(population) >= params.min_cpm


def _claim_genes(table: SiteTable, claim: Claim, measurable: np.ndarray,
                 params: TesterParams) -> tuple[list[int], list[int]] | None:
    """Columns of the claim's measurable up and down genes; None when fewer than
    max(min_genes, min_gene_frac of the claim's genes) are measurable — a site that sees a
    fragment of a program cannot say whether the program holds there."""
    col = {g: i for i, g in enumerate(table.genes)}
    up = [col[g] for g in claim.up_genes if g in col and measurable[col[g]]]
    down = [col[g] for g in claim.down_genes if g in col and measurable[col[g]]]
    n_claim = len(claim.up_genes) + len(claim.down_genes)
    if len(up) + len(down) < max(params.min_genes, math.ceil(params.min_gene_frac * n_claim)):
        return None
    return up, down


def _score(values: np.ndarray, up: list[int] | np.ndarray,
           down: list[int] | np.ndarray) -> np.ndarray:
    """Per-row mean of the up genes minus mean of the down genes; an empty side contributes 0."""
    s = np.zeros(values.shape[0])
    if len(up):
        s += values[:, up].mean(axis=1)
    if len(down):
        s -= values[:, down].mean(axis=1)
    return s


def _null_effects(gene_diff: np.ndarray, pool: np.ndarray, n_up: int, n_down: int,
                  params: TesterParams, seed: int) -> np.ndarray:
    """Module effects of random measurable gene sets with the claim's up/down sizes. The module
    effect is linear in the per-gene mean differences, so a set's effect is computed from them."""
    rng = np.random.default_rng(seed)
    null = np.empty(params.null_sets)
    for j in range(params.null_sets):
        pick = rng.choice(pool, size=n_up + n_down, replace=False)
        null[j] = _score(gene_diff[None, :], pick[:n_up], pick[n_up:])[0]
    return null


def _module_result(claim: Claim, site: str, test: str, table: SiteTable, up: list[int],
                   down: list[int], measurable: np.ndarray, t: np.ndarray, r: np.ndarray,
                   params: TesterParams) -> TestResult:
    """scRNA de: Welch test on donor module scores plus the competitive null — a global or
    compositional shift moves every gene, so the claim's genes must move more than random sets."""
    n = (len(t), len(r))
    score = _score(table.values, up, down)
    diff, se, df = _welch(score[t], score[r])
    if not se > 0:
        return _blocked(claim, site, test, "inconclusive", "variance", n)
    ci, p, mde = _t_inference(diff, se, df)
    gene_diff = table.values[t].mean(axis=0) - table.values[r].mean(axis=0)
    null = _null_effects(gene_diff, np.flatnonzero(measurable), len(up), len(down), params,
                         _seed(claim.id, site))
    hi, lo = np.quantile(null, [params.null_quantile, 1 - params.null_quantile])
    status = _status(ci, p, params.q, params.delta_de, up_ok=diff > hi, down_ok=diff < lo)
    return TestResult(claim.id, site, test, status, diff, se, ci, n, mde, p, bool(diff > hi),
                      "module")


def _boot_auc(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """AUCs of resamples drawn with replacement within each group; computed from ranks so memory
    stays O(n_boot * n) for cohorts of hundreds of samples."""
    rng = np.random.default_rng(seed)
    xa = a[rng.integers(0, len(a), (n_boot, len(a)))]
    xb = b[rng.integers(0, len(b), (n_boot, len(b)))]
    ranks = stats.rankdata(np.hstack([xa, xb]), axis=1)
    u = ranks[:, :len(a)].sum(axis=1) - len(a) * (len(a) + 1) / 2
    return u / (len(a) * len(b))


def _auc_result(claim: Claim, site: str, test: str, a: np.ndarray, b: np.ndarray,
                params: TesterParams) -> TestResult:
    """bulk de: effect = AUC − 0.5 of sample scores (Mann–Whitney), bootstrap 95% CI."""
    n = (len(a), len(b))
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    effect = float(mw.statistic / (n[0] * n[1]) - 0.5)
    boot = _boot_auc(a, b, params.n_boot, _seed(claim.id, site)) - 0.5
    lo, hi = (float(x) for x in np.quantile(boot, [0.025, 0.975]))
    se = float(boot.std(ddof=1))
    mde = float((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se)
    p = float(mw.pvalue)
    status = _status((lo, hi), p, params.q, params.delta_auc)
    return TestResult(claim.id, site, test, status, effect, se, (lo, hi), n, mde, p, None, "auc")


def _test_de(table: SiteTable, claim: Claim, params: TesterParams, site: str, test: str,
             target: str, reference: str) -> TestResult:
    """Bulk sites hold one population (the whole sample), so the claim's population is matched
    only in scRNA."""
    if table.kind != "bulk" and claim.population not in set(table.units["population"]):
        return _blocked(claim, site, test, "untestable", "population")
    measurable = _measurable(table, claim.population, params)
    genes = _claim_genes(table, claim, measurable, params)
    if genes is None:
        return _blocked(claim, site, test, "untestable", "genes")
    up, down = genes
    signed = bool(claim.up_genes) and bool(claim.down_genes)
    if signed and min(len(up), len(down)) < params.min_genes:
        return _blocked(claim, site, test, "untestable", "genes")
    t, r = _unit_rows(table, claim.population, target, reference, params)
    if min(len(t), len(r)) < params.min_units:
        return _blocked(claim, site, test, "inconclusive", "units", (len(t), len(r)))

    def one(u: list[int], d: list[int]) -> TestResult:
        if table.kind == "bulk":
            score = _score(table.values, u, d)
            return _auc_result(claim, site, test, score[t], score[r], params)
        return _module_result(claim, site, test, table, u, d, measurable, t, r, params)

    # A claim with both directions asserts both: each is tested on its own (intersection-union),
    # so genes that do not move cannot ride on a half that does.
    return _weakest([one(up, []), one([], down)]) if signed else one(up, down)


#: Which half of a two-direction claim speaks for it: the first status in this order.
_WEAKEST = ("contradicted", "absent", "inconclusive", "supported")


def _weakest(halves: list[TestResult]) -> TestResult:
    """The claim's result from its halves' results: contradicted if either half is, absent if
    either half is, supported only if both are; within a status the larger p (the
    intersection-union p, so BH stays valid on it)."""
    return min(halves, key=lambda h: (_WEAKEST.index(h.status), -(1.0 if h.p is None else h.p)))


def _test_composition(table: SiteTable, claim: Claim, params: TesterParams, site: str,
                      test: str, target: str, reference: str) -> TestResult:
    """Donor-level logit cell fraction, Welch t, signed so that a positive effect is the claimed
    direction (sign of ``claim.effect``; a missing effect reads as an increase)."""
    comp = table.composition
    if comp is None or claim.population not in comp.columns:
        return _blocked(claim, site, test, "untestable", "population")
    group = comp["group"].to_numpy()
    frac = np.clip(comp[claim.population].to_numpy(dtype=float), 1e-3, 1 - 1e-3)
    logit = np.log(frac / (1 - frac)) * (-1.0 if (claim.effect or 0.0) < 0 else 1.0)
    a, b = logit[group == target], logit[group == reference]
    n = (len(a), len(b))
    if min(n) < params.min_units:
        return _blocked(claim, site, test, "inconclusive", "units", n)
    diff, se, df = _welch(a, b)
    if not se > 0:
        return _blocked(claim, site, test, "inconclusive", "variance", n)
    ci, p, mde = _t_inference(diff, se, df)
    return TestResult(claim.id, site, test, _status(ci, p, params.q, params.delta_comp), diff,
                      se, ci, n, mde, p, None, "logit")


def test_claim(table: SiteTable, claim: Claim, params: TesterParams, *, site: str,
               test: str = "replication", contrast: tuple[str, str] | None = None) -> TestResult:
    """Test ``claim`` on one site's table. ``contrast`` replaces the claim's own (scope tests,
    e.g. mimic vs control) and keeps the claimed direction. Status is provisional on the raw p;
    ``apply_bh`` demotes it within the run's BH family."""
    target, reference = contrast or claim.contrast
    present = set(table.units["group"])
    if target not in present or reference not in present:
        return _blocked(claim, site, test, "untestable", "groups")
    if claim.kind == "composition":
        return _test_composition(table, claim, params, site, test, target, reference)
    return _test_de(table, claim, params, site, test, target, reference)


test_claim.__test__ = False  # imported by test modules; not a pytest test


#: Equivalence band per statistic (the ``reason`` of a computed result).
_BANDS = {"module": "delta_de", "logit": "delta_comp", "auc": "delta_auc"}
_GENE_STATS = ["gene", "effect", "se", "n1", "n2"]


def apply_bh(results: list[TestResult], params: TesterParams) -> list[TestResult]:
    """Benjamini–Hochberg over the results' raw p (the caller passes one family: every
    non-proposer test of a run). Adjustment only demotes: a supported or contradicted result
    whose adjusted p exceeds q becomes absent (CI inside its band) or inconclusive. ``p`` stays
    raw; results without p are unchanged. A result whose ``reason`` names no statistic (built
    outside this module) has no known band and is demoted to inconclusive."""
    idx = [i for i, r in enumerate(results) if r.p is not None]
    adjusted = dict(zip(idx, stats.false_discovery_control([results[i].p for i in idx])))
    out = []
    for i, r in enumerate(results):
        if r.status in ("supported", "contradicted") and adjusted.get(i, 0.0) > params.q:
            band = getattr(params, _BANDS[r.reason]) if r.reason in _BANDS else None
            r = replace(r, status="absent" if band is not None and _within(r.ci95, band)
                        else "inconclusive")
        out.append(r)
    return out


def export_gene_stats(table: SiteTable, population: str, contrast: tuple[str, str],
                      params: TesterParams) -> pd.DataFrame:
    """Per-gene effect (target − reference mean of the table's values) and Welch SE, for
    MA-genome. Same unit and gene filters as ``test_claim``; empty when the site cannot test the
    contrast; genes with zero variance are dropped (they would get infinite weight)."""
    target, reference = contrast
    if table.kind != "bulk" and population not in set(table.units["population"]):
        return pd.DataFrame(columns=_GENE_STATS)
    t, r = _unit_rows(table, population, target, reference, params)
    if min(len(t), len(r)) < params.min_units:
        return pd.DataFrame(columns=_GENE_STATS)
    keep = _measurable(table, population, params)
    a, b = table.values[t][:, keep], table.values[r][:, keep]
    se = np.sqrt(a.var(axis=0, ddof=1) / len(t) + b.var(axis=0, ddof=1) / len(r))
    out = pd.DataFrame({"gene": np.asarray(table.genes)[keep],
                        "effect": a.mean(axis=0) - b.mean(axis=0), "se": se,
                        "n1": len(t), "n2": len(r)})
    return out[out["se"] > 0].reset_index(drop=True)
