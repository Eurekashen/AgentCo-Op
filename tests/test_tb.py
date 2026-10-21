"""Case 2 (TB) on a toy GEO series matrix: offline, no downloads, no agents."""

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path

import pytest

from agentcoop.cases import tb
from agentcoop.coop.claims import Claim
from agentcoop.coop.tester import TestResult


def _series(path: Path, samples: list[tuple[str, str, list[str]]], probes: list[str]) -> None:
    """A series matrix in GEO's layout: (GSM, title, characteristics "key: value" cells)."""
    n_char = len(samples[0][2])
    lines = ["!Series_title\t\"toy\"",
             "!Sample_title\t" + "\t".join(f'"{t}"' for _, t, _ in samples),
             "!Sample_geo_accession\t" + "\t".join(f'"{g}"' for g, _, _ in samples),
             "!Sample_source_name_ch1\t" + "\t".join('"Whole blood"' for _ in samples)]
    lines += ["!Sample_characteristics_ch1\t" + "\t".join(f'"{c[k]}"' for _, _, c in samples)
              for k in range(n_char)]
    lines.append("!series_matrix_table_begin")
    lines.append('"ID_REF"\t' + "\t".join(f'"{g}"' for g, _, _ in samples))
    lines += [f'"{p}"\t' + "\t".join(str(10 * i + j) for j in range(len(samples)))
              for i, p in enumerate(probes)]
    lines.append("!series_matrix_table_end")
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


@pytest.fixture
def raw(tmp_path):
    (tmp_path / "geo").mkdir()
    (tmp_path / "gpl").mkdir()
    _series(tmp_path / "geo" / "GSE22098_series_matrix.txt.gz", [
        ("GSM1", "WB-Still-1", ["age: 40", "gender: Female", "illness: Still"]),
        ("GSM2", "WB-ctl-2", ["age: 41", "gender: Male", "healthy control: Still"]),
        ("GSM3", "WB-Staph-3", ["age: 5", "gender: Male", "illness: Staph"]),
        ("GSM4", "WB-ctl-4", ["age: 6", "gender: Female", "healthy control: Strep and Staph"]),
        ("GSM5", "WB-PSLE-5", ["age: 12", "gender: Female", "illness: PSLE"]),
    ], ["ILMN_1", "ILMN_0012"])
    with gzip.open(tmp_path / "gpl" / "GPL6947.annot.gz", "wt") as f:
        f.write("ID\tGene symbol\nILMN_1\tSTAT1\nILMN_0012\tGBP5\n")
    return tmp_path


S4 = next(s for s in tb.SITES if s.name == "S4")


def test_read_series_parses_metadata_keys_and_keeps_ids_as_strings(raw):
    meta, expr = tb.read_series(raw / "geo" / "GSE22098_series_matrix.txt.gz")
    assert list(meta.index) == ["GSM1", "GSM2", "GSM3", "GSM4", "GSM5"]
    assert meta.at["GSM1", "illness"] == "Still" and meta.at["GSM2", "healthy control"] == "Still"
    assert meta.at["GSM3", "gender"] == "Male" and meta.at["GSM1", "title"] == "WB-Still-1"
    assert list(expr.index) == ["ILMN_1", "ILMN_0012"] and expr.shape == (2, 5)


def test_site_file_has_opaque_tokens_native_codes_and_whitelisted_columns(raw):
    matrix, samples, card = tb.build_site(S4, raw)
    assert list(samples["condition"]) == ["Still", "HC-Still", "Staph", "HC-Bact"]   # pSLE dropped
    assert list(samples.columns) == ["sample", "condition", "sex", "age"]
    assert list(matrix.columns) == list(samples["sample"])
    text = matrix.to_csv(sep="\t") + samples.to_csv(sep="\t") + json.dumps(card)
    assert "GSM" not in text and "WB-" not in text            # no GEO ids or titles leave
    assert set(card["codes"]) == set(samples["condition"])
    assert all(c in tb.TOOL_GROUPS["S4"] for c in card["codes"])
    assert card["files"]["samples"] == "S4_samples.tsv" and "GPL6947" in card["platform"]


def test_build_writes_sites_with_their_files_and_refuses_to_rebuild(raw, tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "SITES", (S4,))
    task = tb.build(tmp_path / "tasks" / "tb-toy", raw)
    case, sites = tb.load_case(task)
    assert case.kind == "bulk" and case.vocab.populations == ("all",)
    assert [s["name"] for s in sites] == ["S4"] and not sites[0]["proposes"]
    assert set(sites[0]["files"]) == {"S4_samples.tsv", "GPL6947.annot.txt"}
    assert all(p.exists() for p in sites[0]["files"].values())
    assert sites[0]["timeout"] == 1800
    assert all(k.startswith("genesets/") for k in case.resources)
    with pytest.raises(FileExistsError):
        tb.build(task, raw)


# ---------------------------------------------------------------------------
# Scope rule (§5.2)
# ---------------------------------------------------------------------------

CLAIM = Claim(id="S1:1", site="S1", kind="de", population="all", target="TB", reference="LTBI",
              up_genes=("GBP5", "STAT1", "BATF2"), down_genes=(), effect=0.4, n_units=72, rank=1)


def scope(site: str, contrast: tuple[str, str], auc: float, half: float = 0.05) -> TestResult:
    e = auc - 0.5
    return TestResult(claim_id=CLAIM.id, site=site, test="scope", status="supported",
                      effect=e, se=half / 1.96, ci95=(e - half, e + half), n_per_group=(30, 30),
                      mde=None, p=0.001, competitive_ok=None, reason="auc", contrast=contrast)


@pytest.mark.parametrize("results, expected", [
    ([scope("S2", tb.OD, 0.80)], "specific vs mixed_OD"),
    ([scope("S2", tb.OD, 0.60)], "not specific vs mixed_OD"),
    ([scope("S2", tb.OD, 0.66, half=0.1)], "inconclusive"),
    # bacterial mimic as strong as TB-vs-HC at the proposer: demoted even with a good OD test
    ([scope("S2", tb.OD, 0.80), scope("S4", tb.MIMICS["bacterial"], 0.85),
      scope("S1", tb.TB_HC, 0.90)], "not specific vs bacterial"),
    # a mimic clearly weaker than TB-vs-HC does not demote
    ([scope("S2", tb.OD, 0.80), scope("S4", tb.MIMICS["autoimmune"], 0.65),
      scope("S1", tb.TB_HC, 0.90)], "specific vs mixed_OD"),
    # only in-sample (proposer) evidence is never specific, but it may demote
    ([scope("S1", tb.OD, 0.90)], "inconclusive"),
    ([scope("S1", tb.OD, 0.55)], "not specific vs mixed_OD"),
    ([], "inconclusive"),
])
def test_scope_rule(results, expected):
    assert tb.scope_rule(CLAIM, results) == expected


def test_scope_table_routes_od_mimics_and_the_proposers_reference():
    class S:
        def __init__(self, name, held):
            self.name, self.held = name, held

        def holds(self, population, contrast):
            return tuple(contrast) in self.held

    sites = [S("S1", {tb.TB_HC, ("TB", "LTBI")}), S("S2", {tb.OD, ("TB", "LTBI")}),
             S("S4", set(tb.MIMICS.values()))]
    routed = [(s.name, c) for s, c in tb.scope_tests(CLAIM, sites)]
    assert routed == [("S2", tb.OD), ("S4", tb.MIMICS["autoimmune"]),
                      ("S4", tb.MIMICS["bacterial"]), ("S1", tb.TB_HC)]
    own = [(s.name, c) for s, c in tb.scope_tests(replace(CLAIM, site="S2"), sites)]
    assert ("S2", tb.OD) in own                     # the proposer's own OD test (demote-only)
    assert tb.scope_tests(replace(CLAIM, target="LTBI", reference="HC"), sites) == []


# ---------------------------------------------------------------------------
# Held-out scoring (eval/tb.py)
# ---------------------------------------------------------------------------

def test_signature_auc_scores_up_minus_down_and_needs_enough_genes():
    import numpy as np
    import pandas as pd

    from agentcoop.eval import tb as ev
    samples = [f"x{i}" for i in range(20)]
    group = pd.Series(["TB"] * 10 + ["PNA"] * 10, index=samples)
    rng = np.random.default_rng(0)
    values = pd.DataFrame(rng.normal(size=(5, 20)), index=["A", "B", "C", "D", "E"],
                          columns=samples)
    values.loc[["A", "B"], samples[:10]] += 3     # up in TB
    values.loc["C", samples[:10]] -= 3            # down in TB
    auc = ev.signature_auc(values, group, ["A", "B"], ["C"], ("TB", "PNA"))
    assert auc[0] == 1.0 and auc[1] > 0.9
    assert ev.signature_auc(values, group, ["C", "D", "E"], [], ("TB", "PNA"))[0] < 0.5
    assert ev.signature_auc(values, group, ["A", "ZZZ"], [], ("TB", "PNA")) is None
    assert ev.signature_auc(values, group, ["A", "B", "C"], [], ("TB", "HC")) is None


def test_scope_correctness_and_the_final_signature_rule():
    from agentcoop.eval import tb as ev
    assert ev.scope_correct("specific vs mixed_OD", (0.75, 0.6, 0.9)) is True
    assert ev.scope_correct("specific vs mixed_OD", (0.60, 0.5, 0.7)) is False
    assert ev.scope_correct("not specific vs bacterial", (0.60, 0.5, 0.7)) is True
    assert ev.scope_correct("not specific vs bacterial", (0.67, 0.5, 0.8)) is None   # grey zone
    assert ev.scope_correct("inconclusive", (0.9, 0.8, 1.0)) is None
    claims = [replace(CLAIM, id=f"c{r}", rank=r, up_genes=(f"U{r}",), down_genes=(f"D{r}", "U1"))
              for r in (3, 1, 7, 2, 5, 4)]
    up, down = ev.final_signature(claims)
    assert up == ["U1", "U2", "U3", "U4", "U5"] and down == ["D1", "D2", "D3", "D4", "D5"]
