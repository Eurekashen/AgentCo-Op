"""Case 2 (TB) held-out validation (spec §5.2): the only writer and reader of the TB references.

    python -m agentcoop.eval.tb          # build data/hidden/tb/ and print the manifest ($0, once)

Confirmatory cohorts, none held by a site (exclusion table below): GSE73408 (US, Affymetrix
HuGene 1.1 ST, log2; TB / LTBI / pneumonia), GSE89403 (South Africa, RNA-seq raw counts per
sample code - lanes already merged; diagnosis visit only; TB vs healthy; "Lung Dx Controls" kept
as other lung disease, exploratory), GSE101705 (India, RNA-seq normalized counts; TB vs LTBI).
Each reference is stored as symbols x samples after a per-gene van der Waerden rank-to-normal
transform, with a sample -> group table; a claim (or signature) is scored as mean(up) - mean(down)
over its genes present in the cohort, and its AUC (TB = positive) gets a bootstrap 95% CI. The
scoring rule is frozen here and is also the scope rule's convention (AUC thresholds 0.65 / 0.70).

Exclusion table (shared patients / consortia; spec §5.2): GSE19442 (subset of S1's SuperSeries)
and GSE107991/2 (same Berry/O'Garra programme as S1 and S4) are never used; GSE39940 (same
consortium as S2) is exploratory only.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from agentcoop.cases.tb import RAW, read_series
from agentcoop.coop.claims import Claim

ROOT = Path(__file__).resolve().parents[2]
HIDDEN = ROOT / "data" / "hidden" / "tb"
MANIFEST = Path(__file__).with_name("tb_reference.json")    # committed before any TB run
MIN_GENES = 3
N_BOOT = 2000
SPECIFIC_AUC, NOT_SPECIFIC_AUC = 0.70, 0.65          # = cases.tb (a priori conventions)
PRIMARY = ("GSE73408", ("TB", "PNA"))
CONFIRMATORY = {"GSE73408": (("TB", "PNA"), ("TB", "LTBI")),
                "GSE89403": (("TB", "HC"), ("TB", "OD")),
                "GSE101705": (("TB", "LTBI"),)}


# ---------------------------------------------------------------------------
# Building the references
# ---------------------------------------------------------------------------

def _annot(platform: str) -> dict[str, str]:
    """Probe -> symbol from a GEO annot table; probes mapping to several symbols are dropped."""
    with gzip.open(RAW / "gpl" / f"{platform}.annot.gz", "rt", encoding="utf-8",
                   errors="replace") as f:
        lines = [line for line in f if not line.startswith(("!", "#", "^"))]
    table = pd.read_csv(io.StringIO("".join(lines)), sep="\t", dtype=str, keep_default_na=False)
    return {i: s for i, s in zip(table["ID"], table["Gene symbol"]) if s and "///" not in s}


def _collapse(expr: pd.DataFrame, symbol_of: dict[str, str]) -> pd.DataFrame:
    """One row per symbol: the probe with the largest IQR (the testers' frozen rule)."""
    x = expr.loc[expr.index.isin(list(symbol_of))].apply(pd.to_numeric, errors="coerce").dropna()
    iqr = x.quantile(0.75, axis=1) - x.quantile(0.25, axis=1)
    order = pd.DataFrame({"symbol": x.index.map(symbol_of), "iqr": iqr.to_numpy(),
                          "probe": x.index}).sort_values(["symbol", "iqr", "probe"],
                                                         ascending=[True, False, True])
    keep = order.drop_duplicates("symbol")
    return x.loc[keep["probe"]].set_axis(keep["symbol"].to_numpy(), axis=0)


def rank_normal(x: pd.DataFrame) -> pd.DataFrame:
    """Per-gene van der Waerden scores across the cohort's samples (genes x samples)."""
    ranks = stats.rankdata(x.to_numpy(dtype=float), axis=1)
    return pd.DataFrame(stats.norm.ppf(ranks / (x.shape[1] + 1)), index=x.index, columns=x.columns)


def _log_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    counts = counts.groupby(level=0).sum()
    counts = counts.loc[counts.sum(axis=1) > 0]
    return np.log2(counts / counts.sum(axis=0) * 1e6 + 1)


def cohort_gse73408() -> tuple[pd.DataFrame, pd.Series]:
    meta, expr = read_series(RAW / "geo" / "GSE73408_series_matrix.txt.gz")
    group = meta["clinical group"].map({"TB": "TB", "LTBI": "LTBI", "PNA": "PNA"}).dropna()
    return _collapse(expr[group.index], _annot("GPL11532")), group


def cohort_gse89403() -> tuple[pd.DataFrame, pd.Series]:
    meta, _ = read_series(RAW / "geo" / "GSE89403_series_matrix.txt.gz")
    baseline = meta[meta["time"] == "DX"].drop_duplicates("sample_code")
    group = (baseline.set_index("sample_code")["disease state"]
             .map({"TB Subjects": "TB", "Healthy Controls": "HC", "Lung Dx Controls": "OD"})
             .dropna())
    counts = pd.read_csv(RAW / "suppl" / "GSE89403_rawCounts_GeneNames_AllSamples.csv.gz")
    counts = counts.drop(columns=counts.columns[0]).set_index("symbol")
    group = group[group.index.isin(counts.columns)]
    return _log_cpm(counts[group.index]), group


def cohort_gse101705() -> tuple[pd.DataFrame, pd.Series]:
    meta, _ = read_series(RAW / "geo" / "GSE101705_series_matrix.txt.gz")
    group = (meta.set_index("title")["condition"]
             .map({"TB": "TB", "latent TB infection": "LTBI"}).dropna())
    counts = pd.read_excel(RAW / "suppl" / "GSE101705_normalizedCounts.xlsx")
    counts = counts.set_index(counts.columns[0])
    group = group[group.index.isin(counts.columns)]
    return np.log2(counts[group.index].groupby(level=0).sum() + 1), group


COHORTS = {"GSE73408": cohort_gse73408, "GSE89403": cohort_gse89403,
           "GSE101705": cohort_gse101705}


def build(hidden: Path = HIDDEN) -> dict:
    """Write each reference (rank-normal matrix + groups) and the committed hash manifest."""
    hidden.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, load in COHORTS.items():
        expr, group = load()
        expr = expr.loc[expr.var(axis=1) > 0]
        values = rank_normal(expr)
        path = hidden / f"{name}.tsv.gz"
        values.round(5).to_csv(path, sep="\t", compression={"method": "gzip", "mtime": 0})
        (hidden / f"{name}_groups.tsv").write_text(group.rename("group").to_csv(sep="\t"))
        manifest[name] = {"genes": int(values.shape[0]),
                          "groups": {g: int(n) for g, n in group.value_counts().sort_index().items()},
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "groups_sha256": hashlib.sha256(
                              (hidden / f"{name}_groups.tsv").read_bytes()).hexdigest()}
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


# ---------------------------------------------------------------------------
# Headroom targets (§6): expansion sites outside the minimal federation
# ---------------------------------------------------------------------------

S5_GROUPS = {"TB": "TB", "Active sarcoidosis": "OD", "Active Sarcoid": "OD", "Pneumonia": "OD",
             "lung cancer": "OD", "Control": "HC"}


def target_sites() -> dict[str, tuple[pd.DataFrame, pd.Series, str]]:
    """{name: (expression: native probe x sample, sample -> case group, platform)} of the gate's
    targets: S3 Malawi (GSE37250 region Malawi) and S5 Bloom (GSE42825/26/30 joined on shared
    probes, exact duplicate columns removed; OD = active sarcoidosis, pneumonia, lung cancer)."""
    meta, expr = read_series(RAW / "geo" / "GSE37250_series_matrix.txt.gz")
    malawi = meta[meta["geographical region"] == "Malawi"]["disease state"].map(
        {"active tuberculosis": "TB", "latent TB infection": "LTBI", "other disease": "OD"})
    out = {"S3": (expr[malawi.index], malawi, "GPL10558")}
    parts = [read_series(RAW / "geo" / f"{g}_series_matrix.txt.gz")
             for g in ("GSE42825", "GSE42826", "GSE42830")]
    meta = pd.concat([m for m, _ in parts])
    expr = pd.concat([e for _, e in parts], axis=1, join="inner")
    expr = expr.loc[:, ~expr.T.duplicated()]
    group = meta.loc[expr.columns, "disease state"].map(S5_GROUPS).dropna()
    out["S5"] = (expr[group.index], group, "GPL10558")
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def load(name: str, hidden: Path = HIDDEN) -> tuple[pd.DataFrame, pd.Series]:
    """A reference, checked against the committed manifest."""
    path = hidden / f"{name}.tsv.gz"
    expected = json.loads(MANIFEST.read_text())[name]["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"{path} differs from the committed reference hash")
    values = pd.read_csv(path, sep="\t", index_col=0)
    group = pd.read_csv(hidden / f"{name}_groups.tsv", sep="\t", index_col=0)["group"]
    return values, group


def signature_auc(values: pd.DataFrame, group: pd.Series, up, down,
                  contrast: tuple[str, str], *, seed: int = 0
                  ) -> tuple[float, float, float] | None:
    """(AUC, CI low, CI high) of score = mean(up) - mean(down) for target vs reference; None when
    fewer than MIN_GENES of the genes are in the cohort or a group is missing."""
    up = [g for g in up if g in values.index]
    down = [g for g in down if g in values.index]
    if len(up) + len(down) < MIN_GENES:
        return None
    score = (values.loc[up].mean(axis=0) if up else 0) - (values.loc[down].mean(axis=0) if down else 0)
    a = score[group[group == contrast[0]].index].to_numpy()
    b = score[group[group == contrast[1]].index].to_numpy()
    if not len(a) or not len(b):
        return None
    auc = float(stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)))
    rng = np.random.default_rng(seed)
    boots = [stats.mannwhitneyu(rng.choice(a, len(a)), rng.choice(b, len(b))).statistic
             / (len(a) * len(b)) for _ in range(N_BOOT)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return auc, float(lo), float(hi)


def claim_auc(claim: Claim, name: str, contrast: tuple[str, str], hidden: Path = HIDDEN):
    """A claim's held-out AUC in its claimed direction."""
    values, group = load(name, hidden)
    return signature_auc(values, group, claim.up_genes, claim.down_genes, contrast)


def scope_correct(scope: str, auc: tuple[float, float, float] | None) -> bool | None:
    """Endpoint (a): a `specific` label is right iff held-out AUC >= 0.70, `not specific` iff
    < 0.65; AUCs in [0.65, 0.70), inconclusive labels and missing AUCs are excluded (None)."""
    if auc is None or scope == "inconclusive":
        return None
    if NOT_SPECIFIC_AUC <= auc[0] < SPECIFIC_AUC:
        return None
    return (auc[0] >= SPECIFIC_AUC) == scope.startswith("specific")


def final_signature(findings: list[Claim], k: int = 5) -> tuple[list[str], list[str]]:
    """Endpoint (b)'s fixed rule: up / down genes of the top-k findings by rank."""
    top = sorted(findings, key=lambda c: (c.rank, c.id))[:k]
    up = sorted({g for c in top for g in c.up_genes})
    down = sorted({g for c in top for g in c.down_genes} - set(up))
    return up, down


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(build(), indent=1))
