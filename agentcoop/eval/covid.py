"""Case 1 (COVID) held-out validation (spec §5.1): the only writer and reader of the references.

    python -m agentcoop.eval.covid        # build data/hidden/covid/ and its committed manifest ($0)

Held-out cohorts, held by no site: Stephenson 2021 (UK, 10x 3'; CELLxGENE c7775e88) and COMBAT
2022 (Oxford, 10x 5' v1; ebc2e1ff). Crosswalks fixed here before any run: case populations from
Stephenson `author_cell_type` and COMBAT `major_subset`; severity - severe = Stephenson Severe +
Critical (day-0 initial samples), COMBAT COVID_SEV + COVID_CRIT; mild = Stephenson Mild, COMBAT
COVID_MILD + COVID_HCW_MILD; healthy = Stephenson Healthy, COMBAT HV (LPS, non-COVID, sepsis,
flu, asymptomatic, moderate excluded). One sample per donor (the first sample id in sort order).

References are donor x population pseudobulk raw counts (genes by Ensembl id, with symbols).
Replication of a de claim, an operator different from the testers' module Welch test: the
claim's up genes are summed into one module count row (and its down genes into another), tested
by pyDESeq2 (Wald, target vs reference) together with a fixed background of the 2,000 most
expressed genes for size factors and dispersions; a half replicates when its log2 fold change
has the claimed sign, and a claim when every half does; the caller applies BH over an arm's
claims to the returned p (the larger p of the halves). Composition claims: donor-level logit
cell fraction, Welch t, as the testers.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from agentcoop.coop.claims import Claim

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "covid" / "h5ad"
HIDDEN = ROOT / "data" / "hidden" / "covid"
MANIFEST = Path(__file__).with_name("covid_reference.json")
MIN_CELLS = 20                   # = TesterParams.min_cells
BACKGROUND = 2000
CHUNK = 50_000

STEPHENSON_POPULATIONS = {
    "CD14_mono": "CD14 Mono", "CD83_CD14_mono": "CD14 Mono", "CD16_mono": "CD16 Mono",
    "C1_CD16_mono": "CD16 Mono", "Treg": "CD4 T", "pDC": "pDC", "Plasmablast": "Plasmablast",
    **{f"CD4.{k}": "CD4 T" for k in ("CM", "EM", "IL22", "Naive", "Prolif", "Tfh", "Th1",
                                       "Th17", "Th2")},
    **{f"CD8.{k}": "CD8 T" for k in ("EM", "Naive", "Prolif", "TE")},
    **{k: "NK" for k in ("NK_16hi", "NK_56hi", "NK_prolif")},
    **{k: "B" for k in ("B_exhausted", "B_immature", "B_naive", "B_non-switched_memory",
                        "B_switched_memory")},
}
COMBAT_POPULATIONS = {"cMono": "CD14 Mono", "ncMono": "CD16 Mono", "CD4": "CD4 T",
                      "CD8": "CD8 T", "NK": "NK", "B": "B", "PB": "Plasmablast"}
COHORTS = {
    "stephenson": {"file": "heldout_stephenson.h5ad", "sample": "sample_id",
                   "label": "author_cell_type", "populations": STEPHENSON_POPULATIONS,
                   "group": ("Status_on_day_collection_summary",
                             {"Severe": "severe", "Critical": "severe", "Mild": "mild",
                              "Healthy": "healthy"}),
                   "keep": {"Collection_Day": "D0", "Resample": "Initial"}},
    "combat": {"file": "heldout_combat.h5ad", "sample": "scRNASeq_sample_ID",
               "label": "major_subset", "populations": COMBAT_POPULATIONS,
               "group": ("Source", {"COVID_SEV": "severe", "COVID_CRIT": "severe",
                                    "COVID_MILD": "mild", "COVID_HCW_MILD": "mild",
                                    "HV": "healthy"}),
               "keep": {}},
}


# ---------------------------------------------------------------------------
# Building the references
# ---------------------------------------------------------------------------

def _cells(spec: dict, obs: pd.DataFrame) -> pd.DataFrame:
    """Kept cells: (row, donor, group, population); one sample per donor."""
    mask = np.ones(len(obs), bool)
    for col, value in spec["keep"].items():
        mask &= (obs[col].astype(str) == value).to_numpy()
    col, groups = spec["group"]
    table = pd.DataFrame({"row": np.arange(len(obs)), "donor": obs["donor_id"].astype(str),
                          "sample": obs[spec["sample"]].astype(str),
                          "group": obs[col].astype(str).map(groups),
                          "population": obs[spec["label"]].astype(str).map(spec["populations"])})
    table = table[mask & table["group"].notna().to_numpy()]
    first = table.groupby("donor")["sample"].min()
    table = table[table["sample"].to_numpy() == first.reindex(table["donor"]).to_numpy()]
    return table[table["population"].notna()]


def pseudobulk(name: str) -> ad.AnnData:
    """Units (donor x population) x genes, summed raw counts; obs: donor, group, population,
    n_cells; var: Ensembl index + feature_name."""
    spec = COHORTS[name]
    src = ad.read_h5ad(RAW / spec["file"], backed="r")
    cells = _cells(spec, src.obs)
    units = cells.groupby(["donor", "population"]).agg(group=("group", "first"),
                                                       n_cells=("row", "size")).reset_index()
    unit_of = pd.Series(np.arange(len(units)),
                        index=pd.MultiIndex.from_frame(units[["donor", "population"]]))
    col = unit_of.reindex(pd.MultiIndex.from_frame(cells[["donor", "population"]])).to_numpy()
    rows = cells["row"].to_numpy()
    X = src.raw.X if src.raw is not None else src.X
    n_genes = X.shape[1]
    total = sp.csr_matrix((len(units), n_genes), dtype=np.float64)
    order = np.argsort(rows)
    rows, col = rows[order], col[order]
    for start in range(0, len(rows), CHUNK):
        r, c = rows[start:start + CHUNK], col[start:start + CHUNK]
        block = sp.csr_matrix(X[r])
        assign = sp.csr_matrix((np.ones(len(r)), (c, np.arange(len(r)))), shape=(len(units), len(r)))
        total = total + assign @ block
    var_names = src.raw.var_names if src.raw is not None else src.var_names
    symbols = src.var["feature_name"].reindex(var_names).astype(str).to_numpy()
    src.file.close()
    obs = units.set_index(units["donor"] + "|" + units["population"])
    return ad.AnnData(X=sp.csr_matrix(np.round(total.toarray()).astype(np.float32)), obs=obs,
                      var=pd.DataFrame({"feature_name": symbols},
                                       index=[str(v) for v in var_names]))


def build(hidden: Path = HIDDEN) -> dict:
    hidden.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in COHORTS:
        pb = pseudobulk(name)
        path = hidden / f"{name}_pseudobulk.h5ad"
        pb.write_h5ad(path, compression="gzip")
        donors = pb.obs.groupby("donor")["group"].first().value_counts().sort_index()
        manifest[name] = {"units": int(pb.n_obs), "genes": int(pb.n_vars),
                          "donors": {g: int(n) for g, n in donors.items()},
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


# ---------------------------------------------------------------------------
# Replication
# ---------------------------------------------------------------------------

def load(name: str, hidden: Path = HIDDEN) -> ad.AnnData:
    path = hidden / f"{name}_pseudobulk.h5ad"
    if hashlib.sha256(path.read_bytes()).hexdigest() != json.loads(MANIFEST.read_text())[name]["sha256"]:
        raise ValueError(f"{path} differs from the committed reference hash")
    return ad.read_h5ad(path)


def _module_tests(pb: ad.AnnData, claim: Claim) -> list[tuple[float, float]] | None:
    """(log2FC in the claimed direction, p) per non-empty half, from pyDESeq2 on module rows."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    obs = pb.obs
    keep = ((obs["population"] == claim.population) & (obs["n_cells"] >= MIN_CELLS)
            & obs["group"].isin([claim.target, claim.reference])).to_numpy()
    sub = pb[keep]
    if min((sub.obs["group"] == g).sum() for g in (claim.target, claim.reference)) < 3:
        return None
    X = sp.csr_matrix(sub.X)
    by_symbol = pd.Series(np.arange(sub.n_vars), index=sub.var["feature_name"].to_numpy())
    halves = [(genes, sign) for genes, sign in ((claim.up_genes, 1), (claim.down_genes, -1))
              if genes]
    modules = {}
    for k, (genes, sign) in enumerate(halves):
        cols = [int(by_symbol[g]) if np.ndim(by_symbol[g]) == 0 else int(by_symbol[g].iloc[0])
                for g in genes if g in by_symbol.index]
        if len(cols) < 3:
            return None
        modules[f"__module{k}"] = np.asarray(X[:, cols].sum(axis=1)).ravel()
    mean = np.asarray(X.mean(axis=0)).ravel()
    background = np.argsort(-mean)[:BACKGROUND]
    counts = pd.DataFrame(X[:, background].toarray(), index=sub.obs_names,
                          columns=[f"g{i}" for i in background])
    for k, v in modules.items():
        counts[k] = v
    meta = pd.DataFrame({"condition": np.where(sub.obs["group"] == claim.target, "B_target",
                                               "A_reference")}, index=sub.obs_names)
    dds = DeseqDataSet(counts=counts.round().astype(int), metadata=meta, design="~condition",
                       quiet=True)
    dds.deseq2()
    res = DeseqStats(dds, contrast=["condition", "B_target", "A_reference"], quiet=True)
    res.summary()
    table = res.results_df
    return [(float(sign * table.at[f"__module{k}", "log2FoldChange"]),
             float(table.at[f"__module{k}", "pvalue"])) for k, (_, sign) in enumerate(halves)]


def _composition(pb: ad.AnnData, claim: Claim) -> tuple[float, float] | None:
    obs = pb.obs[pb.obs["group"].isin([claim.target, claim.reference])]
    cells = obs.pivot_table(index="donor", columns="population", values="n_cells",
                            aggfunc="sum", fill_value=0)
    if claim.population not in cells.columns:
        return None
    frac = np.clip(cells[claim.population] / cells.sum(axis=1), 1e-3, 1 - 1e-3)
    logit = np.log(frac / (1 - frac)) * (-1.0 if (claim.effect or 0) < 0 else 1.0)
    group = obs.groupby("donor")["group"].first().reindex(cells.index)
    a, b = logit[group == claim.target], logit[group == claim.reference]
    if min(len(a), len(b)) < 3:
        return None
    t = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(t.pvalue)


def replicate(claim: Claim, pb: ad.AnnData) -> dict | None:
    """{"effect": in the claimed direction (the weaker half for de), "p": two-sided (the larger
    half's), "same_sign": every half has the claimed sign}; None when untestable here."""
    if claim.kind == "composition":
        out = _composition(pb, claim)
        return None if out is None else {"effect": out[0], "p": out[1], "same_sign": out[0] > 0}
    halves = _module_tests(pb, claim)
    if halves is None or any(math.isnan(p) for _, p in halves):
        return None
    weakest = max(halves, key=lambda h: h[1])
    return {"effect": weakest[0], "p": weakest[1], "same_sign": all(e > 0 for e, _ in halves)}


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(build(), indent=1))
