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
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    if all(n in manifest for n in COHORTS) and "artefact_signature" in manifest:
        return manifest
    sig = artefact_signature()
    path = hidden / "artefact_signature.tsv"
    sig.to_csv(path, sep="\t", index=False)
    manifest["artefact_signature"] = {"genes": int(len(sig)),
                                      "per_population": {p: int(n) for p, n in sig["population"].value_counts().sort_index().items()},
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    if all(n in manifest for n in COHORTS):
        return manifest
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
# Artefact truth (§5.1): a 10x v2 vs v3 technical signature from public healthy PBMC
# ---------------------------------------------------------------------------

TENX = ROOT / "data" / "raw" / "covid" / "tenx"
#: Public 10x healthy PBMC, none of them a Berlin control (Berlin's v3 controls are the 10k and
#: 5k v3 samples): v2 pbmc4k + pbmc8k, v3 pbmc_1k_v3 + pbmc_10k_protein_v3 (gene expression).
TENX_SAMPLES = {"pbmc4k": "v2", "pbmc8k": "v2", "pbmc_1k_v3": "v3", "pbmc_10k_protein_v3": "v3"}
CELLTYPIST_POPULATIONS = {
    "Classical monocytes": "CD14 Mono", "Non-classical monocytes": "CD16 Mono",
    "Tcm/Naive helper T cells": "CD4 T", "Tem/Effector helper T cells": "CD4 T",
    "Regulatory T cells": "CD4 T", "Tem/Temra cytotoxic T cells": "CD8 T",
    "Tcm/Naive cytotoxic T cells": "CD8 T", "Tem/Trm cytotoxic T cells": "CD8 T",
    "CD16+ NK cells": "NK", "CD16- NK cells": "NK", "NK cells": "NK",
    "Naive B cells": "B", "Memory B cells": "B", "pDC": "pDC", "Plasmablasts": "Plasmablast"}
MIN_LOG2 = 1.0
MIN_CPM = 10.0
BIXBENCH_PYTHON = Path.home() / "miniconda3" / "envs" / "bixbench" / "bin" / "python"
CELLTYPIST_MODEL = ROOT / "data" / "markers" / "celltypist" / "Immune_All_Low.pkl"


def _read_tenx(name: str) -> ad.AnnData:
    import scanpy as sc
    if (TENX / f"{name}_filtered_feature_bc_matrix.h5").exists():
        a = sc.read_10x_h5(TENX / f"{name}_filtered_feature_bc_matrix.h5", gex_only=True)
        a.var["gene_symbols"] = a.var_names.astype(str)
    else:
        a = sc.read_10x_mtx(TENX / name / "filtered_gene_bc_matrices" / "GRCh38",
                            var_names="gene_ids")
        a.var["gene_ids"] = a.var_names
    a.var_names = a.var["gene_ids"].astype(str)
    return a[:, ~a.var_names.duplicated()].copy()


def _celltypist(adata: ad.AnnData) -> np.ndarray:
    """CellTypist Immune_All_Low labels (the analysis env has celltypist; best match per cell)."""
    import subprocess
    import tempfile

    import scanpy as sc
    x = adata.copy()
    sc.pp.normalize_total(x, target_sum=1e4)
    sc.pp.log1p(x)
    x.var_names = x.var["gene_symbols"].astype(str).to_numpy() if "gene_symbols" in x.var else x.var_names
    x.var_names_make_unique()
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "x.h5ad", Path(tmp) / "labels.txt"
        ad.AnnData(X=x.X, obs=pd.DataFrame(index=x.obs_names), var=pd.DataFrame(index=x.var_names)).write_h5ad(src)
        code = ("import sys, anndata as ad, celltypist; a = ad.read_h5ad(sys.argv[1]); "
                "p = celltypist.annotate(a, model=sys.argv[2], majority_voting=False); "
                "open(sys.argv[3], 'w').write('\\n'.join(p.predicted_labels['predicted_labels'].astype(str)))")
        run = subprocess.run([str(BIXBENCH_PYTHON), "-I", "-c", code, str(src),
                              str(CELLTYPIST_MODEL), str(out)], capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(f"celltypist failed: {run.stderr[-2000:]}")
        return np.array(out.read_text().split("\n"))


def artefact_signature() -> pd.DataFrame:
    """Per case population, genes whose log2 CPM differs between chemistries in every v3 - v2
    sample pair by >= MIN_LOG2 with one sign (mean CPM >= MIN_CPM in one chemistry): columns
    population, gene (Ensembl), symbol, sign (+1 = higher in v3)."""
    bulks = {}
    symbols: dict[str, str] = {}
    for name in TENX_SAMPLES:
        a = _read_tenx(name)
        symbols.update(dict(zip(a.var_names, a.var.get("gene_symbols", a.var_names).astype(str))))
        population = pd.Series(_celltypist(a)).map(CELLTYPIST_POPULATIONS).to_numpy()
        for pop in sorted(set(population) - {np.nan}):
            if not isinstance(pop, str):
                continue
            rows = population == pop
            if rows.sum() >= MIN_CELLS:
                counts = np.asarray(a.X[rows].sum(axis=0)).ravel()
                bulks[(name, pop)] = pd.Series(counts / counts.sum() * 1e6, index=a.var_names)
    out = []
    for pop in sorted({p for _, p in bulks}):
        v2 = [bulks[(n, pop)] for n, chem in TENX_SAMPLES.items() if chem == "v2" and (n, pop) in bulks]
        v3 = [bulks[(n, pop)] for n, chem in TENX_SAMPLES.items() if chem == "v3" and (n, pop) in bulks]
        if len(v2) < 2 or len(v3) < 2:
            continue
        genes = sorted(set.intersection(*(set(b.index) for b in v2 + v3)))
        cpm = pd.DataFrame({i: b.reindex(genes) for i, b in enumerate(v2 + v3)})
        expressed = (cpm.iloc[:, :len(v2)].mean(axis=1) >= MIN_CPM) | (cpm.iloc[:, len(v2):].mean(axis=1) >= MIN_CPM)
        logs = np.log2(cpm + 1)
        diffs = np.stack([logs[j] - logs[i] for i in range(len(v2)) for j in range(len(v2), len(v2) + len(v3))], axis=1)
        up = (diffs >= MIN_LOG2).all(axis=1) & expressed.to_numpy()
        down = (diffs <= -MIN_LOG2).all(axis=1) & expressed.to_numpy()
        for gene, sign in [(g, 1) for g in np.array(genes)[up]] + [(g, -1) for g in np.array(genes)[down]]:
            out.append({"population": pop, "gene": gene, "symbol": symbols.get(gene, gene), "sign": sign})
    return pd.DataFrame(out, columns=["population", "gene", "symbol", "sign"])


def is_artefact(claim: Claim, replicated: bool, signature: pd.DataFrame) -> bool:
    """§5.1: a claim is an artefact when it fails Stephenson replication and >= 50% of its genes
    are in the v2/v3 signature of its population with the claimed sign (Berlin's COVID-19 samples
    are v3 and its controls largely v2, so `up in severe` reads as `higher in v3`)."""
    if replicated or claim.kind != "de":
        return False
    sig = signature[signature["population"] == claim.population]
    sign_of = dict(zip(sig["symbol"], sig["sign"]))
    genes = [(g, 1) for g in claim.up_genes] + [(g, -1) for g in claim.down_genes]
    hits = sum(sign_of.get(g) == s for g, s in genes)
    return bool(genes) and hits >= 0.5 * len(genes)


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
