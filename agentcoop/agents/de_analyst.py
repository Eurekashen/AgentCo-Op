"""Differential expression analysis component (a deterministic tool agent).

Two methods, both standard practice, which differ in exactly the way that matters for
reproducibility claims:

* ``wilcoxon`` — cell-level Wilcoxon on log-normalized expression (scanpy default). Treats cells
  as replicates, so a strong effect in a single donor can look highly significant.
* ``pseudobulk`` — counts summed per donor × condition, DESeq2 (pydeseq2) with design
  ``~ donor + condition``. Donors are the replicates.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

import numpy as np

from agentcoop.workflow import Artifact, RunContext, agent


@functools.lru_cache(maxsize=4)
def load_counts(path: str):
    import anndata as ad

    return ad.read_h5ad(path)


def _expressed(sub, min_pct: float) -> np.ndarray:
    frac = np.asarray((sub.X > 0).mean(0)).ravel()
    return frac >= min_pct


def _wilcoxon(sub) -> dict[str, Any]:
    import scanpy as sc

    work = sub.copy()
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.tl.rank_genes_groups(work, "condition", groups=["treated"], reference="control",
                                method="wilcoxon", pts=False)
    r = work.uns["rank_genes_groups"]
    genes = [g[0] for g in r["names"]]
    lfc = [float(x[0]) for x in r["logfoldchanges"]]
    p = [float(x[0]) for x in r["pvals"]]
    padj = [float(x[0]) for x in r["pvals_adj"]]
    return {"gene": genes, "log2fc": lfc, "pvalue": p, "padj": padj}


def _pseudobulk(sub) -> dict[str, Any]:
    import pandas as pd
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    keys = sub.obs["donor"].astype(str) + "|" + sub.obs["condition"].astype(str)
    groups = sorted(keys.unique())
    rows = []
    for g in groups:
        m = (keys == g).to_numpy()
        rows.append(np.asarray(sub.X[m].sum(0)).ravel())
    counts = pd.DataFrame(np.vstack(rows).astype(int), index=groups, columns=sub.var_names)
    meta = pd.DataFrame({"donor": [g.split("|")[0] for g in groups],
                         "condition": [g.split("|")[1] for g in groups]}, index=groups)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dds = DeseqDataSet(counts=counts, metadata=meta, design="~donor + condition",
                           refit_cooks=True, quiet=True)
        dds.deseq2()
        stats = DeseqStats(dds, contrast=["condition", "treated", "control"], quiet=True)
        stats.summary()
    res = stats.results_df
    res = res.replace([np.inf, -np.inf], np.nan)
    out = res.fillna({"pvalue": 1.0, "padj": 1.0, "log2FoldChange": 0.0})
    return {"gene": list(out.index), "log2fc": out["log2FoldChange"].astype(float).tolist(),
            "pvalue": out["pvalue"].astype(float).tolist(), "padj": out["padj"].astype(float).tolist()}


@agent("de_analyst")
def de_analyst(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    method = config.get("method", "wilcoxon")
    min_pct = float(config.get("min_pct", 0.05))
    adata = load_counts(str(ctx.task.data_path))
    cell_types = config.get("cell_types") or ctx.task.info["cell_types"]
    task_ids = [a.id for a in inputs if a.kind == "task"]
    out = []
    for ct in cell_types:
        sub = adata[adata.obs["cell_type"] == ct]
        sub = sub[:, _expressed(sub, min_pct)].copy()
        table = _wilcoxon(sub) if method == "wilcoxon" else _pseudobulk(sub)
        order = np.argsort(table["pvalue"], kind="stable")
        rows = [[table["gene"][i], round(table["log2fc"][i], 4), float(f"{table['pvalue'][i]:.3g}"),
                 float(f"{table['padj'][i]:.3g}")] for i in order]
        out.append(Artifact(
            kind="de_table", producer=ctx.node_id,
            data={"cell_type": ct, "method": method, "columns": ["gene", "log2fc", "pvalue", "padj"],
                  "rows": rows},
            derived_from=task_ids,
            meta={"cell_type": ct, "method": method, "n_cells": int(sub.n_obs),
                  "n_donors": int(sub.obs["donor"].nunique()), "n_tested": len(rows)},
        ))
    return out
