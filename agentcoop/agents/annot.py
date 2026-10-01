"""Cell-type annotation as a collaboration (BAIS-DPTA shape): analysis → annotator → assembly.

- ``cluster_analyst`` (in-house, deterministic): standard scanpy workflow — library-size
  normalization, log1p, 2000 HVGs, PCA, kNN graph, Leiden at ``resolution`` — then Wilcoxon
  markers per cluster (log fold change, fraction of cells expressing in / out of the cluster).
- ``gptcelltype`` (external, unmodified logic): GPTCelltype (Hou & Ji, Nature Methods 2024),
  the published LLM annotator. Its interface is a tissue name plus one line of marker genes
  per cluster; it returns one cell-type name per line. Ported verbatim from the R package
  (prompt, positive-logFC top-10 default, 30 clusters per call, re-ask until the line count
  matches); it runs on ``llm.external_model()``.
- ``annot_assemble`` (deterministic): cluster labels → one label per cell.

Handoffs (``handoffs`` below) decide what of the marker tables reaches the annotator.
"""

from __future__ import annotations

import re
from typing import Any

from agentcoop import llm
from agentcoop.workflow import DETERMINISTIC, EXTERNAL, Artifact, RunContext, agent, handoff

DETERMINISTIC.update({"cluster_analyst", "annot_assemble"})
EXTERNAL.add("gptcelltype")
N_MARKERS_STORED = 100


def cluster(path: str, resolution: float, n_hvg: int = 2000, n_pcs: int = 50, seed: int = 0):
    """→ (cluster label per cell, AnnData with log-normalized data and Leiden clusters)."""
    import anndata as ad
    import scanpy as sc

    a = ad.read_h5ad(path)
    a.var_names_make_unique()
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg)
    h = a[:, a.var["highly_variable"]].copy()
    sc.pp.scale(h, max_value=10)
    sc.tl.pca(h, n_comps=n_pcs, random_state=seed)
    sc.pp.neighbors(h, random_state=seed)
    sc.tl.leiden(h, resolution=resolution, flavor="igraph", n_iterations=2, random_state=seed)
    a.obs["cluster"] = h.obs["leiden"].values
    return a


def marker_tables(a, key: str = "cluster", n: int = N_MARKERS_STORED) -> dict[str, list[list]]:
    """Per cluster: [gene, log2fc, frac_in, frac_out, padj], sorted by Wilcoxon score."""
    import scanpy as sc

    sc.tl.rank_genes_groups(a, key, method="wilcoxon", pts=True, n_genes=n)
    res = a.uns["rank_genes_groups"]
    pts, pts_rest = res["pts"], res["pts_rest"]
    out = {}
    for c in res["names"].dtype.names:
        rows = []
        for i in range(n):
            g = str(res["names"][c][i])
            rows.append([g, round(float(res["logfoldchanges"][c][i]), 3),
                         round(float(pts.loc[g, c]), 3), round(float(pts_rest.loc[g, c]), 3),
                         float(f"{float(res['pvals_adj'][c][i]):.3g}")])
        out[str(c)] = rows
    return out


@agent("cluster_analyst")
def cluster_analyst(inputs: list[Artifact], config: dict[str, Any],
                    ctx: RunContext) -> list[Artifact]:
    a = cluster(str(ctx.task.data_path), float(config.get("resolution", 1.0)))
    labels = [str(x) for x in a.obs["cluster"]]
    sizes = {c: labels.count(c) for c in sorted(set(labels), key=int)}
    tables = marker_tables(a)
    clustering = Artifact(kind="clustering", producer=ctx.node_id,
                          data={"cells": labels, "sizes": sizes},
                          meta={"resolution": config.get("resolution", 1.0)})
    out = [clustering]
    for c in sizes:
        out.append(Artifact(kind="marker_table", producer=ctx.node_id,
                            data={"cluster": c, "n_cells": sizes[c],
                                  "columns": ["gene", "log2fc", "frac_in", "frac_out", "padj"],
                                  "rows": tables[c]},
                            derived_from=[clustering.id]))
    return out


# ---------------------------------------------------------------------------
# Handoffs: marker tables → what the annotator sees
# ---------------------------------------------------------------------------

_HOUSEKEEPING = re.compile(r"^(RPL|RPS|MT-|MRPL|MRPS|HB[AB]|MALAT1$|NEAT1$)")


@handoff("markers_to_genelists")
def markers_to_genelists(upstream: list[Artifact], params: dict[str, Any], ctx: RunContext,
                         dst: str) -> list[Artifact]:
    """GPTCelltype input: per cluster, the top ``n`` positive-logFC markers.

    params: n (default 10, GPTCelltype's ``topgenenumber``), rank ("score" = Wilcoxon order,
    "specificity" = frac_in − frac_out), drop_housekeeping (ribosomal/mito/hemoglobin),
    min_frac_in (expressed in at least this fraction of the cluster).
    """
    n = int(params.get("n", 10))
    rank = params.get("rank", "score")
    drop = bool(params.get("drop_housekeeping", False))
    min_in = float(params.get("min_frac_in", 0.0))
    out = []
    for t in (a for a in upstream if a.kind == "marker_table"):
        rows = [r for r in t.data["rows"] if r[1] > 0 and r[2] >= min_in
                and not (drop and _HOUSEKEEPING.match(r[0]))]
        if rank == "specificity":
            rows = sorted(rows, key=lambda r: -(r[2] - r[3]))
        rows = rows[:n]
        out.append(Artifact(kind="marker_list", producer=f"{t.producer}->{dst}",
                            data={"cluster": t.data["cluster"], "genes": [r[0] for r in rows]},
                            derived_from=[t.id],
                            meta={"n_cells": t.data["n_cells"], "rank": rank, "n": n,
                                  "stats": [[r[0], r[1], r[2], r[3]] for r in rows]}))
    return out


@handoff("clustering_only")
def clustering_only(upstream: list[Artifact], params: dict[str, Any], ctx: RunContext,
                    dst: str) -> list[Artifact]:
    return [a for a in upstream if a.kind == "clustering"]


# ---------------------------------------------------------------------------
# GPTCelltype (ported from the R package, gptcelltype())
# ---------------------------------------------------------------------------


def gptcelltype_prompt(tissue: str, lines: list[str]) -> str:
    return ("Identify cell types of " + tissue + " cells using the following markers separately "
            "for each\n row. Only provide the cell type name. Do not show numbers before the "
            "name.\n Some can be a mixture of multiple cell types.\n" + "\n".join(lines))


def tissue_of(ctx: RunContext) -> str:
    return str(ctx.task.info.get("organ", "")).replace("_", " ")


@agent("gptcelltype")
def gptcelltype(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    lists = sorted((a for a in inputs if a.kind == "marker_list"),
                   key=lambda a: int(a.data["cluster"]))
    labels: dict[str, str] = {}
    for start in range(0, len(lists), 30):
        chunk = lists[start:start + 30]
        lines = [f"{a.data['cluster']}:{','.join(a.data['genes'])}" for a in chunk]
        prompt = gptcelltype_prompt(tissue_of(ctx), lines)
        for attempt in range(int(config.get("max_attempts", 5))):
            r = llm.chat([{"role": "user", "content": prompt}], model=llm.external_model(),
                         sample=ctx.sample * 100 + attempt)
            res = [x for x in r.content.split("\n")]
            if len(res) == len(chunk):
                break
            # GPTCelltype re-asks verbatim until the line count matches; blank lines are the
            # common mismatch, so they are dropped before giving up on an attempt.
            res = [x for x in res if x.strip()]
            if len(res) == len(chunk):
                break
        else:
            res = (res + ["Unknown"] * len(chunk))[:len(chunk)]
        for a, name in zip(chunk, res):
            labels[a.data["cluster"]] = name.strip()
    return [Artifact(kind="cluster_labels", producer=ctx.node_id, data={"labels": labels},
                     derived_from=[a.id for a in lists], meta={"annotator": "gptcelltype"})]


@agent("annot_assemble")
def annot_assemble(inputs: list[Artifact], config: dict[str, Any],
                   ctx: RunContext) -> list[Artifact]:
    clustering = next(a for a in inputs if a.kind == "clustering")
    labels = next(a for a in inputs if a.kind == "cluster_labels").data["labels"]
    return [Artifact(kind="annotation", producer=ctx.node_id,
                     data={"cluster_labels": labels, "cells": clustering.data["cells"]},
                     derived_from=[a.id for a in inputs])]


def cell_labels(annotation: dict[str, Any]) -> list[str | None]:
    lab = annotation["cluster_labels"]
    return [lab.get(c) for c in annotation["cells"]]

