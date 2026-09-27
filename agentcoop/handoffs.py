"""Executable handoff policies h_e: what of an upstream agent's output reaches the next agent.

Each policy is a pure function of the upstream artifacts and its params. The produced artifacts
are *views*: new artifacts derived from the upstream ones, carrying only the chosen fields. The
``meta`` of a view records provenance for the audit trail; downstream agents render ``data``
only, so moving a field from ``meta`` into ``data`` is a real change in what they can use.
"""

from __future__ import annotations

from typing import Any

from agentcoop.workflow import Artifact, RunContext, handoff


def _rows(table: Artifact) -> list[list[Any]]:
    return table.data["rows"]


@handoff("task")
def task_passthrough(upstream: list[Artifact], params: dict[str, Any], ctx: RunContext,
                     dst: str) -> list[Artifact]:
    return list(upstream)


@handoff("de_to_genelists")
def de_to_genelists(upstream: list[Artifact], params: dict[str, Any], ctx: RunContext,
                    dst: str) -> list[Artifact]:
    """DE tables → gene lists for a gene-list-only interpreter.

    params: n (genes per list), split_direction, max_padj (significance filter or None),
    min_abs_lfc, min_genes (skip a list shorter than this), cell_types (subset or None).
    """
    n = int(params.get("n", 30))
    split = bool(params.get("split_direction", False))
    max_padj = params.get("max_padj")
    min_abs = float(params.get("min_abs_lfc", 0.0))
    min_genes = int(params.get("min_genes", 1))
    only = params.get("cell_types")
    out = []
    for table in (a for a in upstream if a.kind == "de_table"):
        ct = table.data["cell_type"]
        if only and ct not in only:
            continue
        rows = [r for r in _rows(table)
                if (max_padj is None or r[3] <= max_padj) and abs(r[1]) >= min_abs]
        groups = ({"up": [r for r in rows if r[1] > 0], "down": [r for r in rows if r[1] < 0]}
                  if split else {"mixed": rows})
        for direction, grp in groups.items():
            grp = grp[:n]  # rows are sorted by p-value
            if len(grp) < min_genes:
                continue
            selection = (f"top {n} by p-value" + (f", padj<={max_padj}" if max_padj is not None else "")
                         + (f", |log2FC|>={min_abs}" if min_abs else ""))
            out.append(Artifact(
                kind="gene_list", producer=f"{table.producer}->{dst}",
                data={"genes": [r[0] for r in grp]},
                derived_from=[table.id],
                meta={"cell_type": ct, "direction": direction, "selection": selection,
                      "method": table.data.get("method"),
                      "stats": [[r[0], r[1], r[3]] for r in grp]},
            ))
    return out


@handoff("interpretation_view")
def interpretation_view(upstream: list[Artifact], params: dict[str, Any], ctx: RunContext,
                        dst: str) -> list[Artifact]:
    """Interpreter output → integrator. ``context`` lists which facts about the source gene list
    travel with the interpretation (naive: only the cell type)."""
    fields = params.get("fields", ["process", "narrative"])
    context = params.get("context", ["cell_type"])
    out = []
    for a in (x for x in upstream if x.kind == "interpretation"):
        source = a.meta.get("source", {})
        data = {k: a.data.get(k) for k in fields}
        for key in context:
            if key == "stats":
                data["gene_stats(gene, log2fc, padj)"] = source.get("stats")
            elif key in source:
                data[key] = source[key]
        out.append(Artifact(kind="interpretation_view", producer=f"{a.producer}->{dst}",
                            data=data, derived_from=[a.id], meta={"view_of": a.id}))
    return out


@handoff("de_summary")
def de_summary(upstream: list[Artifact], params: dict[str, Any], ctx: RunContext,
               dst: str) -> list[Artifact]:
    """DE tables → compact per-cell-type summary (top up / down genes with statistics)."""
    k = int(params.get("k", 15))
    max_padj = float(params.get("max_padj", 0.05))
    out = []
    for table in (a for a in upstream if a.kind == "de_table"):
        rows = [r for r in _rows(table) if r[3] <= max_padj]
        up = [r for r in rows if r[1] > 0]
        down = [r for r in rows if r[1] < 0]
        out.append(Artifact(
            kind="de_summary", producer=f"{table.producer}->{dst}",
            data={"cell_type": table.data["cell_type"], "method": table.data.get("method"),
                  f"n_significant(padj<={max_padj})": {"up": len(up), "down": len(down)},
                  "top_up(gene, log2fc, padj)": [[r[0], r[1], r[3]] for r in up[:k]],
                  "top_down(gene, log2fc, padj)": [[r[0], r[1], r[3]] for r in down[:k]]},
            derived_from=[table.id], meta={"cell_type": table.data["cell_type"]},
        ))
    return out


@handoff("passthrough")
def passthrough(upstream: list[Artifact], params: dict[str, Any], ctx: RunContext,
                dst: str) -> list[Artifact]:
    kinds = params.get("kinds")
    return [a for a in upstream if not kinds or a.kind in kinds]
