"""Executable handoff policies h_e: what of an upstream agent's output reaches the next agent.

Each policy is a pure function of the upstream artifacts and its params. The produced artifacts
are *views*: new artifacts derived from the upstream ones, carrying only the chosen fields. The
``meta`` of a view records provenance for the audit trail; downstream agents render ``data``
only, so moving a field from ``meta`` into ``data`` is a real change in what they can use.
"""

from __future__ import annotations

import json
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
    min_genes (skip a list shorter than this).
    """
    n = int(params.get("n", 30))
    split = bool(params.get("split_direction", False))
    max_padj = params.get("max_padj")
    min_genes = int(params.get("min_genes", 1))
    out = []
    for table in (a for a in upstream if a.kind == "de_table"):
        ct = table.data["cell_type"]
        rows = [r for r in _rows(table) if max_padj is None or r[3] <= max_padj]
        groups = ({"up": [r for r in rows if r[1] > 0], "down": [r for r in rows if r[1] < 0]}
                  if split else {"mixed": rows})
        for direction, grp in groups.items():
            grp = grp[:n]  # rows are sorted by p-value
            if len(grp) < min_genes:
                continue
            selection = (f"top {n} by p-value"
                         + (f", padj<={max_padj}" if max_padj is not None else ""))
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
    context = params.get("context", ["cell_type"])
    out = []
    for a in (x for x in upstream if x.kind == "interpretation"):
        source = a.meta.get("source", {})
        data = {k: a.data.get(k) for k in ("process", "narrative")}
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


# ---------------------------------------------------------------------------
# Task shape 2: requests to site agents and cross-site verification
# ---------------------------------------------------------------------------

STRUCTURED = (
    "Report a JSON object {\"findings\": [{\"cell_type\": ..., \"direction\": \"up\"|\"down\", "
    "\"program\": <biological program name>, \"genes\": [10-30 gene symbols that drive it], "
    "\"log2fc\": <typical log2 fold change>, \"method\": <test used>, "
    "\"donors_consistent\": \"k/4\"}]} with one finding per (cell type, program).")


LOCAL_SCOPE = (
    "Your site is one of two sites in a study of an uncharacterized treatment (cells within each "
    "donor are either control or treated). Report the transcriptional programs the treatment "
    "induces or represses at YOUR site, and in which cell types. The study lead will compare "
    "your findings with the other site's to decide what replicates, so report what your data "
    "shows; do not withhold findings because you cannot see the other site.")


@handoff("site_request")
def site_request(upstream: list[Artifact], params: dict[str, Any], ctx: RunContext,
                 dst: str) -> list[Artifact]:
    """Task → site agent. params: scope ("forward" the study question verbatim | "local": ask
    for the effects observed at this site, leaving replication to the lead), template
    ("question" | "structured"), report_weak (bool), method (None | "pseudobulk"), note (free
    text appended to the request; only the prompt-optimizer baseline sets it)."""
    task = next(a for a in upstream if a.kind == "task")
    if params.get("scope", "forward") == "local":
        parts = [LOCAL_SCOPE]
    else:
        parts = [task.data["question"]]
    if params.get("method") == "pseudobulk":
        parts.append("Use donors as the replicates (pseudobulk counts per donor x condition).")
    if params.get("report_weak"):
        parts.append("Also report weaker candidate effects that do not reach significance at "
                     "your site alone, marked as weak: replication will be judged jointly with "
                     "the other site, so a weak but consistent effect is informative.")
    if params.get("note"):
        parts.append(str(params["note"]))
    if params.get("template", "question") == "structured":
        parts.append(STRUCTURED)
    else:
        parts.append("Report your findings as a JSON object.")
    return [Artifact(kind="request", producer=f"task->{dst}", data={"text": "\n\n".join(parts)},
                     derived_from=[task.id], meta={"kind": "analysis", **params})]


@handoff("cross_verify")
def cross_verify(upstream: list[Artifact], params: dict[str, Any], ctx: RunContext,
                 dst: str) -> list[Artifact]:
    """Site report → the *other* site: test these specific findings on your data."""
    out = []
    for rep in (a for a in upstream if a.kind == "site_report"):
        payload = rep.data.get("report", {})
        items = None
        for key in ("findings", "claims", "results", "programs"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
        if not items:
            continue
        text = (f"Site {rep.data['site']} reported the findings below from its own donors. For "
                "EACH finding, test at your site whether its genes change in the same direction "
                "in the same cell type (per-donor consistency and differential expression "
                "statistics). Report a JSON object {\"verifications\": [{\"cell_type\": ..., "
                "\"direction\": ..., \"program\": ..., \"genes\": [...], \"replicated\": "
                "\"yes\"|\"no\"|\"inconclusive\", \"donors_consistent\": \"k/4\", "
                "\"evidence\": ...}]}.\n\nFindings:\n" + json.dumps(items, indent=1)[:12000])
        out.append(Artifact(kind="request", producer=f"{rep.producer}->{dst}",
                            data={"text": text}, derived_from=[rep.id],
                            meta={"kind": "verification", "from_site": rep.data["site"]}))
    return out
