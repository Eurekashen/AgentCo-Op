"""Localize collaboration loss by tracing data-derived signal through artifact lineage.

No reference answer is used. The probes are units of evidence the checks found in the data:

* an **uncovered signal** (a coherent, donor-consistent gene program no claim reports) is
  followed *forward* along the workflow: was it detected by the analysis, handed to the
  interpreter, used by the interpreter, passed on with its context, and claimed? The first stage
  at which the signal disappears is where the collaboration lost it;
* a **weak claim** (unassessed or contradicted by the donor check) is followed *backward* along
  its lineage to the analysis that produced its genes and the handoffs that shaped them.

Each finding names a location (a node or an edge) and a symptom. It is a hypothesis about where
an intervention could help — not a claim that the location is "the" root cause. Whether an edit
there helps is decided afterwards by controlled replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agentcoop.eval_free import canon_cell_type, canon_direction
from agentcoop.workflow import Artifact, Run


@dataclass
class Finding:
    symptom: str
    location: str  # node id or "src->dst"
    probe: str  # which evidence unit revealed it
    detail: str
    weight: float = 1.0
    facts: dict[str, Any] = field(default_factory=dict)


def _edge_id(src: str, dst: str) -> str:
    return f"{src}->{dst}"


def _tables(run: Run) -> dict[str, Artifact]:
    return {a.data["cell_type"]: a for a in run.artifacts.values() if a.kind == "de_table"}


def _descends_from(run: Run, artifact: Artifact, ancestor_id: str) -> bool:
    return ancestor_id in run.lineage(artifact.id)


def _analysis_node(run: Run) -> Optional[str]:
    for n in run.workflow.nodes:
        if n.agent == "de_analyst":
            return n.id
    return None


def _significant(table: Artifact, genes: list[str], direction: str, max_padj: float = 0.05):
    sign = 1 if direction == "up" else -1
    rows = {r[0]: r for r in table.data["rows"]}
    return [g for g in genes if g in rows and rows[g][3] <= max_padj and sign * rows[g][1] > 0]


def _present(table: Artifact, genes: list[str], direction: str, max_p: float = 0.05):
    """Genes the analysis *has* (nominal p and sign). A coherent program can be present at the
    set level while few of its genes survive gene-level FDR — that is not the analysis losing
    the signal, and the stages after it decide whether it travels on."""
    sign = 1 if direction == "up" else -1
    rows = {r[0]: r for r in table.data["rows"]}
    return [g for g in genes if g in rows and rows[g][2] <= max_p and sign * rows[g][1] > 0]


def trace_uncovered(run: Run, unc: dict[str, Any], output_node: str) -> Optional[Finding]:
    ct, direction, genes = unc["cell_type"], unc["direction"], list(unc["genes"])
    probe = f"uncovered {direction} signal in {ct} ({unc['n_genes']} genes)"
    analysis = _analysis_node(run)
    tables = _tables(run)
    if analysis is None or ct not in tables:
        return None
    table = tables[ct]
    sig = _present(table, genes, direction)
    if len(sig) < max(3, 0.3 * len(genes)):
        return Finding("analysis_misses_signal", analysis, probe,
                       f"only {len(sig)}/{len(genes)} of the program's genes change with nominal "
                       f"significance in the {table.data.get('method')} DE table", weight=2.0,
                       facts={"method": table.data.get("method"), "n_sig": len(sig)})

    consumers = [e for e in run.workflow.edges if e.src == analysis]
    # Direct path analysis -> integrator (an evidence branch)?
    direct = [a for a in run.artifacts.values()
              if a.kind == "de_summary" and _descends_from(run, a, table.id)]
    lists = [a for a in run.artifacts.values()
             if a.kind == "gene_list" and _descends_from(run, a, table.id)]
    carrying = [a for a in lists if set(sig) & set(a.data.get("genes", []))]
    if not carrying and not direct:
        edge = next((e for e in consumers if e.handoff == "de_to_genelists"), None)
        loc = _edge_id(edge.src, edge.dst) if edge else analysis
        return Finding("signal_not_handed_off", loc, probe,
                       f"{len(sig)} significant signal genes reach no downstream artifact",
                       weight=1.5, facts={"n_lists": len(lists)})
    if carrying:
        lst = max(carrying, key=lambda a: len(set(sig) & set(a.data["genes"])))
        share = len(set(sig) & set(lst.data["genes"])) / max(len(lst.data["genes"]), 1)
        edge_to_interp = next((e for e in consumers if e.handoff == "de_to_genelists"), None)
        interp_loc = _edge_id(edge_to_interp.src, edge_to_interp.dst) if edge_to_interp else None
        if lst.meta.get("direction") == "mixed" and interp_loc:
            return Finding("direction_dropped", interp_loc, probe,
                           "the signal reached the interpreter inside a list mixing up- and "
                           "down-regulated genes, so its direction cannot travel further",
                           weight=1.5, facts={"share_in_list": share})
        if share < 0.35 and interp_loc:
            return Finding("signal_diluted", interp_loc, probe,
                           f"signal genes are {share:.0%} of the list handed to the interpreter",
                           weight=1.2, facts={"share_in_list": share})
        views = [a for a in run.artifacts.values() if a.kind == "interpretation_view"
                 and _descends_from(run, a, lst.id)]
        for v in views:
            missing = [k for k in ("direction", "method") if k not in v.data]
            if missing:
                edge = next((e for e in run.workflow.edges if e.handoff == "interpretation_view"),
                            None)
                if edge:
                    return Finding("context_dropped", _edge_id(edge.src, edge.dst), probe,
                                   f"the interpretation reached the integrator without "
                                   f"{', '.join(missing)} of the list it came from",
                                   weight=1.3, facts={"missing": missing})
    return Finding("integration_omits_signal", output_node, probe,
                   "the signal reached the integrator with its context but was not claimed; "
                   "the integrator may lack evidence that it is reproducible", weight=1.0,
                   facts={"direct_summary": bool(direct)})


def trace_claim(run: Run, check: dict[str, Any], claim: dict[str, Any],
                output_node: str, cell_types: list[str]) -> Optional[Finding]:
    status = check["status"]
    if status == "supported":
        return None
    probe = f"claim {check['claim_index']} ({check['cell_type']}, {check['direction']}, " \
            f"{check['program']}) is {status}"
    if status == "ungrounded":
        return Finding("ungrounded_claim", output_node, probe, check["detail"], weight=0.5)
    analysis = _analysis_node(run)
    ct = canon_cell_type(str(claim.get("cell_type", "")), cell_types)
    table = _tables(run).get(ct or "")
    if status == "unassessed" and analysis and table is not None \
            and table.data.get("method") != "pseudobulk":
        return Finding("replicate_unit_mismatch", analysis, probe,
                       "the claim rests on a cell-level test while the question asks for "
                       "donor-reproducible effects; the donor check does not support it",
                       weight=1.5, facts={"method": table.data.get("method")})
    if status == "contradicted":
        lists = [a for a in run.artifacts.values() if a.kind == "gene_list"
                 and (a.meta.get("cell_type") == ct)]
        if any(a.meta.get("direction") == "mixed" for a in lists):
            edge = next((e for e in run.workflow.edges if e.handoff == "de_to_genelists"), None)
            if edge:
                return Finding("direction_dropped", _edge_id(edge.src, edge.dst), probe,
                               "the claimed direction is opposite to the data and the lists "
                               "handed to the interpreter mixed directions", weight=1.5)
    return Finding("overclaim", output_node, probe,
                   f"the integrator claimed an effect the data check does not support "
                   f"({check['detail']})", weight=1.0)


def localize(run: Run, profile: dict[str, Any], cell_types: list[str]) -> list[Finding]:
    """All findings, highest weight first; findings at the same location are merged."""
    out_node = run.workflow.output
    report = run.output().data if run.output() else {"claims": []}
    claims = [c for c in report.get("claims", []) if isinstance(c, dict)]
    findings: list[Finding] = []
    for unc in profile["uncovered"]:
        f = trace_uncovered(run, unc, out_node)
        if f:
            findings.append(f)
    for chk in profile["claims"]:
        idx = chk["claim_index"]
        if idx < len(claims):
            f = trace_claim(run, chk, claims[idx], out_node, cell_types)
            if f:
                findings.append(f)
    merged: dict[tuple[str, str], Finding] = {}
    for f in findings:
        key = (f.location, f.symptom)
        if key in merged:
            merged[key].weight += f.weight
            merged[key].detail += f"; {f.probe}"
        else:
            merged[key] = f
    return sorted(merged.values(), key=lambda f: -f.weight)


#: How far a signal travelled before it was lost. A fixed probe that travels further after an
#: edit is answer-free evidence of progress even when the final report does not change yet —
#: which is the normal case when several stages lose the same signal in sequence and each fix
#: is necessary but not sufficient on its own.
STAGE = {"analysis_misses_signal": 0, "signal_not_handed_off": 1, "direction_dropped": 2,
         "signal_diluted": 3, "context_dropped": 4, "integration_omits_signal": 5}
COVERED = 6


def propagation(run: Run, probes: list[dict[str, Any]], profile: dict[str, Any]) -> list[int]:
    """Depth reached by each probe signal in ``run`` (``COVERED`` if its report now claims it)."""
    still = {(u["cell_type"], u["direction"]) for u in profile["uncovered"]}
    depths = []
    for p in probes:
        if (p["cell_type"], p["direction"]) not in still:
            depths.append(COVERED)
            continue
        f = trace_uncovered(run, p, run.workflow.output)
        depths.append(STAGE.get(f.symptom, 0) if f else 0)
    return depths
