"""Our optimizer on BixBench teams: localize where the evidence fails, repair that collaboration.

Start: ``bix_evidence`` (three independent analyses → rubric + code-level audit → arbiter).
Localize from the answer-free evidence alone: if no answer group contains a *clean* analysis
(committed to an answer without violating a stated requirement), or two groups are equally
supported, the arbitration point lacks evidence, and the analyses whose audits failed are where
the collaboration lost it. Edit (executable, recorded in the workflow): route each failing
analysis's audit back to a new run of the same analyst (``audit_feedback`` handoff: the
requirements it missed and the issues found — never another analyst's answer), audit the new
analysis, and let the arbiter see the pooled evidence. Stop when the evidence decides or the
budget of extra analyst runs is spent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentcoop import llm
from agentcoop.pipelines import BIX_ANALYSTS, bix_evidence_team
from agentcoop.workflow import Edge, Node, Run, Task, Workflow, execute


@dataclass
class BixOutcome:
    workflow: Workflow
    run: Run
    rounds: list[dict[str, Any]] = field(default_factory=list)
    extra_runs: int = 0
    nominal_usd: float = 0.0
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"workflow": self.workflow.to_dict(), "run": self.run.to_dict(),
                "rounds": self.rounds, "extra_runs": self.extra_runs,
                "nominal_usd": self.nominal_usd, "stop_reason": self.stop_reason}


def _decided(groups: list[dict[str, Any]]) -> bool:
    if not groups:
        return False
    ranked = sorted(groups, key=lambda g: g["score"], reverse=True)
    best = ranked[0]["score"]
    clean_best = best[0] >= 1
    tie = len(ranked) > 1 and ranked[1]["score"][:2] == best[:2]
    return clean_best and not tie


def _failing(run: Run, audit_nodes: list[str]) -> list[dict[str, Any]]:
    out = []
    for nid in audit_nodes:
        for a in run.of(nid):
            d = a.data
            if a.kind == "audit" and (not d["answered"] or d["violated"] > 0
                                      or d["provenance"] == "unsupported"):
                out.append(d)
    return out


def add_repair(w: Workflow, agent_id: str, round_no: int, audit_node: str) -> Workflow:
    """New run of analyst ``agent_id`` fed by ``audit_node``'s findings on its last analysis,
    with its own audit, pooled at the arbiter."""
    w = w.copy()
    _, agent_name, config = next(x for x in BIX_ANALYSTS if x[0] == agent_id)
    new, audit = f"{agent_id}_r{round_no}", f"audit_r{round_no}"
    w.nodes.insert(len(w.nodes) - 1, Node(new, agent_name, {**config, "replica": 10 + round_no}))
    if not any(n.id == audit for n in w.nodes):
        w.nodes.insert(len(w.nodes) - 1, Node(audit, "bix_audit", {}))
        w.edges += [Edge("requirements", audit, "passthrough"), Edge(audit, w.output, "passthrough")]
    w.edges += [Edge("task", new, "task"),
                Edge(audit_node, new, "audit_feedback", {"agent": _agent_label(agent_name)}),
                Edge(new, audit, "passthrough", {"kinds": ["analysis"]})]
    w.name += f"+repair:{agent_id}@r{round_no}"
    return w


def _agent_label(agent_name: str) -> str:
    return {"finch": "finch", "codex_analyst": "codex", "react_analyst": "react"}[agent_name]


def optimize_bix(task: Task, *, max_extra: int = 3, max_rounds: int = 2,
                 sample: int = 0) -> BixOutcome:
    w = bix_evidence_team()
    with llm.tag("bix_ours/round0"):
        run = execute(w, task, sample=sample)
    out = BixOutcome(workflow=w, run=run, nominal_usd=run.cost["nominal_usd"])
    audit_nodes = ["audit"]
    by_label = {_agent_label(a): i for i, a, _ in BIX_ANALYSTS}
    for r in range(1, max_rounds + 1):
        groups = run.output().data.get("groups", []) if run.output() else []
        if _decided(groups):
            out.stop_reason = "evidence decides"
            break
        failing = [d for d in _failing(run, audit_nodes[-1:]) if d.get("agent") in by_label]
        failing = failing[: max_extra - out.extra_runs]
        if not failing:
            out.stop_reason = "no repairable analysis" if out.extra_runs < max_extra \
                else "budget exhausted"
            break
        cand = w
        for d in failing:
            cand = add_repair(cand, by_label[d["agent"]], r, audit_nodes[-1])
        with llm.tag(f"bix_ours/round{r}"):
            new_run = execute(cand, task, sample=sample)
        out.extra_runs += len(failing)
        out.nominal_usd += new_run.cost["nominal_usd"]
        out.rounds.append({"round": r, "repaired": [d["agent"] for d in failing],
                           "reasons": [{"agent": d["agent"], "answered": d["answered"],
                                        "violated": d["violated"], "provenance": d["provenance"],
                                        "issues": d["issues"][:3]} for d in failing],
                           "groups_before": groups,
                           "groups_after": new_run.output().data.get("groups", [])
                           if new_run.output() else []})
        w, run = cand, new_run
        audit_nodes.append(f"audit_r{r}")
    else:
        out.stop_reason = "round limit"
    out.workflow, out.run = w, run
    return out
