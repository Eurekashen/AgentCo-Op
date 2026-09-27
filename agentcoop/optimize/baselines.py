"""Comparison optimizers that receive the *same* feedback, budget and acceptance rule.

They differ from :func:`agentcoop.optimize.loop.optimize` only in how a change is proposed:

* :func:`reflective_prompt` — GEPA-style: reflect on the feedback and rewrite the integrator's
  instructions (the only text a prompt optimizer can reach; the external agents are black boxes).
* :func:`llm_editor` — AFlow/ADAS-style free-form editing: the model sees the full workflow JSON,
  the documented agent/handoff options and the feedback, and returns a new workflow.
* :func:`best_of_n` — no edits at all: re-draw the incumbent and keep the best by evidence.

All three use ``evidence_rule`` against the incumbent *and* a no-edit re-draw, like the main
loop, so a difference in outcome is attributable to the proposal mechanism.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from agentcoop import llm
from agentcoop.evidence import checks, judge
from agentcoop.optimize.loop import Attempt, Budget, Outcome, _report
from agentcoop.workflow import Task, Workflow, execute


def feedback_text(profile: dict[str, Any]) -> str:
    """The evidence feedback, as text: per-claim check results and uncovered coherent signal."""
    lines = []
    for c in profile["claims"]:
        lines.append(f"- claim {c['claim_index']} ({c['cell_type']}, {c['direction']}, "
                     f"{c['program']}): {c['status']} — {c['detail']}")
    for u in profile["uncovered"]:
        hint = ", ".join(h["term"] for h in u["go_hint"][:2])
        lines.append(f"- uncovered: {u['n_genes']} genes change {u['direction']} consistently "
                     f"across donors in {u['cell_type']} but no claim covers them "
                     f"(e.g. {', '.join(u['genes'][:8])}; GO: {hint})")
    return "\n".join(lines) or "- no claims and no uncovered signal"


def _accept_loop(task: Task, workflow: Workflow, propose, *, budget: Budget, sample: int,
                 label: str) -> Outcome:
    cell_types = task.info.get("cell_types", [])
    run = execute(workflow, task, sample=sample)
    profile = checks.profile_task(_report(run), task).to_dict()
    out = Outcome(workflow=workflow, run=run, profile=profile, nominal_usd=run.cost["nominal_usd"])
    redraw = 1
    for it in range(budget.max_iterations):
        proposals = propose(out.workflow, run, profile, it)
        if not proposals:
            out.stop_reason = "no proposal"
            break
        with llm.tag(f"{label}/iter{it}/control"):
            ctrl = execute(out.workflow, task, sample=sample,
                           samples={n.id: sample + 1000 * redraw for n in out.workflow.nodes})
        redraw += 1
        out.nominal_usd += ctrl.cost["nominal_usd"]
        ctrl_prof = checks.profile_task(_report(ctrl), task).to_dict()
        best = None
        for desc, cand in proposals[: budget.edits_per_iteration]:
            if out.nominal_usd >= budget.max_usd:
                break
            try:
                with llm.tag(f"{label}/iter{it}"):
                    cand_run = execute(cand, task, sample=sample)
            except Exception as exc:  # malformed free-form workflow
                out.attempts.append(Attempt(it, f"{desc} (invalid: {exc})"[:200], label, "", "",
                                            {}, None, "B", None, None, False, 0.0))
                continue
            out.nominal_usd += cand_run.cost["nominal_usd"]
            prof = checks.profile_task(_report(cand_run), task).to_dict()
            vs_inc = judge.evidence_rule(prof, profile)["winner"]
            vs_ctrl = judge.evidence_rule(prof, ctrl_prof)["winner"]
            ok = vs_inc == "A" and vs_ctrl == "A" and not cand_run.errors
            out.attempts.append(Attempt(it, desc[:200], label, "", "", dict(prof["counts"]),
                                        dict(ctrl_prof["counts"]), vs_inc, vs_ctrl, None, False,
                                        cand_run.cost["nominal_usd"]))
            if ok:
                score = judge.evidence_rule(prof, prof)["scores"]
                if best is None or score > best[0]:
                    best = (score, cand, cand_run, prof, len(out.attempts) - 1)
        if best is not None:
            _, out.workflow, run, profile, idx = best
            out.attempts[idx].accepted = True
            out.run, out.profile = run, profile
        if out.nominal_usd >= budget.max_usd:
            out.stop_reason = "budget exhausted"
            break
    out.stop_reason = out.stop_reason or "iteration limit"
    return out


REFLECT_PROMPT = """You are optimizing the instructions of the integrating agent in a workflow of
collaborating scientific agents. The workflow's final report was checked against the data:

{feedback}

Current additional instructions: {current!r}

Reflect on what went wrong and write improved additional instructions for the integrating agent
(at most 100 words). Return JSON {{"instructions": "..."}}."""


def reflective_prompt(task: Task, workflow: Workflow, *, budget: Budget = Budget(),
                      sample: int = 0) -> Outcome:
    target = next(n.id for n in workflow.nodes if n.agent in ("integrator",))

    def propose(w: Workflow, run, profile, it):
        out = []
        for k in range(budget.edits_per_iteration):
            with llm.tag("reflective_prompt/propose"):
                o, _ = llm.chat_json([{"role": "user", "content": REFLECT_PROMPT.format(
                    feedback=feedback_text(profile),
                    current=w.node(target).config.get("instructions", ""))}],
                    reasoning_effort="low", sample=sample * 100 + it * 10 + k)
            cand = w.copy()
            cand.node(target).config = {**cand.node(target).config,
                                        "instructions": str(o.get("instructions", ""))[:1000]}
            out.append((f"instructions: {o.get('instructions', '')[:120]}", cand))
        return out

    return _accept_loop(task, workflow, propose, budget=budget, sample=sample,
                        label="reflective_prompt")


EDITOR_DOC = """Workflow JSON: {"nodes": [{"id", "agent", "config"}], "edges": [{"src", "dst",
"handoff", "params"}], "output": node id}. The special source "task" provides the task.

Agents:
- de_analyst: differential expression per cell type. config.method: "wilcoxon" (cell-level) or
  "pseudobulk" (donor-level DESeq2). Outputs one DE table per cell type.
- geneagent: external gene-set interpretation agent (GeneAgent); input: gene lists only; names
  the dominant biological process with database verification. Cannot see direction or stats.
- integrator: writes the final claims. config.instructions: optional extra instructions text.

Handoffs (edge.handoff and params):
- task: pass the task (task -> de_analyst).
- de_to_genelists (de_analyst -> geneagent): params n (genes per list), split_direction (bool),
  max_padj (float or null), min_genes (int).
- interpretation_view (geneagent -> integrator): params context: list of fields of the source gene
  list to attach, subset of ["cell_type", "direction", "method", "selection", "stats"].
- de_summary (de_analyst -> integrator): params k (top genes per direction), max_padj.
- passthrough: pass artifacts unchanged."""

EDITOR_PROMPT = """You are optimizing a workflow of collaborating scientific agents. {doc}

Current workflow:
{workflow}

The final report was checked against the data:
{feedback}

Propose an improved workflow (you may change any node, edge or parameter). Return JSON
{{"rationale": "...", "workflow": <workflow JSON>}}."""


def llm_editor(task: Task, workflow: Workflow, *, budget: Budget = Budget(),
               sample: int = 0) -> Outcome:
    def propose(w: Workflow, run, profile, it):
        out = []
        for k in range(budget.edits_per_iteration):
            with llm.tag("llm_editor/propose"):
                o, _ = llm.chat_json([{"role": "user", "content": EDITOR_PROMPT.format(
                    doc=EDITOR_DOC, workflow=json.dumps(w.to_dict(), indent=1),
                    feedback=feedback_text(profile))}],
                    reasoning_effort="medium", sample=sample * 100 + it * 10 + k)
            try:
                cand = Workflow.from_dict({**o["workflow"], "name": "llm_edit"})
                cand.order()
            except Exception:
                continue
            out.append((str(o.get("rationale", ""))[:200], cand))
        return out

    return _accept_loop(task, workflow, propose, budget=budget, sample=sample, label="llm_editor")


def best_of_n(task: Task, workflow: Workflow, *, budget: Budget = Budget(),
              sample: int = 0) -> Outcome:
    """Re-draw the whole workflow until the budget is spent; keep the best by evidence."""
    cell_types = task.info.get("cell_types", [])
    run = execute(workflow, task, sample=sample)
    profile = checks.profile_task(_report(run), task).to_dict()
    out = Outcome(workflow=workflow, run=run, profile=profile, nominal_usd=run.cost["nominal_usd"])
    k = 1
    while out.nominal_usd < budget.max_usd and k <= 3 * budget.max_iterations:
        with llm.tag("best_of_n"):
            r = execute(workflow, task, sample=sample,
                        samples={n.id: sample + 1000 * k for n in workflow.nodes})
        k += 1
        cost = r.cost["nominal_usd"] or run.cost["nominal_usd"]
        out.nominal_usd += cost
        prof = checks.profile_task(_report(r), task).to_dict()
        win = judge.evidence_rule(prof, out.profile)["winner"] == "A"
        out.attempts.append(Attempt(k, "re-draw", "resample", "", "", dict(prof["counts"]), None,
                                    "A" if win else "tie", None, None, win, cost))
        if win:
            out.run, out.profile = r, prof
    out.stop_reason = "budget exhausted"
    return out
