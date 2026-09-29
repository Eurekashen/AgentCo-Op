"""Comparison optimizers that receive the same evidence feedback and execution budget.

They differ from :func:`agentcoop.optimize.loop.optimize` in how a change is proposed:

* :func:`reflective_prompt` — GEPA-style: reflect on the feedback and rewrite the integrator's
  instructions (the only text a prompt optimizer can reach; the external agents are black boxes).
* :func:`llm_editor` — AFlow/ADAS-style free-form editing: the model sees the full workflow JSON,
  the documented agent/handoff options and the feedback, and returns a new workflow.
* :func:`best_of_n` — no edits at all: re-draw the incumbent and keep the best by evidence.

Acceptance: the evidence rule against the incumbent and a no-edit re-draw of the whole workflow
(the main loop re-draws only the edited suffix, and additionally has the evidence-judge veto and
the signal-propagation route, which need its localization); best-of-N keeps a re-draw that beats
the incumbent. Proposal LLM calls are not charged to the execution budget in any method.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from agentcoop import llm
from agentcoop.evidence import judge
from agentcoop.optimize.loop import Attempt, Budget, Outcome, _profile, _report, start_outcome
from agentcoop.workflow import HANDOFFS, Task, Workflow, execute

import agentcoop.handoffs  # noqa: E402,F401  (registers the handoffs checked by _parse_workflow)


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


def _invalid(iteration: int, desc: str, label: str) -> Attempt:
    """A proposal that is not a runnable workflow: logged, never executed, never charged."""
    return Attempt(iteration=iteration, edit=desc, family=label, location="", finding="",
                   counts={}, control_counts=None, rule_vs_incumbent="B", rule_vs_control=None,
                   judge=None, accepted=False, nominal_usd=0.0)


def _accept_loop(task: Task, workflow: Workflow, propose, *, budget: Budget, sample: int,
                 label: str, acceptance: str = "evidence",
                 start: Optional[Outcome] = None) -> Outcome:
    """``acceptance``: "evidence" (evidence rule vs incumbent and a no-edit re-draw — the same
    rule as the main loop) or "report_judge" (the common LLM-as-judge setup: accept when a judge
    reading only the two reports prefers the candidate in both presentation orders).

    ``start``: continue another optimizer's outcome (its incumbent, spend and execution count)
    instead of starting from ``workflow`` — the budget is shared, not renewed."""
    if start is None:
        out = start_outcome(task, workflow, sample)
        run, profile, executions, redraw, it0 = out.run, out.profile, 1, 1, 0
    else:
        out, run, profile = start, start.run, start.profile
        executions = start.executions
        # Fresh re-draw indices and iteration numbers, disjoint from the first optimizer's.
        redraw = 100
        it0 = 1 + max((a.iteration for a in start.attempts), default=-1)
        prior_stop, out.stop_reason = start.stop_reason, ""
    for it in range(budget.max_iterations):
        # An iteration needs room for its control (evidence acceptance) and one candidate.
        if not budget.room(out.nominal_usd, executions, 2 if acceptance == "evidence" else 1):
            out.stop_reason = "budget exhausted"
            break
        proposals = propose(out.workflow, run, profile, it)
        for desc, cand in proposals:
            if cand is None:
                out.attempts.append(_invalid(it0 + it, desc[:200], label))
        proposals = [(d, c) for d, c in proposals if c is not None]
        if not proposals:
            out.stop_reason = "no proposal"
            break
        ctrl_prof = None
        if acceptance == "evidence":
            with llm.tag(f"{label}/iter{it}/control"):
                ctrl = execute(out.workflow, task, sample=sample,
                               samples={n.id: sample + 1000 * redraw for n in out.workflow.nodes})
            redraw += 1
            executions += 1
            out.nominal_usd += ctrl.cost["nominal_usd"]
            ctrl_prof = _profile(ctrl, task)
        best = None
        for desc, cand in proposals[: budget.edits_per_iteration]:
            if not budget.room(out.nominal_usd, executions, 1):
                break
            try:
                with llm.tag(f"{label}/iter{it}"):
                    cand_run = execute(cand, task, sample=sample)
            except llm.BudgetExceeded:
                raise
            except Exception as exc:  # malformed free-form workflow
                out.attempts.append(_invalid(it0 + it, f"{desc} (invalid: {exc})"[:200], label))
                continue
            executions += 1
            out.nominal_usd += cand_run.cost["nominal_usd"]
            prof = _profile(cand_run, task)
            if acceptance == "evidence":
                vs_inc = judge.evidence_rule(prof, profile)["winner"]
                vs_ctrl = judge.evidence_rule(prof, ctrl_prof)["winner"]
                ok = vs_inc == "A" and vs_ctrl == "A" and not cand_run.errors
            else:
                with llm.tag(f"{label}/iter{it}/judge"):
                    vs_inc = judge.report_only_judge(task.question, _report(cand_run),
                                                     _report(run), sample=sample)["winner"]
                vs_ctrl = None
                ok = vs_inc == "A" and not cand_run.errors
            out.attempts.append(Attempt(
                iteration=it0 + it, edit=desc[:200], family=label, location="", finding="",
                counts=dict(prof["counts"]),
                control_counts=dict(ctrl_prof["counts"]) if ctrl_prof else None,
                rule_vs_incumbent=vs_inc, rule_vs_control=vs_ctrl, judge=None, accepted=False,
                nominal_usd=cand_run.cost["nominal_usd"], report=_report(cand_run)))
            if ok:
                score = judge.evidence_rule(prof, prof)["scores"] if acceptance == "evidence" \
                    else (0, 0)
                if best is None or score > best[0]:
                    best = (score, cand, cand_run, prof, len(out.attempts) - 1)
        if best is not None:
            _, out.workflow, run, profile, idx = best
            out.attempts[idx].accepted = True
            out.run, out.profile = run, profile
        if budget.exhausted(out.nominal_usd, executions):
            out.stop_reason = "budget exhausted"
            break
    out.stop_reason = out.stop_reason or "iteration limit"
    if start is not None:
        out.stop_reason = f"{prior_stop} -> {label}: {out.stop_reason}"
    out.executions = executions
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
- donor_check: deterministic donor-consistency analysis; input: DE tables (edge from de_analyst
  with handoff "passthrough" and params {"kinds": ["de_table"]}); output: per cell type and
  direction, in how many donors the significant genes change in that direction.

Handoffs (edge.handoff and params):
- task: pass the task (task -> de_analyst).
- de_to_genelists (de_analyst -> geneagent): params n (genes per list), split_direction (bool),
  max_padj (float or null), min_genes (int).
- interpretation_view (geneagent -> integrator): params context: list of fields of the source gene
  list to attach, subset of ["cell_type", "direction", "method", "selection", "stats"].
- de_summary (de_analyst -> integrator): params k (top genes per direction), max_padj.
- passthrough: pass artifacts unchanged."""

EDITOR_DOC_MS = """Workflow JSON: {"nodes": [{"id", "agent", "config"}], "edges": [{"src", "dst",
"handoff", "params"}], "output": node id}. The special source "task" provides the task.
Data is held at two sites (A, B); only site agents can read their own site's data.

Agents:
- site_analyst: config.site "A" or "B". A coding agent that analyses its own site's data and
  answers the request it receives (from a site_request or cross_verify edge) with a JSON report.
- replication_check: deterministic; input: site reports (edges from site_analyst nodes with
  handoff "passthrough", params {"kinds": ["site_report"]}); output: for every reported finding,
  the per-donor direction consistency at every site.
- integrator: the study lead; writes the final claims. config.instructions: optional extra text.

Handoffs (edge.handoff and params):
- site_request (task -> site_analyst): params scope ("forward": the study question verbatim, or
  "local": ask for the site's own effects, replication judged by the lead), template ("question"
  or "structured": findings with cell type, direction, program, genes, log2fc, donors), report_weak
  (bool: also report weak candidate effects), method (null or "pseudobulk").
- cross_verify (site_analyst -> another site_analyst): send one site's findings to the other site
  to be tested on its own donors.
- passthrough: pass artifacts unchanged (e.g. site_analyst -> integrator)."""

#: v2 (after measuring that 6–11% of gpt-6-luna and 3/3 gpt-4.1-mini proposals were cyclic, hence
#: invalid): state the acyclicity constraint and how cross-site verification is wired.
EDITOR_DOC_MS_V2 = EDITOR_DOC_MS + """

Constraints: the workflow must be acyclic (a DAG). Cross-site verification therefore uses
separate verifier nodes: e.g. add node {"id": "verify_B", "agent": "site_analyst", "config":
{"site": "B"}}, edge site_A -> verify_B (handoff "cross_verify"), and edge verify_B -> lead
(handoff "passthrough"); symmetrically site_B -> verify_A. Never connect two site_analyst nodes
in both directions."""


EDITOR_PROMPT = """You are optimizing a workflow of collaborating scientific agents. {doc}

Current workflow:
{workflow}

The final report was checked against the data:
{feedback}

Propose an improved workflow (you may change any node, edge or parameter). Return JSON
{{"rationale": "...", "workflow": <workflow JSON>}}."""


#: Agents a proposal may use, per task shape (a free-form edit must not route raw site data to
#: a pooled-data agent, nor bring in a baseline agent that is not part of the composition).
ALLOWED_AGENTS = {"pooled": {"de_analyst", "geneagent", "integrator", "donor_check"},
                  "multisite": {"site_analyst", "replication_check", "integrator"}}


def _parse_workflow(o: dict, shape: str = "pooled") -> Workflow:
    cand = Workflow.from_dict({**o["workflow"], "name": "llm_edit"})
    cand.order()  # raises on cycles / dangling edges
    bad = [n.agent for n in cand.nodes if n.agent not in ALLOWED_AGENTS[shape]]
    bad += [e.handoff for e in cand.edges if e.handoff not in HANDOFFS]
    if bad:
        raise ValueError(f"not allowed in this workflow: {sorted(set(bad))}")
    return cand


def llm_editor(task: Task, workflow: Workflow, *, budget: Budget = Budget(),
               sample: int = 0, acceptance: str = "evidence",
               start: Optional[Outcome] = None, version: int = 1) -> Outcome:
    """``version`` 1: as evaluated on the held-out suites (tag v3.1-eval). 2: the documented
    DAG constraint and one repair round when a proposal is not a valid workflow."""
    shape = "multisite" if task.info.get("shape") == "multisite" else "pooled"

    def propose(w: Workflow, run, profile, it):
        out = []
        for k in range(budget.edits_per_iteration):
            if shape == "multisite":
                doc = EDITOR_DOC_MS_V2 if version >= 2 else EDITOR_DOC_MS
            else:
                doc = EDITOR_DOC
            messages = [{"role": "user", "content": EDITOR_PROMPT.format(
                doc=doc, workflow=json.dumps(w.to_dict(), indent=1),
                feedback=feedback_text(profile))}]
            with llm.tag("llm_editor/propose"):
                o, _ = llm.chat_json(messages, reasoning_effort="medium",
                                     sample=sample * 100 + it * 10 + k)
            try:
                cand = _parse_workflow(o, shape)
            except Exception as exc:
                if version >= 2:
                    # One repair round: the validator's message goes back to the editor.
                    messages += [{"role": "assistant", "content": json.dumps(o)},
                                 {"role": "user", "content": f"That workflow is invalid: {exc}. "
                                  "Return the corrected JSON in the same format."}]
                    with llm.tag("llm_editor/repair"):
                        o, _ = llm.chat_json(messages, reasoning_effort="medium",
                                             sample=sample * 100 + it * 10 + k)
                    try:
                        out.append((str(o.get("rationale", ""))[:200] + " (repaired)",
                                    _parse_workflow(o, shape)))
                        continue
                    except Exception as exc2:
                        exc = exc2
                # Logged (no execution is charged): how often a model's free-form proposals are
                # not even valid workflows is itself a measure of model dependence.
                out.append((f"{str(o.get('rationale', ''))[:120]} (invalid: {exc})", None))
                continue
            out.append((str(o.get("rationale", ""))[:200], cand))
        return out

    return _accept_loop(task, workflow, propose, budget=budget, sample=sample,
                        label="llm_editor", acceptance=acceptance, start=start)


def best_of_n(task: Task, workflow: Workflow, *, budget: Budget = Budget(),
              sample: int = 0) -> Outcome:
    """Re-draw the whole workflow until the budget is spent; keep the best by evidence."""
    out = start_outcome(task, workflow, sample)
    k = 1  # executions so far; the same execution budget as every other optimizer
    while not budget.exhausted(out.nominal_usd, k) and k <= 3 * budget.max_iterations:
        with llm.tag("best_of_n"):
            r = execute(workflow, task, sample=sample,
                        samples={n.id: sample + 1000 * k for n in workflow.nodes})
        k += 1
        # A re-draw whose every node was already cached still costs what the run costs.
        cost = r.cost["nominal_usd"] or out.run.cost["nominal_usd"]
        out.nominal_usd += cost
        prof = _profile(r, task)
        verdict = judge.evidence_rule(prof, out.profile)["winner"]
        out.attempts.append(Attempt(
            iteration=k - 1, edit="re-draw", family="resample", location="", finding="",
            counts=dict(prof["counts"]), control_counts=None, rule_vs_incumbent=verdict,
            rule_vs_control=None, judge=None, accepted=verdict == "A", nominal_usd=cost,
            report=_report(r)))
        if verdict == "A":
            out.run, out.profile = r, prof
    out.executions = k
    out.stop_reason = ("budget exhausted" if budget.exhausted(out.nominal_usd, k)
                       else "iteration limit")
    return out
