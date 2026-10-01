"""GEPA (Agrawal et al., 2025) as a comparison optimizer — the official ``gepa`` library.

GEPA evolves the *text components* of a system by reflecting on execution feedback (reflective
prompt mutation, Pareto candidate selection). It is given exactly what our optimizer gets: the
same task, the same answer-free evidence (as a score and as textual feedback), the same backbone
(reflection LM = ``llm.backbone()`` through our ledger) and the same budget of workflow
executions. What it can change is what a prompt optimizer can reach in these workflows:

- ``lead_instructions`` — extra instructions for the integrating (lead) agent;
- ``site_request_note`` (multi-site tasks) — text appended to every request sent to a site agent.

The external agents and the workflow's structure are fixed. Score = the evidence rule's net
score (supported − 2·contradicted − uncovered, minus 0.01 per unassessed/ungrounded claim).
Executions are counted per distinct (workflow, sample); GEPA stops when the budget is used.
"""

from __future__ import annotations

from typing import Any

from agentcoop import llm
from agentcoop.optimize.baselines import feedback_text
from agentcoop.optimize.loop import Attempt, Budget, Outcome, _profile, _report, start_outcome
from agentcoop.workflow import Task, Workflow, execute

NONE = "(none)"


def evidence_score(profile: dict[str, Any]) -> float:
    c = profile["counts"]
    return float(c["supported"] - 2 * c["contradicted"] - c["uncovered_groups"]) \
        - 0.01 * (c["unassessed"] + c["ungrounded"])


def components(workflow: Workflow) -> list[str]:
    multisite = any(e.handoff == "site_request" for e in workflow.edges)
    return ["lead_instructions"] + (["site_request_note"] if multisite else [])


def apply(workflow: Workflow, candidate: dict[str, str]) -> Workflow:
    w = workflow.copy()
    lead = w.node(w.output)
    text = candidate.get("lead_instructions", NONE).strip()
    if text and text != NONE:
        lead.config = {**lead.config, "instructions": text[:2000]}
    note = candidate.get("site_request_note", NONE).strip()
    if note and note != NONE:
        for e in w.edges:
            if e.handoff == "site_request":
                e.params = {**e.params, "note": note[:2000]}
    return w


def gepa_optimize(task: Task, workflow: Workflow, *, budget: Budget = Budget(),
                  sample: int = 0, seed: int = 0) -> Outcome:
    import gepa
    from gepa.core.adapter import EvaluationBatch

    out = start_outcome(task, workflow, sample)
    seen: dict[str, tuple[Any, dict[str, Any]]] = {workflow.fingerprint(): (out.run, out.profile)}
    state = {"executions": 1, "iteration": 0}

    def run_candidate(candidate: dict[str, str]):
        w = apply(workflow, candidate)
        key = w.fingerprint()
        if key not in seen:
            if not budget.room(out.nominal_usd, state["executions"], 1):
                return w, None, None
            with llm.tag(f"gepa/iter{state['iteration']}"):
                run = execute(w, task, sample=sample)
            state["executions"] += 1
            out.nominal_usd += run.cost["nominal_usd"]
            prof = _profile(run, task)
            seen[key] = (run, prof)
            out.attempts.append(Attempt(
                iteration=state["iteration"], edit=str(candidate)[:200], family="gepa",
                location="", finding="", counts=dict(prof["counts"]), control_counts=None,
                rule_vs_incumbent="", rule_vs_control=None, judge=None, accepted=False,
                nominal_usd=run.cost["nominal_usd"], report=_report(run)))
            state["iteration"] += 1
        run, prof = seen[key]
        return w, run, prof

    class Adapter:
        propose_new_texts = None  # use GEPA's own reflective instruction proposer

        def evaluate(self, batch, candidate, capture_traces=False):
            w, run, prof = run_candidate(candidate)
            if prof is None:  # out of budget: a candidate GEPA cannot afford scores lowest
                score, fb = -1e9, "not executed (budget exhausted)"
            else:
                score = evidence_score(prof) - (100.0 if run.errors else 0.0)
                fb = feedback_text(prof) + (f"\n- errors: {run.errors}" if run.errors else "")
            traj = [{"feedback": fb, "report": _report(run) if run else None}]
            return EvaluationBatch(outputs=[w.fingerprint()], scores=[score],
                                   trajectories=traj if capture_traces else None)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            t = eval_batch.trajectories[0]
            record = {"Inputs": task.question,
                      "Generated Outputs": str(t["report"])[:6000],
                      "Feedback": "The final report was checked against the data (answer-free "
                                  "evidence):\n" + t["feedback"]}
            return {c: [record] for c in components_to_update}

    def reflection_lm(prompt) -> str:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        with llm.tag("gepa/reflect"):
            return llm.chat(messages, reasoning_effort="low",
                            sample=sample * 1000 + state["iteration"]).content

    def stop(_state) -> bool:
        return not budget.room(out.nominal_usd, state["executions"], 1)

    seed_candidate = {c: NONE for c in components(workflow)}
    result = gepa.optimize(
        seed_candidate=seed_candidate, trainset=[task.task_id], valset=[task.task_id],
        adapter=Adapter(), reflection_lm=reflection_lm, reflection_minibatch_size=1,
        max_metric_calls=4 * (budget.max_executions or 16), stop_callbacks=[stop],
        seed=seed, raise_on_exception=False, display_progress_bar=False)
    best = apply(workflow, result.best_candidate)
    run, prof = seen.get(best.fingerprint(), (out.run, out.profile))
    if best.fingerprint() != workflow.fingerprint():
        for a in out.attempts:
            if a.edit == str(result.best_candidate)[:200]:
                a.accepted = True
    out.workflow, out.run, out.profile = best, run, prof
    out.executions = state["executions"]
    out.stop_reason = "budget exhausted" if stop(None) else "gepa finished"
    return out
