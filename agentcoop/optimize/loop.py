"""Evidence-driven collaboration optimization, training-free and without reference answers.

One iteration:

1. **Profile** the incumbent's report against the data (``evidence.checks``).
2. **Localize** where signal is lost by tracing evidence units through lineage (``trace``).
3. **Propose** edits from the implicated location's declared local space (``edits``); a frozen
   LLM only ranks that menu (or writes instructions for a ``prompt`` edit).
4. **Replay** each edit: the unaffected prefix is reused from cache, the affected suffix is
   re-executed with the incumbent's sample index (paired comparison).
5. **Control**: the same suffix re-executed *without* the edit under a new sample index — what
   plain resampling would have produced.
6. **Accept** an edit if its report's evidence beats both the incumbent and the no-edit control
   and the evidence judge (both presentation orders) does not prefer the incumbent — or, when
   the final report does not move, if the incumbent's uncovered signals now *propagate further*
   through the workflow than under both the incumbent and the control (``trace.propagation``),
   with no loss of report evidence. Otherwise the incumbent stays. Every execution — accepted, rejected and control — is charged
   to the nominal budget.

Ablations: ``locate="random"`` (same edit space, random locations), ``use_propagation=False``.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from agentcoop import llm
from agentcoop.evidence import checks, judge
from agentcoop.optimize.edits import Edit, compound, local_edits, prompt_edit
from agentcoop.optimize import trace as trace_pooled
from agentcoop.optimize import trace_sites
from agentcoop.optimize.trace import Finding
from agentcoop.workflow import Run, Task, Workflow, execute


@dataclass
class Budget:
    max_usd: float = 0.6
    max_iterations: int = 4
    edits_per_iteration: int = 3
    #: Workflow executions (initial + every candidate + every control). A dollar budget buys a
    #: different search depth for models with different prices; an execution budget does not,
    #: so cross-model comparisons use it (None = dollars only).
    max_executions: Optional[int] = None

    def exhausted(self, usd: float, executions: int) -> bool:
        return usd >= self.max_usd or (self.max_executions is not None
                                       and executions >= self.max_executions)


@dataclass
class Attempt:
    iteration: int
    edit: str
    family: str
    location: str
    finding: str
    counts: dict[str, int]
    control_counts: Optional[dict[str, int]]
    rule_vs_incumbent: str
    rule_vs_control: Optional[str]
    judge: Optional[str]
    accepted: bool
    nominal_usd: float
    depth: int = 0
    depth_incumbent: int = 0
    depth_control: Optional[int] = None
    basis: str = ""
    edit_key: str = ""
    from_rule: bool = False
    #: The candidate's report, kept so an evaluation harness can later grade every attempt
    #: (harmful acceptances, missed improvements); the optimizer itself never grades it.
    report: Optional[dict[str, Any]] = None


@dataclass
class Outcome:
    workflow: Workflow
    run: Run
    profile: dict[str, Any]
    attempts: list[Attempt] = field(default_factory=list)
    findings_log: list[list[dict[str, Any]]] = field(default_factory=list)
    nominal_usd: float = 0.0
    stop_reason: str = ""
    initial_report: Optional[dict[str, Any]] = None
    executions: int = 0  # workflow executions charged so far (initial run, candidates, controls)

    def to_dict(self) -> dict[str, Any]:
        return {"workflow": self.workflow.to_dict(), "run": self.run.to_dict(),
                "profile": self.profile, "attempts": [asdict(a) for a in self.attempts],
                "findings_log": self.findings_log, "nominal_usd": self.nominal_usd,
                "stop_reason": self.stop_reason, "initial_report": self.initial_report,
                "executions": self.executions}


RANK_PROMPT = """You are improving a scientific workflow in which independently developed agents
collaborate. A data-derived check localized a problem:

symptom: {symptom}
location: {location}
detail: {detail}

Candidate edits at this location (each changes exactly one thing):
{menu}

Rank the edits by how likely each is to fix the symptom. Return JSON {{"ranking": [indices]}}."""

PROMPT_EDIT_PROMPT = """The integrating agent of a scientific workflow produced a report with this
problem, found by checking its claims against the data:

symptom: {symptom}
detail: {detail}

Its current additional instructions are: {current!r}

Write replacement additional instructions (at most 80 words) that would address the problem.
Return JSON {{"instructions": "..."}}."""


def _report(run: Run) -> dict[str, Any]:
    out = run.output()
    return out.data if out is not None else {"claims": [], "summary": "(workflow failed)"}


def _counts(profile: dict[str, Any]) -> dict[str, int]:
    return dict(profile["counts"])


def _affected(w: Workflow, location: str) -> set[str]:
    if "->" in location:
        dst = location.split("->")[1]
        return {dst} | w.descendants(dst)
    return {location} | w.descendants(location)


def _rank(finding: Finding, edits: list[Edit], sample: int) -> list[Edit]:
    if len(edits) <= 1:
        return edits
    menu = "\n".join(f"{i}. [{e.family}] {e.description}" for i, e in enumerate(edits))
    with llm.tag("optimizer/rank"):
        out, _ = llm.chat_json([{"role": "user", "content": RANK_PROMPT.format(
            symptom=finding.symptom, location=finding.location, detail=finding.detail[:600],
            menu=menu)}], reasoning_effort="low", sample=sample)
    order = [i for i in out.get("ranking", []) if isinstance(i, int) and 0 <= i < len(edits)]
    order += [i for i in range(len(edits)) if i not in order]
    return [edits[i] for i in order]


def _prompt_edit(w: Workflow, node_id: str, finding: Finding, sample: int) -> Edit:
    current = w.node(node_id).config.get("instructions", "")
    with llm.tag("optimizer/prompt"):
        out, _ = llm.chat_json([{"role": "user", "content": PROMPT_EDIT_PROMPT.format(
            symptom=finding.symptom, detail=finding.detail[:600], current=current)}],
            reasoning_effort="low", sample=sample)
    return prompt_edit(node_id, str(out.get("instructions", ""))[:800])


def _safe_fp(edit: Edit, w: Workflow) -> str:
    try:
        return edit.apply(w).fingerprint()
    except (StopIteration, KeyError, ValueError):
        return ""


def _profile(run: Run, task: Task) -> dict[str, Any]:
    return checks.profile_task(_report(run), task).to_dict()


def _localize(run: Run, profile: dict[str, Any], task: Task) -> list[Finding]:
    cts = task.info.get("cell_types", [])
    if task.info.get("shape") == "multisite":
        return trace_sites.localize(run, profile, cts)
    return trace_pooled.localize(run, profile, cts)


def _propagation(run: Run, probes: list[dict[str, Any]], profile: dict[str, Any],
                 task: Task) -> int:
    if task.info.get("shape") == "multisite":
        return sum(trace_sites.propagation(run, probes, profile, task.info.get("cell_types", [])))
    return sum(trace_pooled.propagation(run, probes, profile))


def start_outcome(task: Task, workflow: Workflow, sample: int) -> Outcome:
    """Execute the starting workflow: the incumbent every optimizer begins from (one execution)."""
    run = execute(workflow, task, sample=sample)
    return Outcome(workflow=workflow, run=run, profile=_profile(run, task),
                   nominal_usd=run.cost["nominal_usd"], initial_report=_report(run), executions=1)


def _valid_location(w: Workflow, location: str) -> bool:
    return "->" in location or location in {n.id for n in w.nodes}


def optimize(task: Task, workflow: Workflow, *, budget: Budget = Budget(), sample: int = 0,
             locate: str = "trace", seed: int = 0, rules: Optional[dict] = None,
             use_propagation: bool = True) -> Outcome:
    """``locate="random"``: ablation — the same edit space at every location in random order.
    ``rules``: ``optimize.rules.index(...)`` from source tasks — tried first, still validated.
    ``use_propagation=False``: ablation — accept on report evidence only."""
    rng = random.Random(seed)
    rubric = judge.rubric_for(task.question)

    out = start_outcome(task, workflow, sample)
    run, profile, executions = out.run, out.profile, out.executions
    #: (incumbent fingerprint, edit key): an edit rejected in one context may help in another.
    tried: set[tuple[str, str]] = set()
    #: Workflows already executed on this task: an edit leading back to one is never re-proposed
    #: (e.g. reverting an accepted change).
    visited: set[str] = {workflow.fingerprint()}
    #: (location, parameter) pairs set by an accepted edit; never re-edited (no undoing).
    locked: set[tuple[str, str]] = set()
    redraw = 1

    def is_locked(e: Edit) -> bool:
        return any((e.location, k) in locked for k in e.params)

    for it in range(budget.max_iterations):
        inc_fp = out.workflow.fingerprint()
        findings = _localize(run, profile, task)
        out.findings_log.append([asdict(f) for f in findings])
        menu: list[tuple[Finding, Edit]] = []
        if locate == "random":
            locations = [n.id for n in out.workflow.nodes] + [f"{e.src}->{e.dst}"
                                                               for e in out.workflow.edges]
            rng.shuffle(locations)
            pseudo = Finding("random", "", "random location", "no localization (ablation)")
            for loc in locations:
                menu += [(pseudo, e) for e in local_edits(out.workflow, loc)]
            rng.shuffle(menu)
            findings = [Finding("random", loc, "random location", "") for loc in locations]
        else:
            if not findings:
                out.stop_reason = "no findings: the checks see nothing left to fix"
                break
            seen_locations: set[str] = set()
            for f in findings:
                if not _valid_location(out.workflow, f.location) or f.location in seen_locations:
                    continue  # a location's menu is offered once, for its heaviest finding
                seen_locations.add(f.location)
                edits = local_edits(out.workflow, f.location)
                node = out.workflow.node(f.location) if "->" not in f.location else None
                if node is not None and node.agent == "integrator":
                    edits.append(_prompt_edit(out.workflow, f.location, f, sample * 100 + it))
                edits = [e for e in edits if (inc_fp, e.key) not in tried]
                menu += [(f, e) for e in _rank(f, edits, sample * 100 + it)]
        menu = [(f, e) for f, e in menu if (inc_fp, e.key) not in tried and not is_locked(e)
                and _safe_fp(e, out.workflow) not in visited]
        if not menu:
            # Escalate: every single edit at the implicated locations failed in this context.
            # Try pairs of them (heaviest two locations first) before giving up.
            singles: list[tuple[Finding, Edit]] = []
            seen_loc: set[str] = set()
            for f in findings:
                if f.location in seen_loc or not _valid_location(out.workflow, f.location):
                    continue
                seen_loc.add(f.location)
                singles += [(f, e) for e in local_edits(out.workflow, f.location)
                            if e.family != "prompt" and not is_locked(e)]
                if len(seen_loc) >= 2:
                    break
            for i, (fa, ea) in enumerate(singles):
                for _, eb in singles[i + 1:]:
                    if set(ea.params) & set(eb.params):
                        continue
                    c = compound(ea, eb)
                    if (inc_fp, c.key) not in tried and _safe_fp(c, out.workflow) not in visited:
                        menu.append((fa, c))
        if rules:
            # Known-good (symptom, edit) pairs first, best record first; ties keep LLM order.
            menu.sort(key=lambda fe: -(rules[(fe[0].symptom, fe[1].key)].score
                                       if (fe[0].symptom, fe[1].key) in rules else 0.0))
        menu = menu[: budget.edits_per_iteration]
        if not menu:
            out.stop_reason = "no untried edit at the implicated locations"
            break

        probes = list(profile["uncovered"])
        depth_inc = _propagation(run, probes, profile, task)
        controls: dict[str, tuple[Run, dict[str, Any]]] = {}
        accepted: Optional[tuple[Workflow, Run, dict[str, Any], Edit]] = None
        for f, edit in menu:
            if budget.exhausted(out.nominal_usd, executions):
                break
            tried.add((inc_fp, edit.key))
            try:
                candidate = edit.apply(out.workflow)
            except (StopIteration, KeyError, ValueError):
                continue
            if candidate.fingerprint() in visited:
                continue
            visited.add(candidate.fingerprint())
            with llm.tag(f"optimizer/iter{it}"):
                cand_run = execute(candidate, task, sample=sample)
            executions += 1
            out.nominal_usd += cand_run.cost["nominal_usd"]
            cand_prof = _profile(cand_run, task)

            if edit.location not in controls:
                samples = {n: sample + 1000 * redraw for n in _affected(out.workflow,
                                                                        edit.location)}
                redraw += 1
                with llm.tag(f"optimizer/iter{it}/control"):
                    ctrl_run = execute(out.workflow, task, sample=sample, samples=samples)
                executions += 1
                out.nominal_usd += ctrl_run.cost["nominal_usd"]
                controls[edit.location] = (ctrl_run, _profile(ctrl_run, task))
            ctrl_run, ctrl_prof = controls[edit.location]

            vs_inc = judge.evidence_rule(cand_prof, profile)["winner"]
            vs_ctrl = judge.evidence_rule(cand_prof, ctrl_prof)["winner"]
            depth = _propagation(cand_run, probes, cand_prof, task)
            depth_ctrl = _propagation(ctrl_run, probes, ctrl_prof, task)
            verdict = None
            basis = ""
            ok = vs_inc == "A" and vs_ctrl == "A"
            if ok:
                basis = "report evidence"
                with llm.tag(f"optimizer/iter{it}/judge"):
                    verdict = judge.evidence_judge(task.question, rubric, _report(cand_run),
                                                   cand_prof, _report(run), profile,
                                                   sample=sample)["winner"]
                ok = verdict != "B"
            elif (use_propagation and vs_inc == "tie" and vs_ctrl in ("tie", "A")
                  and depth > depth_inc and depth > depth_ctrl):
                ok, basis = True, "signal propagation"
            out.attempts.append(Attempt(
                iteration=it, edit=edit.description, family=edit.family,
                location=edit.location, finding=f.symptom, counts=_counts(cand_prof),
                control_counts=_counts(ctrl_prof), rule_vs_incumbent=vs_inc,
                rule_vs_control=vs_ctrl, judge=verdict, accepted=ok,
                nominal_usd=cand_run.cost["nominal_usd"], depth=depth, depth_incumbent=depth_inc,
                depth_control=depth_ctrl, basis=basis, edit_key=edit.key,
                from_rule=bool(rules and (f.symptom, edit.key) in rules),
                report=_report(cand_run)))
            if ok:
                accepted = (candidate, cand_run, cand_prof, edit)
                break  # first validated improvement; the rest of the menu waits
        if accepted is None:
            if budget.exhausted(out.nominal_usd, executions):
                out.stop_reason = "budget exhausted"
                break
            continue
        out.workflow, run, profile, accepted_edit = accepted
        locked.update((accepted_edit.location, k) for k in accepted_edit.params)
        out.run, out.profile = run, profile
        if budget.exhausted(out.nominal_usd, executions):
            out.stop_reason = "budget exhausted"
            break
    else:
        out.stop_reason = "iteration limit"
    out.executions = executions
    return out
