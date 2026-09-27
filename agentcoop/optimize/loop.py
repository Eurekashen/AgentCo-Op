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

Variants for ablations: ``locate="random"`` (same edit space, random location),
``control=False``, ``judge=False``, ``families`` (restrict edit families, e.g. prompt-only).
"""

from __future__ import annotations

import json
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


@dataclass
class Outcome:
    workflow: Workflow
    run: Run
    profile: dict[str, Any]
    attempts: list[Attempt] = field(default_factory=list)
    findings_log: list[list[dict[str, Any]]] = field(default_factory=list)
    nominal_usd: float = 0.0
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"workflow": self.workflow.to_dict(), "run": self.run.to_dict(),
                "profile": self.profile, "attempts": [asdict(a) for a in self.attempts],
                "findings_log": self.findings_log, "nominal_usd": self.nominal_usd,
                "stop_reason": self.stop_reason}


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


def optimize(task: Task, workflow: Workflow, *, budget: Budget = Budget(), sample: int = 0,
             locate: str = "trace", control: bool = True, use_judge: bool = True,
             families: Optional[set[str]] = None, seed: int = 0,
             rules: Optional[dict] = None, use_propagation: bool = True,
             first_improvement: bool = True, use_compound: bool = True) -> Outcome:
    """``rules``: ``optimize.rules.index(...)`` from source tasks — tried first, still validated."""
    rng = random.Random(seed)
    cell_types = task.info.get("cell_types", [])
    rubric = judge.rubric_for(task.question) if use_judge else None

    run = execute(workflow, task, sample=sample)
    profile = _profile(run, task)
    out = Outcome(workflow=workflow, run=run, profile=profile, nominal_usd=run.cost["nominal_usd"])
    #: (incumbent fingerprint, edit key): an edit rejected in one context may help in another.
    tried: set[tuple[str, str]] = set()
    #: Workflows already executed on this task: an edit leading back to one is never re-proposed
    #: (e.g. reverting an accepted change).
    visited: set[str] = {workflow.fingerprint()}
    #: (location, parameter) pairs set by an accepted edit; never re-edited (no undoing).
    locked: set[tuple[str, str]] = set()
    redraw = 1

    for it in range(budget.max_iterations):
        inc_fp = out.workflow.fingerprint()
        findings = _localize(run, profile, task)
        out.findings_log.append([asdict(f) for f in findings])
        menu: list[tuple[Finding, Edit]] = []
        if locate == "random":
            # Ablation: the same edit space, every location (incl. edges from the task), in
            # random order — only the choice of *where* differs from the main loop.
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
                if f.location not in [n.id for n in out.workflow.nodes] and "->" not in f.location:
                    continue
                if f.location in seen_locations:
                    continue  # the location's menu is already offered for a heavier finding
                seen_locations.add(f.location)
                edits = local_edits(out.workflow, f.location)
                node = out.workflow.node(f.location) if "->" not in f.location else None
                if node is not None and node.agent == "integrator":
                    edits.append(_prompt_edit(out.workflow, f.location, f, sample * 100 + it))
                edits = [e for e in edits if (inc_fp, e.key) not in tried
                         and (families is None or e.family in families)]
                menu += [(f, e) for e in _rank(f, edits, sample * 100 + it)]
        menu = [(f, e) for f, e in menu if (inc_fp, e.key) not in tried
                and not any((e.location, k) in locked for k in e.params)
                and _safe_fp(e, out.workflow) not in visited]
        if not menu and use_compound:
            # Escalate: every single edit at the implicated locations failed in this context.
            # Try pairs of them (heaviest locations first) before giving up.
            singles: list[tuple[Finding, Edit]] = []
            seen_loc: set[str] = set()
            for f in findings:
                if f.location in seen_loc or (
                        f.location not in [n.id for n in out.workflow.nodes]
                        and "->" not in f.location):
                    continue
                seen_loc.add(f.location)
                singles += [(f, e) for e in local_edits(out.workflow, f.location)
                            if e.family != "prompt"
                            and not any((e.location, k) in locked for k in e.params)]
                if len(seen_loc) >= 2:
                    break
            pairs = []
            for i in range(len(singles)):
                for j in range(i + 1, len(singles)):
                    (fa, ea), (_, eb) = singles[i], singles[j]
                    if set(ea.params) & set(eb.params):
                        continue
                    c = compound(ea, eb)
                    if (inc_fp, c.key) not in tried and _safe_fp(c, out.workflow) not in visited:
                        pairs.append((fa, c))
            menu = pairs
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
        best: Optional[tuple[tuple, Workflow, Run, dict[str, Any]]] = None
        for f, edit in menu:
            if out.nominal_usd >= budget.max_usd:
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
            out.nominal_usd += cand_run.cost["nominal_usd"]
            cand_prof = _profile(cand_run, task)

            ctrl_prof = None
            if control:
                if edit.location not in controls:
                    samples = {n: sample + 1000 * redraw for n in _affected(out.workflow,
                                                                            edit.location)}
                    redraw += 1
                    with llm.tag(f"optimizer/iter{it}/control"):
                        ctrl_run = execute(out.workflow, task, sample=sample, samples=samples)
                    out.nominal_usd += ctrl_run.cost["nominal_usd"]
                    controls[edit.location] = (ctrl_run, _profile(ctrl_run, task))
                ctrl_prof = controls[edit.location][1]

            vs_inc = judge.evidence_rule(cand_prof, profile)["winner"]
            vs_ctrl = judge.evidence_rule(cand_prof, ctrl_prof)["winner"] if ctrl_prof else None
            depth = _propagation(cand_run, probes, cand_prof, task)
            depth_ctrl = (_propagation(controls[edit.location][0], probes, ctrl_prof, task)
                          if ctrl_prof else None)
            verdict = None
            basis = ""
            ok = vs_inc == "A" and (vs_ctrl in (None, "A"))
            if ok:
                basis = "report evidence"
                if use_judge:
                    with llm.tag(f"optimizer/iter{it}/judge"):
                        verdict = judge.evidence_judge(task.question, rubric, _report(cand_run),
                                                       cand_prof, _report(run), profile,
                                                       sample=sample)["winner"]
                    ok = verdict != "B"
            elif (use_propagation and vs_inc == "tie" and vs_ctrl in (None, "tie", "A")
                  and depth > depth_inc
                  and (depth_ctrl is None or depth > depth_ctrl)):
                ok, basis = True, "signal propagation"
            out.attempts.append(Attempt(
                iteration=it, edit=edit.description, family=edit.family,
                location=edit.location, finding=f.symptom, counts=_counts(cand_prof),
                control_counts=_counts(ctrl_prof) if ctrl_prof else None,
                rule_vs_incumbent=vs_inc, rule_vs_control=vs_ctrl, judge=verdict,
                accepted=False, nominal_usd=cand_run.cost["nominal_usd"], depth=depth,
                depth_incumbent=depth_inc, depth_control=depth_ctrl, basis=basis,
                edit_key=edit.key, from_rule=bool(rules and (f.symptom, edit.key) in rules)))
            if ok:
                score = (judge.evidence_rule(cand_prof, cand_prof)["scores"][0], depth)
                if best is None or score > best[0]:
                    best = (score, candidate, cand_run, cand_prof, len(out.attempts) - 1, edit)
                if first_improvement or out.attempts[-1].from_rule:
                    break  # accept the first validated improvement; the rest of the menu waits
        if best is None:
            if out.nominal_usd >= budget.max_usd:
                out.stop_reason = "budget exhausted"
                break
            continue
        _, out.workflow, run, profile, idx, accepted_edit = best
        out.attempts[idx].accepted = True
        locked.update((accepted_edit.location, k) for k in accepted_edit.params)
        out.run, out.profile = run, profile
        if out.nominal_usd >= budget.max_usd:
            out.stop_reason = "budget exhausted"
            break
    else:
        out.stop_reason = out.stop_reason or "iteration limit"
    if not out.stop_reason:
        out.stop_reason = "budget exhausted" if out.nominal_usd >= budget.max_usd else "done"
    return out
