"""Answer-free evidence over independent BixBench analyses, and arbitration on it.

- ``bix_requirements`` (framework, LLM): the explicit requirements a question states (method,
  thresholds, comparison, subset, rounding/units), extracted from the question alone — fixed
  before any analysis is seen, like a rubric.
- ``bix_audit`` (framework, LLM + deterministic): for one analysis, which requirements its
  *executed code* follows (judged from the code and outputs, not from its prose), whether it
  committed to an answer, and the deterministic provenance check (is a numeric answer printed
  by the code it ran?).
- ``bix_arbiter`` (deterministic): groups answers that agree, scores each group by the
  evidence of its members, returns the best-supported answer. ``unassessed`` is never support.
- ``audit_feedback`` (handoff): turns an audit's findings into a repair request for a re-run of
  that analyst — the executable change our optimizer makes when the evidence is insufficient.
"""

from __future__ import annotations

from typing import Any

from agentcoop import llm
from agentcoop.agents.bix_team import clusters, final
from agentcoop.evidence.bix_checks import provenance
from agentcoop.workflow import DETERMINISTIC, Artifact, RunContext, agent, handoff

DETERMINISTIC.add("bix_arbiter")

REQUIREMENTS = """Question about a dataset:
{question}

List the explicit requirements this question places on the analysis: methods or tools named,
thresholds and cutoffs, which groups/samples/subsets to compare or include, the exact quantity
asked for, and rounding or units. Only what the question states, nothing inferred.
Reply as JSON: {{"requirements": ["...", ...]}}"""

AUDIT = """Question:
{question}

Requirements stated by the question:
{requirements}

Below is what an analyst actually executed (code and its printed output), and the answer it
submitted. Judge from the executed code and outputs, not from any prose claims.

{trace}

Submitted answer: {answer!r}

For each requirement, did the executed analysis follow it? Also: did the analyst commit to a
concrete answer to the question (not a refusal or "cannot determine")?
Reply as JSON: {{"requirements": [{{"requirement": "...", "followed": "yes"|"no"|"unclear",
"evidence": "<short quote of code/output>"}}], "answered": true|false,
"issues": ["<concrete problem in the executed analysis, if any>"]}}"""


def _trace_text(analysis: dict[str, Any], chars: int = 24000) -> str:
    parts = []
    for i, s in enumerate(analysis.get("trace", []), 1):
        parts.append(f"[step {i} code]\n{s.get('code', '')[:2500]}\n[step {i} output]\n"
                     f"{s.get('output', '')[:1500]}")
    for name, src in (analysis.get("files") or {}).items():
        parts.append(f"[file {name}]\n{src[:4000]}")
    text = "\n".join(parts)
    return text[-chars:] if len(text) > chars else text


@agent("bix_requirements")
def bix_requirements(inputs: list[Artifact], config: dict[str, Any],
                     ctx: RunContext) -> list[Artifact]:
    out, _ = llm.chat_json([{"role": "user", "content": REQUIREMENTS.format(
        question=ctx.task.question)}], reasoning_effort="low", sample=ctx.sample)
    reqs = [str(r) for r in out.get("requirements") or [] if str(r).strip()][:12]
    return [Artifact(kind="requirements", producer=ctx.node_id, data={"requirements": reqs})]


@agent("bix_audit")
def bix_audit(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    reqs = next((a.data["requirements"] for a in inputs if a.kind == "requirements"), [])
    out = []
    for a in (x for x in inputs if x.kind == "analysis"):
        d = a.data
        prov = provenance(d)
        if not d.get("answer"):
            verdict = {"requirements": [], "answered": False, "issues": ["no answer submitted"]}
        else:
            verdict, _ = llm.chat_json([{"role": "user", "content": AUDIT.format(
                question=ctx.task.question,
                requirements="\n".join(f"- {r}" for r in reqs) or "- (none stated)",
                trace=_trace_text(d), answer=d.get("answer"))}],
                reasoning_effort="low", sample=ctx.sample)
        rows = [r for r in verdict.get("requirements") or [] if isinstance(r, dict)]
        followed = sum(str(r.get("followed")).lower() == "yes" for r in rows)
        violated = sum(str(r.get("followed")).lower() == "no" for r in rows)
        out.append(Artifact(kind="audit", producer=ctx.node_id, data={
            "agent": d.get("agent"), "analysis": a.id, "answer": d.get("answer"),
            "answered": bool(verdict.get("answered", bool(d.get("answer")))) and bool(
                d.get("answer")),
            "provenance": prov, "followed": followed, "violated": violated,
            "requirements": rows, "issues": [str(x) for x in verdict.get("issues") or []][:6]},
            derived_from=[a.id]))
    return out


def evidence_of(audit: dict[str, Any]) -> tuple[int, int, int]:
    """(committed, no violated requirement, provenance supported) — unassessed counts for
    neither side."""
    return (int(audit["answered"]), int(audit["answered"] and audit["violated"] == 0),
            int(audit["provenance"] == "supported"))


@agent("bix_arbiter")
def bix_arbiter(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    audits = [a.data for a in inputs if a.kind == "audit" and a.data.get("answer")]
    if not audits:
        return [final(ctx, None, inputs, groups=[])]
    groups = clusters(ctx.task.question, [a["answer"] for a in audits])

    def group_score(g: list[int]) -> tuple:
        ev = [evidence_of(audits[i]) for i in g]
        clean = sum(1 for e in ev if e[1])            # answered without violating the spec
        grounded = sum(1 for e in ev if e[1] and e[2])  # ... and printed by its own code
        return (clean, grounded, len(g), -min(g))
    best = max(groups, key=group_score)
    return [final(ctx, audits[best[0]]["answer"], inputs,
                  groups=[{"agents": [audits[i]["agent"] for i in g],
                           "answer": audits[g[0]]["answer"], "score": list(group_score(g))}
                          for g in groups])]


@handoff("audit_feedback")
def audit_feedback(upstream: list[Artifact], params: dict[str, Any], ctx: RunContext,
                   dst: str) -> list[Artifact]:
    """Audit of a previous analysis → a repair request for a new run of an analyst. The new run
    sees the question plus the reviewer's findings (requirements not followed, issues), never
    another analyst's answer."""
    agent_name = params.get("agent")
    out = []
    for a in (x for x in upstream if x.kind == "audit"):
        d = a.data
        if agent_name and d.get("agent") != agent_name:
            continue
        missed = [r["requirement"] for r in d["requirements"]
                  if str(r.get("followed")).lower() in ("no", "unclear")]
        notes = missed and ("The question requires: " + "; ".join(missed) + ".") or ""
        issues = "; ".join(d["issues"])
        if not d["answered"]:
            issues = "a previous attempt did not commit to an answer; " + issues
        if d["provenance"] == "unsupported":
            issues += "; report the value exactly as your code computes and prints it"
        text = " ".join(x for x in [notes, f"A reviewer of a previous analysis noted: {issues}."
                                    if issues else ""] if x)
        out.append(Artifact(kind="repair_note", producer=f"{a.producer}->{dst}",
                            data={"text": text}, derived_from=[a.id],
                            meta={"agent": d.get("agent")}))
    return out
