"""Combining independent BixBench analyses: the standard multi-agent baselines.

All of them see the same analyses (the same cached analyst runs) and run on the same backbone;
they differ only in how the analyses are combined:

- ``bix_vote``      majority over answer clusters (self-consistency when the inputs are re-draws
                    of one agent; heterogeneous voting when they come from different agents);
- ``bix_aggregate`` Mixture-of-Agents aggregator (Wang et al. 2024): an LLM synthesizes the final
                    answer from the proposers' responses;
- ``bix_judge``     LLM-as-judge selection: pick the most credible analysis;
- ``bix_debate``    multi-agent debate (Du et al. 2023): each analysis is defended by a debater
                    that sees the others and may revise; final majority.

What reaches them is decided by the ``analysis_view`` handoff (``level``): ``answer`` (the short
answer only), ``summary`` (answer + the last executed steps), ``full`` (the whole trace).
"""

from __future__ import annotations

from typing import Any

from agentcoop import llm
from agentcoop.evidence.bix_checks import same_answer
from agentcoop.workflow import Artifact, RunContext, agent, handoff


def _trace_text(trace: list[dict[str, str]], last: int | None, chars: int = 1500) -> str:
    steps = trace[-last:] if last else trace
    return "\n".join(f"[code]\n{s.get('code', '')[:chars]}\n[output]\n{s.get('output', '')[:chars]}"
                     for s in steps)


@handoff("analysis_view")
def analysis_view(upstream: list[Artifact], params: dict[str, Any], ctx: RunContext,
                  dst: str) -> list[Artifact]:
    level = params.get("level", "answer")
    out = []
    for a in (x for x in upstream if x.kind == "analysis"):
        data: dict[str, Any] = {"agent": a.data["agent"], "answer": a.data.get("answer")}
        if level == "summary":
            data["last_steps"] = _trace_text(a.data.get("trace", []), last=3)
        elif level == "full":
            data["trace"] = _trace_text(a.data.get("trace", []), last=None, chars=2500)
        out.append(Artifact(kind="analysis_view", producer=f"{a.producer}->{dst}", data=data,
                            derived_from=[a.id], meta={"level": level}))
    return out


def final(ctx: RunContext, answer: str | None, inputs: list[Artifact], **extra: Any) -> Artifact:
    return Artifact(kind="final_answer", producer=ctx.node_id,
                    data={"answer": answer, **extra}, derived_from=[a.id for a in inputs])


def clusters(question: str, answers: list[str | None]) -> list[list[int]]:
    """Group answer indices by ``same_answer`` (first member is the representative)."""
    groups: list[list[int]] = []
    for i, a in enumerate(answers):
        if not a:
            continue
        for g in groups:
            if same_answer(question, answers[g[0]], a):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


@agent("bix_vote")
def bix_vote(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    views = [a for a in inputs if a.kind == "analysis_view"]
    answers = [v.data.get("answer") for v in views]
    groups = clusters(ctx.task.question, answers)
    if not groups:
        return [final(ctx, None, inputs, votes=[])]
    best = max(groups, key=len)  # ties: the earliest cluster (input order)
    return [final(ctx, answers[best[0]], inputs,
                  votes=[[views[i].data["agent"] for i in g] for g in groups])]


def _render(views: list[Artifact]) -> str:
    parts = []
    for k, v in enumerate(views, 1):
        d = v.data
        body = f"Analysis {k} (by {d['agent']}): final answer = {d.get('answer')!r}"
        if d.get("last_steps"):
            body += f"\nLast executed steps:\n{d['last_steps']}"
        if d.get("trace"):
            body += f"\nExecuted trace:\n{d['trace']}"
        parts.append(body)
    return "\n\n".join(parts)


ANSWER_RULE = ('Reply as JSON: {"answer": "<concise final answer to the question>", '
               '"rationale": "<one or two sentences>"}')


@agent("bix_aggregate")
def bix_aggregate(inputs: list[Artifact], config: dict[str, Any],
                  ctx: RunContext) -> list[Artifact]:
    views = [a for a in inputs if a.kind == "analysis_view"]
    msg = (f"Question:\n{ctx.task.question}\n\nYou have been provided with analyses of the data "
           "from several independent agents. Synthesize them into a single, high-quality answer. "
           "Critically evaluate the analyses, recognizing that some may be biased or incorrect; "
           "do not simply replicate them.\n\n" + _render(views) + "\n\n" + ANSWER_RULE)
    out, _ = llm.chat_json([{"role": "user", "content": msg}],
                           reasoning_effort=config.get("reasoning_effort", "medium"),
                           sample=ctx.sample)
    return [final(ctx, str(out.get("answer") or "") or None, inputs,
                  rationale=out.get("rationale"))]


@agent("bix_judge")
def bix_judge(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    views = [a for a in inputs if a.kind == "analysis_view" and a.data.get("answer")]
    if not views:
        return [final(ctx, None, inputs)]
    msg = (f"Question:\n{ctx.task.question}\n\nSeveral independent analyses answered this "
           "question. Judge which analysis is most likely correct and pick it.\n\n"
           + _render(views) + '\n\nReply as JSON: {"choice": <analysis number>, '
           '"rationale": "<one or two sentences>"}')
    out, _ = llm.chat_json([{"role": "user", "content": msg}],
                           reasoning_effort=config.get("reasoning_effort", "medium"),
                           sample=ctx.sample)
    try:
        k = int(out.get("choice")) - 1
    except (TypeError, ValueError):
        k = 0
    k = k if 0 <= k < len(views) else 0
    return [final(ctx, views[k].data["answer"], inputs, choice=views[k].data["agent"],
                  rationale=out.get("rationale"))]


@agent("bix_debate")
def bix_debate(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    views = [a for a in inputs if a.kind == "analysis_view"]
    answers = [v.data.get("answer") for v in views]
    history = []
    for rnd in range(int(config.get("rounds", 2))):
        new = []
        for i, v in enumerate(views):
            others = "\n".join(f"- Agent {j + 1} ({views[j].data['agent']}): {answers[j]!r}"
                               for j in range(len(views)) if j != i)
            msg = (f"Question:\n{ctx.task.question}\n\nYour analysis of the data:\n"
                   + _render([v]) + f"\n\nYour current answer: {answers[i]!r}\n\nThese are the "
                   f"answers of the other agents:\n{others}\n\nUsing their answers as additional "
                   "advice, can you give an updated answer? Examine your analysis and theirs "
                   "step by step.\n" + ANSWER_RULE)
            out, _ = llm.chat_json([{"role": "user", "content": msg}],
                                   reasoning_effort=config.get("reasoning_effort", "medium"),
                                   sample=ctx.sample * 100 + rnd * 10 + i)
            new.append(str(out.get("answer") or "") or answers[i])
        answers = new
        history.append(list(answers))
    groups = clusters(ctx.task.question, answers)
    answer = answers[max(groups, key=len)[0]] if groups else None
    return [final(ctx, answer, inputs, rounds=history)]

