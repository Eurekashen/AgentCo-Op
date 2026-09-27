"""Pairwise comparison of two workflow outputs, without a reference answer.

Two judges, differing only in what they may look at:

* :func:`report_only_judge` reads the two final reports — the common LLM-as-judge setup and the
  control condition. It can reward fluent, confident prose.
* :func:`evidence_judge` reads each report *together with* the local checks computed from the
  data (per-claim donor support, uncovered coherent signal). It follows the protocol adapted
  from Kimi K3's agentic GRM, with three changes for scientific outputs:

  1. the rubric is produced from the research question **before** any candidate is seen, and the
     same rubric is used for every comparison in the task;
  2. the judge fills a per-criterion scorepad citing evidence items, and the verdict is derived
     from the scorepad by counting criteria won — no weights, no summed scores;
  3. every criterion records ``unassessed`` when the evidence does not bear on it, and an
     unassessed criterion never counts for either side.

Both judges run in both presentation orders; a verdict that flips with the order becomes a tie.
Judge output is a noisy preference signal: it decides nothing on its own (see
``agentcoop.optimize``) and never enters the final report.
"""

from __future__ import annotations

import json
from typing import Any

from agentcoop import llm

RUBRIC_PROMPT = """You will later compare two analysis reports that answer this research question:

{question}

Before seeing any report, write the evaluation rubric: 4-6 criteria a careful domain scientist
would use to decide which report better answers the question, given that each report will come
with data-derived checks of its claims (whether each claim's genes change in the claimed direction
consistently across donors, and whether reproducible signal in the data was left unreported).
Each criterion must be checkable from the reports and checks, not from prior knowledge of the
answer. Return JSON {{"criteria": [{{"id": "c1", "name": "...", "what_to_check": "..."}}]}}."""

EVIDENCE_PROMPT = """Research question:
{question}

Rubric (fixed before the reports were seen):
{rubric}

Report A with its data-derived checks:
{a}

Report B with its data-derived checks:
{b}

Check semantics: "supported" = the claim's genes change in the claimed direction consistently
across donors; "contradicted" = consistently in the opposite direction; "unassessed" = the data
check could not decide (e.g. effect not consistent across donors, or too few measurable genes);
"ungrounded" = cell type or direction not recognizable. "uncovered signal" = a coherent program
of genes that changes consistently across donors but that no claim in the report covers.

Fill the scorepad: for each criterion, decide A, B, tie, or unassessed, and cite the evidence
(claim indices, check statuses, uncovered groups) that decides it. Read the checks; do not
reward length, confidence, or style. Return JSON
{{"scorepad": [{{"criterion": "c1", "winner": "A|B|tie|unassessed", "evidence": "..."}}]}}"""

REPORT_ONLY_PROMPT = """Research question:
{question}

Report A:
{a}

Report B:
{b}

Which report better answers the research question? Consider correctness, specificity and
scientific rigor. Return JSON {{"winner": "A|B|tie", "reason": "..."}}"""


def rubric_for(question: str, *, sample: int = 0) -> list[dict[str, str]]:
    with llm.tag("judge/rubric"):
        out, _ = llm.chat_json([{"role": "user", "content": RUBRIC_PROMPT.format(question=question)}],
                               reasoning_effort="low", sample=sample)
    crit = out.get("criteria") if isinstance(out, dict) else None
    return crit if isinstance(crit, list) and crit else [
        {"id": "c1", "name": "claims supported by data", "what_to_check": "check statuses"}]


def _report_text(report: dict[str, Any]) -> str:
    return json.dumps({"claims": report.get("claims", []), "summary": report.get("summary", "")},
                      indent=1)


def packet(report: dict[str, Any], profile: dict[str, Any]) -> str:
    """Report + checks, anonymized: no workflow names, agent names or artifact producers."""
    claims = []
    for i, c in enumerate(report.get("claims", []) or []):
        if not isinstance(c, dict):
            continue
        chk = next((x for x in profile["claims"] if x["claim_index"] == i), None)
        claims.append({
            "index": i, "cell_type": c.get("cell_type"), "direction": c.get("direction"),
            "program": c.get("program"), "genes": (c.get("genes") or [])[:15],
            "check": {"status": chk["status"], "detail": chk["detail"]} if chk else None,
        })
    uncovered = [{"cell_type": u["cell_type"], "direction": u["direction"],
                  "n_genes": u["n_genes"], "example_genes": u["genes"][:10],
                  "go_hint": [h["term"] for h in u["go_hint"][:2]]} for u in profile["uncovered"]]
    return json.dumps({"claims": claims, "uncovered_signal": uncovered,
                       "summary": report.get("summary", "")}, indent=1)


def _swap(v: str) -> str:
    return {"A": "B", "B": "A"}.get(v, v)


def _combine(first: str, second_swapped_back: str) -> str:
    return first if first == second_swapped_back else "tie"


def report_only_judge(question: str, a: dict[str, Any], b: dict[str, Any], *,
                      sample: int = 0) -> dict[str, Any]:
    verdicts = []
    for order, (x, y) in enumerate(((a, b), (b, a))):
        with llm.tag("judge/report_only"):
            out, _ = llm.chat_json([{"role": "user", "content": REPORT_ONLY_PROMPT.format(
                question=question, a=_report_text(x), b=_report_text(y))}],
                reasoning_effort="low", sample=sample * 2 + order)
        v = str(out.get("winner", "tie")).strip().upper()
        v = v if v in ("A", "B") else "tie"
        verdicts.append(v if order == 0 else _swap(v))
    return {"winner": _combine(*verdicts), "orders": verdicts}


def evidence_judge(question: str, rubric: list[dict[str, str]], a: dict[str, Any],
                   pa: dict[str, Any], b: dict[str, Any], pb: dict[str, Any], *,
                   sample: int = 0) -> dict[str, Any]:
    packets = (packet(a, pa), packet(b, pb))
    rubric_text = json.dumps(rubric, indent=1)
    per_order = []
    pads = []
    for order, (x, y) in enumerate((packets, packets[::-1])):
        with llm.tag("judge/evidence"):
            out, _ = llm.chat_json([{"role": "user", "content": EVIDENCE_PROMPT.format(
                question=question, rubric=rubric_text, a=x, b=y)}],
                reasoning_effort="low", sample=sample * 2 + order)
        pad = out.get("scorepad", []) if isinstance(out, dict) else []
        wins = {"A": 0, "B": 0}
        for row in pad if isinstance(pad, list) else []:
            w = str(row.get("winner", "")).strip().upper() if isinstance(row, dict) else ""
            if w in wins:
                wins[w if order == 0 else _swap(w)] += 1
        per_order.append("A" if wins["A"] > wins["B"] else "B" if wins["B"] > wins["A"] else "tie")
        pads.append(pad)
    return {"winner": _combine(per_order[0], per_order[1]), "orders": per_order, "scorepads": pads}


def evidence_rule(pa: dict[str, Any], pb: dict[str, Any]) -> dict[str, Any]:
    """Deterministic comparator on the same checks (no LLM): more supported claims, fewer
    contradicted claims, fewer uncovered coherent programs — lexicographic on net score."""
    def score(p: dict[str, Any]) -> tuple[int, int]:
        c = p["counts"]
        return (c["supported"] - 2 * c["contradicted"] - c["uncovered_groups"],
                -c["unassessed"] - c["ungrounded"])
    sa, sb = score(pa), score(pb)
    return {"winner": "A" if sa > sb else "B" if sb > sa else "tie", "scores": [sa, sb]}
