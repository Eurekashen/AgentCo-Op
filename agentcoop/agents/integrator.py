"""The integrating agent: turns whatever reached it into structured, cited claims.

It renders its inputs generically — it sees exactly the fields the incoming handoffs chose to
pass, nothing more — so what it can conclude is bounded by the collaboration interface, not by
anything hidden in this module.
"""

from __future__ import annotations

import json
from typing import Any

from agentcoop import llm
from agentcoop.workflow import Artifact, RunContext, agent

SYSTEM = (
    "You are the lead scientist integrating results from collaborating analysis agents into "
    "a final report. Base every claim only on the evidence items provided; cite them by id."
)

SCHEMA = """Return a JSON object:
{
  "claims": [
    {"cell_type": "<one cell type>",
     "direction": "up" | "down",
     "program": "<short name of the biological program>",
     "genes": ["<supporting gene symbols>"],
     "evidence": ["<evidence item ids>"],
     "confidence": <0..1>}
  ],
  "summary": "<2-4 sentences>"
}
One claim per (cell type, program). "up" means higher in treated than control. Omit anything you
cannot support; an empty claims list is allowed."""


def render(inputs: list[Artifact]) -> str:
    blocks = []
    for a in inputs:
        if a.kind == "task":
            continue
        body = json.dumps(a.data, indent=1, default=str)
        blocks.append(f"[{a.id}] ({a.kind})\n{body}")
    return "\n\n".join(blocks)


@agent("integrator")
def integrator(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    instructions = config.get("instructions", "")
    user = (
        f"Research question:\n{ctx.task.question}\n\n"
        f"Evidence items from collaborating agents:\n\n{render(inputs)}\n\n"
        + (f"Additional instructions:\n{instructions}\n\n" if instructions else "")
        + SCHEMA
    )
    report, _ = llm.chat_json(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        reasoning_effort=config.get("reasoning_effort", "low"),
        sample=ctx.sample,
    )
    claims = report.get("claims") if isinstance(report, dict) else None
    if not isinstance(claims, list):
        claims = []
    report = {"claims": claims, "summary": str(report.get("summary", ""))}
    return [Artifact(kind="report", producer=ctx.node_id, data=report,
                     derived_from=[a.id for a in inputs])]


def report_artifact(report: Any, producer: str, derived_from: list[str],
                    missing: str = "(no report submitted)") -> Artifact:
    """Normalize whatever an agent submitted into the report artifact every check reads."""
    if isinstance(report, list):
        report = {"claims": report}  # an array of claims is still a report
    if not isinstance(report, dict):
        report = {"claims": [], "summary": missing}
    claims = report.get("claims") if isinstance(report.get("claims"), list) else []
    return Artifact(kind="report", producer=producer,
                    data={"claims": claims, "summary": str(report.get("summary", ""))},
                    derived_from=derived_from)
