"""A single autonomous agent that orchestrates the site agents itself (shape-2 control).

Same backbone, same data boundary as every other shape-2 method: it cannot read site data, but it
may send any request it likes to any site agent, as many times as its step budget allows — so it
can, if it thinks of it, ask for structured findings, weak effects, or cross-site verification.
This is the "why not just let a strong agent coordinate them?" arm.
"""

from __future__ import annotations

import json
from typing import Any

from agentcoop import llm
from agentcoop.agents.integrator import SCHEMA
from agentcoop.agents.site import GENESETS, site_brief, site_file
from agentcoop.agents.coder import coding_loop
from agentcoop.workflow import Artifact, RunContext, agent

TOOLS = [
    {"type": "function", "name": "ask_site",
     "description": "Send a request to the analysis agent of one site; it analyses its own data "
                    "and returns a JSON report. Takes a few minutes.",
     "parameters": {"type": "object", "properties": {
         "site": {"type": "string", "enum": ["A", "B"]},
         "request": {"type": "string"}}, "required": ["site", "request"]}},
    {"type": "function", "name": "submit_report",
     "description": "Submit the final report as a JSON string following the required schema.",
     "parameters": {"type": "object", "properties": {"report_json": {"type": "string"}},
                    "required": ["report_json"]}},
]


@agent("orchestrator")
def orchestrator(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    max_steps = int(config.get("max_steps", 12))
    brief = (f"You lead a two-site study.\n\n{ctx.task.question}\n\nYou cannot read any site's "
             "data. Use ask_site to request analyses from each site's analysis agent (you decide "
             "what to ask, and you may ask follow-up questions), then call submit_report.\n\n"
             + SCHEMA)
    items: list[dict[str, Any]] = [{"role": "user", "content": brief}]
    log, report, n_asks = [], None, 0
    for step in range(max_steps):
        r = llm.responses(items, tools=TOOLS, reasoning_effort=config.get("reasoning_effort",
                                                                          "medium"),
                          sample=ctx.sample * 1000 + step)
        items += r["output"]
        calls = [o for o in r["output"] if o.get("type") == "function_call"]
        if not calls:
            items.append({"role": "user", "content": "Continue; when done, call submit_report."})
            continue
        for call in calls:
            try:
                args = json.loads(call.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if call["name"] == "submit_report":
                try:
                    report = json.loads(args.get("report_json", ""))
                    out = "report accepted"
                except (json.JSONDecodeError, TypeError) as exc:
                    out = f"invalid JSON: {exc}"
            else:
                site = str(args.get("site", "A"))
                files = {f"site_{site}.h5ad": site_file(ctx, site),
                         "GO_Biological_Process_2023.txt": GENESETS / "GO_Biological_Process_2023.txt",
                         "Reactome_2022.txt": GENESETS / "Reactome_2022.txt"}
                rep, _ = coding_loop(site_brief(site, str(args.get("request", ""))),
                                     sample=ctx.sample * 100 + n_asks, files=files,
                                     model=llm.external_model())
                n_asks += 1
                out = json.dumps(rep if isinstance(rep, dict) else {"error": "no report"})[:8000]
            log.append({"step": step, "tool": call["name"], "args": args, "output": out[:1500]})
            items.append({"type": "function_call_output", "call_id": call["call_id"],
                          "output": out})
        if report is not None:
            break
    if isinstance(report, list):
        report = {"claims": report}  # an array of claims is still a report
    if not isinstance(report, dict):
        report = {"claims": [], "summary": "(no report submitted)"}
    claims = report.get("claims") if isinstance(report.get("claims"), list) else []
    return [Artifact(kind="report", producer=ctx.node_id,
                     data={"claims": claims, "summary": str(report.get("summary", ""))},
                     derived_from=[a.id for a in inputs]),
            Artifact(kind="transcript", producer=ctx.node_id,
                     data={"steps": log, "n_site_requests": n_asks})]
