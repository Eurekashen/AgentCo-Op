"""A single strong agent: one LLM with a persistent Python kernel does the whole analysis.

This is the "why not just a coding agent" control at the method level: same backbone model,
same public resources as the evidence checks (GO BP and Reactome gene-set libraries, not the
Hallmark library the benchmark planted from), the data, and a standard scientific Python stack.
It receives no external agents.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from agentcoop import llm
from agentcoop.agents.integrator import SCHEMA, report_artifact
from agentcoop.workflow import ROOT, Artifact, RunContext, agent

KERNEL = r'''
import io, json, sys, traceback, contextlib
g = {"__name__": "__main__"}
for line in sys.stdin:
    code = json.loads(line)["code"]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(code, "<cell>", "exec"), g)
    except Exception:
        buf.write(traceback.format_exc())
    sys.__stdout__.write(json.dumps({"out": buf.getvalue()[-6000:]}) + "\n")
    sys.__stdout__.flush()
'''

TOOLS = [
    {"type": "function", "name": "run_python",
     "description": "Execute Python code in a persistent kernel (variables persist between "
                    "calls). Returns captured stdout/stderr (last 6000 chars).",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                    "required": ["code"]}},
    {"type": "function", "name": "submit_report",
     "description": "Submit the final report as a JSON string following the required schema.",
     "parameters": {"type": "object", "properties": {"report_json": {"type": "string"}},
                    "required": ["report_json"]}},
]


class Kernel:
    def __init__(self, cwd: Path, timeout: int = 600) -> None:
        self.timeout = timeout
        self.proc = subprocess.Popen([sys.executable, "-u", "-c", KERNEL], cwd=cwd,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True)

    def run(self, code: str) -> str:
        import select

        self.proc.stdin.write(json.dumps({"code": code}) + "\n")
        self.proc.stdin.flush()
        ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout)
        if not ready:
            self.close()
            raise TimeoutError("kernel call timed out")
        line = self.proc.stdout.readline()
        return json.loads(line)["out"] if line else "(kernel died)"

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()


def brief(ctx: RunContext) -> str:
    genesets = ROOT / "data" / "genesets"
    return (
        f"Research question:\n{ctx.task.question}\n\n"
        f"Data: an AnnData file at {ctx.task.data_path} (raw UMI counts in .X; obs columns "
        "'cell_type', 'donor', 'condition' with values control|treated).\n"
        f"Gene-set libraries (Enrichr text format: name<TAB><TAB>gene<TAB>gene...):\n"
        f"  {genesets / 'GO_Biological_Process_2023.txt'}\n  {genesets / 'Reactome_2022.txt'}\n"
        "Python packages available: scanpy, anndata, numpy, pandas, scipy, statsmodels, "
        "sklearn, pydeseq2.\n\nAnalyze the data with run_python, then call submit_report with "
        "the final report.\n\n" + SCHEMA
    )


def coding_loop(brief_text: str, *, sample: int, max_steps: int = 30, effort: str = "medium",
                files: dict[str, Path] | None = None,
                model: str | None = None) -> tuple[dict[str, Any] | None, list]:
    """Run one coding agent to completion. ``files`` are copied (not linked) into a private
    working directory, which is the only data the agent is given a path to."""
    items: list[dict[str, Any]] = [{"role": "user", "content": brief_text}]
    transcript: list[dict[str, Any]] = []
    report: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="coder_") as tmp:
        for name, src in (files or {}).items():
            shutil.copy(src, Path(tmp) / name)
        kernel = Kernel(Path(tmp))
        try:
            for step in range(max_steps):
                r = llm.responses(items, tools=TOOLS, reasoning_effort=effort,
                                  sample=sample * 1000 + step, model=model)
                items += r["output"]
                calls = [o for o in r["output"] if o.get("type") == "function_call"]
                if not calls:
                    items.append({"role": "user", "content":
                                  "Continue; when done, call submit_report."})
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
                        try:
                            out = kernel.run(args.get("code", ""))
                        except TimeoutError as exc:
                            out = str(exc)
                            kernel = Kernel(Path(tmp))
                    transcript.append({"step": step, "tool": call["name"],
                                       "input": args, "output": out[:2000]})
                    items.append({"type": "function_call_output", "call_id": call["call_id"],
                                  "output": out})
                if report is not None:
                    break
        finally:
            kernel.close()
    return report, transcript


@agent("coder")
def coder(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    report, transcript = coding_loop(brief(ctx), sample=ctx.sample,
                                     max_steps=int(config.get("max_steps", 30)),
                                     effort=config.get("reasoning_effort", "medium"))
    return [report_artifact(report, ctx.node_id, [a.id for a in inputs]),
            Artifact(kind="transcript", producer=ctx.node_id, data={"steps": transcript})]
