"""Independent analysis agents for BixBench questions.

Three independently developed agents, all on the same backbone, all executing code in the same
analysis environment (``external_agents/bixbench_env``, a native replica of BixBench's notebook
image) on a private copy-on-write clone of the capsule's data:

- ``finch``  — FutureHouse's BixBench agent (data-analysis-crow), unmodified, BixBench's own
  configuration; it reaches the model through ``agentcoop.proxy``.
- ``codex_analyst`` — the Codex CLI on the user's local login (``-m <backbone>``).
- ``react_analyst`` — our in-house ReAct coder (Responses API, persistent Python kernel, shell).

Each produces one ``analysis`` artifact: the short answer, plus the executed trace (code and its
output) and the code files it wrote — the material an answer-free check can re-examine. An agent
that fails *as an agent* (no answer) yields an analysis with ``answer=None``; an infrastructure
failure (spending cap, proxy, login) raises so it is never cached as a result.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator

from agentcoop import llm, proxy
from agentcoop.agents.codex_agent import run_codex
from agentcoop.agents.coder import Kernel
from agentcoop.workflow import EXTERNAL, ROOT, Artifact, RunContext, agent

BIXENV = Path(os.environ.get("AGENTCOOP_BIXBENCH_ENV",
                             Path.home() / "miniconda3" / "envs" / "bixbench"))
FINCH_PY = ROOT / "external_agents" / "finch" / ".venv" / "bin" / "python"
RUN_FINCH = ROOT / "external_agents" / "run_finch.py"
KERNELS = ROOT / "external_agents" / "bixbench_env" / "jupyter"
CODE_SUFFIXES = {".py", ".R", ".r", ".sh", ".ipynb"}

EXTERNAL.update({"finch", "codex_analyst"})

TOOLS_TEXT = (
    "Installed: Python 3.12 (`python`: pandas, numpy, scipy, statsmodels, scikit-learn, scanpy, "
    "anndata, pydeseq2, gseapy, biopython, mygene, rpy2) and R 4.4 (`Rscript`: DESeq2, "
    "clusterProfiler, org.Hs.eg.db, apeglm, tidyverse, data.table, rstatix, multcomp), plus "
    "mafft, clipkit, iqtree2, phykit, blast, fastqc and trim_galore on the command line."
)


def draw(ctx: RunContext, config: dict[str, Any]) -> int:
    """Sample index for the agent's model calls: ``config["replica"]`` makes otherwise identical
    nodes independent re-draws (self-consistency), not cache replays of one run."""
    return ctx.sample * 100 + int(config.get("replica", 0))


def question_for(ctx: RunContext, inputs: list[Artifact]) -> str:
    """The question, plus any repair note routed to this analyst (``audit_feedback``)."""
    notes = [a.data["text"] for a in inputs if a.kind == "repair_note" and a.data.get("text")]
    return ctx.task.question + ("\n\nNote: " + " ".join(notes) if notes else "")


class InfraError(RuntimeError):
    """The harness, not the agent, failed (budget cap, proxy, login): never cache it."""


def exec_env(ws: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if k != "OPENAI_API_KEY" and not k.startswith("AGENTCOOP_")}
    env["PATH"] = f"{BIXENV / 'bin'}:{env.get('PATH', '/usr/bin:/bin')}"
    # The shared env is read-only (setup.sh): a package an agent installs goes to its own
    # workspace (pip falls back to the user site, R to R_LIBS_USER), so no run changes another.
    tmp = ws / ".tmp"
    env.update(TMPDIR=str(tmp), MPLCONFIGDIR=str(tmp / "mpl"), NUMBA_CACHE_DIR=str(tmp / "numba"),
               JUPYTER_PATH=str(KERNELS), PYTHONUSERBASE=str(tmp / "pyuser"),
               R_LIBS_USER=str(tmp / "Rlib"))
    for d in (tmp / "mpl", tmp / "numba", tmp / "Rlib",
              tmp / "pyuser" / "lib" / "python3.12" / "site-packages"):
        d.mkdir(parents=True, exist_ok=True)
    return env


@contextlib.contextmanager
def workspace(ctx: RunContext, prefix: str) -> Iterator[Path]:
    """A private copy-on-write clone of the capsule's data files."""
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        ws = Path(tmp) / "work"
        ws.mkdir()
        subprocess.run(["cp", "-cR", f"{ctx.task.data_path}/.", str(ws)], check=True)
        yield ws


def short_answer(text: Any) -> str | None:
    if text is None:
        return None
    s = str(text)
    m = re.search(r"<answer>(.*?)</answer>", s, re.DOTALL)
    return (m.group(1) if m else s).strip() or None


def code_files(ws: Path, before: set[Path], limit: int = 40_000) -> dict[str, str]:
    """Code files the agent wrote (not the capsule's own), truncated."""
    out = {}
    for p in sorted(ws.rglob("*")):
        if p.is_file() and p.suffix in CODE_SUFFIXES and p not in before \
                and ".tmp" not in p.parts and ".home" not in p.parts:
            out[str(p.relative_to(ws))] = p.read_text(errors="replace")[:limit]
    return out


def leak_suspect(texts: list[str]) -> bool:
    """Did the agent reach for the benchmark's answers (outside its workspace)?"""
    pat = re.compile(r"data/tasks|BixBench\.jsonl|bench/bixbench|/reference/|truth\.json|"
                     r"huggingface\.co/datasets/futurehouse|hf_hub_download", re.I)
    return any(pat.search(t or "") for t in texts)


def analysis(ctx: RunContext, agent_name: str, answer: Any, trace: list[dict[str, str]],
             files: dict[str, str], inputs: list[Artifact], **extra: Any) -> Artifact:
    texts = [t.get("code", "") for t in trace] + list(files.values())
    return Artifact(kind="analysis", producer=ctx.node_id, data={
        "agent": agent_name, "answer": short_answer(answer), "raw_answer": answer,
        "trace": trace, "files": files, **extra},
        derived_from=[a.id for a in inputs],
        meta={"agent": agent_name, "n_steps": len(trace), "leak_suspect": leak_suspect(texts)})


# ---------------------------------------------------------------------------


def _notebook_trace(nb_path: Path) -> list[dict[str, str]]:
    if not nb_path.exists():
        return []
    nb = json.loads(nb_path.read_text())
    trace = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        outs = []
        for o in cell.get("outputs", []):
            if "text" in o:
                outs.append("".join(o["text"]) if isinstance(o["text"], list) else o["text"])
            elif "data" in o and "text/plain" in o["data"]:
                v = o["data"]["text/plain"]
                outs.append("".join(v) if isinstance(v, list) else v)
            elif o.get("output_type") == "error":
                outs.append(f"{o.get('ename')}: {o.get('evalue')}")
        src = cell.get("source", "")
        trace.append({"code": "".join(src) if isinstance(src, list) else src,
                      "output": "\n".join(outs)[-3000:]})
    return trace


@agent("finch")
def finch(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    with workspace(ctx, "finch_") as ws:
        before = set(ws.rglob("*"))
        req = {"work_dir": str(ws), "question": question_for(ctx, inputs),
               "base_url": proxy.base_url(proxy.shared(), llm.current_tag(), draw(ctx, config)),
               "model": llm.external_model(), "max_steps": int(config.get("max_steps", 40)),
               "language": config.get("language", "python")}
        env = exec_env(ws)
        env["PATH"] = f"{FINCH_PY.parent}:{env['PATH']}"
        try:
            proc = subprocess.run([str(FINCH_PY), str(RUN_FINCH)], input=json.dumps(req),
                                  capture_output=True, text=True, env=env, cwd=ws,
                                  timeout=int(config.get("timeout_s", 3600)))
            lines = [x for x in proc.stdout.splitlines() if x.startswith("AGENTCOOP_RESULT ")]
            result = json.loads(lines[-1].removeprefix("AGENTCOOP_RESULT ")) if lines else {
                "answer": None, "error": f"finch exited {proc.returncode}: {proc.stderr[-500:]}"}
        except subprocess.TimeoutExpired:
            result = {"answer": None, "error": "timeout"}
        err = result.get("error") or ""
        if "agentcoop budget" in err or "Connection refused" in err or "APIConnectionError" in err:
            raise InfraError(err[:500])
        trace = _notebook_trace(ws / "notebook.ipynb")
        files = code_files(ws, before | {ws / "notebook.ipynb"})
    return [analysis(ctx, "finch", result.get("answer"), trace, files, inputs,
                     error=err[:500] or None, steps=result.get("steps"))]


def codex_brief(question: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        "The data files for this question are in the current directory. Analyze them to answer "
        f"the question. {TOOLS_TEXT}\n"
        "Save the code you run as files in this directory (e.g. analysis.py / analysis.R). When "
        'you are done, write answer.json containing {"answer": "<your concise final answer>"}.'
    )


@agent("codex_analyst")
def codex_analyst(inputs: list[Artifact], config: dict[str, Any],
                  ctx: RunContext) -> list[Artifact]:
    model = config.get("model") or llm.external_model()
    with workspace(ctx, "codexbix_") as ws:
        before = set(ws.rglob("*"))
        env = exec_env(ws)
        result = run_codex(ws, codex_brief(question_for(ctx, inputs)), model,
                           int(config.get("timeout_s", 3600)), network=True,
                           report_name="answer.json", extra_env={
                               "PATH": env["PATH"], "TMPDIR": env["TMPDIR"],
                               "JUPYTER_PATH": env["JUPYTER_PATH"]})
        files = code_files(ws, before)
    report = result["report"]
    answer = report.get("answer") if isinstance(report, dict) else report
    trace = [{"code": e["command"], "output": e["output"]} for e in result["executions"]]
    return [analysis(ctx, "codex", answer, trace, files, inputs, timed_out=result["timed_out"],
                     usage=result["usage"])]


REACT_TOOLS = [
    {"type": "function", "name": "run_python",
     "description": "Execute Python code in a persistent kernel (variables persist). Returns "
                    "captured stdout/stderr (last 6000 chars).",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                    "required": ["code"]}},
    {"type": "function", "name": "run_shell",
     "description": "Run a bash command in the working directory (e.g. `Rscript script.R`). "
                    "Returns stdout/stderr (last 6000 chars).",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                    "required": ["command"]}},
    {"type": "function", "name": "submit_answer",
     "description": "Submit your concise final answer to the question (ends the task).",
     "parameters": {"type": "object", "properties": {"answer": {"type": "string"}},
                    "required": ["answer"]}},
]


def react_brief(question: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        "The data files for this question are in your working directory. Analyze them with "
        f"run_python / run_shell to answer the question. {TOOLS_TEXT}\n"
        "When done, call submit_answer with your concise final answer."
    )


def _shell(cmd: str, ws: Path, env: dict[str, str], timeout: int) -> str:
    try:
        p = subprocess.run(["bash", "-lc", cmd], cwd=ws, env=env, capture_output=True,
                           text=True, errors="replace", timeout=timeout)
        return ((p.stdout or "") + (p.stderr or ""))[-6000:] + f"\n[exit {p.returncode}]"
    except subprocess.TimeoutExpired:
        return f"(timed out after {timeout}s)"


@agent("react_analyst")
def react_analyst(inputs: list[Artifact], config: dict[str, Any],
                  ctx: RunContext) -> list[Artifact]:
    max_steps = int(config.get("max_steps", 40))
    effort = config.get("reasoning_effort", "medium")
    items: list[dict[str, Any]] = [{"role": "user",
                                    "content": react_brief(question_for(ctx, inputs))}]
    trace: list[dict[str, str]] = []
    answer = None
    with workspace(ctx, "react_") as ws:
        before = set(ws.rglob("*"))
        env = exec_env(ws)
        kernel = Kernel(ws, timeout=1200, python=str(BIXENV / "bin" / "python"), env=env)
        try:
            for step in range(max_steps):
                r = llm.responses(items, tools=REACT_TOOLS, reasoning_effort=effort,
                                  sample=draw(ctx, config) * 1000 + step)
                items += r["output"]
                calls = [o for o in r["output"] if o.get("type") == "function_call"]
                if not calls:
                    items.append({"role": "user",
                                  "content": "Continue; when done, call submit_answer."})
                    continue
                for call in calls:
                    try:
                        args = json.loads(call.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    if call["name"] == "submit_answer":
                        answer, out = args.get("answer"), "answer submitted"
                    elif call["name"] == "run_shell":
                        out = _shell(args.get("command", ""), ws, env, 1200)
                        trace.append({"code": "$ " + args.get("command", ""), "output": out[-3000:]})
                    else:
                        try:
                            out = kernel.run(args.get("code", ""))
                        except TimeoutError as exc:
                            out = str(exc)
                            kernel = Kernel(ws, timeout=1200, python=str(BIXENV / "bin" / "python"),
                                            env=env)
                        trace.append({"code": args.get("code", ""), "output": out[-3000:]})
                    items.append({"type": "function_call_output", "call_id": call["call_id"],
                                  "output": out})
                if answer is not None:
                    break
        finally:
            kernel.close()
        files = code_files(ws, before)
    return [analysis(ctx, "react", answer, trace, files, inputs)]
