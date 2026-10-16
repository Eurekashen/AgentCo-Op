"""The in-house ReAct coder (robustness arm only, spec §4): one LLM with a persistent Python kernel
and a shell, both inside the workspace sandbox, on the site agents' analysis env.

Its deliverables are the files it writes in the workspace; ``finish(summary)`` ends the run.
Model calls go through ``llm.responses`` (cache, ledger, cap) with sample index
``sample * 1000 + step``: the same sample replays a run exactly, a new one re-draws it.
"""

from __future__ import annotations

import contextlib
import json
import os
import select
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from agentcoop import llm
from agentcoop.agents.workspace import BIXENV, exec_env, run_group, sandboxed

PYTHON = str(BIXENV / "bin" / "python")

#: The protocol has its own descriptor; output written below Python (subprocesses, C code) is
#: captured per cell; a cell cannot end the kernel.
KERNEL = r'''
import io, json, os, sys, tempfile, traceback, contextlib
proto = os.fdopen(os.dup(1), "w"); g = {"__name__": "__main__"}
for line in sys.stdin:
    code, buf, raw = json.loads(line)["code"], io.StringIO(), tempfile.TemporaryFile()
    saved = os.dup(1), os.dup(2); os.dup2(raw.fileno(), 1); os.dup2(raw.fileno(), 2)
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(code, "<cell>", "exec"), g)
    except KeyboardInterrupt:
        raise
    except BaseException:
        buf.write(traceback.format_exc())
    finally:
        os.dup2(saved[0], 1); os.dup2(saved[1], 2); os.close(saved[0]); os.close(saved[1])
    raw.seek(0)
    out = buf.getvalue() + raw.read().decode(errors="replace")
    raw.close()
    proto.write(json.dumps({"out": out[-6000:]}) + "\n")
    proto.flush()
'''

TOOLS = [
    {"type": "function", "name": "run_python",
     "description": "Execute Python code in a persistent kernel (variables persist between "
                    "calls). Returns captured stdout/stderr (last 6000 chars).",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                    "required": ["code"]}},
    {"type": "function", "name": "run_shell",
     "description": "Run a bash command in the working directory (e.g. `Rscript script.R`). "
                    "Returns stdout/stderr (last 6000 chars).",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                    "required": ["command"]}},
    {"type": "function", "name": "finish",
     "description": "End the task once every file the instructions ask for is written in the "
                    "working directory. summary: a short account of what you did.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}},
                    "required": ["summary"]}},
]


class Kernel:
    """A persistent Python process under the workspace sandbox, in its own process group."""

    def __init__(self, ws: Path, env: dict[str, str]) -> None:
        """Raises RuntimeError if the kernel cannot start (missing interpreter, a sandbox that
        blocks it): an infrastructure failure, not one of the agent's analysis steps."""
        self.ws, self.env = ws, env
        self._start()
        out = self.run("pass", 120)
        if out:
            self.close()
            raise RuntimeError(f"analysis kernel failed to start in the sandbox: {out[:300]}")

    def _start(self) -> None:
        self.proc = subprocess.Popen(sandboxed([PYTHON, "-u", "-c", KERNEL], self.ws),
                                     cwd=self.ws, env=self.env, text=True,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, start_new_session=True)

    def run(self, code: str, timeout: float) -> str:
        restarted = ""
        if self.proc.poll() is not None:
            self._start()
            restarted = "(the kernel had died and was restarted; variables were lost)\n"
        try:
            self.proc.stdin.write(json.dumps({"code": code}) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            return restarted + "(kernel died)"
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            self.close()
            raise TimeoutError(f"kernel call timed out after {timeout:.0f}s")
        line = self.proc.stdout.readline()
        return restarted + (json.loads(line)["out"] if line else "(kernel died)")

    def close(self) -> None:
        """Kill the kernel and everything it started."""
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc.wait()


def _shell(command: str, ws: Path, env: dict[str, str], timeout: float) -> str:
    """One command in the sandbox, in a non-login bash: a login shell's ``path_helper`` would put
    the system Python ahead of the analysis env."""
    code, out, err, timed_out = run_group(sandboxed(["/bin/bash", "-c", command], ws), ws, env,
                                          timeout)
    if timed_out:
        return f"(timed out after {timeout:.0f}s)"
    return (out + err)[-6000:] + f"\n[exit {code}]"


def coding_loop(prompt: str, ws: Path, *, sample: int, timeout: int, max_steps: int = 40,
                effort: str = "medium", model: str | None = None) -> dict[str, Any]:
    """Run the coder in ``ws`` until it calls ``finish`` or runs out of steps or time.

    Returns ``{"transcript", "timed_out", "error": None, "nominal_usd_external": 0.0}`` (its model
    calls are ledgered by ``llm`` itself); ``harness.run_agent`` wraps it into an ``AgentRun``.
    ``llm.BudgetExceeded`` and API errors propagate.
    """
    deadline = time.monotonic() + timeout
    env = exec_env(ws)
    items: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    transcript: list[dict[str, str]] = []
    summary: str | None = None
    kernel = Kernel(ws, env)
    try:
        for step in range(max_steps):
            if time.monotonic() >= deadline:
                break
            r = llm.responses(items, tools=TOOLS, reasoning_effort=effort,
                              sample=sample * 1000 + step, model=model)
            items += r["output"]
            calls = [o for o in r["output"] if o.get("type") == "function_call"]
            if not calls:
                items.append({"role": "user", "content": "Continue; when done, call finish."})
                continue
            for call in calls:
                try:
                    args = json.loads(call.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                left = max(deadline - time.monotonic(), 1.0)
                if call["name"] == "finish":
                    summary, out = str(args.get("summary", "")), "finished"
                elif call["name"] == "run_shell":
                    command = str(args.get("command", ""))
                    out = _shell(command, ws, env, left)
                    transcript.append({"code": "$ " + command, "output": out[-3000:]})
                else:
                    code = str(args.get("code", ""))
                    try:
                        out = kernel.run(code, left)
                    except TimeoutError as exc:
                        out = str(exc)
                        kernel = Kernel(ws, env)
                    transcript.append({"code": code, "output": out[-3000:]})
                items.append({"type": "function_call_output", "call_id": call["call_id"],
                              "output": out})
            if summary is not None:
                break
    finally:
        kernel.close()
    if summary is not None:
        transcript.append({"code": "(final)", "output": summary[-3000:]})
    return {"transcript": transcript,
            "timed_out": summary is None and time.monotonic() >= deadline,
            "error": None, "nominal_usd_external": 0.0}
