"""One entry point for every site agent: ``run_agent`` runs Codex, Finch or the ReAct coder in a
workspace under a wall-clock cap and reports the outcome as data.

An agent that fails — times out, crashes, exits abnormally, crosses the silo boundary — yields
``ok=False`` with the reason, never an exception, so one failed site cannot stop a protocol run
(the caller charges the call, counts it against the budget and does not cache it). Only
``llm.BudgetExceeded`` propagates: the spending cap stops everything. The wall-clock cap is
enforced twice: each agent stops its own processes at ``timeout``, and a backstop returns
``GRACE`` seconds later even if the agent is stuck in a model call.
"""

from __future__ import annotations

import contextvars
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agentcoop import llm
from agentcoop.agents.codex import CodexFailed, run_codex
from agentcoop.agents.finch import run_finch
from agentcoop.agents.react import coding_loop

GRACE = 60.0


@dataclass
class AgentRun:
    agent: str               # "codex" | "finch" | "react"
    ok: bool                 # False on timeout / harness failure
    transcript: list[dict]   # [{"code": str, "output": str}], truncated
    error: str | None
    # Codex API-equivalent cost (0 for finch/react; their LLM calls go through llm)
    nominal_usd_external: float
    seconds: float


def _codex(ws: Path, prompt: str, sample: int, timeout: int) -> dict[str, Any]:
    # Codex cannot be seeded: ``sample`` only separates cache entries upstream (store.memo keys).
    return run_codex(ws, prompt, llm.external_model(), timeout)


def _finch(ws: Path, prompt: str, sample: int, timeout: int) -> dict[str, Any]:
    return run_finch(ws, prompt, sample=sample, timeout=timeout)


def _react(ws: Path, prompt: str, sample: int, timeout: int) -> dict[str, Any]:
    return coding_loop(prompt, ws, sample=sample, timeout=timeout)


ENTRY: dict[str, Callable[[Path, str, int, int], dict[str, Any]]] = {
    "codex": _codex, "finch": _finch, "react": _react}


def run_agent(agent: str, ws: Path, prompt: str, *, sample: int, timeout: int) -> AgentRun:
    """Run ``agent`` on ``prompt`` in ``ws``; its deliverables are the files it leaves there.

    The agent runs on a worker thread that carries the caller's context, so its model calls are
    ledgered under the caller's tag (``store.memo`` reads its cost from that tag).
    """
    entry = ENTRY[agent]
    box: dict[str, Any] = {}
    context = contextvars.copy_context()

    def work() -> None:
        try:
            box["out"] = context.run(entry, ws, prompt, sample, timeout)
        except BaseException as exc:  # handed over to the caller's thread
            box["exc"] = exc

    start = time.monotonic()
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(timeout + GRACE)
    seconds = time.monotonic() - start
    if worker.is_alive():
        return AgentRun(agent, False, [], f"timeout after {timeout}s (agent did not stop)", 0.0,
                        seconds)
    exc = box.get("exc")
    if isinstance(exc, llm.BudgetExceeded) or (exc is not None and not isinstance(exc, Exception)):
        raise exc
    if exc is not None:
        nominal = exc.nominal_usd_external if isinstance(exc, CodexFailed) else 0.0
        return AgentRun(agent, False, [], f"{type(exc).__name__}: {exc}"[:500], nominal, seconds)
    out = box["out"]
    error = f"timeout after {timeout}s" if out["timed_out"] else out["error"]
    return AgentRun(agent, error is None, out["transcript"], error, out["nominal_usd_external"],
                    seconds)
