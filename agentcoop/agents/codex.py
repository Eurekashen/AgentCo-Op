"""The Codex CLI as a site agent, on the user's local login.

Hermetic except for that login: ``--ignore-user-config``, ``project_doc_max_bytes=0``, a
throwaway HOME, no API key or AGENTCOOP_* settings in its environment. Codex reaches its API
itself, so it cannot run inside the network-denying ``sandbox-exec`` profile; every command it
runs is confined by Codex's own permission profile instead: the filesystem readable, the
workspace writable, the silo paths (``workspace.DENY``) unreadable, no network, web search off.
Commands run in a non-login shell, because a login shell's ``path_helper`` puts the system Python
ahead of the analysis env. An item type outside the plain coding loop (web search, MCP tools, ...)
would cross the silo boundary, so it fails the run.

``model="native"`` runs the user's own config and default model (a product-level arm). Usage is
billed to the user's ChatGPT plan: it is ledgered as nominal (API-equivalent) cost, $0 actual.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentcoop import llm
from agentcoop.agents import workspace

ITEM_TYPES = frozenset({"agent_message", "reasoning", "command_execution", "file_change",
                        "todo_list"})


def _native_model() -> str:
    """The top-level model the user's Codex config selects (for costing a native run)."""
    import tomllib

    try:
        model = tomllib.loads((workspace.CODEX_HOME / "config.toml").read_text()).get("model")
    except (OSError, tomllib.TOMLDecodeError):
        model = None
    return model if model in llm.PRICES else llm.backbone()


class CodexFailed(RuntimeError):
    """Codex failed as infrastructure (login, quota, network) or used a tool outside the silo
    boundary: never a result to cache. Carries the call's nominal cost, which is still charged."""

    def __init__(self, message: str, nominal_usd_external: float = 0.0) -> None:
        super().__init__(message)
        self.nominal_usd_external = nominal_usd_external


def _events(stdout: str) -> list[dict[str, Any]]:
    out = []
    for line in stdout.splitlines():
        if line.strip().startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # a truncated last line of a killed run
                pass
    return out


def permission_args(ws: Path) -> list[str]:
    """Config overrides for Codex's commands: read anywhere, write only ``ws``, no access to the
    silo paths, no network (the profile's default), no web search, no login shell. Paths are
    resolved (Seatbelt matches real paths); a JSON string is a valid TOML basic string."""
    fs = {":root": "read", str(ws.resolve()): "write",
          **{str(p.resolve()): "none" for p in workspace.DENY}}
    table = ",".join(f"{json.dumps(k)}={json.dumps(v)}" for k, v in fs.items())
    return ["-c", 'default_permissions="site"', "-c", f"permissions.site.filesystem={{{table}}}",
            "-c", 'web_search="disabled"', "-c", "allow_login_shell=false"]


def run_codex(ws: Path, prompt: str, model: str, timeout: int) -> dict[str, Any]:
    """Run ``codex exec`` in ``ws``; its deliverables are the files it writes there.

    Returns ``{"transcript": [{"code", "output"}], "timed_out", "error": None, "usage",
    "nominal_usd_external"}``. A timeout is an outcome, not an exception: the process group is
    killed and the files written so far stay. Raises :class:`CodexFailed` when Codex exits
    abnormally or emits an item outside ``ITEM_TYPES``; the usage is ledgered first either way.
    """
    env = workspace.exec_env(ws)
    env["CODEX_HOME"] = str(workspace.CODEX_HOME)
    native = model == "native"
    if native:
        env["HOME"] = str(Path.home())  # the user's own setup, as they would run it
    cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral",
           *([] if native else ["--ignore-user-config", "-c", "project_doc_max_bytes=0",
                                "-m", model]),
           "-c", "shell_environment_policy.inherit=all", *permission_args(ws), "--json", prompt]
    code, stdout, stderr, timed_out = workspace.run_group(cmd, ws, env, timeout)
    events = _events(stdout)
    usage = dict.fromkeys(("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
                           "output_tokens"), 0)
    for e in events:
        if e.get("type") == "turn.completed":
            for k in usage:
                usage[k] += int(e.get("usage", {}).get(k, 0) or 0)
    nominal = llm.record_external(_native_model() if native else model, usage["input_tokens"],
                                  usage["cached_input_tokens"], usage["cache_write_input_tokens"],
                                  usage["output_tokens"], billed=False)
    items = [e for e in events if str(e.get("type", "")).startswith("item.")
             and isinstance(e.get("item"), dict)]
    breach = sorted({str(e["item"].get("type")) for e in items} - ITEM_TYPES)
    if breach:
        raise CodexFailed(f"codex used item types outside the silo boundary: {breach}", nominal)
    if not timed_out and (code != 0 or any(e.get("type") in ("turn.failed", "error")
                                           for e in events)):
        raise CodexFailed(f"codex exit {code}: {stderr.strip()[-300:]}", nominal)
    done = {e["item"].get("id", n): e["item"] for n, e in enumerate(items)
            if e["type"] == "item.completed"}
    transcript = [{"code": str(i.get("command", "")),
                   "output": str(i.get("aggregated_output", ""))[-3000:]}
                  for i in done.values() if i.get("type") == "command_execution"]
    messages = [str(i.get("text", "")) for i in done.values() if i.get("type") == "agent_message"]
    if messages:
        transcript.append({"code": "(final)", "output": messages[-1][-3000:]})
    return {"transcript": transcript, "timed_out": timed_out, "error": None, "usage": usage,
            "nominal_usd_external": nominal}
