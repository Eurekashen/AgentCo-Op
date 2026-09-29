"""The Codex CLI as a single-agent baseline.

Runs the user's locally installed, logged-in ``codex exec`` headless in a throwaway workspace
containing the data and the same public gene-set libraries the coding-agent baseline gets, with
the same task brief. Only the login is borrowed from the user's ``CODEX_HOME``; the run is
otherwise hermetic: ``--ignore-user-config`` (no personal config.toml), a throwaway ``HOME`` (no
personal skills), ``project_doc_max_bytes=0``.

``config["model"]``: default = the experiment backbone (same-model comparison); ``"native"`` =
Codex exactly as the user runs it (their config and default model) — a product-level arm.
Usage is billed to the user's ChatGPT plan, so it enters the ledger as nominal (API-equivalent)
cost with $0 actual.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agentcoop import llm
from agentcoop.agents.integrator import SCHEMA, report_artifact
from agentcoop.workflow import ROOT, Artifact, RunContext, agent

CODEX_HOME = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))


def brief(question: str, extra: str = "") -> str:
    return (
        f"Research question:\n{question}\n\n"
        "Data: counts.h5ad in this directory (AnnData; raw UMI counts in .X; obs columns "
        "'cell_type', 'donor', 'condition' with values control|treated).\n"
        "Gene-set libraries in ./genesets (Enrichr text format: name<TAB><TAB>gene<TAB>gene...): "
        "GO_Biological_Process_2023.txt, Reactome_2022.txt.\n"
        "Python with scanpy, anndata, numpy, pandas, scipy, statsmodels, sklearn and pydeseq2 is "
        "available (`python`).\n" + extra +
        "\nAnalyze the data, then write the final report to report.json in this directory.\n\n"
        + SCHEMA
    )


def _native_model() -> str:
    """The top-level model the user's Codex config selects (for costing a native run)."""
    import tomllib

    try:
        model = tomllib.loads((CODEX_HOME / "config.toml").read_text()).get("model")
    except (OSError, tomllib.TOMLDecodeError):
        model = None
    return model if model in llm.PRICES else llm.backbone()


class CodexFailed(RuntimeError):
    """Codex exited abnormally without a report (login, quota or network failure): an
    infrastructure error, recorded as a node failure rather than cached as a result."""


def _events(stdout: str) -> list[dict[str, Any]]:
    out = []
    for line in stdout.splitlines():
        if line.strip().startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # a truncated last line of a killed run
                pass
    return out


def run_codex(ws: Path, prompt: str, model: str, timeout: int, *,
              network: bool = False) -> dict[str, Any]:
    """Run ``codex exec`` in ``ws`` on the user's local login, hermetic except for that login.

    Returns events, usage, the executed commands (unique, by item id), ``timed_out`` and the
    parsed ``report.json`` (``None`` if missing or invalid). Raises :class:`CodexFailed` when
    Codex fails without writing a report. Usage enters the ledger as nominal cost only (billed
    to the user's ChatGPT plan).
    """
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    # A throwaway HOME keeps the user's global skills/config out of the run; the sandbox needs
    # writable caches for numba/matplotlib or scanpy fails to import. No API key is passed on:
    # Codex must use the local login.
    home = ws / ".home"
    for d in (home, ws / ".cache" / "numba", ws / ".cache" / "mpl"):
        d.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if k != "OPENAI_API_KEY" and not k.startswith("AGENTCOOP_")}
    env.update(CODEX_HOME=str(CODEX_HOME), HOME=str(home),
               NUMBA_CACHE_DIR=str(ws / ".cache" / "numba"), MPLCONFIGDIR=str(ws / ".cache" / "mpl"))
    native = model == "native"
    if native:
        env["HOME"] = str(Path.home())  # the user's own setup, as they would run it
    cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral",
           *([] if native else ["--ignore-user-config", "-c", "project_doc_max_bytes=0",
                                "-m", model]),
           "-c", "shell_environment_policy.inherit=all",
           *(["-c", "sandbox_workspace_write.network_access=true"] if network else []),
           "-s", "workspace-write", "--json", prompt]
    proc = subprocess.Popen(cmd, cwd=ws, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            errors="replace", start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)  # codex and everything it started
        stdout, stderr = proc.communicate()
        timed_out = True
    events = _events(stdout)
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "cache_write_input_tokens": 0,
             "output_tokens": 0}
    for e in events:
        if e.get("type") == "turn.completed":
            for k in usage:
                usage[k] += int(e.get("usage", {}).get(k, 0) or 0)
    llm.record_external(_native_model() if native else model, usage["input_tokens"],
                        usage["cached_input_tokens"], usage["cache_write_input_tokens"],
                        usage["output_tokens"], billed=False)
    commands = {e["item"].get("id", i): str(e["item"].get("command", ""))
                for i, e in enumerate(events) if e.get("type") == "item.completed"
                and e.get("item", {}).get("type") == "command_execution"}
    try:
        report = json.loads((ws / "report.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        report = None
    failed = any(e.get("type") in ("turn.failed", "error") for e in events)
    if report is None and not timed_out and (proc.returncode != 0 or failed):
        raise CodexFailed(f"codex exit {proc.returncode}: {stderr.strip()[-300:]}")
    return {"events": events, "usage": usage, "commands": list(commands.values()),
            "timed_out": timed_out, "report": report}


def report_of(result: dict[str, Any], producer: str, derived_from: list[str]) -> Artifact:
    report = result["report"]
    if report is None:
        report = {"claims": [], "summary": "(no valid report.json written)"}
    return report_artifact(report, producer, derived_from,
                           missing="(report.json is not a JSON object)")


@agent("codex")
def codex(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    model = config.get("model") or llm.backbone()
    with tempfile.TemporaryDirectory(prefix="codex_") as tmp:
        ws = Path(tmp)
        # A copy, not a link: nothing in the workspace points back into the task directory.
        shutil.copy(ctx.pooled_data(), ws / "counts.h5ad")
        (ws / "genesets").mkdir()
        for lib in ("GO_Biological_Process_2023.txt", "Reactome_2022.txt"):
            shutil.copy(ROOT / "data" / "genesets" / lib, ws / "genesets" / lib)
        result = run_codex(ws, brief(ctx.task.question, config.get("extra_brief", "")), model,
                           int(config.get("timeout_s", 2400)))
    return [
        report_of(result, ctx.node_id, [a.id for a in inputs]),
        Artifact(kind="transcript", producer=ctx.node_id,
                 data={"usage": result["usage"], "n_events": len(result["events"]),
                       "n_commands": len(result["commands"]), "commands": result["commands"],
                       "timed_out": result["timed_out"]}),
    ]
