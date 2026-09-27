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
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agentcoop import llm
from agentcoop.agents.integrator import SCHEMA
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
    """The model the user's Codex config selects (for costing a native run)."""
    try:
        for line in (CODEX_HOME / "config.toml").read_text().splitlines():
            if line.strip().startswith("model") and "=" in line and "_" not in line.split("=")[0]:
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return llm.backbone()


@agent("codex")
def codex(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    model = config.get("model") or llm.backbone()
    timeout = int(config.get("timeout_s", 2400))
    with tempfile.TemporaryDirectory(prefix="codex_") as tmp:
        ws = Path(tmp)
        os.symlink(ctx.task.data_path, ws / "counts.h5ad")
        (ws / "genesets").mkdir()
        for lib in ("GO_Biological_Process_2023.txt", "Reactome_2022.txt"):
            shutil.copy(ROOT / "data" / "genesets" / lib, ws / "genesets" / lib)
        subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
        # Hermetic: a throwaway HOME keeps the user's global skills/config out of the run, and
        # the sandbox needs writable caches for numba/matplotlib or scanpy fails to import.
        home = ws / ".home"
        for d in (home, ws / ".cache" / "numba", ws / ".cache" / "mpl"):
            d.mkdir(parents=True)
        env = dict(os.environ, CODEX_HOME=str(CODEX_HOME), HOME=str(home),
                   NUMBA_CACHE_DIR=str(ws / ".cache" / "numba"),
                   MPLCONFIGDIR=str(ws / ".cache" / "mpl"))
        native = model == "native"
        if native:
            env["HOME"] = str(Path.home())  # the user's own setup, as they would run it
        cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral",
               *([] if native else ["--ignore-user-config", "-c", "project_doc_max_bytes=0",
                                    "-m", model]),
               "-c", "shell_environment_policy.inherit=all",
               "-s", "workspace-write", "--json",
               brief(ctx.task.question, config.get("extra_brief", ""))]
        try:
            proc = subprocess.run(cmd, cwd=ws, env=env, stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=timeout)
            events = [json.loads(l) for l in proc.stdout.splitlines() if l.strip().startswith("{")]
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            events = [json.loads(l) for l in out.splitlines() if l.strip().startswith("{")]
            timed_out = True
        usage = {"input_tokens": 0, "cached_input_tokens": 0, "cache_write_input_tokens": 0,
                 "output_tokens": 0}
        for e in events:
            if e.get("type") == "turn.completed":
                for k in usage:
                    usage[k] += int(e.get("usage", {}).get(k, 0) or 0)
        priced = _native_model() if native else model
        llm.record_external(priced, usage["input_tokens"], usage["cached_input_tokens"],
                            usage["cache_write_input_tokens"], usage["output_tokens"],
                            billed=False)
        report_path = ws / "report.json"
        try:
            report = json.loads(report_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            report = {"claims": [], "summary": "(no valid report.json written)"}
        n_commands = sum(1 for e in events if e.get("type") == "item.completed"
                         and e.get("item", {}).get("type") == "command_execution")
    if isinstance(report, list):
        report = {"claims": report}  # an array of claims is still a report
    if not isinstance(report, dict):
        report = {"claims": [], "summary": "(report.json is not a JSON object)"}
    claims = report.get("claims") if isinstance(report.get("claims"), list) else []
    return [
        Artifact(kind="report", producer=ctx.node_id,
                 data={"claims": claims, "summary": str(report.get("summary", ""))},
                 derived_from=[a.id for a in inputs]),
        Artifact(kind="transcript", producer=ctx.node_id,
                 data={"usage": usage, "n_events": len(events), "n_commands": n_commands,
                       "timed_out": timed_out}),
    ]
