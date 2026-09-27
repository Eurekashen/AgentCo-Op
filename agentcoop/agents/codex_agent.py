"""The Codex CLI as a product-level single-agent baseline (same backbone, gpt-6-luna).

Runs ``codex exec`` headless in a throwaway workspace containing the data and the same public
gene-set libraries the coding-agent baseline gets, with the same task brief. Authentication uses
an isolated ``CODEX_HOME`` under ``.secrets/`` logged in with the project API key, so the user's
own Codex configuration and login are never read or changed. Token usage from the event stream
is priced with the same table as every other method.
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

CODEX_HOME = ROOT / ".secrets" / "codex_home"


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


@agent("codex")
def codex(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    model = config.get("model", llm.DEFAULT_MODEL)
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
        cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config",
               "-c", "shell_environment_policy.inherit=all",
               "-m", model, "-s", "workspace-write", "--json",
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
        llm.record_external(model, usage["input_tokens"], usage["cached_input_tokens"],
                            usage["cache_write_input_tokens"], usage["output_tokens"])
        report_path = ws / "report.json"
        try:
            report = json.loads(report_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            report = {"claims": [], "summary": "(no valid report.json written)"}
        n_commands = sum(1 for e in events if e.get("type") == "item.completed"
                         and e.get("item", {}).get("type") == "command_execution")
    claims = report.get("claims") if isinstance(report.get("claims"), list) else []
    return [
        Artifact(kind="report", producer=ctx.node_id,
                 data={"claims": claims, "summary": str(report.get("summary", ""))},
                 derived_from=[a.id for a in inputs]),
        Artifact(kind="transcript", producer=ctx.node_id,
                 data={"usage": usage, "n_events": len(events), "n_commands": n_commands,
                       "timed_out": timed_out}),
    ]
