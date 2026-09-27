"""GeneAgent (ncbi-nlp/GeneAgent) as an external black-box component.

Upstream code runs unmodified in a subprocess (``external_agents/run_geneagent.py``); only its
model transport is replaced by the openai-0.28 shim, and its web-database calls go through a
record/replay HTTP cache. Its interface is what upstream offers: a comma-separated gene list in,
"Process: <name>" plus a verified narrative out. It knows nothing about direction, contrast,
cell type or the tested-gene universe — whatever the handoff drops is simply unavailable to it.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from agentcoop import llm
from agentcoop.workflow import ROOT, Artifact, RunContext, agent

EXTERNAL = ROOT / "external_agents"
RESULT_CACHE = ROOT / "runs" / "geneagent_cache"
HTTP_CACHE = ROOT / "runs" / "http_cache"
#: Bounds concurrent GeneAgent processes across the whole Python process (NCBI rate limits).
SLOTS = threading.BoundedSemaphore(int(os.environ.get("AGENTCOOP_GENEAGENT_SLOTS", "8")))


class GeneAgentError(RuntimeError):
    pass


def call_geneagent(genes: list[str], *, sample: int = 0, timeout: int = 1800) -> dict[str, Any]:
    genes_arg = ",".join(genes)
    model = os.environ.get("AGENTCOOP_EXTERNAL_MODEL", llm.DEFAULT_MODEL)
    key = hashlib.sha256(json.dumps([genes_arg, sample, model]).encode()).hexdigest()
    cached = RESULT_CACHE / f"{key}.json"
    if cached.exists():
        result = json.loads(cached.read_text())
        result["cached"] = True
        return result
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": f"{EXTERNAL / 'shims'}{os.pathsep}{ROOT}",
        "AGENTCOOP_TAG": llm.current_tag(),
        "AGENTCOOP_SAMPLE": str(sample),
        "AGENTCOOP_HTTP_CACHE": str(HTTP_CACHE),
    })
    with SLOTS, tempfile.TemporaryDirectory(prefix="geneagent_") as tmp:
        out = Path(tmp) / "out.json"
        proc = subprocess.run(
            [sys.executable, str(EXTERNAL / "run_geneagent.py"), genes_arg, str(out)],
            cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout,
        )
        if not out.exists():
            raise GeneAgentError(f"GeneAgent exited {proc.returncode}: {proc.stderr[-800:]}")
        result = json.loads(out.read_text())
    if result.get("ok"):
        RESULT_CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(result))
    result["cached"] = False
    return result


@agent("geneagent")
def geneagent(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    lists = [a for a in inputs if a.kind == "gene_list" and a.data.get("genes")]
    tag = llm.current_tag()

    def one(item: Artifact) -> Artifact:
        with llm.tag("geneagent") if not tag else _same_tag(tag):
            result = call_geneagent(item.data["genes"], sample=ctx.sample)
        return Artifact(
            kind="interpretation", producer=ctx.node_id,
            data={"process": result.get("process", ""), "narrative": result.get("final", ""),
                  "genes": item.data["genes"], "ok": bool(result.get("ok"))},
            derived_from=[item.id],
            meta={"source": item.meta, "agent": "GeneAgent"},
        )

    workers = int(config.get("max_workers", 8))
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, lists))


class _same_tag:
    """Re-enter a tag inside worker threads (contextvars do not cross thread pools)."""

    def __init__(self, full: str) -> None:
        self.full = full

    def __enter__(self):
        self.token = llm._tag.set(self.full)

    def __exit__(self, *exc):
        llm._tag.reset(self.token)
