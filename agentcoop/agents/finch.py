"""FutureHouse's Finch (data-analysis-crow, the BixBench agent) as a site agent.

``external_agents/run_finch.py`` runs Finch unmodified with its own interpreter, under the
workspace sandbox. Finch reaches the model through ``agentcoop.proxy`` on loopback (allowed by
the sandbox), so its calls use the shared backbone, cache, ledger and spending cap; the proxy URL
carries the caller's ledger tag and the sample index (same sample = replay, new = re-draw). Its
deliverables are the files it writes in the workspace; its notebook is the transcript.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentcoop import llm, proxy
from agentcoop.agents.workspace import exec_env, run_group, sandboxed
from agentcoop.llm import ROOT

FINCH_PY = ROOT / "external_agents" / "finch" / ".venv" / "bin" / "python"
RUN_FINCH = ROOT / "external_agents" / "run_finch.py"


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


def run_finch(ws: Path, prompt: str, *, sample: int, timeout: int, max_steps: int = 40,
              notebook: str = "notebook.ipynb") -> dict[str, Any]:
    """Run Finch on ``prompt`` (verbatim, not BixBench's question template) in ``ws``.

    ``notebook`` names the notebook Finch writes, so a follow-up call does not overwrite the
    first one's. Returns ``{"transcript", "timed_out", "error", "nominal_usd_external": 0.0}``
    (Finch's model calls are ledgered by the proxy). Raises ``llm.BudgetExceeded`` when the proxy
    refused a call at the spending cap: Finch itself only sees an API error.
    """
    req = {"work_dir": str(ws), "prompt": prompt, "notebook": notebook,
           "base_url": proxy.base_url(proxy.shared(), llm.current_tag(), sample),
           "model": llm.external_model(), "max_steps": max_steps, "language": "python"}
    env = exec_env(ws)
    env["PATH"] = f"{FINCH_PY.parent}:{env['PATH']}"
    code, stdout, stderr, timed_out = run_group(sandboxed([str(FINCH_PY), str(RUN_FINCH)], ws),
                                                ws, env, timeout, json.dumps(req))
    lines = [x for x in stdout.splitlines() if x.startswith("AGENTCOOP_RESULT ")]
    result = json.loads(lines[-1].removeprefix("AGENTCOOP_RESULT ")) if lines else {
        "error": f"finch exited {code}: {stderr[-500:]}"}
    error = str(result.get("error") or "")
    if "agentcoop budget" in error:
        raise llm.BudgetExceeded(error[:500])
    transcript = _notebook_trace(ws / notebook)
    if result.get("answer"):
        transcript.append({"code": "(final)", "output": str(result["answer"])[-3000:]})
    return {"transcript": transcript, "timed_out": timed_out,
            "error": None if timed_out or not error else error[:500],
            "nominal_usd_external": 0.0}
