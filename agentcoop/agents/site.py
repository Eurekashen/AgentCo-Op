"""Site components for task shape 2: data stays at its site.

* ``site_analyst`` — a coding agent that can see only its own site's data (a private copy in its
  working directory). Its interface is a request in, a JSON report out; *what* it reports is
  whatever the incoming request asks for, which is exactly the handoff policy under study.
* ``replication_check`` — a deterministic check each site runs on its own data for findings
  proposed by anyone: per-donor direction consistency of a finding's genes. Only summary
  statistics leave the site.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from agentcoop import llm
from agentcoop.agents.coder import coding_loop
from agentcoop.eval_free import canon_cell_type, canon_direction
from agentcoop.evidence.checks import donor_tables
from agentcoop.workflow import Artifact, RunContext, agent

GENESETS = Path(__file__).resolve().parents[2] / "data" / "genesets"


def site_file(ctx: RunContext, site: str) -> Path:
    return ctx.task.data_path.parent / ctx.task.info["sites"][site]


def site_brief(site: str, request: str) -> str:
    return (
        f"You are the analysis agent of site {site}. Your site's data is site_{site}.h5ad in "
        "your working directory (AnnData; raw UMI counts in .X; obs columns 'cell_type', 'donor', "
        "'condition' with values control|treated; 4 donors). You cannot access any other site's "
        "data. Gene-set libraries (Enrichr text format: name<TAB><TAB>gene<TAB>gene...): "
        "GO_Biological_Process_2023.txt and Reactome_2022.txt in your working directory. Python "
        "packages: scanpy, anndata, numpy, pandas, scipy, statsmodels, sklearn, pydeseq2.\n\n"
        f"Request from the study lead:\n{request}\n\n"
        "Analyze with run_python, then call submit_report with a JSON object answering the "
        "request."
    )


def run_site_agent(ctx: RunContext, site: str, request: str, sample: int, *,
                   max_steps: int = 30, effort: str = "medium"):
    """One site agent on its own site's data (a private copy) → (report, transcript)."""
    files = {f"site_{site}.h5ad": site_file(ctx, site),
             "GO_Biological_Process_2023.txt": GENESETS / "GO_Biological_Process_2023.txt",
             "Reactome_2022.txt": GENESETS / "Reactome_2022.txt"}
    return coding_loop(site_brief(site, request), sample=sample, max_steps=max_steps,
                       effort=effort, files=files, model=llm.external_model())


@agent("site_analyst")
def site_analyst(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    site = config["site"]
    out = []
    for req in (a for a in inputs if a.kind == "request"):
        report, transcript = run_site_agent(ctx, site, req.data["text"], ctx.sample,
                                            max_steps=int(config.get("max_steps", 30)),
                                            effort=config.get("reasoning_effort", "medium"))
        if isinstance(report, list):
            # A JSON array is a legitimate report format; the lead reads reports generically.
            report = {"findings": report}
        out.append(Artifact(kind="site_report", producer=ctx.node_id,
                            data={"site": site, "request_kind": req.meta.get("kind"),
                                  "report": report if isinstance(report, dict)
                                  else {"error": "no report submitted"}},
                            derived_from=[req.id], meta={"site": site,
                                                         "n_steps": len(transcript)}))
    return out


def _findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("findings", "claims", "results", "programs"):
        v = report.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


@agent("replication_check")
def replication_check(inputs: list[Artifact], config: dict[str, Any],
                      ctx: RunContext) -> list[Artifact]:
    """For every finding any site reported, the per-donor consistency at *every* site."""
    sites = list(ctx.task.info["sites"])
    tables = {s: donor_tables(str(site_file(ctx, s))) for s in sites}
    cell_types = ctx.task.info["cell_types"]
    out = []
    for rep in (a for a in inputs if a.kind == "site_report"):
        for f in _findings(rep.data.get("report", {})):
            ct = canon_cell_type(str(f.get("cell_type", "")), cell_types)
            d = canon_direction(f.get("direction", ""))
            genes = [str(g).upper() for g in (f.get("genes") or [])]
            if ct is None or d not in ("up", "down") or len(genes) < 3:
                continue
            sign = 1 if d == "up" else -1
            per_site = {}
            for s in sites:
                t = tables[s][ct]
                idx = t.index()
                rows = [idx[g] for g in genes if g in idx and t.mean_cpm[idx[g]] >= 20]
                if len(rows) < 3:
                    per_site[s] = "not measurable"
                    continue
                agree = int(np.sum(np.sign(t.lfc[rows].mean(axis=0)) == sign))
                per_site[s] = f"{agree}/{len(t.donors)} donors change {d}"
            out.append(Artifact(
                kind="replication_evidence", producer=ctx.node_id,
                data={"reported_by": rep.data["site"], "cell_type": ct, "direction": d,
                      "program": f.get("program") or f.get("name"), "genes": genes[:15],
                      "per_site": per_site},
                derived_from=[rep.id]))
    return out
