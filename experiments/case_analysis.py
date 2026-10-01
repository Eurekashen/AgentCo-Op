"""Validate every case-study report against the held-out donors (evaluation side).

    python experiments/case_analysis.py [--task kangreal-0]

Reads phase-1 runs (fixed workflows, single agents) and phase-2 outcomes (optimizers) for the
real-data case and prints the independent validation of each final report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentcoop.eval.case_validation import validate  # noqa: E402
from agentcoop.evidence.checks import profile  # noqa: E402
from agentcoop.workflow import Run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="kangreal-0")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    td = ROOT / "data" / "tasks" / "kangreal" / args.task
    rows = {}
    for phase in ("phase1", "phase2"):
        for f in sorted((ROOT / "runs" / phase / "kangreal" / args.task).glob("*.json")):
            d = json.loads(f.read_text())
            run = Run.from_dict(d["run"] if "run" in d else d)
            rep = run.output().data if run.output() else {"claims": []}
            v = validate(rep, td)
            disc = profile(rep, td / "public" / "counts.h5ad").counts()
            rows[f.stem] = {"phase": phase, "executions": d.get("executions"),
                            "nominal_usd": d.get("nominal_usd", run.cost.get("nominal_usd")),
                            "discovery": disc, **{k: v[k] for k in (
                                "n_claims", "replicated", "contradicted", "unassessed",
                                "ungrounded", "ifn_cell_types", "missed_programs")}}
            print(f"{f.stem:32s} claims={v['n_claims']:2d} replicated={v['replicated']:2d} "
                  f"contra={v['contradicted']} IFN={len(v['ifn_cell_types'])}/6 "
                  f"missed={len(v['missed_programs']):2d} | discovery {disc}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
