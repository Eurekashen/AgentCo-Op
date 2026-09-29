"""Grade one method's outcomes from an alternative output root against stored references.

    python experiments/compare_version.py --suite mstest2 --root runs/phase2_v34 --method ours_v31 \
        --refs ours_v31 llm_editor_v2 ms_codex_orchestrator__mean ...

References come from runs/phase2/<suite>_analysis.json (written by phase2_analysis.py); the
new outcomes are graded here. Per-task paired W/L (ties below 1e-9) and Wilcoxon, over the tasks
the new root has.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from agentcoop.eval.grader import grade  # noqa: E402
from agentcoop.tasks.planted import load_truth  # noqa: E402
from agentcoop.workflow import Run  # noqa: E402
from phase1_analysis import wilcoxon_p  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--analysis", type=Path, default=None)
    args = ap.parse_args()
    ref = json.loads((args.analysis or ROOT / "runs" / "phase2" / f"{args.suite}_analysis.json")
                     .read_text())["grades"]
    new = {}
    for f in sorted((args.root / args.suite).glob(f"*/{args.method}.json")):
        t = f.parent.name
        report = Run.from_dict(json.loads(f.read_text())["run"]).output().data
        new[t] = grade(report, load_truth(ROOT / "data" / "tasks" / args.suite / t))["f1"]
    tasks = sorted(new)
    print(f"{args.method} @ {args.root.name}: n={len(tasks)} mean F1 {np.mean(list(new.values())):.3f}")
    print("   per task:", " ".join(f"{new[t]:.2f}" for t in tasks))
    for r in args.refs:
        v = np.array([ref[r][t]["f1"] for t in tasks])
        d = np.array([new[t] for t in tasks]) - v
        print(f"   vs {r:36s} ref mean {v.mean():.3f}  Δ {d.mean():+.3f}  "
              f"W/L {(d > 1e-9).sum()}/{(d < -1e-9).sum()}  p {wilcoxon_p(d):.3f}")


if __name__ == "__main__":
    main()
