"""Pool graded suites (task = unit) and compare methods pairwise.

    python experiments/pool_suites.py mstest2 mstest3 [--methods ours_v31 gepa ...]

Reads runs/phase2/<suite>_analysis.json written by phase2_analysis.py (per-task F1 under
"grades"; the Codex-lead mean is taken from "<name>__mean" when present). Reports pooled mean
F1 with a task-bootstrap CI and, for every pair of methods, wins/losses and the Wilcoxon p.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from phase1_analysis import bootstrap_ci, wilcoxon_p  # noqa: E402

DEFAULT = ["ms_codex_orchestrator__mean", "ms_orchestrator__s0", "best_of_n", "gepa",
           "ms_local+structured+weak+repl__s0", "llm_editor_v2", "ours_v31"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("suites", nargs="+")
    ap.add_argument("--methods", nargs="*", default=DEFAULT)
    ap.add_argument("--focus", default="ours_v31")
    args = ap.parse_args()
    per: dict[str, dict[str, float]] = {}
    cost: dict[str, list[float]] = {}
    for suite in args.suites:
        d = json.loads((ROOT / "runs" / "phase2" / f"{suite}_analysis.json").read_text())
        for m in args.methods:
            g = d["grades"].get(m)
            if g is None:
                continue
            for t, v in g.items():
                per.setdefault(m, {})[t] = v["f1"] if isinstance(v, dict) else float(v)
            cost.setdefault(m, []).append(d["summary"][m]["nominal_usd"] * len(g))
    tasks = sorted(set.intersection(*(set(v) for v in per.values())))
    print(f"suites {args.suites}: {len(tasks)} tasks common to {len(per)} methods\n")
    for m in args.methods:
        if m not in per:
            continue
        x = np.array([per[m][t] for t in tasks])
        lo, hi = bootstrap_ci(x)
        print(f"{m:36s} F1 {x.mean():.3f} [{lo:.2f},{hi:.2f}]  ${sum(cost[m]) / len(tasks):.3f}/task")
    print()
    for a, b in itertools.combinations([m for m in args.methods if m in per], 2):
        if args.focus not in (a, b):
            continue
        xa = np.array([per[a][t] for t in tasks])
        xb = np.array([per[b][t] for t in tasks])
        d = xa - xb
        w, l = int((d > 1e-9).sum()), int((d < -1e-9).sum())
        print(f"{a:34s} vs {b:34s} Δ {d.mean():+.3f}  W/L {w}/{l}  p {wilcoxon_p(d):.3g}")


if __name__ == "__main__":
    main()
