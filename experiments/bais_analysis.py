"""Grade BAIS-DPTA runs (evaluation harness: reads the hidden expert labels).

    python experiments/bais_analysis.py [--tasks bais-trachea ...] [--candidates ...]
                                         [--root runs/phase1]

Every stored run ``<root>/bais/<task>/<name>__s<k>.json`` (phase 1) or ``<name>.json`` (phase 2
optimizer outcome) whose output is an ``annotation`` artifact is scored with the benchmark's
hierarchical uHAF metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentcoop.agents.annot import cell_labels  # noqa: E402
from agentcoop.eval.bais_grader import load_labels, score  # noqa: E402
from agentcoop.workflow import Run  # noqa: E402

TASKS = ROOT / "data" / "tasks" / "bais"


def run_of(path: Path) -> Run:
    d = json.loads(path.read_text())
    return Run.from_dict(d["run"] if "run" in d else d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--candidates", nargs="*")
    ap.add_argument("--roots", nargs="*", type=Path,
                    default=[ROOT / "runs" / "phase1", ROOT / "runs" / "phase2"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    table: dict[str, dict[str, float]] = {}
    for task_dir in sorted(TASKS.iterdir()):
        if args.tasks and task_dir.name not in args.tasks:
            continue
        labels = load_labels(task_dir)
        for root in args.roots:
            for f in sorted((root / "bais" / task_dir.name).glob("*.json")) \
                    if (root / "bais" / task_dir.name).exists() else []:
                name = f.stem
                if args.candidates and name.split("__s")[0] not in args.candidates:
                    continue
                out = run_of(f).output()
                if out is None or out.kind != "annotation":
                    continue
                s = score(labels["organ"], labels["cell_type"], cell_labels(out.data))
                table.setdefault(name, {})[task_dir.name] = s["score"]
                print(f"{task_dir.name:22s} {name:34s} {s['score']:.3f}  "
                      f"labels={sorted(set(out.data['cluster_labels'].values()))[:8]}", flush=True)
    print()
    for name, per in sorted(table.items()):
        print(f"{name:34s} n={len(per):2d} mean={sum(per.values()) / len(per):.3f}")
    if args.out:
        args.out.write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
