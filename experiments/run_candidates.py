"""Execute a fixed set of candidate workflows on a task suite (no grading here).

    python experiments/run_candidates.py --suite dev --candidates naive split ... --samples 0

Runs land in runs/phase1/<suite>/<task>/<candidate>__s<sample>.json. Grading is a separate
step (experiments/grade_runs.py) so nothing on this path touches the hidden answers.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentcoop import llm  # noqa: E402
from agentcoop.pipelines import (  # noqa: E402
    BASELINES, CANDIDATES, MS_BASELINES, MS_CANDIDATES,
)

WORKFLOWS = {**CANDIDATES, **BASELINES, **MS_CANDIDATES, **MS_BASELINES}
from agentcoop.workflow import Task, execute  # noqa: E402

#: Outputs are separated per backbone so a second model never overwrites the first.
OUT = ROOT / "runs" / ("phase1" if llm.backbone() == llm.DEFAULT_MODEL
                        else f"phase1@{llm.backbone()}")


def run_task(suite: str, task_dir: Path, candidates: list[str], samples: list[int]) -> list[str]:
    task = Task.load(task_dir)
    done = []
    for sample in samples:
        for name in candidates:
            path = OUT / suite / task.task_id / f"{name}__s{sample}.json"
            if path.exists():
                continue
            t0 = time.time()
            with llm.tag(f"phase1/{suite}/{task.task_id}/{name}/s{sample}"):
                run = execute(WORKFLOWS[name](), task, sample=sample)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(run.to_dict()))
            done.append(f"{task.task_id} {name} s{sample} {time.time() - t0:.0f}s "
                        f"${run.cost['actual_usd']:.3f} errors={list(run.errors)}")
            print(done[-1], flush=True)
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="dev")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--candidates", nargs="*", default=list(CANDIDATES))
    ap.add_argument("--samples", nargs="*", type=int, default=[0])
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()

    task_dirs = sorted((ROOT / "data" / "tasks" / args.suite).iterdir())
    if args.tasks:
        task_dirs = [d for d in task_dirs if d.name in args.tasks]
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_task, args.suite, d, args.candidates, args.samples): d
                   for d in task_dirs}
        for fut in cf.as_completed(futures):
            try:
                fut.result()
            except Exception:
                print(f"FAILED {futures[fut].name}\n{traceback.format_exc()}", flush=True)
    print("spent:", llm.spent("phase1/"), flush=True)


if __name__ == "__main__":
    main()
