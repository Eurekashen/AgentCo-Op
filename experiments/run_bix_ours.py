"""Run our optimizer on BixBench questions (no grading here).

    python experiments/run_bix_ours.py --tasks bixdev [--max-extra 3] [--parallel 3]

Outcomes land in runs/phase2/bixbench/<qid>/bix_ours.json.
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
from agentcoop.optimize.bix_loop import optimize_bix  # noqa: E402
from agentcoop.tasks.bixbench import question_ids  # noqa: E402
from agentcoop.workflow import Task  # noqa: E402

OUT = ROOT / "runs" / "phase2" / "bixbench"


def run_one(qid: str, max_extra: int, name: str) -> str:
    path = OUT / qid / f"{name}.json"
    if path.exists():
        return f"{qid} (exists)"
    t0 = time.time()
    with llm.tag(f"phase2/bixbench/{qid}/{name}"):
        out = optimize_bix(Task.load(ROOT / "data" / "tasks" / "bixbench" / qid),
                           max_extra=max_extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out.to_dict()))
    return (f"{qid} {name} {time.time() - t0:.0f}s ${out.nominal_usd:.3f} extra={out.extra_runs} "
            f"stop={out.stop_reason} answer={str(out.run.output().data.get('answer'))[:60]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--max-extra", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--name", default="bix_ours")
    args = ap.parse_args()
    qids = question_ids(args.tasks[0]) if args.tasks[0] in ("bixdev", "bixtest") else args.tasks
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {pool.submit(run_one, q, args.max_extra, args.name): q for q in qids}
        for fut in cf.as_completed(futs):
            try:
                print(fut.result(), flush=True)
            except Exception:
                print(f"FAILED {futs[fut]}\n{traceback.format_exc()}", flush=True)
    print("spent:", llm.spent("phase2/bixbench/"), flush=True)


if __name__ == "__main__":
    main()
