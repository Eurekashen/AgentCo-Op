"""Run optimizer variants from the naive composition on a suite (no grading here).

    python experiments/run_optimizers.py --suite test --methods ours random_loc ... --budget 0.6

Outcomes land in runs/phase2/<suite>/<task>/<method>.json.
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
from agentcoop.optimize import baselines as ob  # noqa: E402
from agentcoop.optimize.loop import Budget, optimize  # noqa: E402
from agentcoop.pipelines import naive_composition  # noqa: E402
from agentcoop.workflow import Task  # noqa: E402

#: Outputs are separated per backbone so a second model never overwrites the first.
OUT = ROOT / "runs" / ("phase2" if llm.backbone() == llm.DEFAULT_MODEL
                        else f"phase2@{llm.backbone()}")

RULES = {"pooled": ROOT / "runs" / "rules" / "dev_rules.json",
         "multisite": ROOT / "runs" / "rules" / "ms_rules.json"}


def _rules(task: Task):
    from agentcoop.optimize import rules as R

    return R.index(R.load(RULES["multisite" if task.info.get("shape") == "multisite"
                                else "pooled"]))


def _ours_escalate(t: Task, b: Budget):
    """v3.3: localized typed search first; when it runs out of edits at the implicated locations
    while findings remain, the unused budget goes to free-form editing from its incumbent (the
    typed space cannot express multi-location changes). Same budget, same acceptance rule."""
    out = optimize(t, _start(t), budget=b)
    if out.stop_reason.startswith("no untried edit") and not b.exhausted(out.nominal_usd,
                                                                         out.executions):
        out = ob.llm_editor(t, out.workflow, budget=b, start=out, version=2)
    return out


#: Methods reproducible with this code. Earlier variants (v3 "ours", v3.2 lookahead "ours_v32")
#: are reproducible from their git tags (v3-final-eval, v3.1-eval); their outputs keep their names.
METHODS = {
    # The typed, localized search (v3.1: competitive claim support + execution budget).
    "ours_v31": lambda t, b: optimize(t, _start(t), budget=b),
    "ours_escalate": _ours_escalate,
    "ours_rules": lambda t, b: optimize(t, _start(t), budget=b, rules=_rules(t)),
    "ours_no_propagation": lambda t, b: optimize(t, _start(t), budget=b, use_propagation=False),
    "random_loc": lambda t, b: optimize(t, _start(t), budget=b, locate="random"),
    "reflective_prompt": lambda t, b: ob.reflective_prompt(t, _start(t), budget=b),
    "llm_editor": lambda t, b: ob.llm_editor(t, _start(t), budget=b),
    # v2: documented DAG constraint + one repair round for invalid proposals.
    "llm_editor_v2": lambda t, b: ob.llm_editor(t, _start(t), budget=b, version=2),
    "llm_editor_report_judge": lambda t, b: ob.llm_editor(t, _start(t), budget=b,
                                                          acceptance="report_judge"),
    "best_of_n": lambda t, b: ob.best_of_n(t, _start(t), budget=b),
}


def _start(task: Task):
    """The naive composition for the task's shape."""
    from agentcoop.pipelines import naive_multisite

    return naive_multisite() if task.info.get("shape") == "multisite" else naive_composition()


def run_one(suite: str, task_dir: Path, method: str, budget: Budget, out: Path = OUT) -> str:
    task = Task.load(task_dir)
    path = out / suite / task.task_id / f"{method}.json"
    if path.exists():
        return f"{task.task_id} {method} (exists)"
    t0 = time.time()
    with llm.tag(f"phase2/{suite}/{task.task_id}/{method}"):
        outcome = METHODS[method](task, budget)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = outcome.to_dict()
    d["seconds"] = time.time() - t0
    path.write_text(json.dumps(d))
    acc = [a.family for a in outcome.attempts if a.accepted]
    return (f"{task.task_id} {method} {time.time() - t0:.0f}s nominal ${outcome.nominal_usd:.3f} "
            f"accepted={acc} stop={outcome.stop_reason}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="dev")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=["ours"])
    ap.add_argument("--budget", type=float, default=0.6)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--max-executions", type=int, default=None,
                    help="model-agnostic budget (workflow executions); --budget stays a $ cap")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="output root (e.g. to replay finished runs from the caches with newer "
                         "instrumentation without touching the originals)")
    args = ap.parse_args()
    budget = Budget(max_usd=args.budget, max_iterations=args.iterations,
                    max_executions=args.max_executions)
    task_dirs = sorted((ROOT / "data" / "tasks" / args.suite).iterdir())
    if args.tasks:
        task_dirs = [d for d in task_dirs if d.name in args.tasks]
    jobs = [(d, m) for m in args.methods for d in task_dirs]
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {pool.submit(run_one, args.suite, d, m, budget, args.out): (d.name, m) for d, m in jobs}
        for fut in cf.as_completed(futs):
            try:
                print(fut.result(), flush=True)
            except Exception:
                print(f"FAILED {futs[fut]}\n{traceback.format_exc()}", flush=True)
    print("spent:", llm.spent(f"phase2/{args.suite}"), flush=True)


if __name__ == "__main__":
    main()
