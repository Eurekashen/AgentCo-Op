"""Re-draw chosen nodes of fixed candidates under new sample indices (no-edit rerun control).

    python experiments/redraw.py --suite dev --candidates naive pseudobulk+no_interp --nodes report --samples 1 2 3
Writes runs/phase1/<suite>/<task>/<candidate>__s<k>.json with only ``nodes`` re-drawn.
"""
import argparse, concurrent.futures as cf, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentcoop import llm  # noqa: E402
from agentcoop.pipelines import CANDIDATES  # noqa: E402
from agentcoop.workflow import Task, execute  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="dev"); ap.add_argument("--candidates", nargs="+")
ap.add_argument("--nodes", nargs="+", default=["report"]); ap.add_argument("--samples", nargs="+", type=int)
args = ap.parse_args()

def one(job):
    d, name, k = job
    task = Task.load(d)
    path = (ROOT / "runs" / ("phase1" if llm.backbone() == llm.DEFAULT_MODEL
                             else f"phase1@{llm.backbone()}") / args.suite / task.task_id
            / f"{name}__s{k}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with llm.tag(f"phase1/{args.suite}/{task.task_id}/{name}/redraw{k}"):
        run = execute(CANDIDATES[name](), task, sample=0, samples={n: k for n in args.nodes})
    path.write_text(json.dumps(run.to_dict()))

jobs = [(d, n, k) for d in sorted((ROOT / "data" / "tasks" / args.suite).iterdir())
        for n in args.candidates for k in args.samples]
with cf.ThreadPoolExecutor(8) as pool:
    list(pool.map(one, jobs))
print("redraws done", llm.spent(f"phase1/{args.suite}/"))
