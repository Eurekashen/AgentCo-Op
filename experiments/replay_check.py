"""Refactoring guard: re-execute stored runs from the caches and demand identical results.

    python experiments/replay_check.py runs/phase1/mstest/mstest-400/ms_naive__s0.json \
        runs/phase2/mstest/mstest-404/ours_v31.json ...

A phase-1 run (``<candidate>__s<k>.json``) is re-executed from its stored workflow and sample; a
phase-2 outcome (``<method>.json``) is re-optimized with the 16-execution protocol. The backbone
is read from the directory name (``phase2@<model>``). The spending cap is set to the current
ledger total, so any call that is not already cached fails instead of being paid for — a
behaviour change shows up as a cache miss or as a different artifact / attempt sequence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _model_of(path: Path) -> str | None:
    m = re.match(r"phase[12]@(.+)", path.parts[-4])
    return m.group(1) if m else None


def check(path: Path) -> str:
    from agentcoop.optimize.loop import Budget
    from agentcoop.workflow import Run, Task, Workflow, execute

    suite, task_id, name = path.parts[-3], path.parts[-2], path.stem
    task = Task.load(ROOT / "data" / "tasks" / suite / task_id)
    stored = json.loads(path.read_text())
    if path.parts[-4].startswith("phase1"):
        run = Run.from_dict(stored)
        new = execute(Workflow.from_dict(stored["workflow"]), task, samples=stored["samples"])
        same = ({k: a.to_dict() for k, a in new.artifacts.items()}
                == {k: a.to_dict() for k, a in run.artifacts.items()} and new.errors == run.errors)
    else:
        sys.path.insert(0, str(ROOT / "experiments"))
        from run_optimizers import METHODS

        out = METHODS[name](task, Budget(max_usd=5.0, max_iterations=8, max_executions=16))
        new_d = out.to_dict()

        def sig(d):
            # Invalid free-form proposals cost no execution (and were not logged by older runs).
            tried = [(a["iteration"], a["edit"], a["accepted"], a.get("rule_vs_control"),
                      a.get("control_counts")) for a in d["attempts"] if "(invalid:" not in a["edit"]]
            return d["workflow"], tried, Run.from_dict(d["run"]).output().data

        same = sig(new_d) == sig(stored)
    return f"{'OK  ' if same else 'DIFF'} {path}"


def run_checks(paths: list[str]) -> int:
    """In-process: the caller has already set AGENTCOOP_MODEL for these paths."""
    from agentcoop import llm

    os.environ["AGENTCOOP_BUDGET_USD"] = str(llm.spent()["actual_usd"])  # no new spending
    failed = 0
    for p in paths:
        try:
            line = check(Path(p))
        except Exception as exc:  # a cache miss surfaces as BudgetExceeded
            line = f"FAIL {p}: {type(exc).__name__}: {str(exc)[:200]}"
        failed += not line.startswith("OK")
        print(line, flush=True)
    return failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.in_process:
        sys.exit(1 if run_checks([str(p) for p in args.paths]) else 0)
    # llm.backbone() is read from the environment: one subprocess per backbone.
    by_model: dict[str | None, list[str]] = {}
    for p in args.paths:
        by_model.setdefault(_model_of(p.resolve()), []).append(str(p.resolve()))
    failed = False
    for model, paths in by_model.items():
        env = {k: v for k, v in os.environ.items() if k != "AGENTCOOP_MODEL"}
        if model:
            env["AGENTCOOP_MODEL"] = model
        failed |= subprocess.run([sys.executable, __file__, "--in-process", *paths],
                                 env=env).returncode != 0
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
