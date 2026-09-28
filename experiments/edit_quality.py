"""Quality of the optimizer's *decisions*, graded after the fact (plan §9).

    python experiments/edit_quality.py --root runs/phase2_attempts --suite mstest

Needs outcomes that recorded every candidate report (``Attempt.report``, ``initial_report``);
older outcomes are replayed from the caches with ``run_optimizers.py --out <root>``.

An attempt is compared with the incumbent it was judged against: the report after the last
edit accepted in an *earlier* iteration (both loops change the incumbent only between
iterations; best-of-N gives every re-draw its own iteration). Per method, over all tasks:

* harmful acceptance  — accepted, hidden F1 lower than the incumbent's
* helpful acceptance  — accepted, hidden F1 higher
* missed improvement  — rejected by the acceptance rule, hidden F1 higher than the incumbent's
* outranked           — passed the rule but a sibling candidate of the same iteration was kept

The hidden grader is used here only; nothing flows back into any optimizer.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentcoop.eval.grader import grade  # noqa: E402
from agentcoop.tasks.planted import load_truth  # noqa: E402


def attempts_with_incumbent(d: dict) -> list[tuple[dict, dict]]:
    """[(attempt, incumbent report it was judged against)] in order."""
    inc = d["initial_report"]
    out, pending, it = [], None, None
    for a in d["attempts"]:
        if a["iteration"] != it:
            if pending is not None:
                inc = pending
                pending = None
            it = a["iteration"]
        out.append((a, inc))
        if a["accepted"]:
            pending = a["report"]
    return out


def passed(a: dict) -> bool:
    """Did the attempt pass its method's acceptance rule (whether or not it was kept)?"""
    if a["accepted"]:
        return True
    if a.get("basis") == "signal propagation":
        return True
    return (a["rule_vs_incumbent"] == "A" and a.get("rule_vs_control") in (None, "A")
            and a.get("judge") != "B")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "phase2_attempts")
    ap.add_argument("--suite", default="mstest")
    ap.add_argument("--methods", nargs="*", default=None)
    args = ap.parse_args()
    suite_dir = ROOT / "data" / "tasks" / args.suite

    rows = []  # (method, task, attempt, incumbent)
    for f in sorted((args.root / args.suite).glob("*/*.json")):
        d = json.loads(f.read_text())
        if d.get("initial_report") is None or (args.methods and f.stem not in args.methods):
            continue
        for a, inc in attempts_with_incumbent(d):
            if a.get("report") is None:
                continue
            rows.append((f.stem, f.parent.name, a, inc))

    def f1(job):
        report, task = job
        return grade(report, load_truth(suite_dir / task))["f1"]

    jobs = [(r[2]["report"], r[1]) for r in rows] + [(r[3], r[1]) for r in rows]
    with cf.ThreadPoolExecutor(8) as pool:
        scores = list(pool.map(f1, jobs))
    cand, inc = scores[: len(rows)], scores[len(rows):]

    stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    tasks: dict[str, set] = collections.defaultdict(set)
    for (m, t, a, _), fc, fi in zip(rows, cand, inc):
        s = stats[m]
        tasks[m].add(t)
        s["attempts"] += 1
        delta = fc - fi
        if a["accepted"]:
            s["accepted"] += 1
            s["helpful"] += delta > 1e-9
            s["harmful"] += delta < -1e-9
        elif passed(a):
            s["outranked"] += 1
            s["outranked_better"] += delta > 1e-9
        else:
            s["missed"] += delta > 1e-9
            s["rejected_worse"] += delta < -1e-9

    print(f"suite {args.suite} ({args.root})\n")
    print(f"{'method':26s} {'tasks':>5s} {'tries':>5s} {'acc':>4s} {'helpful':>7s} "
          f"{'harmful':>7s} {'missed':>6s} {'rej.worse':>9s} {'outranked(better)':>17s}")
    for m, s in sorted(stats.items()):
        print(f"{m:26s} {len(tasks[m]):5d} {s['attempts']:5d} {s['accepted']:4d} "
              f"{s['helpful']:7d} {s['harmful']:7d} {s['missed']:6d} {s['rejected_worse']:9d} "
              f"{s['outranked']:9d} ({s['outranked_better']})")
    out = args.root / f"{args.suite}_edit_quality.json"
    out.write_text(json.dumps({m: dict(s) | {"tasks": len(tasks[m])} for m, s in stats.items()},
                              indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
