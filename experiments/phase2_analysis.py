"""Phase-2 analysis: grade optimizer outcomes and fixed references on a held-out suite.

    python experiments/phase2_analysis.py --suite test [--reference naive]

Reads runs/phase2/<suite>/<task>/<method>.json (optimizer outcomes) and, for references,
runs/phase1/<suite>/<task>/<candidate>__s0.json (fixed workflows and single-agent baselines).
Statistics are per task: mean hidden F1 with a bootstrap CI, paired win/loss and Wilcoxon p
against the reference, nominal cost, and what the optimizer changed.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentcoop.eval.grader import grade  # noqa: E402
from agentcoop.tasks.planted import load_truth  # noqa: E402
from agentcoop.workflow import Run  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments"))
from phase1_analysis import bootstrap_ci, wilcoxon_p  # noqa: E402


def report_from_outcome(d: dict) -> dict:
    run = Run.from_dict(d["run"])
    out = run.output()
    return out.data if out else {"claims": [], "summary": ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="test")
    ap.add_argument("--reference", default="naive__s0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--single", default="coder_single__s0",
                    help="single-agent run used for the answer-free select step")
    args = ap.parse_args()
    suite_dir = ROOT / "data" / "tasks" / args.suite
    tasks = sorted(p.name for p in suite_dir.iterdir() if p.is_dir())

    entries: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for t in tasks:
        for f in sorted((ROOT / "runs" / "phase2" / args.suite / t).glob("*.json")):
            d = json.loads(f.read_text())
            entries[f.stem][t] = {"report": report_from_outcome(d), "cost": d["nominal_usd"],
                                  "attempts": d["attempts"], "stop": d["stop_reason"],
                                  "workflow": d["workflow"]}
        for f in sorted((ROOT / "runs" / "phase1" / args.suite / t).glob("*__s0.json")):
            run = Run.from_dict(json.loads(f.read_text()))
            out = run.output()
            entries[f.stem][t] = {"report": out.data if out else {"claims": []},
                                  "cost": run.cost["nominal_usd"], "attempts": [], "stop": "",
                                  "workflow": run.workflow.to_dict()}

    # "Decline to compose" as a measured decision: after optimizing, compare the optimized
    # workflow's report with the single-agent report using the answer-free evidence rule only,
    # keep the winner (incumbent composition on ties), and charge both runs.
    from agentcoop.evidence import checks as _checks, judge as _judge
    from agentcoop.workflow import Task as _Task

    single = args.single
    if single in entries:
        for m in [k for k in list(entries) if k in ("ours", "ours_rules")]:
            sel = f"{m}+select_{single.split('__')[0]}"
            for t, e in entries[m].items():
                if t not in entries[single]:
                    continue
                task = _Task.load(suite_dir / t)
                pa = _checks.profile_task(e["report"], task).to_dict()
                pb = _checks.profile_task(entries[single][t]["report"], task).to_dict()
                pick = entries[single][t] if _judge.evidence_rule(pb, pa)["winner"] == "A" else e
                entries[sel][t] = {**pick, "cost": e["cost"] + entries[single][t]["cost"],
                                   "attempts": [], "picked": "single" if pick is not e else m}

    jobs = [(m, t) for m, per in entries.items() for t in per]

    def g(job):
        m, t = job
        return m, t, grade(entries[m][t]["report"], load_truth(suite_dir / t))

    grades: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    with cf.ThreadPoolExecutor(8) as pool:
        for m, t, res in pool.map(g, jobs):
            grades[m][t] = res

    ref = args.reference
    print(f"suite {args.suite}: {len(tasks)} tasks; reference = {ref}\n")
    print(f"{'method':38s} {'n':>3s} {'F1':>6s} {'95% CI':>13s} {'P':>5s} {'R':>5s} {'decoy':>5s} "
          f"{'W/L vs ref':>10s} {'p':>7s} {'$nominal':>9s}")
    summary = {}
    for m in sorted(entries, key=lambda k: np.mean([grades[k][t]["f1"] for t in grades[k]])):
        ts = [t for t in tasks if t in grades[m]]
        f1 = np.array([grades[m][t]["f1"] for t in ts])
        lo, hi = bootstrap_ci(f1) if len(f1) > 1 else (float("nan"),) * 2
        shared = [t for t in ts if t in grades.get(ref, {})]
        d = np.array([grades[m][t]["f1"] - grades[ref][t]["f1"] for t in shared])
        cost = np.mean([entries[m][t]["cost"] for t in ts])
        summary[m] = {"n": len(ts), "f1": float(f1.mean()), "ci": [lo, hi],
                      "precision": float(np.mean([grades[m][t]["precision"] for t in ts])),
                      "recall": float(np.mean([grades[m][t]["recall"] for t in ts])),
                      "decoys": int(sum(grades[m][t]["decoy_claimed"] for t in ts)),
                      "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                      "p": wilcoxon_p(d) if len(d) else float("nan"), "nominal_usd": float(cost)}
        s = summary[m]
        print(f"{m:38s} {s['n']:3d} {s['f1']:6.3f} [{lo:4.2f},{hi:4.2f}] {s['precision']:5.2f} "
              f"{s['recall']:5.2f} {s['decoys']:5d} {s['wins']:4d}/{s['losses']:<4d} "
              f"{s['p']:7.3g} {cost:9.3f}")

    print("\naccepted edits by optimizer method (family @ location, basis)")
    for m, per in sorted(entries.items()):
        acc = collections.Counter()
        for t, e in per.items():
            for a in e["attempts"]:
                if a.get("accepted"):
                    acc[f"{a['family']}@{a.get('location', '')} [{a.get('basis', '')}]"] += 1
        if acc:
            print(f"  {m}: " + "; ".join(f"{k} ×{v}" for k, v in acc.most_common()))

    for m in [k for k in entries if "+select_" in k]:
        picks = collections.Counter(e.get("picked") for e in entries[m].values())
        print(f"  {m}: picked {dict(picks)}")

    out = Path(args.out or ROOT / "runs" / "phase2" / f"{args.suite}_analysis.json")
    out.write_text(json.dumps({"summary": summary, "grades": grades}, indent=1, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
