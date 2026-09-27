"""Q2 on a second model: agreement of each evaluator with the hidden ΔF1 on a fixed pair sample.

    AGENTCOOP_MODEL=gpt-4.1-mini python experiments/judge_robustness.py --n 60

Pairs come from the phase-1 dev candidates (gpt-6-luna outputs, graded once); the sample is fixed
(seed 0) so every judge model is scored on the same pairs.
"""
import argparse, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentcoop import llm  # noqa: E402
from agentcoop.evidence import checks, judge  # noqa: E402
from agentcoop.workflow import Run, Task  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); args = ap.parse_args()
A = json.loads((ROOT / "runs/phase1/dev_analysis.json").read_text())
grades, verdicts = A["grades"], A["verdicts"]
pairs = []
for key, v in verdicts.items():
    t, a, b = key.split("|")
    d = grades[t][a]["f1"] - grades[t][b]["f1"]
    if abs(d) >= 0.1 and a.endswith("__s0") and b.endswith("__s0"):
        pairs.append((t, a, b, d, v))
random.Random(0).shuffle(pairs)
pairs = sorted(pairs[: args.n])

def report(t, n):
    r = Run.from_dict(json.loads((ROOT / "runs/phase1/dev" / t / f"{n}.json").read_text()))
    return r.output().data if r.output() else {"claims": [], "summary": ""}

res = {"report_only": [], "evidence_judge": [], "luna_report_only": [], "luna_evidence_judge": [],
       "evidence_rule": []}
with llm.tag(f"judge-robustness/{llm.backbone()}"):
    for t, a, b, d, v in pairs:
        task = Task.load(ROOT / "data/tasks/dev" / t)
        ra, rb = report(t, a), report(t, b)
        pa = checks.profile_task(ra, task).to_dict(); pb = checks.profile_task(rb, task).to_dict()
        rub = judge.rubric_for(task.question)
        w = {"report_only": judge.report_only_judge(task.question, ra, rb)["winner"],
             "evidence_judge": judge.evidence_judge(task.question, rub, ra, pa, rb, pb)["winner"],
             "luna_report_only": v["report_only"], "luna_evidence_judge": v["evidence_judge"],
             "evidence_rule": judge.evidence_rule(pa, pb)["winner"]}
        for k, x in w.items():
            res[k].append(x != "tie" and (x == "A") == (d > 0))
print(f"judge model: {llm.backbone()}  pairs: {len(pairs)}")
for k, v in res.items():
    print(f"  {k:20s} accuracy {sum(v)/len(v):.2f}")
print("spend:", llm.spent(f"judge-robustness/{llm.backbone()}"))
