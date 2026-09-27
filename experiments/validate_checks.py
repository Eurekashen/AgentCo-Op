"""Re-score the phase-1 dev pairs with the current evidence checks (no LLM, deterministic).

    AGENTCOOP_COMPETITIVE_SUPPORT=0|1 python experiments/validate_checks.py
"""
import itertools, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentcoop.evidence import checks, judge  # noqa: E402
from agentcoop.workflow import Run, Task  # noqa: E402

A = json.loads((ROOT / "runs/phase1/dev_analysis.json").read_text())
grades = A["grades"]
prof = {}
for t in A["tasks"]:
    task = Task.load(ROOT / "data/tasks/dev" / t)
    for f in (ROOT / "runs/phase1/dev" / t).glob("*__s0.json"):
        r = Run.from_dict(json.loads(f.read_text()))
        rep = r.output().data if r.output() else {"claims": []}
        prof[(t, f.stem)] = checks.profile_task(rep, task).to_dict()
agree = n = decided = 0
for key in A["verdicts"]:
    t, a, b = key.split("|")
    if (t, a) not in prof or (t, b) not in prof:
        continue
    d = grades[t][a]["f1"] - grades[t][b]["f1"]
    if abs(d) < 0.1:
        continue
    n += 1
    w = judge.evidence_rule(prof[(t, a)], prof[(t, b)])["winner"]
    if w != "tie":
        decided += 1
        agree += (w == "A") == (d > 0)
print(f"competitive={checks.COMPETITIVE}: accuracy {agree/n:.3f} (decided {decided}/{n}, "
      f"error-when-decided {(decided-agree)/max(decided,1):.3f})")
