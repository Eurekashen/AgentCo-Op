"""Grade BixBench runs (evaluation harness: reads the hidden answers).

    python experiments/bix_analysis.py --split bixdev [--candidates bix_finch bix_codex ...]

Every stored run ``runs/phase1/bixbench/<qid>/<candidate>__s<k>.json`` is graded with the
BixBench open-answer grader. Reports accuracy per candidate (mean over samples per question, then
over questions), nominal cost, and — for the single-agent arms — how often the agents agree and
the oracle (any agent correct). Statistics are per question; capsule-clustered bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentcoop.eval.bixbench_grader import grade, grade_mcq, load_truth  # noqa: E402
from agentcoop.tasks.bixbench import question_ids  # noqa: E402
from agentcoop.workflow import Run, Task  # noqa: E402

RUNS = ROOT / "runs" / "phase1" / "bixbench"
RUNS2 = ROOT / "runs" / "phase2" / "bixbench"
TASKS = ROOT / "data" / "tasks" / "bixbench"


def answer_of(run: Run) -> str | None:
    out = run.output()
    return out.data.get("answer") if out is not None else None


def graded(split: str, candidates: list[str] | None) -> dict[str, dict[str, list[dict]]]:
    """candidate -> qid -> [{sample, answer, correct, usd}]"""
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for qid in question_ids(split):
        task, truth = Task.load(TASKS / qid), load_truth(TASKS / qid)
        files = sorted((RUNS / qid).glob("*__s*.json")) if (RUNS / qid).exists() else []
        files += sorted((RUNS2 / qid).glob("*.json")) if (RUNS2 / qid).exists() else []
        for f in files:
            name, s = f.stem.rsplit("__s", 1) if "__s" in f.stem else (f.stem, "0")
            if candidates and name not in candidates:
                continue
            d = json.loads(f.read_text())
            run = Run.from_dict(d["run"] if "run" in d else d)
            ans = answer_of(run)
            out[name][qid].append({"sample": int(s), "answer": ans,
                                   "correct": grade(task.question, truth, ans),
                                   "mcq": grade_mcq(task.question, truth, ans, qid),
                                   "usd": run.cost.get("nominal_usd", 0.0),
                                   "error": next(iter(run.errors.values()), None)})
    return out


def capsule_of(qid: str) -> str:
    return json.loads((TASKS / qid / "public" / "task.json").read_text())["capsule"]


def boot_ci(per_q: dict[str, float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    by_cap: dict[str, list[float]] = defaultdict(list)
    for q, v in per_q.items():
        by_cap[capsule_of(q)].append(v)
    caps, rng, means = list(by_cap), random.Random(seed), []
    for _ in range(n):
        vals = [v for c in rng.choices(caps, k=len(caps)) for v in by_cap[c]]
        means.append(sum(vals) / len(vals))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="bixdev")
    ap.add_argument("--candidates", nargs="*")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    g = graded(args.split, args.candidates)
    summary = {}
    for name, per in sorted(g.items()):
        acc = {q: sum(r["correct"] for r in rs) / len(rs) for q, rs in per.items()}
        mcq = {q: sum(r["mcq"] for r in rs) / len(rs) for q, rs in per.items()}
        usd = sum(r["usd"] for rs in per.values() for r in rs) / max(1, sum(map(len, per.values())))
        no_ans = sum(r["answer"] is None for rs in per.values() for r in rs)
        lo, hi = boot_ci(acc)
        mlo, mhi = boot_ci(mcq)
        summary[name] = {"n": len(acc), "acc": sum(acc.values()) / len(acc), "ci": [lo, hi],
                         "mcq": sum(mcq.values()) / len(mcq), "mcq_ci": [mlo, mhi],
                         "usd_per_q": usd, "no_answer": no_ans}
        print(f"{name:28s} n={len(acc):3d} open={summary[name]['acc']:.3f} [{lo:.2f},{hi:.2f}] "
              f"mcq={summary[name]['mcq']:.3f} [{mlo:.2f},{mhi:.2f}] ${usd:.4f}/q "
              f"no-answer={no_ans}")
    singles = [c for c in ("bix_finch", "bix_codex", "bix_react") if c in g]
    if len(singles) > 1:
        qs = sorted(set.intersection(*(set(g[c]) for c in singles)))
        first = {c: {q: g[c][q][0] for q in qs} for c in singles}
        oracle = sum(any(first[c][q]["correct"] for c in singles) for q in qs) / len(qs)
        print(f"oracle(any of {singles}, s0) = {oracle:.3f} over {len(qs)} questions")
        summary["_oracle_s0"] = oracle
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "graded": g}, indent=1, default=str))


if __name__ == "__main__":
    main()
