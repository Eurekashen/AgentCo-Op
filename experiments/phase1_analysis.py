"""Phase-1 feasibility analysis over the fixed candidate set.

Q1  Is there room for improvement?   hidden F1 per candidate vs naive, paired by task.
Q2  Can answer-free feedback find it? agreement of each evaluator's pairwise preference with the
    sign of the hidden F1 difference, and hidden F1 of the candidate each evaluator selects.
Q3  Is it more than resampling?       naive re-drawn K times (new sample index), best-of-K picked
    by the same evaluators, at equal or greater nominal cost.

Evaluators never see grades; grades are joined afterwards, only for scoring the evaluators.

    python experiments/phase1_analysis.py --suite dev [--judges] [--retry-samples 1 2 3]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentcoop import llm  # noqa: E402
from agentcoop.eval.grader import grade  # noqa: E402
from agentcoop.evidence import checks, judge  # noqa: E402
from agentcoop.tasks.planted import load_truth  # noqa: E402
from agentcoop.workflow import Run, Task  # noqa: E402

RUNS = ROOT / "runs" / "phase1"


def load_runs(suite: str) -> dict[str, dict[str, Run]]:
    out: dict[str, dict[str, Run]] = {}
    for task_dir in sorted((RUNS / suite).iterdir()):
        if not task_dir.is_dir():
            continue
        for f in sorted(task_dir.glob("*.json")):
            out.setdefault(task_dir.name, {})[f.stem] = Run.from_dict(json.loads(f.read_text()))
    return out


def report_of(run: Run) -> dict:
    out = run.output()
    return out.data if out is not None else {"claims": [], "summary": "(workflow failed)"}


def bootstrap_ci(x: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(d: np.ndarray) -> float:
    from scipy.stats import wilcoxon

    d = d[np.abs(d) > 1e-9]  # float noise (e.g. a mean of equal repeats) is a tie
    if len(d) < 3:
        return float("nan")
    return float(wilcoxon(d).pvalue)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="dev")
    ap.add_argument("--judges", action="store_true", help="run the pairwise LLM judges (Q2/Q3)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suite_dir = ROOT / "data" / "tasks" / args.suite
    runs = load_runs(args.suite)

    # ---- hidden grading (evaluation side only) -------------------------------------------
    grades: dict[str, dict[str, dict]] = {}
    with llm.tag(f"phase1-analysis/{args.suite}/grade"):
        def g(item):
            tid, name, run = item
            return tid, name, grade(report_of(run), load_truth(suite_dir / tid))
        items = [(t, n, r) for t, rs in runs.items() for n, r in rs.items()]
        with cf.ThreadPoolExecutor(8) as pool:
            for tid, name, res in pool.map(g, items):
                grades.setdefault(tid, {})[name] = res

    from agentcoop.pipelines import BASELINES, CANDIDATES, MS_BASELINES, MS_CANDIDATES

    multisite = Task.load(next(p for p in suite_dir.iterdir() if p.is_dir())).info.get(
        "shape") == "multisite"
    cands, bases = (MS_CANDIDATES, MS_BASELINES) if multisite else (CANDIDATES, BASELINES)
    base_names = [f"{n}__s0" for n in cands]
    tasks = sorted(t for t in runs if all(n in runs[t] for n in base_names))
    baseline_names = [f"{n}__s0" for n in bases
                      if all(f"{n}__s0" in runs[t] for t in tasks)]
    print(f"tasks with all edit candidates: {len(tasks)}/{len(runs)}; "
          f"baselines on all of them: {baseline_names}")

    print("\nQ1  hidden F1 by candidate (mean over tasks, 95% bootstrap CI; Δ vs naive, Wilcoxon p)")
    ref = base_names[0]
    naive = np.array([grades[t][ref]["f1"] for t in tasks])
    summary = {}
    for name in base_names + baseline_names:
        f1 = np.array([grades[t][name]["f1"] for t in tasks])
        prec = np.mean([grades[t][name]["precision"] for t in tasks])
        rec = np.mean([grades[t][name]["recall"] for t in tasks])
        wd = np.sum([grades[t][name]["wrong_direction"] for t in tasks])
        dec = np.sum([grades[t][name]["decoy_claimed"] for t in tasks])
        cost = np.mean([runs[t][name].cost["nominal_usd"] for t in tasks])
        lo, hi = bootstrap_ci(f1)
        d = f1 - naive
        summary[name] = {"f1": float(f1.mean()), "ci": [lo, hi], "precision": float(prec),
                         "recall": float(rec), "wrong_direction": int(wd), "decoys": int(dec),
                         "delta_vs_naive": float(d.mean()), "p": wilcoxon_p(d),
                         "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()),
                         "nominal_usd": float(cost)}
        print(f"  {name:40s} F1={f1.mean():.3f} [{lo:.2f},{hi:.2f}] P={prec:.2f} R={rec:.2f} "
              f"wrongdir={wd:2d} decoy={dec:2d} Δ={d.mean():+.3f} (W/L {int((d>0).sum())}/"
              f"{int((d<0).sum())}, p={summary[name]['p']:.3g}) ${cost:.3f}")
    oracle = np.array([max(grades[t][n]["f1"] for n in base_names) for t in tasks])
    print(f"  per-task oracle best candidate: F1={oracle.mean():.3f}  (naive {naive.mean():.3f})")

    result = {"suite": args.suite, "tasks": tasks, "q1": summary,
              "oracle_f1": float(oracle.mean()), "grades": grades}

    if args.judges:
        result.update(evaluate_judges(args.suite, suite_dir, runs, grades, tasks, base_names))

    out = Path(args.out or RUNS / f"{args.suite}_analysis.json")
    out.write_text(json.dumps(result, indent=1, default=str))
    print(f"\nwrote {out}; analysis spend {llm.spent('phase1-analysis/')}")


def evaluate_judges(suite, suite_dir, runs, grades, tasks, base_names) -> dict:
    """Q2/Q3: round-robin pairwise comparisons by each evaluator, without grades."""
    profiles = {t: {n: checks.profile_task(report_of(runs[t][n]), Task.load(suite_dir / t))
                    .to_dict() for n in runs[t]} for t in tasks}

    def compare(t: str, a: str, b: str) -> dict:
        task = Task.load(suite_dir / t)
        ra, rb = report_of(runs[t][a]), report_of(runs[t][b])
        pa, pb = profiles[t][a], profiles[t][b]
        with llm.tag(f"phase1-analysis/{suite}/judge"):
            rub = judge.rubric_for(task.question)
            return {
                "report_only": judge.report_only_judge(task.question, ra, rb)["winner"],
                "evidence_judge": judge.evidence_judge(task.question, rub, ra, pa, rb, pb)["winner"],
                "evidence_rule": judge.evidence_rule(pa, pb)["winner"],
            }

    # Q2 pairs: every pair of edit candidates (sample 0). Q3 pairs: re-draws of one candidate.
    pairs = [(t, a, b) for t in tasks for a, b in itertools.combinations(sorted(base_names), 2)]
    q2_pairs = set(pairs)
    groups = {}
    for t in tasks:
        for n in runs[t]:
            groups.setdefault((t, n.rsplit("__s", 1)[0]), []).append(n)
    for (t, _), members in groups.items():
        if len(members) > 1:
            pairs += [(t, a, b) for a, b in itertools.combinations(sorted(members), 2)
                      if (t, a, b) not in pairs]
    with cf.ThreadPoolExecutor(12) as pool:
        verdicts = dict(zip(pairs, pool.map(lambda p: compare(*p), pairs)))

    evaluators = ["report_only", "evidence_judge", "evidence_rule"]
    print("\nQ2  pairwise agreement with hidden ΔF1 (pairs with |ΔF1| ≥ 0.1; tie counts as wrong)")
    q2 = {}
    for ev in evaluators:
        agree = decided = n = false_promote = 0
        for (t, a, b), v in verdicts.items():
            d = grades[t][a]["f1"] - grades[t][b]["f1"]
            if (t, a, b) not in q2_pairs or abs(d) < 0.1:
                continue  # Q2 is about edit candidates; re-draw pairs belong to Q3
            n += 1
            w = v[ev]
            if w != "tie":
                decided += 1
                agree += (w == "A") == (d > 0)
                false_promote += (w == "A") != (d > 0)
        q2[ev] = {"n": n, "accuracy": agree / n if n else None,
                  "decided": decided / n if n else None,
                  "wrong_when_decided": false_promote / decided if decided else None}
        print(f"  {ev:16s} accuracy={q2[ev]['accuracy']:.2f} decided={q2[ev]['decided']:.2f} "
              f"error-when-decided={q2[ev]['wrong_when_decided']:.2f} (n={n})")

    def select(t: str, ev: str, pool_names: list[str]) -> str:
        """Copeland winner among ``pool_names``; ties → keep the incumbent (first name)."""
        score = {n: 0 for n in pool_names}
        for a, b in itertools.combinations(sorted(pool_names), 2):
            w = verdicts[(t, a, b)][ev]
            if w == "A":
                score[a] += 1; score[b] -= 1
            elif w == "B":
                score[b] += 1; score[a] -= 1
        best = max(score.values())
        winners = [n for n in pool_names if score[n] == best]
        return pool_names[0] if pool_names[0] in winners else sorted(winners)[0]

    print("\nQ2b hidden F1 of the candidate each evaluator selects (incumbent = naive on ties)")
    sel = {}
    edits = base_names
    for ev in evaluators:
        chosen = [select(t, ev, edits) for t in tasks]
        f1 = np.array([grades[t][c]["f1"] for t, c in zip(tasks, chosen)])
        sel[ev] = {"f1": float(f1.mean()), "chosen": dict(zip(tasks, chosen))}
        print(f"  {ev:16s} selected F1={f1.mean():.3f}")

    q3 = {}
    redrawn = sorted({n.rsplit("__s", 1)[0] for t in tasks for n in runs[t] if not n.endswith("__s0")})
    for cand in redrawn:
        members = sorted(n for n in runs[tasks[0]] if n.rsplit("__s", 1)[0] == cand)
        if not all(all(m in runs[t] for m in members) for t in tasks):
            continue
        single = np.array([grades[t][f"{cand}__s0"]["f1"] for t in tasks])
        spread = np.array([np.std([grades[t][m]["f1"] for m in members]) for t in tasks])
        row = {"single_f1": float(single.mean()), "rerun_std": float(spread.mean()),
               "mean_over_redraws": float(np.mean([[grades[t][m]["f1"] for m in members]
                                                   for t in tasks]))}
        print(f"\nQ3  {cand}: best-of-{len(members)} integrator re-draws (single draw "
              f"F1={single.mean():.3f}, mean over draws {row['mean_over_redraws']:.3f}, "
              f"per-task std {spread.mean():.3f})")
        for ev in evaluators:
            chosen = [select(t, ev, [f"{cand}__s0"] + [m for m in members if m != f"{cand}__s0"])
                      for t in tasks]
            f1 = np.array([grades[t][c]["f1"] for t, c in zip(tasks, chosen)])
            row[ev] = float(f1.mean())
            print(f"  {ev:16s} picked F1={f1.mean():.3f}")
        oracle = np.array([max(grades[t][m]["f1"] for m in members) for t in tasks])
        row["oracle"] = float(oracle.mean())
        print(f"  {'oracle':16s} picked F1={oracle.mean():.3f}")
        q3[cand] = row
    return {"q2": q2, "q2_selection": sel, "q3": q3,
            "verdicts": {f"{t}|{a}|{b}": v for (t, a, b), v in verdicts.items()},
            "profiles_counts": {t: {n: p["counts"] for n, p in ps.items()}
                                for t, ps in profiles.items()}}


if __name__ == "__main__":
    main()
