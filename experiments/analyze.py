"""Grade stored v4 runs and tabulate them (evaluation side: reads the hidden answers).

    python experiments/analyze.py --suite fsdev [--tasks fsdev-800 fsdev-801]

Reads runs/v4/<suite>/<task>/*.json (written by experiments/run.py), grades every stored result
with eval.planted.grade and averages a task's draws, because the task is the statistical unit
(§7). Precision, recall, F1, artefact rejection and scope accuracy are over the stored findings
(``claims``). Cause accuracy is over every claim the arm labelled: for ours and no_resolution,
Task 9 stores them with their labels, causes and scopes in meta["all_claims"], because causes
mostly explain demoted claims, which are not findings. Derived rows best_single / best_single_codex take, per task, the maximum over proposing
sites on each quality endpoint (an after-the-fact choice that favours the baseline, §4); their
claims, calls and cost are the mean of one site's run. Printed, and written to
runs/v4/<suite>_analysis.json:

* per arm: mean precision, recall and F1 with task-bootstrap 95% CIs, artefact rejection rate,
  scope and cause accuracy (over the tasks where they are defined), mean claims, site-agent
  calls and nominal USD;
* ours vs every other arm on F1: paired Wilcoxon signed-rank over tasks, differences within
  1e-9 dropped as ties (a mean of equal draws is not exactly equal);
* silo-cost recovery on F1 = (ours - best single) / (oracle - best single), from task means,
  with a task-bootstrap 95% CI; undefined when the oracle does not beat the best single site;
* the mechanism decomposition (§4): ours, no_resolution, ma_claims, ma_genome,
  ma_genome_repaired, no_routing, each with its paired F1 comparison to ours;
* the view-error rate of agent-built views (§6) per stage, harness and ids_labels knob: every
  view an arm recorded in meta["views"] (step 1 from single / single_codex, self-checked from
  ma_genome, repaired from ours) is restored from the store and compared with the task's correct
  codebook by eval.planted.view_errors. If it is about 0 with the knob on, §6 drops the
  view-repair claim and the knob.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentcoop import llm, store  # noqa: E402
from agentcoop.cases.planted import TASKS, load_case  # noqa: E402
from agentcoop.coop.arms import ARMS  # noqa: E402
from agentcoop.coop.claims import claim_from_dict  # noqa: E402
from agentcoop.coop.view import load_view  # noqa: E402
from agentcoop.eval.planted import grade, load_truth, view_errors  # noqa: E402

OUT = ROOT / "runs" / ("v4" if llm.backbone() == llm.DEFAULT_MODEL else f"v4@{llm.backbone()}")
ENDPOINTS = ("precision", "recall", "f1")
RATES = ("artefact_rejected", "scope_accuracy", "cause_accuracy")
COUNTS = ("n_claims", "calls", "nominal_usd")
MECHANISM = ("ours", "no_resolution", "ma_claims", "ma_genome", "ma_genome_repaired", "no_routing")
#: Arm family -> which views its meta["views"] holds.
STAGES = {"single": "step 1", "single_codex": "step 1 (all Codex)", "ma_genome": "self-checked",
          "ours": "repaired"}
TIE = 1e-9
N_BOOT = 10_000

PerTask = dict[str, dict[str, dict]]     # task -> arm -> metric -> value


def load_results(suite: str, tasks: list[str] | None) -> dict[str, list[dict]]:
    """task -> its stored results (every arm and draw)."""
    return {d.name: [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
            for d in sorted((OUT / suite).iterdir())
            if d.is_dir() and (not tasks or d.name in tasks)}


def _mean(values: list) -> float | None:
    """Mean over the defined values (booleans count 0/1); None when none is defined."""
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def grade_task(task_dir: Path, results: list[dict]) -> dict[str, dict]:
    """arm -> graded metrics averaged over the arm's draws, plus the best-single rows."""
    truth = load_truth(task_dir)
    draws: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        g = grade([claim_from_dict(c) for c in r["claims"]], truth, labels=r["labels"] or None,
                  causes=r["causes"] or None, scopes=r["scopes"] or None)
        every = r["meta"].get("all_claims")
        if every:
            claims = [claim_from_dict(d["claim"]) for d in every]
            full = grade(claims, truth,
                         labels={c.id: d["label"] for c, d in zip(claims, every)},
                         causes={c.id: d["cause"] for c, d in zip(claims, every) if d["cause"]},
                         scopes={c.id: d["scope"] for c, d in zip(claims, every) if d["scope"]})
            g["cause_accuracy"] = full["cause_accuracy"]
        draws[r["arm"]].append({**g, "calls": r["calls"], "nominal_usd": r["nominal_usd"]})
    keys = (*ENDPOINTS, *RATES, *COUNTS)
    per_arm = {arm: {k: _mean([d[k] for d in ds]) for k in keys} for arm, ds in draws.items()}
    for family in ("single", "single_codex"):
        sites = [m for arm, m in per_arm.items() if arm.startswith(family + ":")]
        if sites:
            per_arm[f"best_{family}"] = {
                **{k: max((m[k] for m in sites if m[k] is not None), default=None)
                   for k in (*ENDPOINTS, *RATES)},
                **{k: _mean([m[k] for m in sites]) for k in COUNTS}}
    return per_arm


def _boot_ci(x: np.ndarray) -> list[float] | None:
    """Percentile 95% CI of the mean over resampled tasks (seeded, so tables reproduce)."""
    if len(x) < 2:
        return None
    idx = np.random.default_rng(0).integers(0, len(x), (N_BOOT, len(x)))
    return [float(v) for v in np.quantile(x[idx].mean(axis=1), [0.025, 0.975])]


def summarize(per_task: PerTask, arm: str) -> dict:
    rows = [arms[arm] for arms in per_task.values() if arm in arms]
    out: dict = {"n_tasks": len(rows)}
    for k in ENDPOINTS:
        x = np.array([r[k] for r in rows if r[k] is not None], dtype=float)
        out[k] = float(x.mean()) if len(x) else None
        out[f"{k}_ci95"] = _boot_ci(x)
    for k in (*RATES, *COUNTS):
        out[k] = _mean([r[k] for r in rows])
    return out


def paired(per_task: PerTask, a: str, b: str, metric: str = "f1") -> dict:
    """Wilcoxon signed-rank of a - b over the tasks holding both; |difference| <= TIE is a tie
    and is dropped before the test. p is None when no difference is left."""
    d = np.array([arms[a][metric] - arms[b][metric] for arms in per_task.values()
                  if a in arms and b in arms and arms[a][metric] is not None
                  and arms[b][metric] is not None], dtype=float)
    nz = d[np.abs(d) > TIE]
    return {"n": len(d), "wins": int((nz > 0).sum()), "losses": int((nz < 0).sum()),
            "ties": int(len(d) - len(nz)), "mean_diff": float(d.mean()) if len(d) else None,
            "p": float(stats.wilcoxon(nz).pvalue) if len(nz) else None}


def silo_cost_recovery(per_task: PerTask, metric: str = "f1") -> dict:
    """(ours - best single) / (oracle - best single) on task means, over tasks holding all three;
    a resample (or the estimate) whose oracle gap is <= TIE has no silo cost to recover."""
    x = np.array([[arms[a][metric] for a in ("ours", "best_single", "oracle")]
                  for arms in per_task.values()
                  if all(a in arms and arms[a][metric] is not None
                         for a in ("ours", "best_single", "oracle"))], dtype=float).reshape(-1, 3)
    if not len(x):
        return {"n": 0, "value": None, "ci95": None}
    ours, single, oracle = (float(v) for v in x.mean(axis=0))
    value = (ours - single) / (oracle - single) if oracle - single > TIE else None
    ci = None
    if len(x) > 1:
        m = x[np.random.default_rng(0).integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
        gap = m[:, 2] - m[:, 1]
        ok = gap > TIE
        if ok.any():
            ci = [float(v) for v in np.quantile((m[ok, 0] - m[ok, 1]) / gap[ok], [0.025, 0.975])]
    return {"n": len(x), "ours": ours, "best_single": single, "oracle": oracle, "value": value,
            "ci95": ci}


def view_rows(task_dir: Path, results: list[dict]) -> list[dict]:
    """One row per distinct (stage, site, view) among the views the results recorded: stage,
    harness, the task's knobs and eval.planted.view_errors of the restored view. A site that had
    no view (failed call) or an invalid one is a row with valid False and no rates."""
    entries = {(STAGES[r["arm"].split(":")[0]], site, blob,
                "codex" if r["arm"].startswith("single_codex:") else r["agents"][site])
               for r in results if r["arm"].split(":")[0] in STAGES
               for site, blob in r["meta"].get("views", {}).items()}
    if not entries:
        return []
    case, _ = load_case(task_dir)
    knobs = load_truth(task_dir)["knobs"]
    rows = []
    for stage, site, blob, agent in sorted(entries, key=str):
        errors = None
        if blob is not None:
            with tempfile.TemporaryDirectory() as tmp:
                store.get_blob(blob, Path(tmp) / "view")
                view, _ = load_view(Path(tmp) / "view", kind=case.kind, vocab=case.vocab)
                errors = None if view is None else view_errors(view, task_dir, site)
        rows.append({"task": task_dir.name, "stage": stage, "site": site, "agent": agent,
                     "view": blob, **knobs, "valid": errors is not None, **(errors or {})})
    return rows


def view_error_summary(rows: list[dict]) -> list[dict]:
    """Per (stage, harness, ids_labels knob): views, invalid views, mean error rate of the valid
    ones and the share of valid views with any error."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["stage"], row["agent"], row["ids_labels"])].append(row)
    return [{"stage": stage, "agent": agent, "ids_labels": knob, "n_views": len(rs),
             "n_invalid": sum(not r["valid"] for r in rs),
             "error_rate": _mean([r["error_rate"] for r in rs if r["valid"]]),
             "any_error": _mean([r["error_rate"] > 0 for r in rs if r["valid"]])}
            for (stage, agent, knob), rs in sorted(groups.items(), key=str)]


def _order(arm: str) -> tuple[int, str]:
    base = arm.split(":")[0].removeprefix("best_")
    return (ARMS.index(base) if base in ARMS else len(ARMS), arm)


def analyze(per_task: PerTask) -> dict:
    arms = sorted({a for t in per_task.values() for a in t}, key=_order)
    summary = {a: summarize(per_task, a) for a in arms}
    vs_ours = {a: paired(per_task, "ours", a) for a in arms if a != "ours"}
    return {"tasks": sorted(per_task), "per_task": per_task, "arms": summary,
            "paired_f1_vs_ours": vs_ours, "silo_cost_recovery_f1": silo_cost_recovery(per_task),
            "mechanism": [{"arm": a, **{k: summary[a][k] for k in ENDPOINTS},
                           "vs_ours": vs_ours.get(a)} for a in MECHANISM if a in summary]}


def _f(x: float | None, nd: int = 2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _ci(s: dict, k: str) -> str:
    ci = s[f"{k}_ci95"]
    return f"{_f(s[k])} " + ("[-]" if ci is None else f"[{ci[0]:.2f},{ci[1]:.2f}]")


def print_report(res: dict) -> None:
    print(f"{'arm':<22}{'n':>3}  {'precision':<18}{'recall':<18}{'F1':<18}"
          f"{'art.rej':>7}{'scope':>7}{'cause':>7}{'claims':>7}{'calls':>7}{'nominal$':>9}")
    for arm, s in res["arms"].items():
        print(f"{arm:<22}{s['n_tasks']:>3}  {_ci(s, 'precision'):<18}{_ci(s, 'recall'):<18}"
              f"{_ci(s, 'f1'):<18}{_f(s['artefact_rejected']):>7}{_f(s['scope_accuracy']):>7}"
              f"{_f(s['cause_accuracy']):>7}{_f(s['n_claims'], 1):>7}{_f(s['calls'], 1):>7}"
              f"{_f(s['nominal_usd'], 3):>9}")
    print("\nours vs arm on F1 (paired Wilcoxon over tasks; W/L/T)")
    for arm, p in res["paired_f1_vs_ours"].items():
        print(f"  {arm:<22} n={p['n']:<3} {p['wins']}/{p['losses']}/{p['ties']}  "
              f"mean diff {_f(p['mean_diff'], 3)}  p={_f(p['p'], 4)}")
    r = res["silo_cost_recovery_f1"]
    ci = "-" if r["ci95"] is None else f"[{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"
    print(f"\nsilo-cost recovery (F1, n={r['n']}): {_f(r['value'])} {ci}")
    print("\nmechanism decomposition (F1; ours - arm)")
    for m in res["mechanism"]:
        p = m["vs_ours"]
        diff = "" if p is None else (f"  diff {_f(p['mean_diff'], 3)} "
                                     f"({p['wins']}/{p['losses']}/{p['ties']}, p={_f(p['p'], 4)})")
        print(f"  {m['arm']:<20} F1 {_f(m['f1'])}  P {_f(m['precision'])}  "
              f"R {_f(m['recall'])}{diff}")
    print("\nview-error rate of agent-built views (§6)")
    for v in res["view_errors"]:
        print(f"  {v['stage']:<20} {v['agent']:<6} ids_labels={str(v['ids_labels']):<5} "
              f"views={v['n_views']:<3} invalid={v['n_invalid']:<3} "
              f"error rate {_f(v['error_rate'], 3)}  any error {_f(v['any_error'])}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tasks", nargs="*", help="default: every task with stored results")
    args = ap.parse_args(argv)
    if not (OUT / args.suite).is_dir():
        ap.error(f"no stored runs in {OUT / args.suite} (experiments/run.py writes them)")
    results = load_results(args.suite, args.tasks)
    per_task = {t: grade_task(TASKS / args.suite / t, rs) for t, rs in results.items()}
    views = [row for t, rs in results.items() for row in view_rows(TASKS / args.suite / t, rs)]
    res = {"suite": args.suite, **analyze(per_task), "view_errors": view_error_summary(views),
           "views": views}
    (OUT / f"{args.suite}_analysis.json").write_text(json.dumps(res, indent=1))
    print_report(res)


if __name__ == "__main__":
    main()
