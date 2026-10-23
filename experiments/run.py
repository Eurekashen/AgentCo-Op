"""Run arms x tasks x draws on a v4 suite and store every ArmResult (no grading here).

    AGENTCOOP_BUDGET_USD=180 python experiments/run.py --suite fsdev --tasks fsdev-800 \
        --arms ours ma_genome single --draws 0 [--parallel 2] [--agents A=codex,B=finch,C=codex]

Writes runs/v4/<suite>/<task>/<arm>__s<k>.json (the per-site key "single:A" is stored as
single-A__s0.json; the key itself is inside the file) and never recomputes a file that exists, so
an interrupted run resumes where it stopped. Step-1 proposals are shared through the store, so
arms requested in separate invocations still see identical step-1 claims. A non-default
backbone writes to runs/v4@<model>/, so a robustness run never skips over or overwrites the main
one.

The site -> harness assignment is fixed per suite: the first invocation records it in
runs/v4/<suite>/agents.json and a later invocation with a different one is refused (mixing
assignments inside a suite would confound arm differences with harness choice). Default: codex
and finch alternating over the case's proposing sites in site order; a tester-only site (the
planted mimic M) defaults to codex. Task 8's load_case gives site specs without an "agent" key;
run_task adds it from this assignment before calling run_arms.

Printed per arm: its site-agent calls, nominal USD and the wall time of the run_arms call that
produced it (one call per task and draw computes all missing arms together, sharing their sites,
so per-arm time is not separable). A task whose run_arms raises stores nothing, the other tasks
finish, and the run exits non-zero after printing the spend (a harness failure is never a
result). Grading is experiments/analyze.py, so nothing on this path touches hidden answers.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentcoop import llm  # noqa: E402
from agentcoop.cases import covid, planted, tb  # noqa: E402
from agentcoop.cases.planted import TASKS  # noqa: E402
from agentcoop.coop.arms import ARMS, ArmResult, run_arms  # noqa: E402
from agentcoop.coop.claims import claim_to_dict  # noqa: E402

OUT = ROOT / "runs" / ("v4" if llm.backbone() == llm.DEFAULT_MODEL else f"v4@{llm.backbone()}")
#: A robustness variant (e.g. another site -> harness assignment) gets its own directory.
if os.environ.get("AGENTCOOP_VARIANT"):
    OUT = OUT.with_name(f"{OUT.name}+{os.environ['AGENTCOOP_VARIANT']}")
HARNESSES = ("codex", "finch", "react")
#: Arms that yield one result per proposing site, keyed "<arm>:<site>".
PER_SITE = ("single", "single_codex")


def parse_agents(text: str | None) -> dict[str, str]:
    """Parse "A=codex,B=finch" into {"A": "codex", "B": "finch"}."""
    out: dict[str, str] = {}
    for item in (text or "").split(","):
        if not item.strip():
            continue
        site, _, agent = (s.strip() for s in item.partition("="))
        if not site or agent not in HARNESSES:
            raise ValueError(f"bad --agents item {item!r}: expected SITE={'|'.join(HARNESSES)}")
        out[site] = agent
    return out


def site_agents(site_specs: list[dict], overrides: dict[str, str]) -> dict[str, str]:
    """Harness per site: codex/finch alternating over the proposing sites in site order, codex
    for a tester-only site, then overrides."""
    names = [s["name"] for s in site_specs]
    unknown = sorted(set(overrides) - set(names))
    if unknown:
        raise ValueError(f"--agents names unknown sites {unknown}; the suite's sites: {names}")
    proposers = [s["name"] for s in site_specs if s["proposes"]]
    default = {n: ("codex", "finch")[i % 2] for i, n in enumerate(proposers)}
    return {n: overrides.get(n, default.get(n, "codex")) for n in names}


def fix_agents(suite: str, agents: dict[str, str]) -> None:
    """Record the suite's assignment on first use; refuse a different one afterwards."""
    path = OUT / suite / "agents.json"
    if path.exists():
        stored = json.loads(path.read_text())
        if stored != agents:
            raise SystemExit(f"suite {suite} runs with agents {stored}, not {agents} "
                             f"(the assignment is fixed per suite: {path})")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(agents, indent=1))


def result_keys(arm: str, site_specs: list[dict]) -> list[str]:
    if arm in PER_SITE:
        return [f"{arm}:{s['name']}" for s in site_specs if s["proposes"]]
    return [arm]


def result_path(suite: str, task_id: str, key: str, sample: int) -> Path:
    return OUT / suite / task_id / f"{key.replace(':', '-')}__s{sample}.json"


def result_record(key: str, r: ArmResult, *, task_id: str, sample: int, agents: dict[str, str],
                  seconds: float) -> dict:
    return {"arm": key, "task": task_id, "sample": sample, "agents": agents,
            "claims": [claim_to_dict(c) for c in r.claims], "labels": r.labels,
            "causes": r.causes, "scopes": r.scopes, "calls": r.calls,
            "nominal_usd": r.nominal_usd, "meta": r.meta, "seconds": round(seconds, 1)}


def write_json(path: Path, record: dict) -> None:
    """Atomic, so a crash mid-write cannot leave a partial file that later runs would skip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, path)


#: Real-data suites (data/tasks/<suite>/<task>) and their case loaders; any other suite is planted.
CASES = {"covid": covid.load_case, "tb": tb.load_case}


def load_case(task_dir: Path):
    return CASES.get(task_dir.parent.name, planted.load_case)(task_dir)


def run_task(suite: str, task_dir: Path, arms: list[str], draws: list[int],
             agents: dict[str, str]) -> None:
    task_id = task_dir.name
    case, site_specs = load_case(task_dir)
    names = [s["name"] for s in site_specs]
    if sorted(names) != sorted(agents):
        raise ValueError(f"{task_id}: sites {names} differ from the suite's agents {agents}")
    site_specs = [{**s, "agent": agents[s["name"]]} for s in site_specs]
    with llm.tag(f"v4/{suite}/{task_id}"):
        for sample in draws:
            todo = [a for a in arms if any(not result_path(suite, task_id, k, sample).exists()
                                           for k in result_keys(a, site_specs))]
            if not todo:
                continue
            t0 = time.time()
            results = run_arms(todo, site_specs, case, task_id=task_id, sample=sample)
            seconds = time.time() - t0
            for key, r in results.items():
                path = result_path(suite, task_id, key, sample)
                if path.exists():
                    continue
                write_json(path, result_record(key, r, task_id=task_id, sample=sample,
                                               agents=agents, seconds=seconds))
                print(f"{task_id} s{sample} {key:<24} calls={r.calls:<3} "
                      f"nominal ${r.nominal_usd:.3f} {seconds:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tasks", nargs="+", default=["all"], help="task names, or all")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--draws", nargs="+", type=int, default=[0], help="sample indices")
    ap.add_argument("--parallel", type=int, default=2, help="tasks run in parallel")
    ap.add_argument("--agents", help="site -> harness, e.g. A=codex,B=finch,C=codex")
    args = ap.parse_args(argv)
    if not os.environ.get("AGENTCOOP_BUDGET_USD"):
        raise SystemExit("set AGENTCOOP_BUDGET_USD (hard cap on actual spend) before running arms")
    suite_dir = TASKS / args.suite
    task_dirs = sorted(d for d in suite_dir.iterdir() if d.is_dir()) if suite_dir.is_dir() else []
    if args.tasks != ["all"]:
        missing = sorted(set(args.tasks) - {d.name for d in task_dirs})
        if missing:
            ap.error(f"no such task(s) in {suite_dir}: {missing}")
        task_dirs = [d for d in task_dirs if d.name in args.tasks]
    if not task_dirs:
        ap.error(f"no tasks in {suite_dir}")
    try:
        agents = site_agents(load_case(task_dirs[0])[1], parse_agents(args.agents))
    except ValueError as e:
        ap.error(str(e))
    fix_agents(args.suite, agents)

    failed = []
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_task, args.suite, d, args.arms, args.draws, agents): d
                   for d in task_dirs}
        for fut in cf.as_completed(futures):
            try:
                fut.result()
            except Exception:
                failed.append(futures[fut].name)
                print(f"FAILED {futures[fut].name}\n{traceback.format_exc()}", flush=True)
    print("spent:", llm.spent("v4/"), flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} task(s) failed: {sorted(failed)}")


if __name__ == "__main__":
    main()
