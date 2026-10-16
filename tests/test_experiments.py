"""experiments/ and the code they add to the arms and the grader: recorded views, runner
bookkeeping, grading on every labelled claim, paired statistics, view-error rates and the
headroom gate.
Offline: fakes, synthetic records and one planted task built by Task 8's generator on its fake
base (no API key, no agent, no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from agentcoop import store
from agentcoop.cases import planted
from agentcoop.coop import arms
from agentcoop.coop.arms import ArmResult
from agentcoop.coop.claims import claim_to_dict
from agentcoop.coop.protocol import ProtocolRun
from agentcoop.coop.tester import TestResult
from agentcoop.coop.view import load_view, write_view
from agentcoop.eval import planted as grader
from tests.fakes import (PROGRAMS, FakeMemo, SiteFactory, case_spec, claim, fake_workspace,
                         site_specs, standard_sites)
from tests.test_planted import _fake_base, _fake_libraries

ROOT = Path(__file__).resolve().parents[1]
IFN = sorted(PROGRAMS["IFN"])


def _script(name: str):
    """experiments/ is not a package: load a script as a module under a unique name (registered
    in sys.modules, which dataclasses need to resolve the script's annotations)."""
    spec = importlib.util.spec_from_file_location(f"experiments_{name}",
                                                  ROOT / "experiments" / f"{name}.py")
    module = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------- arms: recorded views
# (MA-genome's min_sites=1 single-site mode, which the headroom gate uses, is Task 9's and is
# tested in tests/test_arms.py.)

def test_arms_record_the_view_each_site_used(tmp_path, monkeypatch):
    """single / single_codex: the site's step-1 view; ma_genome: the self-checked views; ours:
    the views after the protocol. A site without a view id (here M) records None."""
    factory = SiteFactory(standard_sites())

    def make_sites(specs, case, *, task_id, run_id, agents=None):
        sites = factory(specs, case, task_id=task_id, run_id=run_id, agents=agents)
        for s in sites[:3]:
            s.view_blob = f"{run_id.split('/')[1]}-{s.name}"
        return sites

    monkeypatch.setattr(arms, "make_sites", make_sites)
    monkeypatch.setattr(arms.store, "memo", FakeMemo())
    monkeypatch.setattr(arms, "new_workspace", fake_workspace(tmp_path / "ws"))
    monkeypatch.setattr(arms, "Coordinator", lambda case: None)
    monkeypatch.setattr(arms, "run_protocol",
                        lambda sites, case, **kw: ProtocolRun([], [], 0, 0.0))
    out = arms.run_arms(["single", "single_codex", "ma_genome", "ours"], site_specs(tmp_path),
                        case_spec(), task_id="t0", sample=0)
    assert out["single:B"].meta["views"] == {"B": "base-B"}
    assert out["single_codex:C"].meta["views"] == {"C": "single_codex-C"}
    assert out["ma_genome"].meta["views"] == {"A": "ma-A", "B": "ma-B", "C": "ma-C", "M": None}
    assert "tables" in out["ma_genome"].meta
    assert out["ours"].meta["views"] == {"A": "ours-A", "B": "ours-B", "C": "ours-C", "M": None}


# ---------------------------------------------------------------------------- run.py

run = _script("run")

SITES = [{"name": "A", "data_path": "a.h5ad", "card": {}, "proposes": True},
         {"name": "B", "data_path": "b.h5ad", "card": {}, "proposes": True},
         {"name": "M", "data_path": "m.h5ad", "card": {}, "proposes": False}]


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """run.py on a fake suite of two tasks; the fake run_arms records every call."""
    for task in ("fx-000", "fx-001"):
        (tmp_path / "tasks" / "fx" / task).mkdir(parents=True)
    calls = []

    def fake_run_arms(arm_names, specs, case, *, task_id, sample):
        calls.append((task_id, sample, list(arm_names), {s["name"]: s["agent"] for s in specs}))
        out = {}
        for arm in arm_names:
            keys = ([f"{arm}:{s['name']}" for s in specs if s["proposes"]]
                    if arm in ("single", "single_codex") else [arm])     # Task 9's key scheme
            for key in keys:
                out[key] = ArmResult(arm=key, claims=[claim(f"{key}:1", "A", tuple(IFN[:3]))],
                                     labels={}, causes={}, scopes={}, calls=2, nominal_usd=0.01,
                                     meta={"k": 1})
        return out

    monkeypatch.setattr(run, "TASKS", tmp_path / "tasks")
    monkeypatch.setattr(run, "OUT", tmp_path / "out")
    monkeypatch.setattr(run, "load_case", lambda d: (case_spec(), [dict(s) for s in SITES]))
    monkeypatch.setattr(run, "run_arms", fake_run_arms)
    monkeypatch.setenv("AGENTCOOP_BUDGET_USD", "1")
    monkeypatch.setenv("AGENTCOOP_LLM_DB", str(tmp_path / "llm.sqlite"))
    return calls


def test_run_writes_one_file_per_arm_and_skips_existing(runner, tmp_path):
    out = tmp_path / "out" / "fx"
    (out / "fx-000").mkdir(parents=True)
    (out / "fx-000" / "ours__s0.json").write_text('{"old": true}')
    run.main(["--suite", "fx", "--arms", "single", "ours", "--parallel", "1"])

    assert sorted(p.name for p in (out / "fx-000").iterdir()) == [
        "ours__s0.json", "single-A__s0.json", "single-B__s0.json"]
    assert json.loads((out / "fx-000" / "ours__s0.json").read_text()) == {"old": True}
    rec = json.loads((out / "fx-001" / "single-A__s0.json").read_text())
    assert (rec["arm"], rec["task"], rec["sample"], rec["calls"]) == ("single:A", "fx-001", 0, 2)
    assert rec["claims"][0]["id"] == "single:A:1" and rec["claims"][0]["up_genes"] == IFN[:3]
    assert rec["meta"] == {"k": 1}
    assert rec["agents"] == {"A": "codex", "B": "finch", "M": "codex"}
    assert sorted((t, a) for t, _, a, _ in runner) == [
        ("fx-000", ["single"]), ("fx-001", ["single", "ours"])]
    assert json.loads((out / "agents.json").read_text()) == rec["agents"]

    runner.clear()
    run.main(["--suite", "fx", "--arms", "single", "ours"])
    assert runner == []                                   # everything exists: nothing re-run


def test_site_agents_default_alternates_proposers_and_gives_testers_codex():
    """load_case's site specs carry no "agent" key; the runner adds it before run_arms."""
    specs = [{"name": n, "proposes": n != "M"} for n in ("A", "B", "C", "M")]
    assert run.site_agents(specs, {}) == {"A": "codex", "B": "finch", "C": "codex", "M": "codex"}
    assert run.site_agents(specs, {"M": "finch"})["M"] == "finch"


def test_run_draws_agent_overrides_and_guards(runner, tmp_path, monkeypatch):
    run.main(["--suite", "fx", "--tasks", "fx-001", "--arms", "vote", "--draws", "0", "1",
              "--agents", "B=codex"])
    assert [(t, s) for t, s, _, _ in runner] == [("fx-001", 0), ("fx-001", 1)]
    assert runner[0][3] == {"A": "codex", "B": "codex", "M": "codex"}
    assert (tmp_path / "out" / "fx" / "fx-001" / "vote__s1.json").exists()

    with pytest.raises(SystemExit, match="fixed per suite"):
        run.main(["--suite", "fx", "--arms", "vote"])      # default assignment differs
    with pytest.raises(SystemExit):
        run.main(["--suite", "fx", "--arms", "vote", "--agents", "Z=codex"])
    with pytest.raises(SystemExit):
        run.main(["--suite", "fx", "--tasks", "fx-999", "--arms", "vote"])
    monkeypatch.delenv("AGENTCOOP_BUDGET_USD")
    with pytest.raises(SystemExit, match="AGENTCOOP_BUDGET_USD"):
        run.main(["--suite", "fx", "--arms", "vote", "--agents", "B=codex"])


def test_run_failed_task_stores_nothing_and_the_others_finish(runner, tmp_path, monkeypatch):
    """A harness failure inside run_arms (Review Focus 3) must not become a stored result."""
    fake = run.run_arms

    def flaky(arm_names, specs, case, *, task_id, sample):
        if task_id == "fx-000":
            raise RuntimeError("codex timed out")
        return fake(arm_names, specs, case, task_id=task_id, sample=sample)

    monkeypatch.setattr(run, "run_arms", flaky)
    with pytest.raises(SystemExit, match="1 task"):
        run.main(["--suite", "fx", "--arms", "vote"])
    assert not (tmp_path / "out" / "fx" / "fx-000").exists()
    assert (tmp_path / "out" / "fx" / "fx-001" / "vote__s0.json").exists()
