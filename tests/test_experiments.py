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
