"""Every test is hermetic: no AGENTCOOP_* setting from the developer's shell leaks in, and the
LLM ledger/cache lives in the test's tmp_path, never in runs/."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch, tmp_path):
    for k in [k for k in os.environ if k.startswith("AGENTCOOP_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("AGENTCOOP_LLM_DB", str(tmp_path / "llm.sqlite"))
