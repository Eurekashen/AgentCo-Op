"""Every test is hermetic: no AGENTCOOP_* setting from the developer's shell leaks in, and the
LLM ledger/cache and the memo store live in the test's tmp_path, never in runs/."""

from __future__ import annotations

import os

import pytest

from agentcoop import store


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch, tmp_path):
    for k in [k for k in os.environ if k.startswith("AGENTCOOP_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("AGENTCOOP_LLM_DB", str(tmp_path / "llm.sqlite"))
    monkeypatch.setattr(store, "STORE", tmp_path / "store")
