"""Transferable collaboration rules: what repeatedly worked, stated as symptom → edit.

A rule is distilled from optimizer runs on source tasks: for a trace symptom (e.g.
``direction_dropped``) and an edit (by its key, e.g. ``handoff@de->interp:[('split_direction',
True), ...]``), how often was it accepted when tried, and how often rejected. On a new task a
rule never bypasses validation — the edit is still replayed, controlled and judged — it only
decides what is tried *first*, so a correct rule saves search cost and a wrong one costs one
rejected attempt.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Rule:
    symptom: str
    edit_key: str
    description: str
    accepted: int
    tried: int
    source_tasks: list[str]

    @property
    def success_rate(self) -> float:
        return self.accepted / self.tried if self.tried else 0.0

    @property
    def score(self) -> float:
        """Laplace-smoothed acceptance rate, so a single lucky acceptance ranks below a record."""
        return (self.accepted + 1) / (self.tried + 2)


def edit_key_of(attempt: dict[str, Any]) -> str:
    return attempt.get("edit_key") or f"{attempt['family']}@{attempt['location']}:{attempt['edit']}"


def distill(outcomes: Iterable[tuple[str, dict[str, Any]]], *, min_accepted: int = 2,
            min_rate: float = 0.5) -> list[Rule]:
    """``outcomes``: (task_id, Outcome.to_dict()) pairs from source tasks."""
    stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"accepted": 0, "tried": 0, "tasks": set(), "desc": ""})
    for task_id, out in outcomes:
        for a in out["attempts"]:
            key = (a["finding"], edit_key_of(a))
            s = stats[key]
            s["tried"] += 1
            s["desc"] = a["edit"]
            if a["accepted"]:
                s["accepted"] += 1
                s["tasks"].add(task_id)
    rules = [Rule(sym, ek, s["desc"], s["accepted"], s["tried"], sorted(s["tasks"]))
             for (sym, ek), s in stats.items()]
    return sorted([r for r in rules if r.accepted >= min_accepted and r.success_rate >= min_rate],
                  key=lambda r: -r.score)


def save(rules: list[Rule], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in rules], indent=1))


def load(path: Path) -> list[Rule]:
    return [Rule(**r) for r in json.loads(Path(path).read_text())]


def index(rules: list[Rule]) -> dict[tuple[str, str], Rule]:
    return {(r.symptom, r.edit_key): r for r in rules}


if __name__ == "__main__":  # pragma: no cover
    # python -m agentcoop.optimize.rules <suite> <method> <out.json>   (e.g. msdev ours)
    import sys

    from agentcoop.workflow import ROOT

    suite, method, dest = sys.argv[1:4]
    found = sorted((ROOT / "runs" / "phase2" / suite).glob(f"*/{method}.json"))
    distilled = distill((f.parent.name, json.loads(f.read_text())) for f in found)
    save(distilled, Path(dest))
    print(f"{len(distilled)} rules from {len(found)} outcomes -> {dest}")
