"""Truth-free normalization helpers shared by the optimizer-side checks.

Kept separate from ``agentcoop.eval`` (the hidden grader) on purpose: anything the optimizer
imports must not have a path to the answer sheet.
"""

from __future__ import annotations

from typing import Any, Optional

_CT_KEYS = [
    ("FCGR3A+ Monocytes", ("fcgr3a", "cd16", "non-classical", "nonclassical")),
    ("CD14+ Monocytes", ("cd14", "classical mono")),
    ("CD8 T cells", ("cd8",)),
    ("CD4 T cells", ("cd4",)),
    ("NK cells", (" nk ", "natural killer")),
    ("B cells", ("b cell", "b-cell")),
]


def canon_cell_type(text: str, allowed: list[str]) -> Optional[str]:
    t = f" {text.lower().replace('nk cells', 'nk ').replace('nk cell', 'nk ')} "
    for name, keys in _CT_KEYS:
        if name in allowed and any(k in t for k in keys):
            if name == "CD14+ Monocytes" and "non-classical" in t:
                continue
            return name
    for name in allowed:
        if name.lower() in t:
            return name
    return None


def canon_direction(text: Any) -> str:
    t = str(text).lower()
    if t.startswith(("up", "induc", "increas", "activ", "higher")):
        return "up"
    if t.startswith(("down", "repress", "decreas", "suppress", "lower", "reduc")):
        return "down"
    return "unknown"
