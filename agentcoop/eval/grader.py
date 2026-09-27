"""Hidden grader. Reads the planted answer; only the evaluation harness may import this module.

A claim is correct when its cell type, direction and program *family* all match a reproducible
planted program. Family is assigned from the claim's program *name* alone (not its genes), by an
LLM classifier over the fixed family list — naming is the scientific content of an
interpretation, so a vague or wrong name earns nothing even if the cited genes happen to be right.

Precision is computed over de-duplicated claims (cell type, family, direction), so repeating a
claim neither helps nor hurts; recall is over planted (program, cell type) items.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agentcoop import llm
from agentcoop.tasks.planted import CELL_TYPES, FAMILIES, TaskTruth

_CT_RULES = [
    ("FCGR3A+ Monocytes", ("fcgr3a", "cd16", "non-classical", "nonclassical")),
    ("CD14+ Monocytes", ("cd14", "classical mono")),
    ("CD8 T cells", ("cd8",)),
    ("CD4 T cells", ("cd4",)),
    ("NK cells", ("nk ", "nk_", "natural killer")),
    ("B cells", ("b cell", "b-cell", "b_cell")),
]


def canon_cell_types(text: str) -> list[str]:
    t = f" {str(text).lower()} "
    t = t.replace("nk cells", "nk ").replace("nk cell", "nk ")
    found = []
    for name, keys in _CT_RULES:
        if any(k in t for k in keys) and name not in found:
            found.append(name)
    if "FCGR3A+ Monocytes" in found and "CD14+ Monocytes" in found and "cd14" not in t:
        found.remove("CD14+ Monocytes")
    return found


def canon_direction(text: Any) -> str:
    t = str(text).lower()
    if t.startswith(("up", "induc", "increas", "activ", "higher")):
        return "up"
    if t.startswith(("down", "repress", "decreas", "suppress", "lower", "reduc")):
        return "down"
    return "unknown"


FAMILY_PROMPT = """Classify the biological program named below into exactly one family.

Families:
{families}
other: none of the above, too vague to place (e.g. "immune activation", "stress response"),
       or a mixture of several families

Program name: "{name}"

Return JSON {{"family": "<family key>"}}."""


def classify_family(name: str) -> str:
    families = "\n".join(f"{k}: {v['description']}" for k, v in FAMILIES.items())
    out, _ = llm.chat_json(
        [{"role": "user", "content": FAMILY_PROMPT.format(families=families, name=name)}],
        reasoning_effort="low",
    )
    fam = str(out.get("family", "other")).strip()
    return fam if fam in FAMILIES else "other"


def grade(report: dict[str, Any], truth: TaskTruth) -> dict[str, Any]:
    with llm.tag("grader"):
        return _grade(report, truth)


def _grade(report: dict[str, Any], truth: TaskTruth) -> dict[str, Any]:
    items = {(ct, p.family, p.direction) for p in truth.reproducible() for ct in p.cell_types}
    decoys = {(ct, p.family) for p in truth.programs if p.kind == "decoy" for ct in p.cell_types}
    reproducible_pairs = {(ct, fam) for ct, fam, _ in items}

    claims = set()
    parsed = []
    for c in report.get("claims", []) or []:
        if not isinstance(c, dict):
            continue
        fam = classify_family(str(c.get("program", "")))
        direction = canon_direction(c.get("direction", ""))
        for ct in canon_cell_types(c.get("cell_type", "")) or ["unknown"]:
            claims.add((ct, fam, direction))
            parsed.append({"cell_type": ct, "family": fam, "direction": direction,
                           "program": c.get("program", "")})

    tp = {c for c in claims if c in items}
    wrong_direction = {c for c in claims if c not in items and (c[0], c[1]) in reproducible_pairs}
    decoy_claims = {c for c in claims if (c[0], c[1]) in decoys}
    precision = len(tp) / len(claims) if claims else (1.0 if not items else 0.0)
    recall = len(tp) / len(items) if items else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "f1": f1, "precision": precision, "recall": recall,
        "n_claims": len(claims), "n_items": len(items), "tp": len(tp),
        "wrong_direction": len(wrong_direction), "decoy_claimed": len(decoy_claims),
        "other_fp": len(claims) - len(tp) - len(wrong_direction - decoy_claims) - len(decoy_claims),
        "parsed": parsed,
        "missed": sorted(map(list, items - tp)),
    }
