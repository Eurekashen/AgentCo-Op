"""Independent validation for the real-data case study (evaluation side only).

A final report from the 4 discovery donors is checked against evidence no component saw:

1. **Held-out replication** — every claim is re-tested on the 4 validation donors with the same
   per-donor, competitive check the optimizer used on the discovery donors (now on new people):
   supported / contradicted / unassessed / ungrounded.
2. **Missed signal** — coherent programs (GO enrichment) that change reproducibly in the
   validation donors in a (cell type, direction) where no claim replicates. (Gene-level
   coverage is too strict on real data: a claim cites a handful of the dozens of induced ISGs,
   and the rest still form an enriched group.)
3. **Known biology** — IFN-β induces the type-I interferon program in every PBMC type (Kang et
   al. 2018, Fig. 2). Coverage = cell types with an interferon-family claim (classified by the
   pinned grader from the program name) that also replicates in held-out donors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentcoop.eval.grader import classify_family
from agentcoop.eval_free import canon_cell_type
from agentcoop.evidence.checks import profile
from agentcoop.tasks.planted import CELL_TYPES


def validate(report: dict[str, Any], task_dir: Path) -> dict[str, Any]:
    prof = profile(report, Path(task_dir) / "hidden" / "validation.h5ad", CELL_TYPES)
    claims = [c for c in (report.get("claims") or []) if isinstance(c, dict)]
    counts = prof.counts()
    n = len(claims)
    ifn_types, replicated_pairs = set(), set()
    for claim, check in zip(claims, prof.claims):
        if check.status == "supported":
            replicated_pairs.add((check.cell_type, check.direction))
        ct = canon_cell_type(str(claim.get("cell_type", "")), CELL_TYPES)
        if ct and check.status == "supported" and check.direction == "up" \
                and classify_family(str(claim.get("program", ""))) == "interferon":
            ifn_types.add(ct)
    return {
        "n_claims": n,
        "replicated": counts["supported"], "contradicted": counts["contradicted"],
        "unassessed": counts["unassessed"], "ungrounded": counts["ungrounded"],
        "replication_rate": counts["supported"] / n if n else 0.0,
        "missed_programs": [{"cell_type": u.cell_type, "direction": u.direction,
                             "term": u.go_hint[0]["term"] if u.go_hint else None}
                            for u in prof.uncovered
                            if (u.cell_type, u.direction) not in replicated_pairs],
        "ifn_cell_types": sorted(ifn_types), "ifn_coverage": len(ifn_types) / len(CELL_TYPES),
        "claims": [c.__dict__ for c in prof.claims],
    }
