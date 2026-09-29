"""Donor-consistency analysis: an evidence-producing component an optimizer can add as a branch.

For each DE table it takes the significant genes per direction and reports in how many donors
the group changes in that direction (per-donor pseudobulk log fold change of the group mean).
It answers exactly the question an integrator must settle before calling an effect
"reproducible across donors", which neither a DE p-value nor a gene-set name answers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agentcoop.evidence.checks import donor_tables
from agentcoop.workflow import Artifact, RunContext, agent


@agent("donor_check")
def donor_check(inputs: list[Artifact], config: dict[str, Any], ctx: RunContext) -> list[Artifact]:
    max_padj = float(config.get("max_padj", 0.05))
    k = int(config.get("k", 30))
    tables = donor_tables(str(ctx.pooled_data()))
    out = []
    for de in (a for a in inputs if a.kind == "de_table"):
        ct = de.data["cell_type"]
        table = tables.get(ct)
        if table is None:
            continue
        idx = table.index()
        for direction, sign in (("up", 1), ("down", -1)):
            genes = [r[0] for r in de.data["rows"] if r[3] <= max_padj and sign * r[1] > 0][:k]
            rows = [idx[g] for g in genes if g in idx]
            if len(rows) < 3:
                continue
            group = table.lfc[rows].mean(axis=0)
            agree = int(np.sum(np.sign(group) == sign))
            per_gene = np.sum(np.sign(table.lfc[rows]) == sign, axis=1)
            out.append(Artifact(
                kind="donor_evidence", producer=ctx.node_id,
                data={"cell_type": ct, "direction": direction, "n_genes": len(rows),
                      "donors_where_group_changes_in_this_direction":
                          f"{agree}/{len(table.donors)}",
                      "median_donors_agreeing_per_gene": int(np.median(per_gene)),
                      "genes": genes[:15]},
                derived_from=[de.id], meta={"cell_type": ct, "direction": direction}))
    return out
