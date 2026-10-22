"""Grade Case 2 (TB) runs against the held-out cohorts (spec §5.2). Evaluation side; $0.

    python experiments/analyze_tb.py [--task tb-min]

Descriptive (user decision 2026-10-22: TB's headroom gate failed). Per arm and draw:
  (a) scope labels where ours differs from S2-alone, judged on GSE73408 TB vs pneumonia
      (eval.tb.scope_correct: `specific` right iff held-out AUC >= 0.70, `not specific` iff
      < 0.65; [0.65, 0.70) excluded). S2-alone's label for a claim is the in-sample form of the
      scope rule on S2's own view (definition fixed before any TB run): `specific vs mixed_OD`
      when S2's TB-vs-OD AUC is >= 0.70 with CI lower > 0.5, `not specific vs mixed_OD` when its
      CI upper < 0.65, else `inconclusive` - what a single site can say about scope alone;
  (b) the final signature (eval.tb.final_signature: top-5 findings by rank) scored on GSE73408
      TB vs pneumonia, AUC with bootstrap CI;
  secondary: the final signature on every confirmatory cohort and contrast, and the claim
  replication rate (a finding replicates in a cohort when its AUC CI lower > 0.5; replicated
  when it does so in >= 2/3 of the confirmatory cohorts holding its contrast).
Writes runs/v4/tb_analysis.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentcoop import store  # noqa: E402
from agentcoop.cases import tb  # noqa: E402
from agentcoop.coop.claims import Claim  # noqa: E402
from agentcoop.coop.tester import test_claim  # noqa: E402
from agentcoop.coop.view import load_view, site_table  # noqa: E402
from agentcoop.eval import tb as ev  # noqa: E402

RUNS = ROOT / "runs" / "v4" / "tb"
OUT = ROOT / "runs" / "v4" / "tb_analysis.json"


def claim_of(d: dict) -> Claim:
    fields = Claim.__dataclass_fields__
    return Claim(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()
                    if k in fields})


def findings(result: dict) -> list[Claim]:
    labels = result.get("labels") or {}
    claims = [claim_of(c) for c in result["claims"]]
    return sorted((c for c in claims if labels.get(c.id, "corroborated") == "corroborated"),
                  key=lambda c: (c.rank, c.id))


def s2_alone_scope(claim: Claim, table, case) -> str:
    """The in-sample scope label S2 alone gives a claim (module docstring)."""
    r = test_claim(table, claim, case.tester, site="S2", test="scope", contrast=tb.OD)
    auc = tb._auc(r)
    if auc is None:
        return "inconclusive"
    if auc[2] < tb.NOT_SPECIFIC_AUC:
        return "not specific vs mixed_OD"
    if auc[0] >= tb.SPECIFIC_AUC and auc[1] > 0.5:
        return "specific vs mixed_OD"
    return "inconclusive"


def s2_table(result: dict, case, specs):
    spec = next(s for s in specs if s["name"] == "S2")
    blob = result["meta"]["views"]["S2"]
    with tempfile.TemporaryDirectory() as tmp:
        store.get_blob(blob, Path(tmp) / "view")
        view, problems = load_view(Path(tmp) / "view", kind="bulk", vocab=case.vocab)
        if view is None:
            raise ValueError(f"S2's stored view is invalid: {problems}")
        return site_table(view, spec["data_path"], kind="bulk")


def cohort_auc(claim_like, name: str, contrast, refs):
    values, group = refs[name]
    return ev.signature_auc(values, group, claim_like[0], claim_like[1], contrast)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="tb-min")
    args = ap.parse_args(argv)
    case, specs = tb.load_case()
    refs = {name: ev.load(name) for name in ev.CONFIRMATORY}
    primary, contrast = ev.PRIMARY
    results = [json.loads(p.read_text()) for p in sorted((RUNS / args.task).glob("*__s*.json"))]
    rows = {}
    for r in results:
        name = f"{r['arm']}|s{r['sample']}"
        found = findings(r)
        up, down = ev.final_signature(found)
        signature = {f"{n}:{t}>{ref}": cohort_auc((up, down), n, (t, ref), refs)
                     for n, contrasts in ev.CONFIRMATORY.items() for t, ref in contrasts}
        replicated = []
        for c in found:
            holding = [(n, c.contrast) for n, cs in ev.CONFIRMATORY.items() if c.contrast in cs]
            aucs = [cohort_auc((c.up_genes, c.down_genes), n, k, refs) for n, k in holding]
            scored = [a for a in aucs if a is not None]
            if scored:
                replicated.append(sum(a[1] > 0.5 for a in scored) >= 2 / 3 * len(scored))
        rows[name] = {"arm": r["arm"], "sample": r["sample"], "n_findings": len(found),
                      "signature_genes": {"up": up, "down": down},
                      "primary_auc": signature.get(f"{primary}:{contrast[0]}>{contrast[1]}"),
                      "signature_auc": signature,
                      "replication_rate": (sum(replicated) / len(replicated) if replicated
                                           else None)}
        if r["arm"] == "ours" and r.get("scopes"):
            table = s2_table(r, case, specs)
            diffs = []
            for c in found:
                ours = r["scopes"].get(c.id, "inconclusive")
                alone = s2_alone_scope(c, table, case)
                if ours != alone:
                    auc = cohort_auc((c.up_genes, c.down_genes), primary, contrast, refs)
                    diffs.append({"claim": c.id, "ours": ours, "s2_alone": alone, "auc": auc,
                                  "ours_correct": ev.scope_correct(ours, auc),
                                  "s2_correct": ev.scope_correct(alone, auc)})
            judged = [d for d in diffs if d["ours_correct"] is not None]
            rows[name]["scope_differences"] = diffs
            rows[name]["scope_correct_ours"] = (sum(d["ours_correct"] for d in judged) / len(judged)
                                                if judged else None)
    OUT.write_text(json.dumps({"task": args.task, "rows": rows}, indent=1, default=str))
    for name, row in sorted(rows.items()):
        auc = row["primary_auc"]
        print(f"{name:26s} findings {row['n_findings']:3d}  GSE73408 TB vs PNA AUC "
              f"{'-' if auc is None else f'{auc[0]:.2f} [{auc[1]:.2f}, {auc[2]:.2f}]'}  "
              f"replication {_f(row['replication_rate'])}"
              + (f"  scope diffs {len(row['scope_differences'])}, ours correct "
                 f"{_f(row['scope_correct_ours'])}" if "scope_differences" in row else ""),
              flush=True)


def _f(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


if __name__ == "__main__":
    main()
