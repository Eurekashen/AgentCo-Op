"""Grade Case 1 (COVID) runs against the held-out cohorts (spec §5.1). Evaluation side; $0.

    python experiments/analyze_covid.py [--task covid-4]

Per arm and draw, the findings (claims labelled `corroborated`; every claim of an unlabelled arm)
in the arm's rank order. Primary:
  (a) prior-adjusted replication - each finding loses the genes any prior-only draw named for the
      same population, contrast and direction; residual de lists with >= 3 genes (and every
      composition claim the prior-only arm did not make) are scored on Stephenson
      (eval.covid.replicate, BH over the arm's scored claims; replicated = same sign and
      FDR < 0.05); residual precision@k and replicated residual claims at k = 5, 10, 20;
  (b) artefact rate among the top m findings, m = the smallest finding count over the arms
      (matched claim count), with eval.covid.is_artefact.
Secondary: plain (unadjusted) precision@k on Stephenson; COMBAT same-sign rate of the
Stephenson-replicated findings. Writes runs/v4/covid_analysis.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentcoop.coop.claims import Claim  # noqa: E402
from agentcoop.eval import covid as ev  # noqa: E402

RUNS = ROOT / "runs" / "v4" / "covid"
OUT = ROOT / "runs" / "v4" / "covid_analysis.json"
KS = (5, 10, 20)
Q = 0.05


def claim_of(d: dict) -> Claim:
    fields = Claim.__dataclass_fields__
    return Claim(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()
                    if k in fields})


def findings(result: dict) -> list[Claim]:
    labels = result.get("labels") or {}
    claims = [claim_of(c) for c in result["claims"]]
    kept = [c for c in claims if labels.get(c.id, "corroborated") == "corroborated"]
    return sorted(kept, key=lambda c: (c.rank, c.id))


def prior_genes(results: list[dict]) -> set[tuple]:
    """(population, target, reference, direction, gene) named by any prior-only draw, plus
    (population, target, reference, "composition") for its composition claims."""
    named = set()
    for r in results:
        for c in (claim_of(d) for d in r["claims"]):
            if c.kind == "composition":
                named.add((c.population, c.target, c.reference, "composition"))
            named |= {(c.population, c.target, c.reference, "up", g) for g in c.up_genes}
            named |= {(c.population, c.target, c.reference, "down", g) for g in c.down_genes}
    return named


def residual(c: Claim, named: set[tuple]) -> Claim | None:
    if c.kind == "composition":
        return None if (c.population, c.target, c.reference, "composition") in named else c
    key = (c.population, c.target, c.reference)
    up = tuple(g for g in c.up_genes if (*key, "up", g) not in named)
    down = tuple(g for g in c.down_genes if (*key, "down", g) not in named)
    if len(up) + len(down) < 3 or (c.up_genes and c.down_genes and min(len(up), len(down)) < 3):
        return None
    return Claim(**{**c.__dict__, "up_genes": up, "down_genes": down})


def replicated(claims: list[Claim], pb) -> list[bool]:
    """Stephenson replication of each claim (BH over the list); untestable = not replicated."""
    tests = [ev.replicate(c, pb) for c in claims]
    p = [t["p"] for t in tests if t is not None]
    q = iter(false_discovery_control(p) if p else [])
    return [t is not None and bool(next(q) < Q) and t["same_sign"] for t in tests]


def at_k(flags: list[bool]) -> dict:
    return {k: {"precision": (float(np.mean(flags[:k])) if flags[:k] else None),
                "replicated": int(sum(flags[:k]))} for k in KS}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="covid-4")
    args = ap.parse_args(argv)
    task_dir = RUNS / args.task
    results = [json.loads(p.read_text()) for p in sorted(task_dir.glob("*__s*.json"))]
    named = prior_genes([r for r in results if r["arm"] == "prior_only"])
    stephenson, combat = ev.load("stephenson"), ev.load("combat")
    signature = pd.read_csv(ev.HIDDEN / "artefact_signature.tsv", sep="\t")
    rows = {}
    for r in results:
        arm = r["arm"] if r["arm"] not in ("single", "single_codex") else r["arm"]
        name = f"{arm}|s{r['sample']}"
        found = findings(r)
        plain = replicated(found, stephenson)
        resid = [x for x in (residual(c, named) for c in found) if x is not None]
        adjusted = replicated(resid, stephenson)
        hits = [c for c, ok in zip(found, plain) if ok]
        combat_tests = [ev.replicate(c, combat) for c in hits]
        artefacts = [ev.is_artefact(c, ok, signature) for c, ok in zip(found, plain)]
        rows[name] = {"arm": r["arm"], "sample": r["sample"], "n_findings": len(found),
                      "n_residual": len(resid), "adjusted": at_k(adjusted), "plain": at_k(plain),
                      "combat_same_sign": (float(np.mean([t["same_sign"] for t in combat_tests
                                                          if t is not None]))
                                           if any(t is not None for t in combat_tests) else None),
                      "artefact_flags": artefacts}
    m = min((row["n_findings"] for row in rows.values() if row["arm"] != "prior_only"
             and row["n_findings"]), default=0)
    for row in rows.values():
        flags = row.pop("artefact_flags")
        row["artefact_rate_at_m"] = float(np.mean(flags[:m])) if m and flags[:m] else None
    OUT.write_text(json.dumps({"task": args.task, "matched_m": m, "rows": rows}, indent=1))
    print(f"matched claim count m = {m}")
    for name, row in sorted(rows.items()):
        a, p = row["adjusted"], row["plain"]
        print(f"{name:28s} findings {row['n_findings']:3d} residual {row['n_residual']:3d} | "
              + " ".join(f"adj@{k} {a[k]['replicated']}/{_f(a[k]['precision'])}" for k in KS)
              + f" | plain@10 {_f(p[10]['precision'])} | artefact@m {_f(row['artefact_rate_at_m'])}"
              + f" | COMBAT same-sign {_f(row['combat_same_sign'])}", flush=True)


def _f(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


if __name__ == "__main__":
    main()
