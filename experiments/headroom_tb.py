"""Headroom gate (spec §6) for Case 2 (TB): $0, fixed tools, no agent, before any TB LLM spend.

    python experiments/headroom_tb.py

Design-time and evaluation side (it reads the gate's targets from agentcoop/eval/tb.py). Fixed
tools build each site's view: probes -> symbols through the site's GEO annot table (probes with
one symbol), native condition codes -> case groups through the committed crosswalk
cases.tb.TOOL_GROUPS. Three claim sets, scored on targets outside the federation (S3 Malawi,
S5 Bloom; TB vs OD) by the testers' bulk AUC:

  (i)   each proposing site alone: MA statistics + enrichment on its own table, kept by its own
        tester (experiments/headroom.single_site_claims); best = per target the best site;
  (ii)  MA-tools: arms.ma_genome over S1, S2, S6;
  (iii) scripted ours: MA-tools' claims labelled by protocol.label_claims (most significant
        supporting site as proposer), corroborated ones kept, then pruned by the TB scope rule
        (S2's TB-vs-OD and S4's mimic tests): "not specific vs ..." claims are dropped.

Pass rule (findings.md, written before the first run): (iii) - (ii) >= 0.05 precision on both
targets and the mean over targets of (iii) > that of (i). Writes runs/v4/tb_headroom.json.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from headroom import TableSite, single_site_claims  # noqa: E402

from agentcoop.cases import tb  # noqa: E402
from agentcoop.coop.arms import ma_genome  # noqa: E402
from agentcoop.coop.claims import Claim  # noqa: E402
from agentcoop.coop.protocol import label_claims  # noqa: E402
from agentcoop.coop.tester import TestResult, apply_bh  # noqa: E402
from agentcoop.coop.view import SiteTable, load_view, site_table, write_view  # noqa: E402
from agentcoop.eval import tb as ev  # noqa: E402

OUT = ROOT / "runs" / "v4" / "tb_headroom.json"
GAIN, TIE = 0.05, 1e-9


def symbols(annot: Path) -> dict[str, str]:
    """Probe -> symbol from a GEO annot table (header lines !, #, ^ skipped; one symbol only)."""
    lines = [line for line in annot.read_text(encoding="utf-8", errors="replace").splitlines(True)
             if not line.startswith(("!", "#", "^"))]
    table = pd.read_csv(io.StringIO("".join(lines)), sep="\t", dtype=str, keep_default_na=False)
    return {i: s for i, s in zip(table["ID"], table["Gene symbol"]) if s and "///" not in s}


def tool_table(expr_path: Path, group: pd.Series, symbol_of: dict[str, str],
               case) -> SiteTable:
    """The site table fixed tools build from an expression TSV and sample -> case group."""
    ids = pd.read_csv(expr_path, sep="\t", usecols=[0], dtype=str).iloc[:, 0]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "view"
        write_view(root, genes={f: symbol_of[f] for f in ids if f in symbol_of},
                   samples=pd.DataFrame({"sample": group.index, "donor": group.index,
                                         "group": group.to_numpy()}),
                   cells=None, spec={"built_by": "fixed tools (TB headroom gate)"})
        view, problems = load_view(root, kind="bulk", vocab=case.vocab)
        if view is None:
            raise ValueError(f"invalid tool view for {expr_path.name}: {problems}")
        return site_table(view, expr_path, kind="bulk")


def federation(case, specs) -> dict[str, TableSite]:
    sites = {}
    for s in specs:
        samples = pd.read_csv(s["files"][f"{s['name']}_samples.tsv"], sep="\t", dtype=str)
        group = samples.set_index("sample")["condition"].map(tb.TOOL_GROUPS[s["name"]])
        annot = next(p for n, p in s["files"].items() if n.endswith(".annot.txt"))
        sites[s["name"]] = TableSite(s["name"], tool_table(s["data_path"], group, symbols(annot),
                                                           case), case.tester)
    return sites


def targets(case) -> dict[str, TableSite]:
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, (expr, group, platform) in ev.target_sites().items():
            path = Path(tmp) / f"{name}.tsv"
            expr.rename_axis("ID_REF").to_csv(path, sep="\t")
            symbol_of = ev._annot(platform)
            out[name] = TableSite(name, tool_table(path, group, symbol_of, case), case.tester)
    return out


def scripted_ours(claims: list[Claim], sites: dict[str, TableSite], case) -> list[Claim]:
    """(iii): the shared label rule over every proposing site holding a claim's contrast, then
    the TB scope rule with the chosen proposer; "not specific" claims are dropped."""
    pairs = [(c, s.test(c)) for c in claims for s in sites.values()
             if s.holds(c.population, c.contrast)]
    adjusted = apply_bh([r for _, r in pairs], case.tester)
    results: dict[str, list[TestResult]] = {}
    for (c, _), r in zip(pairs, adjusted):
        results.setdefault(c.id, []).append(r)
    items = []
    for c in claims:
        supported = [r for r in results.get(c.id, []) if r.status == "supported"]
        if supported:
            proposer = min(supported, key=lambda r: r.p)
            items.append((replace(c, site=proposer.site),
                          (proposer, [r for r in results[c.id] if r is not proposer])))
    labels = label_claims([item for _, item in items], case.tester)
    kept = []
    for (c, _), label in zip(items, labels):
        if label != "corroborated":
            continue
        routed = tb.scope_tests(c, list(sites.values()))
        scope = [s.test(c, test="scope", contrast=contrast) for s, contrast in routed]
        if not tb.scope_rule(c, scope).startswith("not specific"):
            kept.append(c)
    return kept


def precision(claims: list[Claim], target: TableSite) -> tuple[float | None, int]:
    """Fraction of the claims (target group TB, >= 3 measurable genes there) whose TB-vs-OD AUC
    in the claimed direction on the target is >= 0.70."""
    scored = []
    for c in claims:
        if c.target != "TB":
            continue
        r = target.test(c, contrast=tb.OD)
        if r.effect is not None:
            scored.append(r.effect + 0.5 >= tb.SPECIFIC_AUC)
    return (float(np.mean(scored)) if scored else None), len(scored)


def main() -> None:
    case, specs = tb.load_case()
    sites = federation(case, specs)
    proposers = {n: s for n, s in sites.items() if n != "S4"}
    goals = targets(case)
    sets = {f"single:{n}": single_site_claims(s, case) for n, s in proposers.items()}
    sets["ma_tools"], _ = ma_genome(list(proposers.values()), case, "ma_tools")
    sets["scripted_ours"] = scripted_ours(sets["ma_tools"], sites, case)
    rows = {name: {t: precision(claims, goal) for t, goal in goals.items()} | {"n": len(claims)}
            for name, claims in sets.items()}
    for name, row in rows.items():
        print(f"{name:14s} n={row['n']:3d}  " + "  ".join(
            f"{t}: P={'-' if row[t][0] is None else f'{row[t][0]:.2f}'} ({row[t][1]} scored)"
            for t in goals), flush=True)
    best = {t: max((rows[n][t][0] for n in rows if n.startswith("single:")
                    and rows[n][t][0] is not None), default=None) for t in goals}

    def p(name: str, t: str) -> float | None:
        return rows[name][t][0]

    gain = {t: p("scripted_ours", t) is not None and p("ma_tools", t) is not None
            and p("scripted_ours", t) - p("ma_tools", t) >= GAIN - TIE for t in goals}
    ours_mean = np.mean([p("scripted_ours", t) or 0.0 for t in goals])
    best_mean = np.mean([best[t] or 0.0 for t in goals])
    verdict = {"gain_vs_ma": gain, "scripted_ours_mean": float(ours_mean),
               "best_single_mean": float(best_mean),
               "pass": all(gain.values()) and ours_mean - best_mean > TIE}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "best_single": best, "verdict": verdict}, indent=1))
    print(f"gate {'PASS' if verdict['pass'] else 'FAIL'}: gain vs MA {gain}; mean precision "
          f"scripted ours {ours_mean:.3f} vs best single {best_mean:.3f}", flush=True)


if __name__ == "__main__":
    main()
