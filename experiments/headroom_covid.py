"""Headroom gate (spec §6) for Case 1 (COVID): $0, fixed tools, no agent, before any COVID spend.

    python experiments/headroom_covid.py

Fixed tools build each site's view from its site file: symbols from var["feature_name"],
severity codes -> case groups through cases.covid.TOOL_GROUPS, cell populations from Tian's
harmonized Cell.group in the raw files (cases.covid.TOOL_POPULATIONS), matched to the site's
cells by their tokens. Folds hold out Stanford, Emory and Seoul in turn; the federation is Berlin
plus the other two. Per fold: (i) each federation site alone, best = the best site; (ii)
MA-tools over the federation as built; (iii) scripted ours = MA-tools' claims on the resolved
views (Berlin's 10x v2 controls excluded: the correct resolution, applied by script), labelled
by protocol.label_claims, corroborated kept. A claim replicates when the held-out site's tester
supports it (BH over the arm's claims there).

Pass rule (findings.md, written before the first run): (iii) - (ii) >= 0.05 precision in >= 2
of 3 folds and mean precision (iii) > mean (i). Writes runs/v4/covid_headroom.json.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from headroom import TableSite, single_site_claims  # noqa: E402

from agentcoop.cases import covid  # noqa: E402
from agentcoop.coop.arms import ma_genome  # noqa: E402
from agentcoop.coop.claims import Claim  # noqa: E402
from agentcoop.coop.protocol import label_claims  # noqa: E402
from agentcoop.coop.tester import TestResult, apply_bh  # noqa: E402
from agentcoop.coop.view import EXCLUDE, OTHER, load_view, site_table, write_view  # noqa: E402

OUT = ROOT / "runs" / "v4" / "covid_headroom.json"
HELD_OUT = ("Stanford", "Emory", "Seoul")
GAIN, TIE = 0.05, 1e-9


def tool_site(spec: dict, case, *, resolved: bool) -> TableSite:
    """A site seen through fixed tools; ``resolved`` excludes Berlin's 10x v2 controls."""
    name, data = spec["name"], Path(spec["data_path"])
    site = next(s for s in covid.SITES if s.name == name)
    raw = ad.read_h5ad(covid.RAW / site.file, backed="r")
    population_of = {covid._token("cell", name, str(c), width=16): covid.TOOL_POPULATIONS.get(str(g), OTHER)
                     for c, g in zip(raw.obs_names, raw.obs["Cell.group"])}
    raw.file.close()
    a = ad.read_h5ad(data, backed="r")
    obs, genes = a.obs.astype(str), dict(zip(a.var_names.astype(str), a.var["feature_name"].astype(str)))
    a.file.close()
    samples = obs.groupby("sample").first().reset_index()
    group = samples["condition"].map(covid.TOOL_GROUPS[name])
    if resolved and name == "Berlin":
        group = group.where(~((group == "healthy") & (samples["assay"] == "10x 3' v2")), EXCLUDE)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "view"
        write_view(root, genes=genes,
                   samples=pd.DataFrame({"sample": samples["sample"], "donor": samples["donor"],
                                         "group": group}),
                   cells=pd.DataFrame({"cell": obs.index, "sample": obs["sample"].to_numpy(),
                                       "population": [population_of[c] for c in obs.index]}),
                   spec={"built_by": "fixed tools (COVID headroom gate)", "resolved": resolved})
        view, problems = load_view(root, kind="scrna", vocab=case.vocab)
        if view is None:
            raise ValueError(f"invalid tool view for {name}: {problems}")
        return TableSite(name, site_table(view, data, kind="scrna"), case.tester)


def scripted_ours(claims: list[Claim], sites: list[TableSite], case) -> list[Claim]:
    pairs = [(c, s.test(c)) for c in claims for s in sites if s.holds(c.population, c.contrast)]
    results: dict[str, list[TestResult]] = {}
    for (c, _), r in zip(pairs, apply_bh([r for _, r in pairs], case.tester)):
        results.setdefault(c.id, []).append(r)
    items = []
    for c in claims:
        supported = [r for r in results.get(c.id, []) if r.status == "supported"]
        if supported:
            proposer = min(supported, key=lambda r: r.p)
            items.append((replace(c, site=proposer.site),
                          (proposer, [r for r in results[c.id] if r is not proposer])))
    labels = label_claims([item for _, item in items], case.tester)
    return [c for (c, _), label in zip(items, labels) if label == "corroborated"]


def precision(claims: list[Claim], held_out: TableSite, case) -> tuple[float | None, int]:
    tested = [held_out.test(c) for c in claims if held_out.holds(c.population, c.contrast)]
    tested = [r for r in apply_bh(tested, case.tester) if r.status != "untestable"]
    if not tested:
        return None, 0
    return float(np.mean([r.status == "supported" for r in tested])), len(tested)


def _f(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def main() -> None:
    case, specs = covid.load_case()
    by_name = {s["name"]: s for s in specs}
    built = {n: tool_site(by_name[n], case, resolved=False) for n in by_name}
    resolved_berlin = tool_site(by_name["Berlin"], case, resolved=True)
    folds = {}
    for out in HELD_OUT:
        fed = [built[n] for n in by_name if n != out]
        fed_resolved = [resolved_berlin if s.name == "Berlin" else s for s in fed]
        sets = {f"single:{s.name}": single_site_claims(s, case) for s in fed}
        sets["ma_tools"], _ = ma_genome(fed, case, "ma_tools")
        candidates, _ = ma_genome(fed_resolved, case, "ma_tools_resolved")
        sets["scripted_ours"] = scripted_ours(candidates, fed_resolved, case)
        rows = {k: {"precision": precision(v, built[out], case), "n": len(v)}
                for k, v in sets.items()}
        best = max((r["precision"][0] for k, r in rows.items() if k.startswith("single:")
                    and r["precision"][0] is not None), default=None)
        folds[out] = {"rows": rows, "best_single": best}
        print(f"held out {out}: " + "  ".join(
            f"{k} P={_f(r['precision'][0])} ({r['precision'][1]}/{r['n']})"
            for k, r in rows.items()), flush=True)

    def p(fold: str, arm: str) -> float | None:
        return folds[fold]["rows"][arm]["precision"][0]

    gains = [f for f in folds if p(f, "scripted_ours") is not None and p(f, "ma_tools") is not None
             and p(f, "scripted_ours") - p(f, "ma_tools") >= GAIN - TIE]
    ours = float(np.mean([p(f, "scripted_ours") or 0.0 for f in folds]))
    best = float(np.mean([folds[f]["best_single"] or 0.0 for f in folds]))
    verdict = {"folds_with_gain": gains, "scripted_ours_mean": ours, "best_single_mean": best,
               "pass": len(gains) >= 2 and ours - best > TIE}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"folds": folds, "verdict": verdict}, indent=1, default=str))
    print(f"gate {'PASS' if verdict['pass'] else 'FAIL'}: gain vs MA in {gains}; mean precision "
          f"scripted ours {ours:.3f} vs best single {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
