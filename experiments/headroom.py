"""Headroom gate (spec §6) on a planted suite: is there anything left for cross-examination? $0.

    python experiments/headroom.py --suite fsdev [--tasks fsdev-800 ...]

Design-time and evaluation side: it reads the hidden truth and never calls a model, and its
verdict decides, before any LLM spend, whether the suite is worth running. Fixed tools (no agent)
build each site's *tool view*: feature ids (symbol, Ensembl or alias) -> symbols through the
task's symbol table, condition
codes and cell labels -> case vocabulary through the codebook tables committed in
agentcoop/cases/planted.py (GROUP_TEXT, NATIVE_LABELS). The tool views are checked against the
truth (eval.planted.view_errors must be 0: the gate measures headroom, not tool failures) and the
testers' site tables are computed from them. Three analyses are graded with eval.planted.grade:

  (i)   best single site: each proposing site alone runs MA-genome's statistics and enrichment on
        its own table (arms.ma_genome(..., min_sites=1)) and keeps the claims its own tester
        supports; best = per task, the maximum over sites on each endpoint;
  (ii)  MA-tools: arms.ma_genome over every site's tool view (the ma_genome arm's code);
  (iii) scripted ours: MA-tools' claims tested at every site holding them (one BH family), the
        artefact site's `contradicted` results set aside using the truth (the correct
        resolution, applied by script), kept when the shared label rule, with the most
        significant supporting site as proposer, says `corroborated`. A ceiling for ours:
        passing is necessary, not sufficient.

Pass rule (fixed before computing; copy it into findings.md before the first run):
    PASS iff (iii) - (ii) >= 0.05 in precision or in recall on at least half of the tasks,
    and the mean F1 of (iii) over the tasks exceeds the mean F1 of (i).
A difference within 1e-9 of the 0.05 margin reaches it (float ties); an undefined precision (no
claims) shows no gain. Also recorded per task: the fraction of MA-tools' true hits that some
single-site claim matches (if it is low, ours is capped below MA on recall and the suite can
support precision claims only). The view-error rate of *agent-built* views, the other §6
quantity, needs agent runs and is reported by experiments/analyze.py. Writes
runs/v4/<suite>_headroom.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentcoop.cases.planted import (  # noqa: E402
    GROUP_TEXT, NATIVE_LABELS, POPULATIONS, TASKS, load_case,
)
from agentcoop.coop.arms import ma_genome  # noqa: E402
from agentcoop.coop.claims import Claim, same_claim  # noqa: E402
from agentcoop.coop.protocol import label_claim  # noqa: E402
from agentcoop.coop.site import CaseSpec  # noqa: E402
from agentcoop.coop.tester import (  # noqa: E402
    TesterParams, TestResult, apply_bh, export_gene_stats, test_claim,
)
from agentcoop.coop.view import SiteTable, load_view, site_table, write_view  # noqa: E402
from agentcoop.eval.planted import grade, load_truth, view_errors  # noqa: E402

OUT = ROOT / "runs" / "v4"
#: Resource name of the task's symbol table (columns ensembl_id, symbol, alias).
SYMBOLS = "symbols.tsv"
#: Site-file obs columns by role (the planted layout of cases/planted.plant_task). Planted sites
#: hold one sample per donor, so the sample id is the donor id.
SAMPLE, CONDITION, LABEL = "sample", "condition", "cell_label"
GAIN = 0.05
TIE = 1e-9
ENDPOINTS = ("precision", "recall", "f1")


def tool_codebook(card: dict) -> tuple[dict[str, str], dict[str, str]]:
    """(condition code -> case group, cell label -> case population) from the committed tables: a
    code's card description is GROUP_TEXT's, a label is a NATIVE_LABELS name or a case
    population. A code or label outside them raises: fixed tools must cover the whole suite."""
    group_of_text = {text: group for group, text in GROUP_TEXT.items()}
    population_of = {p: p for p in POPULATIONS} | {
        name: p for p, names in NATIVE_LABELS.items() for name in names}
    return ({code: group_of_text[text] for code, text in card["codes"].items()},
            {label: population_of[label] for label in card["cell_labels"]})


def tool_view(root: Path, data_path: Path, card: dict, symbols: pd.DataFrame) -> None:
    """Write the view fixed tools build for one site. Symbols, Ensembl ids and aliases all
    resolve through the symbol table (Task 8 keeps symbol <-> alias a bijection and no alias is
    another gene's symbol), so the site's gene id scheme needs no detection."""
    import anndata as ad    # heavy; only scRNA cases get here

    adata = ad.read_h5ad(data_path, backed="r")
    obs, features = adata.obs.copy(), [str(v) for v in adata.var_names]
    adata.file.close()
    symbol_of = (dict(zip(symbols["symbol"], symbols["symbol"]))
                 | dict(zip(symbols["alias"], symbols["symbol"]))
                 | dict(zip(symbols["ensembl_id"], symbols["symbol"])))
    groups, populations = tool_codebook(card)
    sample = obs[SAMPLE].astype(str).to_numpy()
    samples = (pd.DataFrame({"sample": sample, "donor": sample,
                             "group": obs[CONDITION].astype(str).map(groups).to_numpy()})
               .drop_duplicates().sort_values("sample", ignore_index=True))
    cells = pd.DataFrame({"cell": obs.index.astype(str), "sample": sample,
                          "population": obs[LABEL].astype(str).map(populations).to_numpy()})
    write_view(root, genes={f: symbol_of[f] for f in features if f in symbol_of},
               samples=samples, cells=cells,
               spec={"built_by": "fixed tools (headroom gate)", "gene_mapping": SYMBOLS,
                     "labels": "committed codebook (GROUP_TEXT, NATIVE_LABELS)",
                     "unit": "donor", "normalization": "CPM"})


@dataclass
class TableSite:
    """A site seen through fixed tools: the Site methods MA-genome and cross-testing call
    (holds / export_stats / test, as coop.site.Site answers them), from a tool-view table."""

    name: str
    table: SiteTable
    params: TesterParams

    def holds(self, population: str, contrast: tuple[str, str]) -> bool:
        units = self.table.units[self.table.units["population"] == population]
        return all((units["group"] == g).any() for g in contrast)

    def export_stats(self, population: str, contrast: tuple[str, str]) -> pd.DataFrame:
        return export_gene_stats(self.table, population, contrast, self.params)

    def test(self, claim: Claim, *, test: str = "replication",
             contrast: tuple[str, str] | None = None) -> TestResult:
        return test_claim(self.table, claim, self.params, site=self.name, test=test,
                          contrast=contrast)


def tool_sites(case: CaseSpec, site_specs: list[dict], task_dir: Path) -> dict[str, TableSite]:
    """Tool view and site table of every site; the views live in a temporary directory (the
    tables are all the gate needs). A view the truth calls wrong raises."""
    symbols = pd.read_csv(case.resources[SYMBOLS], sep="\t", dtype=str)
    sites = {}
    with tempfile.TemporaryDirectory() as tmp:
        for spec in site_specs:
            name, root, data = spec["name"], Path(tmp) / spec["name"], Path(spec["data_path"])
            tool_view(root, data, spec["card"], symbols)
            view, problems = load_view(root, kind=case.kind, vocab=case.vocab)
            if view is None:
                raise ValueError(f"{task_dir.name} site {name}: invalid tool view: {problems}")
            errors = view_errors(view, task_dir, name)
            if errors["error_rate"] > 0:
                raise ValueError(f"{task_dir.name} site {name}: tool view differs from the "
                                 f"truth: {errors}")
            sites[name] = TableSite(name, site_table(view, data, kind=case.kind), case.tester)
    return sites


def single_site_claims(site: TableSite, case: CaseSpec) -> list[Claim]:
    """(i): MA-genome's statistics and enrichment on one site's table, kept when the site's own
    tester supports the claim (the gate of the single-site arm)."""
    claims, _ = ma_genome([site], case, f"single:{site.name}", min_sites=1)
    return [c for c in claims if site.test(c).status == "supported"]


def scripted_ours(claims: list[Claim], sites: list, case: CaseSpec,
                  artefact: str | None) -> list[Claim]:
    """(iii): each claim tested at every site holding its population and contrast, one BH
    family; the artefact site's contradictions are set aside (the correct resolution); a claim
    is kept when the shared label rule, with its most significant supporting site as proposer,
    says `corroborated`."""
    pairs = [(c, s.test(c)) for c in claims for s in sites if s.holds(c.population, c.contrast)]
    results: dict[str, list[TestResult]] = defaultdict(list)
    for (c, _), r in zip(pairs, apply_bh([r for _, r in pairs], case.tester)):
        if not (r.site == artefact and r.status == "contradicted"):
            results[c.id].append(r)
    kept = []
    for c in claims:
        supported = [r for r in results[c.id] if r.status == "supported"]
        if not supported:
            continue
        proposer = min(supported, key=lambda r: r.p)
        others = [r for r in results[c.id] if r is not proposer]
        if label_claim(proposer, others, case.tester) == "corroborated":
            kept.append(c)
    return kept


def union_coverage(ma_claims: list[Claim], single: list[Claim], truth: dict) -> float | None:
    """Fraction of MA-tools' true hits that some single-site claim matches (None: no hits)."""
    hits = [c for c in ma_claims if grade([c], truth)["precision"] == 1.0]
    if not hits:
        return None
    return sum(any(same_claim(c, s) for s in single) for c in hits) / len(hits)


def evaluate_task(task_dir: Path) -> dict:
    case, site_specs = load_case(task_dir)
    truth = load_truth(task_dir)
    sites = tool_sites(case, site_specs, task_dir)
    single = {s["name"]: single_site_claims(sites[s["name"]], case)
              for s in site_specs if s["proposes"]}
    ma_tools, _ = ma_genome(list(sites.values()), case, "ma_tools")
    ours = scripted_ours(ma_tools, list(sites.values()), case, truth["artefact_site"])
    graded = {name: grade(claims, truth) for name, claims in single.items()}
    return {
        "knobs": truth["knobs"],
        "single": graded,
        "best_single": {m: max((g[m] for g in graded.values() if g[m] is not None), default=None)
                        for m in ENDPOINTS},
        "ma_tools": grade(ma_tools, truth),
        "scripted_ours": grade(ours, truth),
        "n_claims": {"ma_tools": len(ma_tools), "scripted_ours": len(ours),
                     **{f"single:{n}": len(c) for n, c in single.items()}},
        "union_covers_ma_hits": union_coverage(
            ma_tools, [c for claims in single.values() for c in claims], truth),
    }


def _gain(ours: float | None, ma: float | None) -> bool:
    return ours is not None and ma is not None and ours - ma >= GAIN - TIE


def verdict(rows: dict[str, dict]) -> dict:
    """The pass rule of the module docstring."""
    gain = sorted(t for t, r in rows.items()
                  if any(_gain(r["scripted_ours"][m], r["ma_tools"][m])
                         for m in ("precision", "recall")))
    ours = float(np.mean([r["scripted_ours"]["f1"] for r in rows.values()]))
    single = float(np.mean([r["best_single"]["f1"] for r in rows.values()]))
    return {"tasks_with_gain": gain, "n_tasks": len(rows), "scripted_ours_f1": ours,
            "best_single_f1": single, "pass": 2 * len(gain) >= len(rows) and ours - single > TIE}


def _f(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tasks", nargs="*", help="default: every task of the suite")
    args = ap.parse_args(argv)
    rows = {}
    for task_dir in sorted(d for d in (TASKS / args.suite).iterdir() if d.is_dir()):
        if args.tasks and task_dir.name not in args.tasks:
            continue
        rows[task_dir.name] = r = evaluate_task(task_dir)
        print(f"{task_dir.name} {r['knobs']}: " + "  ".join(
            f"{k} P={_f(r[k]['precision'])} R={_f(r[k]['recall'])} F1={_f(r[k]['f1'])}"
            for k in ("best_single", "ma_tools", "scripted_ours"))
            + f"  union covers MA hits: {_f(r['union_covers_ma_hits'])}", flush=True)
    if not rows:
        ap.error(f"no tasks in {TASKS / args.suite}")
    v = verdict(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.suite}_headroom.json").write_text(
        json.dumps({"suite": args.suite, "tasks": rows, "verdict": v}, indent=1))
    print(f"gate {'PASS' if v['pass'] else 'FAIL'}: (iii)-(ii) >= {GAIN} in "
          f"{len(v['tasks_with_gain'])}/{v['n_tasks']} tasks; mean F1 scripted ours "
          f"{v['scripted_ours_f1']:.3f} vs best single {v['best_single_f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
