"""Answer key and grader of the planted suite (spec §7). Only the evaluation harness imports this.

Suites are built here because this side owns the answer path (spec §8): the generator in
`agentcoop.cases.planted` hands each task's planting record to `write_truth`.

A DE claim matches a planted program when its population is one of the program's cell types,
its contrast is (treated, control) and its signed genes overlap the program's with Jaccard >= 0.2
(frozen); a composition claim matches the composition shift by population and sign. A claim that
matches several planted items counts once, for its best match.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import pandas as pd

from agentcoop.cases.planted import build_suite
from agentcoop.coop.claims import Claim
from agentcoop.coop.view import SiteView

MIN_JACCARD = 0.2
#: The 2 x 2 knob cells (artefact, ids_labels); `build` cycles through them per seed.
KNOB_CELLS = ((False, False), (True, False), (False, True), (True, True))
#: What a correct cause record says about a claim that matches a non-reproducible program.
EXPECTED_CAUSE = {"artefact": "site-artefact", "site_specific": "site-biology"}


def write_truth(task_dir: Path, truth: dict) -> None:
    hidden = Path(task_dir) / "hidden"
    hidden.mkdir()
    (hidden / "truth.json").write_text(json.dumps(truth, indent=2))


def build(name: str, seeds: list[int], *, pool: str) -> list[Path]:
    """A suite balanced over the knob cells: seed i gets KNOB_CELLS[i % 4]."""
    knobs = [KNOB_CELLS[i % len(KNOB_CELLS)] for i in range(len(seeds))]
    return build_suite(name, seeds, pool=pool, knobs=knobs, seal=write_truth)


def load_truth(task_dir: Path) -> dict:
    return json.loads((Path(task_dir) / "hidden" / "truth.json").read_text())


def _signed(up: list[str] | tuple[str, ...], down: list[str] | tuple[str, ...]) -> frozenset:
    return frozenset({(g.upper(), 1) for g in up} | {(g.upper(), -1) for g in down})


def match(claim: Claim, truth: dict) -> dict | None:
    """The planted record (program or composition shift) a claim best matches, or None."""
    if claim.contrast != ("treated", "control"):
        return None
    if claim.kind == "composition":
        shift = truth["composition"]
        sign = 0 if claim.effect is None else (claim.effect > 0) - (claim.effect < 0)
        return shift if claim.population == shift["population"] and sign == shift["sign"] else None
    genes = _signed(claim.up_genes, claim.down_genes)
    best, best_j = None, 0.0
    for prog in truth["programs"]:
        if claim.population in prog["cell_types"]:
            planted = _signed(prog["up_genes"], prog["down_genes"])
            j = len(genes & planted) / len(genes | planted)
            if j >= MIN_JACCARD and j > best_j:
                best, best_j = prog, j
    return best


def items(truth: dict) -> set[tuple[str, str]]:
    """What recall is over: (program id, cell type) of every reproducible program, plus the
    composition shift."""
    out = {(p["id"], ct) for p in truth["programs"] if p["kind"] == "reproducible"
           for ct in p["cell_types"]}
    out.add((truth["composition"]["id"], truth["composition"]["population"]))
    return out


def _mean(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def grade(claims: list[Claim], truth: dict, *, labels: dict[str, str] | None = None,
          causes: dict[str, dict] | None = None, scopes: dict[str, str] | None = None) -> dict:
    """Grade one arm's claims on one task.

    `claims` are the claims the arm reports. With `labels`, its findings are the claims labelled
    `corroborated` (an unlabelled claim counts as a finding); without, every claim is a finding.
    Precision, recall, F1, artefact rejection and scope accuracy are over findings. Cause accuracy
    is over every claim with a recorded cause (causes mostly explain demoted claims); a cause is
    correct when both its category and its site match the planted record.
    """
    labels, causes, scopes = labels or {}, causes or {}, scopes or {}
    best = {c.id: match(c, truth) for c in claims}
    findings = [c for c in claims if labels.get(c.id, "corroborated") == "corroborated"]
    wanted = items(truth)
    item_of = {c.id: (best[c.id]["id"], c.population) for c in claims if best[c.id] is not None}
    hits = [c for c in findings if item_of.get(c.id) in wanted]
    precision = len(hits) / len(findings) if findings else None
    recall = len({item_of[c.id] for c in hits}) / len(wanted)
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    artefact = next((p["id"] for p in truth["programs"] if p["kind"] == "artefact"), None)
    rejected = (None if artefact is None
                else not any(best[c.id] is not None and best[c.id]["id"] == artefact
                             for c in findings))
    scoped = [(scopes[c.id] == "shared") == best[c.id]["shared_with_mimic"] for c in findings
              if best[c.id] is not None and best[c.id]["kind"] == "reproducible"
              and scopes.get(c.id) in ("shared", "specific")]
    caused = [causes[c.id].get("cause") == EXPECTED_CAUSE[best[c.id]["kind"]]
              and causes[c.id].get("site") == best[c.id]["sites"][0] for c in claims
              if causes.get(c.id) and best[c.id] is not None
              and best[c.id]["kind"] in EXPECTED_CAUSE]
    return {"precision": precision, "recall": recall, "f1": f1, "n_claims": len(findings),
            "artefact_rejected": rejected, "scope_accuracy": _mean(scoped),
            "cause_accuracy": _mean(caused)}


#: Components of a site view the view-error rate is measured on (spec §6).
VIEW_PARTS = ("genes", "groups", "populations")


def view_errors(view: SiteView, task_dir: Path, site: str) -> dict:
    """How far a site view is from the site's correct codebook (spec §6: the view-error rate).

    Per component, the fraction of the site's items the view maps wrongly (``<part>_wrong``) and
    the fraction it leaves out (``<part>_missing``): genes over the site file's feature ids (the
    correct symbol comes from the task's symbol table under the site's scheme: symbol, Ensembl or
    alias; symbols compare case-insensitively), groups over the site's samples,
    populations over its cells. Planted site files hold no cell a correct analysis would exclude,
    so a left-out item is an error too. ``error_rate`` is the worst component's wrong + missing.
    """
    task_dir = Path(task_dir)
    truth = load_truth(task_dir)["sites"][site]
    adata = ad.read_h5ad(task_dir / "sites" / f"{site}.h5ad", backed="r")
    obs, features = adata.obs.copy(), [str(v) for v in adata.var_names]
    adata.file.close()
    # Task 8's gene-ID schemes: the site's var_names are the symbol table's symbol, ensembl_id or
    # alias column (a bijection with symbol), so the correct symbol is one lookup away.
    table = pd.read_csv(task_dir / "public" / "symbols.tsv", sep="\t", dtype=str)
    column = {"symbol": "symbol", "ensembl": "ensembl_id", "alias": "alias"}[truth["scheme"]]
    symbol_of = dict(zip(table[column], table["symbol"].str.upper()))
    group_of = {donor: group for group, donors in truth["donors"].items() for donor in donors}
    correct = {
        "genes": {f: symbol_of[f] for f in features},
        "groups": {sample: group_of[donor] for sample, donor in truth["samples"].items()},
        "populations": dict(zip(obs.index.astype(str),
                                obs["cell_label"].astype(str).map(truth["labels"]))),
    }
    seen = {
        "genes": {f: str(s).upper() for f, s in view.genes.items()},   # claims upper-case too
        "groups": dict(zip(view.samples["sample"], view.samples["group"])),
        "populations": ({} if view.cells is None
                        else dict(zip(view.cells["cell"], view.cells["population"]))),
    }
    out = {}
    for part in VIEW_PARTS:
        items, mapped = correct[part], seen[part]
        out[f"{part}_wrong"] = sum(k in mapped and mapped[k] != v
                                   for k, v in items.items()) / len(items)
        out[f"{part}_missing"] = sum(k not in mapped for k in items) / len(items)
    out["error_rate"] = max(out[f"{p}_wrong"] + out[f"{p}_missing"] for p in VIEW_PARTS)
    return out


if __name__ == "__main__":  # pragma: no cover
    import sys

    # python -m agentcoop.eval.planted fsdev dev 800 808   (suite, donor pool, seeds [lo, hi))
    suite, pool, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    for path in build(suite, list(range(lo, hi)), pool=pool):
        print(path, flush=True)
