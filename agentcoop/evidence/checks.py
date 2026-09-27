"""Local scientific checks computed from the task's own data — no reference answer involved.

Each check can *exclude* some errors; none can certify a conclusion. They are the partial
feedback the optimizer is allowed to use:

* ``claim_support`` — for a claim (cell type, direction, genes): per-donor pseudobulk log fold
  changes of the cited genes. A claim is ``supported`` when the program-level change has the
  claimed sign in nearly every donor, ``contradicted`` when it has the opposite sign in nearly
  every donor, and ``unassessed`` otherwise or when it cites too few measurable genes. Donors are
  the replicate unit because the question asks for effects reproducible across donors.
* ``uncovered_signal`` — genes whose change is consistent across donors but that no claim
  covers, grouped by cell type and direction, with GO enrichment (not Hallmark, the library the
  benchmark planted from) as a hint of what the signal might be.

Status values follow the rule that *not checked is not "no problem"*: ``unassessed`` is its own
state and is never counted as support.
"""

from __future__ import annotations

import functools
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GO_BP = ROOT / "data" / "genesets" / "GO_Biological_Process_2023.txt"

def min_agree(n_donors: int) -> int:
    """Donors that must agree in sign: 7 of 8 (chance ~3.5%), 4 of 4 (chance 6.25%)."""
    return math.ceil(0.875 * n_donors)
MIN_GENES = 3
#: An uncovered group is reported only if it is a coherent program (GO enrichment).
MAX_GO_P = 1e-4


@dataclass
class DonorTable:
    genes: list[str]
    donors: list[str]
    lfc: np.ndarray  # genes × donors, log2((cpm_treated+1)/(cpm_control+1))
    mean_cpm: np.ndarray  # genes

    def index(self) -> dict[str, int]:
        return {g: i for i, g in enumerate(self.genes)}


@functools.lru_cache(maxsize=8)
def donor_tables(data_path: str) -> dict[str, DonorTable]:
    import anndata as ad

    adata = ad.read_h5ad(data_path)
    genes = list(adata.var_names)
    out = {}
    for ct in sorted(adata.obs["cell_type"].unique()):
        sub = adata[adata.obs["cell_type"] == ct]
        donors = sorted(sub.obs["donor"].unique())
        cols, cpms = [], []
        for d in donors:
            vals = {}
            for cond in ("treated", "control"):
                m = ((sub.obs["donor"] == d) & (sub.obs["condition"] == cond)).to_numpy()
                counts = np.asarray(sub.X[m].sum(0)).ravel().astype(float)
                vals[cond] = counts / max(counts.sum(), 1.0) * 1e6
            cols.append(np.log2((vals["treated"] + 1) / (vals["control"] + 1)))
            cpms.append((vals["treated"] + vals["control"]) / 2)
        out[ct] = DonorTable(genes=genes, donors=[str(d) for d in donors],
                             lfc=np.vstack(cols).T, mean_cpm=np.mean(cpms, axis=0))
    return out


def _binom_sf(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 0.5)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


@dataclass
class ClaimCheck:
    claim_index: int
    cell_type: str
    direction: str
    program: str
    status: str  # supported | contradicted | unassessed | ungrounded
    n_genes_cited: int = 0
    n_genes_measured: int = 0
    donors_agree: int = 0
    donors_total: int = 0
    gene_fraction_agree: float = 0.0
    sign_test_p: float = 1.0
    detail: str = ""


def check_claim(i: int, claim: dict[str, Any], tables: dict[str, DonorTable],
                cell_types: list[str]) -> ClaimCheck:
    from agentcoop.eval_free import canon_cell_type, canon_direction  # local, truth-free

    ct = canon_cell_type(str(claim.get("cell_type", "")), cell_types)
    direction = canon_direction(claim.get("direction", ""))
    program = str(claim.get("program", ""))
    genes = [str(g).upper() for g in (claim.get("genes") or [])]
    base = ClaimCheck(i, ct or str(claim.get("cell_type", "")), direction, program, "ungrounded",
                      n_genes_cited=len(genes))
    if ct is None or direction not in ("up", "down"):
        base.detail = "cell type or direction not recognizable"
        return base
    table = tables[ct]
    idx = table.index()
    rows = [idx[g] for g in genes if g in idx and table.mean_cpm[idx[g]] >= 20]
    base.n_genes_measured = len(rows)
    if len(rows) < MIN_GENES:
        base.status = "unassessed"
        base.detail = f"only {len(rows)} cited genes are measurable in {ct}"
        return base
    sign = 1 if direction == "up" else -1
    program_lfc = table.lfc[rows].mean(axis=0)  # per donor
    agree = int(np.sum(np.sign(program_lfc) == sign))
    oppose = int(np.sum(np.sign(program_lfc) == -sign))
    n = len(table.donors)
    gene_median = np.median(table.lfc[rows], axis=1)
    base.donors_agree, base.donors_total = agree, n
    base.gene_fraction_agree = float(np.mean(np.sign(gene_median) == sign))
    base.sign_test_p = _binom_sf(agree, n)
    if agree >= min_agree(n) and base.gene_fraction_agree >= 0.6:
        base.status = "supported"
    elif oppose >= min_agree(n):
        base.status = "contradicted"
    else:
        base.status = "unassessed"
    base.detail = (f"{agree}/{n} donors change in the claimed direction; "
                   f"{base.gene_fraction_agree:.0%} of {len(rows)} measured genes agree")
    return base


@functools.lru_cache(maxsize=1)
def _go_library() -> dict[str, frozenset[str]]:
    out = {}
    for line in GO_BP.read_text().strip().split("\n"):
        parts = line.split("\t")
        genes = frozenset(g for g in parts[2:] if g)
        if 10 <= len(genes) <= 300:
            out[parts[0]] = genes
    return out


def go_enrichment(genes: list[str], universe: list[str], top: int = 3) -> list[dict[str, Any]]:
    """Hypergeometric enrichment against GO BP, restricted to the tested universe."""
    from scipy.stats import hypergeom

    U = set(universe)
    query = set(genes) & U
    if len(query) < 3:
        return []
    results = []
    for term, members in _go_library().items():
        m = members & U
        k = len(query & m)
        if k < 3:
            continue
        p = float(hypergeom.sf(k - 1, len(U), len(m), len(query)))
        results.append({"term": term, "overlap": k, "p": p, "genes": sorted(query & m)[:8]})
    results.sort(key=lambda r: r["p"])
    return results[:top]


@dataclass
class Uncovered:
    cell_type: str
    direction: str
    n_genes: int
    genes: list[str]
    go_hint: list[dict[str, Any]] = field(default_factory=list)


def uncovered_signal(claims: list[dict[str, Any]], tables: dict[str, DonorTable],
                     cell_types: list[str], min_genes: int = 6) -> list[Uncovered]:
    from agentcoop.eval_free import canon_cell_type, canon_direction

    claimed: dict[tuple[str, str], set[str]] = {}
    for c in claims:
        ct = canon_cell_type(str(c.get("cell_type", "")), cell_types)
        d = canon_direction(c.get("direction", ""))
        if ct:
            claimed.setdefault((ct, d), set()).update(str(g).upper() for g in c.get("genes") or [])
    out = []
    for ct, table in tables.items():
        measurable = table.mean_cpm >= 20
        n = len(table.donors)
        med = np.median(table.lfc, axis=1)
        for direction, sign in (("up", 1), ("down", -1)):
            agree = np.sum(np.sign(table.lfc) == sign, axis=1)
            hits = np.flatnonzero(measurable & (agree >= min_agree(n)) & (sign * med >= 0.4))
            genes = [table.genes[i] for i in hits]
            covered = claimed.get((ct, direction), set())
            free = [g for g in genes if g not in covered]
            if len(free) < min_genes:
                continue
            # Single genes pass this filter by chance (a few per cell type and direction even
            # with no effect at all); a *coherent* program does not: require GO enrichment.
            universe = [table.genes[i] for i in np.flatnonzero(measurable)]
            hint = [h for h in go_enrichment(free, universe) if h["p"] <= MAX_GO_P
                    and h["overlap"] >= 4]
            if hint:
                out.append(Uncovered(ct, direction, len(free), sorted(free), hint))
    return out


@dataclass
class EvidenceProfile:
    claims: list[ClaimCheck]
    uncovered: list[Uncovered]

    def counts(self) -> dict[str, int]:
        out = {"supported": 0, "contradicted": 0, "unassessed": 0, "ungrounded": 0}
        for c in self.claims:
            out[c.status] += 1
        out["uncovered_groups"] = len(self.uncovered)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"claims": [asdict(c) for c in self.claims],
                "uncovered": [asdict(u) for u in self.uncovered], "counts": self.counts()}


def profile(report: dict[str, Any], data_path: str | Path,
            cell_types: Optional[list[str]] = None) -> EvidenceProfile:
    tables = donor_tables(str(data_path))
    cell_types = cell_types or sorted(tables)
    claims = [c for c in (report.get("claims") or []) if isinstance(c, dict)]
    checks = [check_claim(i, c, tables, cell_types) for i, c in enumerate(claims)]
    return EvidenceProfile(checks, uncovered_signal(claims, tables, cell_types))


def profile_sites(report: dict[str, Any], site_paths: dict[str, str | Path],
                  cell_types: list[str]) -> EvidenceProfile:
    """Shape 2: the same checks run at every site on its own data (only summaries leave).

    A claim is ``supported`` only if it is supported at every site, ``contradicted`` if it is
    contradicted at any, and otherwise ``unassessed`` — including when it holds at one site only
    (reported in ``detail`` as site-specific). An uncovered signal is a coherent program that
    every site's scan finds (same cell type and direction, >= 3 shared genes) but no claim covers.
    """
    tables = {s: donor_tables(str(p)) for s, p in site_paths.items()}
    claims = [c for c in (report.get("claims") or []) if isinstance(c, dict)]
    merged: list[ClaimCheck] = []
    for i, c in enumerate(claims):
        per_site = {s: check_claim(i, c, t, cell_types) for s, t in tables.items()}
        first = next(iter(per_site.values()))
        statuses = {s: chk.status for s, chk in per_site.items()}
        detail = "; ".join(f"site {s}: {chk.detail}" for s, chk in per_site.items())
        if all(v == "supported" for v in statuses.values()):
            status = "supported"
        elif any(v == "contradicted" for v in statuses.values()):
            status = "contradicted"
        elif any(v == "ungrounded" for v in statuses.values()):
            status = "ungrounded"
        else:
            status = "unassessed"
            if any(v == "supported" for v in statuses.values()):
                detail = "supported at one site only (site-specific); " + detail
        merged.append(ClaimCheck(i, first.cell_type, first.direction, first.program, status,
                                 n_genes_cited=first.n_genes_cited,
                                 n_genes_measured=min(x.n_genes_measured for x in per_site.values()),
                                 donors_agree=sum(x.donors_agree for x in per_site.values()),
                                 donors_total=sum(x.donors_total for x in per_site.values()),
                                 detail=detail))
    per_site_unc = {s: {(u.cell_type, u.direction): u for u in uncovered_signal(claims, t, cell_types)}
                    for s, t in tables.items()}
    sites = list(per_site_unc)
    uncovered = []
    for key, u in per_site_unc[sites[0]].items():
        others = [per_site_unc[s].get(key) for s in sites[1:]]
        if all(o is not None and len(set(o.genes) & set(u.genes)) >= 3 for o in others):
            shared = sorted(set(u.genes).intersection(*[set(o.genes) for o in others]))
            uncovered.append(Uncovered(u.cell_type, u.direction, len(shared), shared, u.go_hint))
    return EvidenceProfile(merged, uncovered)


def profile_task(report: dict[str, Any], task) -> EvidenceProfile:
    """Dispatch on task shape: pooled data (shape 1) or data held at separate sites (shape 2)."""
    cts = task.info.get("cell_types", [])
    if task.info.get("shape") == "multisite":
        sites = {s: task.data_path.parent / f for s, f in task.info["sites"].items()}
        return profile_sites(report, sites, cts)
    return profile(report, task.data_path, cts)
