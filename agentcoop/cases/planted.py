"""Case 3: a controlled multi-site suite planted into real healthy PBMCs (spec §7).

Cells, donors and gene-gene structure are real (a frozen OneK1K donor subset, used by no case or
held-out cohort); only the effects are planted, by binomial thinning (Gerard 2020): to make gene
g β log2 units higher in the target group, its counts in the lower group's cells are thinned with
probability 2^-β. Thinned counts are still counts, and the planted log fold change is exact in
expectation. Condition is a donor attribute (unpaired), as in the real cases.

Each task has three proposing sites A, B, C (4 control + 4 treated donors each) and a tester-only
site M (4 control + 4 mimic donors). It plants
* 1-3 reproducible programs at A, B and C, each also planted in M's mimic donors with
  probability 0.5 (so scope has a known answer: shared with the mimic, or specific);
* 0-1 site-specific program (one proposing site: real there, not replicable);
* one reproducible composition shift (40% of one population's cells dropped in treated donors);
* knob `artefact`: a coherent program thinned in one proposing site's control donors only, with
  an `assay` column that differs between that site's control and treated donors (as in Berlin);
* knob `ids_labels`: per-site gene ID scheme (symbol / Ensembl / alias), site-native cell-type
  names and condition codes that only the site's card explains.

This module writes only what sites and the coordinator may see. Each task's planting record is
handed to a caller-supplied `seal`, so `cases/` holds no path to answers (spec §8); suites are
built from the evaluation side, which owns that path.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from agentcoop.coop.claims import Claim, Vocabulary
from agentcoop.coop.protocol import SiteHandle
from agentcoop.coop.site import CaseSpec, file_digest
from agentcoop.coop.tester import TesterParams, TestResult

ROOT = Path(__file__).resolve().parents[2]
BASE: Path = ROOT / "data" / "bench" / "onek1k" / "base.h5ad"
TASKS = ROOT / "data" / "tasks"
HALLMARK = ROOT / "data" / "genesets" / "hallmark_enrichr.txt"
GO_BP = ROOT / "data" / "genesets" / "GO_Biological_Process_2023.txt"
#: HGNC complete set, a dated (immutable) monthly release: previous and alias symbols for the
#: alias gene-ID scheme. Read once, by prepare_base; the chosen aliases are frozen into the base.
HGNC_URL = ("https://storage.googleapis.com/public-download-files/hgnc/archive/archive/monthly/"
            "tsv/hgnc_complete_set_2026-10-06.txt")
HGNC = ROOT / "data" / "bench" / "onek1k" / "hgnc_complete_set_2026-10-06.txt"

POPULATIONS = ("CD4 T", "CD8 T", "NK", "B", "CD14 Mono", "CD16 Mono")
CASE_VOCAB = Vocabulary(populations=POPULATIONS, groups=("control", "treated", "mimic"),
                        contrasts=(("treated", "control"), ("mimic", "control")))
MIMIC = ("mimic", "control")

#: OneK1K Cell Ontology labels pooled into the case populations; every other label is dropped.
CELL_TYPES: dict[str, str] = {label: population for population, labels in {
    "CD4 T": ("CD4-positive, alpha-beta T cell",
              "naive thymus-derived CD4-positive, alpha-beta T cell",
              "central memory CD4-positive, alpha-beta T cell",
              "effector memory CD4-positive, alpha-beta T cell",
              "regulatory T cell", "CD4-positive, alpha-beta cytotoxic T cell"),
    "CD8 T": ("CD8-positive, alpha-beta T cell",
              "naive thymus-derived CD8-positive, alpha-beta T cell",
              "central memory CD8-positive, alpha-beta T cell",
              "effector memory CD8-positive, alpha-beta T cell"),
    "NK": ("natural killer cell", "CD16-negative, CD56-bright natural killer cell, human"),
    "B": ("memory B cell", "naive B cell", "transitional stage B cell"),
    "CD14 Mono": ("CD14-positive monocyte",),
    "CD16 Mono": ("CD14-low, CD16-positive monocyte",),
}.items() for label in labels}

MIN_DONOR_CELLS = 1000   # spec §7: donors with >= 1,000 cells
MIN_POP_CELLS = 20       # = TesterParams.min_cells: every population stays testable in every donor
MAX_DONOR_CELLS = 1500   # a site file stays ~10^4 cells
ROW_BLOCK = 20_000       # CSR rows per h5py read (~250 MB at OneK1K density)

PROPOSERS = ("A", "B", "C")
MIMIC_SITE = "M"
PER_GROUP = 4
MIN_CORE = 8             # plantable in a population only with >= 8 expressed core genes
DROP = 0.75              # composition shift: fraction of the population's cells removed; single-
                         # site power ~0.5-0.8 on the dev pool (findings 2026-10-21; 0.4 gave <= 0.37)
#: Populations a program is planted in: median >= 50 cells per donor in the base (CD16 Mono: 34,
#: too few for the donor-pseudobulk tester at some sites).
PROGRAM_POPULATIONS = ("CD4 T", "CD8 T", "NK", "B", "CD14 Mono")
#: Populations a composition shift is planted in: minor ones (CD4 T is ~46% of cells, so removing
#: it would shift every other population's fraction) with enough cells per donor.
COMPOSITION_POPULATIONS = ("CD8 T", "NK", "B", "CD14 Mono")
ASSAY, ASSAY_ALT = "10x 3' v2", "10x 3' v3"
#: Gene ID schemes of the ids_labels knob and what a site card says about each.
SCHEMES = ("symbol", "ensembl", "alias")
FEATURES = {"symbol": "var_names are HGNC gene symbols",
            "ensembl": "var_names are Ensembl gene IDs",
            "alias": "var_names are legacy gene names: a previous or alias HGNC symbol where one "
                     "exists, else the current symbol"}

#: Plantable programs, grouped into families a biologist can recognize from a gene subset; at
#: most one program per family per task, so planted programs never describe the same biology.
FAMILIES: dict[str, dict] = {
    "interferon": {"programs": ["Interferon Alpha Response", "Interferon Gamma Response"],
                   "go_terms": ["Response To Type I Interferon", "Defense Response To Virus",
                                "Type II Interferon-Mediated Signaling Pathway",
                                "Cellular Response To Type II Interferon",
                                "Negative Regulation Of Viral Genome Replication"]},
    "proliferation": {"programs": ["E2F Targets", "G2-M Checkpoint", "Mitotic Spindle"],
                      "go_terms": ["DNA Replication", "Mitotic Cell Cycle Phase Transition",
                                   "Mitotic Spindle Organization",
                                   "Mitotic Sister Chromatid Segregation"]},
    "myc_ribosome": {"programs": ["Myc Targets V1"],
                     "go_terms": ["Ribosome Biogenesis", "rRNA Processing",
                                  "Ribosomal Large Subunit Biogenesis",
                                  "Ribosomal Small Subunit Biogenesis"]},
    "oxphos": {"programs": ["Oxidative Phosphorylation"],
               "go_terms": ["Oxidative Phosphorylation",
                            "Mitochondrial Electron Transport, NADH To Ubiquinone",
                            "Mitochondrial Electron Transport, Cytochrome C To Oxygen",
                            "Mitochondrial Electron Transport, Ubiquinol To Cytochrome C",
                            "Aerobic Electron Transport Chain",
                            "Mitochondrial ATP Synthesis Coupled Electron Transport",
                            "Proton Motive Force-Driven ATP Synthesis"]},
    "hypoxia_glycolysis": {"programs": ["Hypoxia", "Glycolysis"],
                           "go_terms": ["Glycolytic Process", "Cellular Response To Hypoxia"]},
    "cholesterol": {"programs": ["Cholesterol Homeostasis"],
                    "go_terms": ["Cholesterol Biosynthetic Process",
                                 "Sterol Biosynthetic Process", "Cholesterol Metabolic Process",
                                 "Secondary Alcohol Biosynthetic Process"]},
    "upr": {"programs": ["Unfolded Protein Response", "Protein Secretion"],
            "go_terms": ["Response To Endoplasmic Reticulum Stress",
                         "Response To Unfolded Protein",
                         "IRE1-mediated Unfolded Protein Response"]},
    "nfkb_inflammation": {
        "programs": ["TNF-alpha Signaling via NF-kB", "Inflammatory Response",
                     "IL-6/JAK/STAT3 Signaling"],
        "go_terms": ["Inflammatory Response", "Cytokine-Mediated Signaling Pathway",
                     "Tumor Necrosis Factor-Mediated Signaling Pathway",
                     "Positive Regulation Of NF-kappaB Transcription Factor Activity",
                     "Positive Regulation Of I-kappaB kinase/NF-kappaB Signaling"]},
    "fatty_acid": {"programs": ["Fatty Acid Metabolism"],
                   "go_terms": ["Fatty Acid Beta-Oxidation", "Fatty Acid Metabolic Process",
                                "Fatty Acid Catabolic Process"]},
    "dna_repair": {"programs": ["DNA Repair"],
                   "go_terms": ["DNA Repair", "Double-Strand Break Repair",
                                "Nucleotide-Excision Repair", "Base-Excision Repair"]},
    "heme": {"programs": ["heme Metabolism"],
             "go_terms": ["Heme Biosynthetic Process", "Erythrocyte Differentiation",
                          "Porphyrin-Containing Compound Biosynthetic Process"]},
}

#: Site-native cell-type names under the ids_labels knob. Each is unambiguous to an immunologist:
#: mapping them is a handoff step, not a riddle.
NATIVE_LABELS: dict[str, tuple[str, ...]] = {
    "CD4 T": ("helper T cells", "CD4+ T lymphocytes", "T helper cells"),
    "CD8 T": ("cytotoxic T cells", "CD8+ T lymphocytes", "killer T cells"),
    "NK": ("natural killer cells", "NK lymphocytes", "CD56+ NK cells"),
    "B": ("B lymphocytes", "CD19+ B cells", "B-cells"),
    "CD14 Mono": ("classical monocytes", "CD14+ monocytes", "CD14++ CD16- monocytes"),
    "CD16 Mono": ("non-classical monocytes", "FCGR3A+ monocytes", "CD14+ CD16++ monocytes"),
}
GROUP_TEXT = {"control": "untreated control donors",
              "treated": "donors exposed to the study treatment",
              "mimic": "donors exposed to a comparator stimulus (mimic), not the study treatment"}

QUESTION = (
    "Four sites each profiled PBMCs from 8 donors by scRNA-seq (one sample per donor). At sites "
    "A, B and C, 4 donors received an uncharacterized study treatment and 4 are untreated "
    "controls; at site M, 4 donors received a comparator stimulus (mimic) and 4 are untreated "
    "controls. Raw data cannot leave a site. Which cell-type-specific transcriptional programs "
    "and cell-type composition changes does the treatment cause that replicate across sites, "
    "and which of them does the mimic share?"
)
DEFAULT_SPEC = {"unit": "donor", "normalization": "CPM", "one_sample_per_donor": True,
                "groups": "map site codes per the codebook"}


def load_gene_sets(path: Path) -> dict[str, list[str]]:
    sets = {}
    for line in path.read_text().strip().split("\n"):
        parts = line.split("\t")
        sets[parts[0]] = [g for g in parts[2:] if g]
    return sets


def core_gene_sets(hallmark: dict[str, list[str]], go_path: Path) -> dict[str, list[str]]:
    """Core genes of each plantable program: Hallmark members that are also annotated to one of
    the family's defining GO terms. Two independent libraries agree these genes *are* the
    program, which is what makes a planted subset recognizable."""
    go = load_gene_sets(go_path)
    by_name = {k.split(" (GO:")[0].lower(): set(v) for k, v in go.items()}
    core = {}
    for spec in FAMILIES.values():
        defining = set()
        for term in spec["go_terms"]:
            defining |= by_name.get(term.lower(), set())
        for program in spec["programs"]:
            core[program] = [g for g in hallmark.get(program, []) if g in defining]
    return core


def _expressed(counts: sp.csr_matrix, min_mean: float, min_frac: float) -> np.ndarray:
    mean = np.asarray(counts.mean(0)).ravel()
    frac = np.asarray((counts > 0).mean(0)).ravel()
    return (mean >= min_mean) & (frac >= min_frac)


def _thin(X: sp.csc_matrix, row_mask: np.ndarray, col: int, p: float,
          rng: np.random.Generator) -> None:
    """In place on a CSC matrix: X[row_mask, col] ~ Binomial(X[row_mask, col], p)."""
    lo, hi = X.indptr[col], X.indptr[col + 1]
    hit = row_mask[X.indices[lo:hi]]
    values = X.data[lo:hi]
    values[hit] = rng.binomial(values[hit].astype(np.int64), p)


# ---------------------------------------------------------------------------
# Frozen base: a donor subset of OneK1K
# ---------------------------------------------------------------------------

def _strings(node: h5py.Dataset | h5py.Group) -> np.ndarray:
    """An h5ad string or categorical column as an object array (anndata >= 0.8 encoding)."""
    if isinstance(node, h5py.Group):
        categories = node["categories"].asstr()[...]
        codes = node["codes"][...]
        out = np.full(codes.shape, None, dtype=object)
        out[codes >= 0] = categories[codes[codes >= 0]]
        return out
    return node.asstr()[...]


def _read_rows(X: h5py.Group, rows: np.ndarray) -> sp.csr_matrix:
    """Sorted rows of an on-disk CSR matrix as int32 counts, reading ROW_BLOCK rows per slice so
    the full matrix is never in memory."""
    indptr = X["indptr"][...]
    data, indices, lengths = [], [], []
    for start in range(0, len(indptr) - 1, ROW_BLOCK):
        i0, i1 = np.searchsorted(rows, [start, start + ROW_BLOCK])
        if i0 == i1:
            continue
        lo, hi = indptr[rows[i0]], indptr[rows[i1 - 1] + 1]
        block_data = np.rint(X["data"][lo:hi]).astype(np.int32)
        block_indices = X["indices"][lo:hi].astype(np.int32)
        for r in rows[i0:i1]:
            a, b = indptr[r] - lo, indptr[r + 1] - lo
            data.append(block_data[a:b])
            indices.append(block_indices[a:b])
            lengths.append(b - a)
    return sp.csr_matrix((np.concatenate(data), np.concatenate(indices),
                          np.concatenate([[0], np.cumsum(lengths)])),
                         shape=(len(rows), int(X.attrs["shape"][1])))


def _allocate(counts: np.ndarray, cap: int, floor: int) -> np.ndarray:
    """Cells kept per population: proportional to the donor's composition, at least `floor` (so
    rare populations stay testable), at most `cap` in total."""
    if counts.sum() <= cap:
        return counts
    keep = np.maximum(np.minimum(counts, floor), counts * cap // counts.sum())
    keep[np.argmax(keep)] -= max(0, int(keep.sum()) - cap)
    return keep


def _aliases(gene_ids: np.ndarray, symbols: np.ndarray, hgnc: Path) -> np.ndarray:
    """One frozen legacy name per gene for the alias ID scheme: its first previous HGNC symbol,
    else its first alias symbol, else the symbol itself. A name qualifies only if HGNC lists it
    for exactly one gene and it is no approved or base symbol (case-insensitive), so
    symbol <-> alias is a bijection and an alias never reads as another gene."""
    table = pd.read_csv(hgnc, sep="\t", dtype=str, keep_default_na=False,
                        usecols=["symbol", "prev_symbol", "alias_symbol", "ensembl_gene_id"])
    names = [[n for n in f"{r.prev_symbol}|{r.alias_symbol}".split("|") if n]
             for r in table.itertuples()]
    listed = Counter(u for ns in names for u in {n.upper() for n in ns})
    taken = {s.upper() for s in table["symbol"]} | {s.upper() for s in symbols}
    names_of = {eid: ns for eid, ns in zip(table["ensembl_gene_id"], names) if eid}
    out = []
    for eid, symbol in zip(gene_ids, symbols):
        usable = [n for n in names_of.get(eid, []) if listed[n.upper()] == 1
                  and n.upper() not in taken]
        out.append(usable[0] if usable else symbol)
    return np.asarray(out, dtype=object)


def prepare_base(src: Path, dst: Path = BASE, *, n_donors: int = 80, seed: int = 0) -> dict:
    """Freeze the donor subset every planted suite draws from; writes `dst` and `dst`.json.

    Streams `src` with h5py (OneK1K is 1.2M cells: only obs and the chosen CSR rows are read),
    keeps the case populations, donors with >= MIN_DONOR_CELLS such cells and >= MIN_POP_CELLS in
    every population, draws `n_donors`, splits them half dev / half test (disjoint pools for
    fsdev and fstest) and subsamples each donor to <= MAX_DONOR_CELLS, stratified by population.
    Genes with an ambiguous symbol or no counts are dropped, so symbol <-> Ensembl ID is a
    bijection (the ID knob relies on it); each gene's legacy name for the alias scheme is chosen
    from HGNC once, here, and frozen into var["alias"].
    """
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(f"{dst} exists; the base is frozen (suites record its hash)")
    if not HGNC.exists():
        raise FileNotFoundError(f"{HGNC} is missing; download {HGNC_URL}")
    rng = np.random.default_rng(seed)
    with h5py.File(src, "r") as f:
        obs = pd.DataFrame({
            "donor": _strings(f["obs"]["donor_id"]),
            "population": pd.Series(_strings(f["obs"]["cell_type"])).map(CELL_TYPES).to_numpy(),
        }).dropna()
        counts = (obs.groupby(["donor", "population"]).size().unstack(fill_value=0)
                  .reindex(columns=list(POPULATIONS), fill_value=0))
        eligible = counts.index[(counts.sum(1) >= MIN_DONOR_CELLS)
                                & (counts.min(1) >= MIN_POP_CELLS)]
        if len(eligible) < n_donors:
            raise ValueError(f"{len(eligible)} eligible donors, {n_donors} requested")
        chosen = [str(d) for d in rng.choice(sorted(eligible), size=n_donors, replace=False)]
        cells = obs.groupby(["donor", "population"]).indices
        source_row = obs.index.to_numpy()
        picked = []
        for donor in sorted(chosen):
            pools = [source_row[cells[(donor, p)]] for p in POPULATIONS]
            keep = _allocate(np.array([len(x) for x in pools]), MAX_DONOR_CELLS, MIN_POP_CELLS)
            picked += [rng.choice(x, size=k, replace=False) for x, k in zip(pools, keep)]
        rows = np.sort(np.concatenate(picked))
        X = _read_rows(f["X"], rows)
        gene_ids = _strings(f["var"][f["var"].attrs["_index"]])
        symbols = _strings(f["var"]["feature_name"])
    X.eliminate_zeros()
    genes = (np.bincount(X.indices, minlength=X.shape[1]) > 0) & ~pd.Series(symbols).duplicated(
        keep=False).to_numpy()
    names = symbols[genes].astype(str)
    base = ad.AnnData(
        X=X[:, genes],
        obs=pd.DataFrame({"donor": obs.loc[rows, "donor"].to_numpy(),
                          "population": obs.loc[rows, "population"].to_numpy()},
                         index=rows.astype(str)).astype("category"),
        var=pd.DataFrame({"gene_ids": gene_ids[genes],
                          "alias": _aliases(gene_ids[genes], names, HGNC)}, index=names))
    dst.parent.mkdir(parents=True, exist_ok=True)
    base.write_h5ad(dst, compression="gzip")
    half = n_donors // 2
    meta = {"source": src.name, "seed": seed, "dev_donors": sorted(chosen[:half]),
            "test_donors": sorted(chosen[half:]), "n_cells": int(base.n_obs),
            "n_genes": int(base.n_vars),
            "n_aliased": int((base.var["alias"] != base.var_names).sum()),
            "thresholds": {"min_donor_cells": MIN_DONOR_CELLS, "min_pop_cells": MIN_POP_CELLS,
                           "max_donor_cells": MAX_DONOR_CELLS},
            "hgnc": HGNC.name, "hgnc_sha256": file_digest(HGNC), "sha256": file_digest(dst)}
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def _target(site: str) -> str:
    return "mimic" if site == MIMIC_SITE else "treated"


def _tokens(rng: np.random.Generator, n: int) -> list[str]:
    """Opaque unique IDs: nothing in a sample or cell ID reveals donor, group or order."""
    return [f"{int(v):08x}" for v in rng.choice(16 ** 8, size=n, replace=False)]


def _draw_program(rng: np.random.Generator, kind: str, X: sp.csr_matrix, obs: pd.DataFrame,
                  gene_sets: dict[str, list[str]], index: dict[str, int],
                  taken: list[dict], populations: list[str]) -> dict:
    """A program from a family not used yet in this task, on genes not used yet, planted in
    ``populations`` where at least MIN_CORE of its core genes are expressed. One-site programs (site-specific,
    artefact) never share a site: at an assay-confounded site an effect seen only there could
    not be told apart from the artefact, so its cause would be ungradable."""
    used = {g for p in taken for g in p["up_genes"] + p["down_genes"]}
    families = sorted(set(FAMILIES) - {p["family"] for p in taken})
    busy = {s for p in taken if p["kind"] != "reproducible" for s in p["sites"]}
    free_sites = [s for s in PROPOSERS if s not in busy]
    for _ in range(200):
        family = str(rng.choice(families))
        program = str(rng.choice(FAMILIES[family]["programs"]))
        n_types = 2 if kind == "reproducible" and rng.random() < 0.4 else 1
        cell_types = sorted(str(c) for c in rng.choice(populations, size=n_types, replace=False))
        ok = _expressed(X[obs["population"].isin(cell_types).to_numpy()], 0.08, 0.05)
        candidates = [g for g in gene_sets.get(program, [])
                      if g in index and ok[index[g]] and g not in used]
        if len(candidates) < MIN_CORE:
            continue
        n = int(np.clip(round(rng.uniform(0.7, 1.0) * len(candidates)), MIN_CORE, 30))
        genes = sorted(str(g) for g in rng.choice(candidates, size=n, replace=False))
        up = kind == "artefact" or bool(rng.random() < 0.5)
        return {"id": f"P{len(taken) + 1}", "kind": kind, "program": program, "family": family,
                "cell_types": cell_types, "up_genes": genes if up else [],
                "down_genes": [] if up else genes,
                "log2fc": round(float(rng.uniform(0.8, 1.4)), 3),
                "sites": (list(PROPOSERS) if kind == "reproducible"
                          else [str(rng.choice(free_sites))]),
                "shared_with_mimic": kind == "reproducible" and bool(rng.random() < 0.5)}
    raise RuntimeError(f"no plantable {kind} program in 200 draws")


def plant_task(base: ad.AnnData, gene_sets: dict[str, list[str]], *, task_id: str, seed: int,
               donors: list[str], artefact: bool, ids_labels: bool,
               ) -> tuple[dict[str, ad.AnnData], dict[str, dict], dict]:
    """One task: (site file by site, site card by site, planting record)."""
    rng = np.random.default_rng(seed)
    sites = (*PROPOSERS, MIMIC_SITE)
    need = 2 * PER_GROUP * len(sites)
    if len(donors) < need:
        raise ValueError(f"the pool has {len(donors)} donors; a task needs {need}")
    drawn = [str(d) for d in rng.choice(sorted(donors), size=need, replace=False)]
    blocks = [drawn[i: i + 2 * PER_GROUP] for i in range(0, need, 2 * PER_GROUP)]
    layout = {site: {"control": sorted(block[:PER_GROUP]), _target(site): sorted(block[PER_GROUP:])}
              for site, block in zip(sites, blocks)}
    where = {d: (s, g) for s, groups in layout.items() for g, ds in groups.items() for d in ds}
    cells = base[base.obs["donor"].isin(drawn).to_numpy()]
    obs = pd.DataFrame({"donor": cells.obs["donor"].astype(str).to_numpy(),
                        "population": cells.obs["population"].astype(str).to_numpy()})
    obs["site"] = [where[d][0] for d in obs["donor"]]
    obs["group"] = [where[d][1] for d in obs["donor"]]
    X = sp.csr_matrix(cells.X, dtype=np.int32)
    index = {g: i for i, g in enumerate(cells.var_names)}

    # The composition shift's population carries no program, so DE power is untouched by it.
    population = str(rng.choice(COMPOSITION_POPULATIONS))
    kinds = ["reproducible"] * int(rng.integers(1, 4))
    if rng.random() < 0.5:
        kinds.append("site_specific")
    if artefact:
        kinds.append("artefact")
    programs: list[dict] = []
    plantable = [p for p in PROGRAM_POPULATIONS if p != population]
    for kind in kinds:
        programs.append(_draw_program(rng, kind, X, obs, gene_sets, index, programs, plantable))

    # Thin the lower group: control donors for an up program, target donors for a down one.
    Xc = X.tocsc()
    for prog in programs:
        planted = prog["sites"] + ([MIMIC_SITE] if prog["shared_with_mimic"] else [])
        lower = (obs["population"].isin(prog["cell_types"]) & obs["site"].isin(planted)
                 & ((obs["group"] == "control") == bool(prog["up_genes"]))).to_numpy()
        for g in prog["up_genes"] + prog["down_genes"]:
            beta = max(0.3, prog["log2fc"] + rng.normal(0, 0.2))
            _thin(Xc, lower, index[g], 2.0 ** -beta, rng)
    Xc.eliminate_zeros()
    X = Xc.tocsr()

    keep = np.ones(len(obs), dtype=bool)
    for site in PROPOSERS:
        for donor in layout[site]["treated"]:
            idx = np.flatnonzero(((obs["donor"] == donor)
                                  & (obs["population"] == population)).to_numpy())
            keep[rng.choice(idx, size=int(round(DROP * len(idx))), replace=False)] = False
    artefact_site = next((p["sites"][0] for p in programs if p["kind"] == "artefact"), None)

    # Every scheme at one proposing site, so per-scheme view-error rates (§6) rest on equal counts.
    schemes = dict.fromkeys(sites, "symbol")
    if ids_labels:
        schemes = {**{s: str(x) for s, x in zip(PROPOSERS, rng.permutation(SCHEMES))},
                   MIMIC_SITE: str(rng.choice(SCHEMES))}
    native_ids = {"symbol": np.asarray(cells.var_names, dtype=str),
                  "ensembl": cells.var["gene_ids"].to_numpy(dtype=str),
                  "alias": cells.var["alias"].to_numpy(dtype=str)}

    files, cards, maps = {}, {}, {}
    for site in sites:
        target = _target(site)
        if ids_labels:
            first, second = (str(c) for c in rng.permutation(["arm A", "arm B"]))
            codes = {first: "control", second: target}
            labels = {str(rng.choice(NATIVE_LABELS[p])): p for p in POPULATIONS}
        else:
            codes = {"control": "control", target: target}
            labels = {p: p for p in POPULATIONS}
        code_of = {g: c for c, g in codes.items()}
        label_of = {p: n for n, p in labels.items()}
        site_donors = layout[site]["control"] + layout[site][target]
        sample_of = dict(zip(site_donors, _tokens(rng, len(site_donors))))
        assay = {d: ASSAY_ALT if site == artefact_site and where[d][1] == target else ASSAY
                 for d in site_donors}
        rows = rng.permutation(np.flatnonzero(keep & (obs["site"] == site).to_numpy()))
        donor = obs["donor"].to_numpy()[rows]
        files[site] = ad.AnnData(
            X=X[rows].astype(np.float32),
            obs=pd.DataFrame({
                "sample": [sample_of[d] for d in donor],
                "condition": [code_of[where[d][1]] for d in donor],
                "cell_label": [label_of[p] for p in obs["population"].to_numpy()[rows]],
                "assay": [assay[d] for d in donor],
            }, index=_tokens(rng, len(rows))).astype("category"),
            var=pd.DataFrame(index=native_ids[schemes[site]]))
        cards[site] = {
            "site": site,
            "columns": {"sample": "sample ID (one sample per donor)",
                        "condition": "condition code; see codes",
                        "cell_label": "cell-type annotation made at this site",
                        "assay": "library chemistry"},
            "features": FEATURES[schemes[site]],
            "codes": {c: GROUP_TEXT[g] for c, g in codes.items()},
            "donors_per_group": {c: PER_GROUP for c in codes},
            "assay_values": {c: sorted({assay[d] for d in layout[site][g]})
                             for c, g in codes.items()},
            "cell_labels": sorted(labels),
        }
        maps[site] = {"scheme": schemes[site], "codes": codes, "labels": labels,
                      "donors": layout[site], "samples": {s: d for d, s in sample_of.items()}}

    record = {"task_id": task_id, "seed": seed,
              "knobs": {"artefact": artefact, "ids_labels": ids_labels},
              "programs": programs,
              "composition": {"id": "C1", "kind": "composition", "population": population,
                              "sign": -1, "fraction_removed": DROP, "sites": list(PROPOSERS)},
              "artefact_site": artefact_site, "sites": maps}
    return files, cards, record


def _write_task(task_dir: Path, files: dict[str, ad.AnnData], cards: dict[str, dict],
                var: pd.DataFrame) -> None:
    (task_dir / "sites").mkdir(parents=True)
    (task_dir / "public" / "cards").mkdir(parents=True)
    for site, adata in files.items():
        adata.write_h5ad(task_dir / "sites" / f"{site}.h5ad", compression="gzip")
        card = task_dir / "public" / "cards" / f"{site}.json"
        card.write_text(json.dumps(cards[site], indent=2))
    pd.DataFrame({"ensembl_id": var["gene_ids"].to_numpy(), "symbol": var.index.to_numpy(),
                  "alias": var["alias"].to_numpy()}).to_csv(
        task_dir / "public" / "symbols.tsv", sep="\t", index=False)
    (task_dir / "public" / "task.json").write_text(json.dumps({
        "task_id": task_dir.name, "question": QUESTION,
        "sites": [{"name": s, "data": f"sites/{s}.h5ad", "card": f"public/cards/{s}.json",
                   "proposes": s != MIMIC_SITE} for s in files],
        "symbols": "public/symbols.tsv",
    }, indent=2))


def build_suite(name: str, seeds: list[int], *, pool: str, knobs: list[tuple[bool, bool]],
                seal: Callable[[Path, dict], None]) -> list[Path]:
    """Write tasks `<name>-<seed>` under TASKS/<name>; `knobs[i]` = (artefact, ids_labels) of
    seed i. Each task's planting record (with its pool and the base's sha256) goes to
    `seal(task_dir, record)`.

    Refuses existing task directories: runs and grades are stored by task id, so a task rebuilt
    in place would pair old runs with new data. Build a new suite instead.
    """
    if pool not in ("dev", "test"):
        raise ValueError(f"pool must be 'dev' or 'test', not {pool!r}")
    if len(knobs) != len(seeds):
        raise ValueError("one (artefact, ids_labels) pair per seed")
    dirs = [TASKS / name / f"{name}-{seed:03d}" for seed in seeds]
    for task_dir in dirs:
        if task_dir.exists():
            raise FileExistsError(f"{task_dir} exists; build a new suite instead")
    frozen = json.loads(BASE.with_suffix(".json").read_text())
    base = ad.read_h5ad(BASE)
    gene_sets = core_gene_sets(load_gene_sets(HALLMARK), GO_BP)
    for task_dir, seed, (artefact, ids_labels) in zip(dirs, seeds, knobs):
        files, cards, record = plant_task(base, gene_sets, task_id=task_dir.name, seed=seed,
                                          donors=frozen[f"{pool}_donors"], artefact=artefact,
                                          ids_labels=ids_labels)
        _write_task(task_dir, files, cards, base.var)
        seal(task_dir, {**record, "pool": pool, "base_sha256": frozen["sha256"]})
    return dirs


# ---------------------------------------------------------------------------
# Case spec
# ---------------------------------------------------------------------------

def scope_tests(claim: Claim, sites: list[SiteHandle]) -> list[tuple[SiteHandle, tuple[str, str]]]:
    """Fixed scope table: a treated-vs-control claim is tested as mimic-vs-control wherever a site
    holds that contrast for the claim's population."""
    if claim.contrast != ("treated", "control"):
        return []
    return [(s, MIMIC) for s in sites if s.name != claim.site and s.holds(claim.population, MIMIC)]


def scope_rule(claim: Claim, results: list[TestResult]) -> str:
    """`shared` if any scope test supports the claim in the mimic; `specific` if every scope test
    that could run found it absent or reversed; otherwise `inconclusive`."""
    scope = [r for r in results if r.test == "scope"]
    if any(r.status == "supported" for r in scope):
        return "shared"
    testable = [r for r in scope if r.status != "untestable"]
    if testable and all(r.status in ("absent", "contradicted") for r in testable):
        return "specific"
    return "inconclusive"


def load_case(task_dir: Path) -> tuple[CaseSpec, list[dict]]:
    """The case spec of a planted task and its site specs ({"name", "data_path", "card",
    "proposes"}); M is tester-only. Resource keys are workspace paths; gene-set libraries go
    under `genesets/`, which is how MA-genome finds the library the site agents were given."""
    task_dir = Path(task_dir)
    meta = json.loads((task_dir / "public" / "task.json").read_text())
    case = CaseSpec(name="planted", kind="scrna", vocab=CASE_VOCAB, question=meta["question"],
                    default_spec=dict(DEFAULT_SPEC), tester=TesterParams(),
                    resources={"symbols.tsv": task_dir / meta["symbols"],
                               f"genesets/{GO_BP.name}": GO_BP},
                    scope_tests=scope_tests, scope_rule=scope_rule)
    sites = [{"name": s["name"], "data_path": task_dir / s["data"],
              "card": json.loads((task_dir / s["card"]).read_text()), "proposes": s["proposes"]}
             for s in meta["sites"]]
    return case, sites


if __name__ == "__main__":  # pragma: no cover
    import sys

    # python -m agentcoop.cases.planted data/bench/onek1k/onek1k.h5ad
    meta = prepare_base(Path(sys.argv[1]))
    print(json.dumps({k: meta[k] for k in ("n_cells", "n_genes", "n_aliased", "sha256")}, indent=2))
