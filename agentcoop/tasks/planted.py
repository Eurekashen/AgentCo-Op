"""Semi-synthetic single-cell tasks with a hidden, exactly known answer.

Real cells, real donors, real gene–gene structure; only the treatment effect is planted, so the
answer sheet is known by construction and the optimizer can be evaluated without ever seeing it.

Base data: Kang et al. 2018 PBMCs, *control* cells only (8 donors), so no real stimulation signal
is present. Within every donor × cell type, cells are split at random into ``control`` and
``treated``. Effects are planted by **binomial thinning** (Gerard 2020, seqgendiff): to make gene
g ``β`` log2 units higher in treated cells, the control cells' counts of g are thinned with
probability 2^-β (and vice versa for down-regulation). Thinned Poisson/NB counts are still valid
counts, and the planted log fold change is exact in expectation.

A task plants 1–3 *reproducible* programs (all donors) and, sometimes, one *decoy*: a strong
program planted in a single donor. A correct analysis reports the reproducible programs with the
right direction and cell type, and does not report the decoy as a treatment effect.

Programs are the *core* of MSigDB Hallmark sets (members also annotated to the family's defining
GO terms), planted only where at least MIN_CORE core genes are expressed. The evidence checks available to the optimizer deliberately
use a *different* library (GO / Reactome), so the checks are informative without being an oracle.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "kang_2018.h5ad"
HALLMARK = ROOT / "data" / "genesets" / "hallmark_enrichr.txt"
TASKS = ROOT / "data" / "tasks"

CELL_TYPES = [
    "CD14+ Monocytes",
    "CD4 T cells",
    "B cells",
    "NK cells",
    "CD8 T cells",
    "FCGR3A+ Monocytes",
]

QUESTION = (
    "PBMCs from 8 donors were profiled by scRNA-seq; within each donor, cells were either "
    "untreated (control) or exposed to an uncharacterized treatment (treated). Characterize the "
    "transcriptional response to the treatment across the major cell types: which biological "
    "programs are induced and which are repressed, and in which cell types? Only report effects "
    "that are reproducible across donors."
)


#: Plantable programs, grouped into families a biologist can recognize from a gene subset.
#: Grading matches a claim to the planted *family*, so naming "interferon gamma response" for a
#: planted "Interferon Alpha Response" subset is correct, while "inflammation" is not. Hallmark
#: sets that are not recognizable from a gene subset (mTORC1, KRAS, estrogen, ...) are excluded:
#: no analyst could name them, so they would measure grader noise, not collaboration.
FAMILIES: dict[str, dict] = {
    "interferon": {"programs": ["Interferon Alpha Response", "Interferon Gamma Response"],
               "go_terms": ['Response To Type I Interferon', 'Defense Response To Virus', 'Type II Interferon-Mediated Signaling Pathway', 'Cellular Response To Type II Interferon', 'Negative Regulation Of Viral Genome Replication'],
                   "description": "type I/II interferon response, antiviral interferon-stimulated genes"},
    "proliferation": {"programs": ["E2F Targets", "G2-M Checkpoint", "Mitotic Spindle"],
                  "go_terms": ['DNA Replication', 'Mitotic Cell Cycle Phase Transition', 'Mitotic Spindle Organization', 'Mitotic Sister Chromatid Segregation'],
                      "description": "cell cycle, DNA replication, mitosis, proliferation"},
    "myc_ribosome": {"programs": ["Myc Targets V1"],
                 "go_terms": ['Ribosome Biogenesis', 'rRNA Processing', 'Ribosomal Large Subunit Biogenesis', 'Ribosomal Small Subunit Biogenesis'],
                     "description": "MYC targets, ribosome biogenesis, RNA processing, translation machinery"},
    "oxphos": {"programs": ["Oxidative Phosphorylation"],
           "go_terms": ['Oxidative Phosphorylation', 'Mitochondrial Electron Transport, NADH To Ubiquinone', 'Mitochondrial Electron Transport, Cytochrome C To Oxygen', 'Mitochondrial Electron Transport, Ubiquinol To Cytochrome C', 'Aerobic Electron Transport Chain', 'Mitochondrial ATP Synthesis Coupled Electron Transport', 'Proton Motive Force-Driven ATP Synthesis'],
               "description": "oxidative phosphorylation, mitochondrial respiration, electron transport chain"},
    "hypoxia_glycolysis": {"programs": ["Hypoxia", "Glycolysis"],
                       "go_terms": ['Glycolytic Process', 'Cellular Response To Hypoxia'],
                           "description": "hypoxia response, glycolysis, HIF1A targets"},
    "cholesterol": {"programs": ["Cholesterol Homeostasis"],
                "go_terms": ['Cholesterol Biosynthetic Process', 'Sterol Biosynthetic Process', 'Cholesterol Metabolic Process', 'Secondary Alcohol Biosynthetic Process'],
                    "description": "cholesterol/sterol biosynthesis and homeostasis, SREBP targets"},
    "upr": {"programs": ["Unfolded Protein Response", "Protein Secretion"],
        "go_terms": ['Response To Endoplasmic Reticulum Stress', 'Response To Unfolded Protein', 'IRE1-mediated Unfolded Protein Response'],
            "description": "unfolded protein response, ER stress, protein folding and secretion"},
    "nfkb_inflammation": {"programs": ["TNF-alpha Signaling via NF-kB", "Inflammatory Response",
                                       "IL-6/JAK/STAT3 Signaling"],
                      "go_terms": ['Inflammatory Response', 'Cytokine-Mediated Signaling Pathway', 'Tumor Necrosis Factor-Mediated Signaling Pathway', 'Positive Regulation Of NF-kappaB Transcription Factor Activity', 'Positive Regulation Of I-kappaB kinase/NF-kappaB Signaling'],
                          "description": "TNF/NF-kB signaling, inflammatory cytokine response, JAK/STAT3"},
    "fatty_acid": {"programs": ["Fatty Acid Metabolism"],
               "go_terms": ['Fatty Acid Beta-Oxidation', 'Fatty Acid Metabolic Process', 'Fatty Acid Catabolic Process'],
                   "description": "fatty acid metabolism, beta-oxidation, lipid metabolism"},
    "dna_repair": {"programs": ["DNA Repair"],
               "go_terms": ['DNA Repair', 'Double-Strand Break Repair', 'Nucleotide-Excision Repair', 'Base-Excision Repair'],
                   "description": "DNA repair, nucleotide metabolism"},
    "heme": {"programs": ["heme Metabolism"],
         "go_terms": ['Heme Biosynthetic Process', 'Erythrocyte Differentiation', 'Porphyrin-Containing Compound Biosynthetic Process'],
             "description": "heme metabolism, erythroid differentiation, hemoglobin"},
}


def family_of(program: str) -> str:
    for family, spec in FAMILIES.items():
        if program in spec["programs"]:
            return family
    raise KeyError(program)


GO_BP = ROOT / "data" / "genesets" / "GO_Biological_Process_2023.txt"
#: A program is plantable in a cell type only with at least this many expressed core genes.
MIN_CORE = 8


def load_gene_sets(path: Path = HALLMARK) -> dict[str, list[str]]:
    sets = {}
    for line in path.read_text().strip().split("\n"):
        parts = line.split("\t")
        sets[parts[0]] = [g for g in parts[2:] if g]
    return sets


def core_gene_sets(hallmark: dict[str, list[str]], go_path: Path = GO_BP) -> dict[str, list[str]]:
    """Core genes of each plantable program: Hallmark members that are also annotated to one of
    the family's defining GO terms. Two independent libraries agree these genes *are* the
    program, which is what makes a planted subset recognizable by name."""
    go = load_gene_sets(go_path)
    by_name = {k.split(" (GO:")[0].lower(): set(v) for k, v in go.items()}
    core = {}
    for spec in FAMILIES.values():
        defining = set()
        for term in spec["go_terms"]:
            defining |= by_name.get(term.lower(), set())
        for program in spec["programs"]:
            core[program] = [g for g in hallmark[program] if g in defining]
    return core


@dataclass
class PlantedProgram:
    program: str
    family: str
    direction: str  # "up" | "down"
    cell_types: list[str]
    genes: list[str]
    log2fc: float
    donors: list[str]
    kind: str  # "reproducible" | "decoy"


@dataclass
class TaskTruth:
    task_id: str
    seed: int
    programs: list[PlantedProgram] = field(default_factory=list)

    def reproducible(self) -> list[PlantedProgram]:
        return [p for p in self.programs if p.kind == "reproducible"]


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


def plant_task(base, task_id: str, seed: int, gene_sets: dict[str, list[str]]):
    """Return (planted AnnData, TaskTruth). ``base`` is Kang control cells, 6 cell types."""
    rng = np.random.default_rng(seed)
    adata = base.copy()
    X = adata.X.tocsr().astype(np.float32)
    obs = adata.obs
    donors = sorted(obs["donor"].unique())

    # Random within-donor × cell-type split into control / treated.
    condition = np.empty(adata.n_obs, dtype=object)
    for (_, _), idx in obs.groupby(["donor", "cell_type"], observed=True).indices.items():
        idx = rng.permutation(idx)
        half = len(idx) // 2
        condition[idx[:half]] = "control"
        condition[idx[half:]] = "treated"
    obs["condition"] = condition

    var = np.asarray(adata.var_names)
    gene_index = {g: i for i, g in enumerate(var)}
    used_genes: set[str] = set()

    n_reproducible = int(rng.choice([1, 2, 2, 3]))
    specs = [("reproducible", None) for _ in range(n_reproducible)]
    if rng.random() < 0.5:
        specs.append(("decoy", None))

    truth = TaskTruth(task_id=task_id, seed=seed)
    families = list(FAMILIES)
    for kind, _ in specs:
        for _attempt in range(200):
            family = str(rng.choice(families))
            if any(p.family == family for p in truth.programs):
                continue
            program = str(rng.choice(FAMILIES[family]["programs"]))
            n_ct = 1 if (kind == "decoy" or rng.random() < 0.6) else 2
            cts = list(rng.choice(CELL_TYPES, size=n_ct, replace=False))
            mask = obs["cell_type"].isin(cts).to_numpy()
            ok = _expressed(X[mask], min_mean=0.08, min_frac=0.05)
            candidates = [g for g in gene_sets[program]
                          if g in gene_index and ok[gene_index[g]] and g not in used_genes]
            if len(candidates) < MIN_CORE:
                continue
            frac = rng.uniform(0.7, 1.0)
            n = int(np.clip(round(frac * len(candidates)), MIN_CORE, 30))
            genes = sorted(rng.choice(candidates, size=n, replace=False).tolist())
            direction = str(rng.choice(["up", "down"]))
            if kind == "decoy":
                # the donor with the most cells in the target type: loud in pooled analyses
                counts = obs[mask].groupby("donor", observed=True).size()
                target_donors = [str(counts.idxmax())]
                log2fc = float(rng.uniform(1.8, 2.4))
            else:
                target_donors = donors
                log2fc = float(rng.uniform(0.8, 1.4))
            used_genes.update(genes)
            truth.programs.append(PlantedProgram(
                program=program, family=family, direction=direction, cell_types=[str(c) for c in cts],
                genes=genes, log2fc=round(log2fc, 3), donors=[str(d) for d in target_donors],
                kind=kind,
            ))
            break

    # Apply: per gene jitter around the program effect; thin the group that should be lower.
    Xc = X.tocsc()
    for prog in truth.programs:
        lower = "control" if prog.direction == "up" else "treated"
        in_scope = (obs["cell_type"].isin(prog.cell_types) & obs["donor"].isin(prog.donors)
                    & (obs["condition"] == lower)).to_numpy()
        for g in prog.genes:
            beta = max(0.3, prog.log2fc + rng.normal(0, 0.2))
            _thin(Xc, in_scope, gene_index[g], 2.0 ** (-beta), rng)
    Xc.eliminate_zeros()

    adata.X = Xc.tocsr()
    adata.obs = obs
    return adata, truth


def load_base():
    import anndata as ad

    a = ad.read_h5ad(RAW)
    a = a[(a.obs["label"] == "ctrl") & a.obs["cell_type"].isin(CELL_TYPES)].copy()
    a.obs = a.obs[["cell_type", "replicate"]].rename(columns={"replicate": "donor"}).copy()
    a.obs["donor"] = a.obs["donor"].astype(str).str.replace("patient_", "D", regex=False)
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)
    a.var = a.var[[]]
    a.X = a.X.tocsr()
    return a


def build_suite(name: str, seeds: list[int], out: Path = TASKS) -> list[Path]:
    base = load_base()
    gene_sets = core_gene_sets(load_gene_sets())
    paths = []
    for seed in seeds:
        task_id = f"{name}-{seed:03d}"
        adata, truth = plant_task(base, task_id, seed, gene_sets)
        public = out / name / task_id / "public"
        hidden = out / name / task_id / "hidden"
        public.mkdir(parents=True, exist_ok=True)
        hidden.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(public / "counts.h5ad", compression="gzip")
        (public / "task.json").write_text(json.dumps({
            "task_id": task_id,
            "question": QUESTION,
            "data": "counts.h5ad",
            "obs_columns": {"cell_type": "annotated cell type", "donor": "donor id",
                            "condition": "control | treated"},
            "cell_types": CELL_TYPES,
        }, indent=2))
        (hidden / "truth.json").write_text(json.dumps(asdict(truth), indent=2))
        paths.append(out / name / task_id)
    return paths


def load_truth(task_dir: Path) -> TaskTruth:
    raw = json.loads((Path(task_dir) / "hidden" / "truth.json").read_text())
    return TaskTruth(task_id=raw["task_id"], seed=raw["seed"],
                     programs=[PlantedProgram(**p) for p in raw["programs"]])


if __name__ == "__main__":  # pragma: no cover
    import sys

    suite, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    for p in build_suite(suite, list(range(lo, hi))):
        print(p)
