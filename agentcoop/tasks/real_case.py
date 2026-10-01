"""Open-science case study: the *real* IFN-β stimulation in Kang et al. 2018 (no planted answer).

    python -m agentcoop.tasks.real_case

The 8 donors are split once (pre-registered, by sorted donor id: even positions = discovery,
odd = validation). Agents, evidence checks and the optimizer see only the 4 discovery donors, in
the same task format as the planted suites (``condition`` control | treated, treatment blinded).
The 4 validation donors are written to ``hidden/validation.h5ad`` and are used only after the
fact, by ``agentcoop.eval.case_validation``, as independent evidence: does each claimed effect
replicate in donors no component ever saw?
"""

from __future__ import annotations

import json
from pathlib import Path

from agentcoop.tasks.planted import CELL_TYPES, RAW, TASKS

SUITE = "kangreal"
TASK_ID = "kangreal-0"
QUESTION = (
    "PBMCs from 4 donors were profiled by scRNA-seq; within each donor, cells were either "
    "untreated (control) or exposed to an uncharacterized treatment (treated). Characterize the "
    "transcriptional response to the treatment across the major cell types: which biological "
    "programs are induced and which are repressed, and in which cell types? Only report effects "
    "that are reproducible across donors."
)


def load_real():
    import anndata as ad

    a = ad.read_h5ad(RAW)
    a = a[a.obs["cell_type"].isin(CELL_TYPES)].copy()
    obs = a.obs[["cell_type", "replicate", "label"]].copy()
    obs["donor"] = obs["replicate"].astype(str).str.replace("patient_", "D", regex=False)
    obs["condition"] = obs["label"].map({"ctrl": "control", "stim": "treated"}).astype(str)
    obs["cell_type"] = obs["cell_type"].astype(str)
    a.obs = obs[["cell_type", "donor", "condition"]]
    a.var = a.var[[]]
    a.X = a.X.tocsr()
    return a


def donor_split(donors: list[str]) -> tuple[list[str], list[str]]:
    ordered = sorted(donors)
    return ordered[0::2], ordered[1::2]


def build(out: Path = TASKS) -> Path:
    task_dir = out / SUITE / TASK_ID
    if task_dir.exists():
        raise FileExistsError(f"{task_dir} exists")
    a = load_real()
    discovery, validation = donor_split(list(a.obs["donor"].unique()))
    (task_dir / "public").mkdir(parents=True)
    (task_dir / "hidden").mkdir()
    a[a.obs["donor"].isin(discovery)].copy().write_h5ad(task_dir / "public" / "counts.h5ad",
                                                        compression="gzip")
    a[a.obs["donor"].isin(validation)].copy().write_h5ad(task_dir / "hidden" / "validation.h5ad",
                                                         compression="gzip")
    (task_dir / "public" / "task.json").write_text(json.dumps({
        "task_id": TASK_ID, "question": QUESTION, "data": "counts.h5ad",
        "obs_columns": {"cell_type": "annotated cell type", "donor": "donor id",
                        "condition": "control | treated"},
        "cell_types": CELL_TYPES}, indent=2))
    (task_dir / "hidden" / "split.json").write_text(json.dumps(
        {"discovery": discovery, "validation": validation}, indent=2))
    return task_dir


if __name__ == "__main__":
    print(build())
