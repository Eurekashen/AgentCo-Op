"""BAISBench BAIS-DPTA (Luo et al.; HF ``EperLuo/BaisBench``): annotate every cell of a real
single-cell dataset from one human organ (15 organs).

    python -m agentcoop.tasks.bais

Each organ becomes a task ``data/tasks/bais/bais-<organ>``. ``public/`` gets the benchmark's own
agent input (the ``<organ>_raw.h5ad`` file: counts and gene symbols, no metadata) and the
question; the expert labels (``cell_type`` of ``<organ>.h5ad``, in cell order) go to
``hidden/labels.json``. The raw file is linked, not copied (it is the public input as released).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data" / "bench" / "bais" / "Task1_data"
TASKS = ROOT / "data" / "tasks"
SUITE = "bais"
ORGANS = ["Adipose", "Blood", "Bone_marrow", "Breast", "Eye", "Heart", "Intestine", "Kidney",
          "Liver", "Ovary", "Pancreas", "Salivary_gland", "Skin", "Testis", "Trachea"]


def question(organ: str) -> str:
    return (f"This is a scRNA-seq dataset (raw counts) from a human {organ.replace('_', ' ')} "
            "sample. No preprocessing has been done. Annotate the cell type of every cell.")


def build(organs: list[str] = ORGANS, out: Path = TASKS) -> list[Path]:
    import anndata as ad

    paths = []
    for organ in organs:
        task_dir = out / SUITE / f"bais-{organ.lower()}"
        if task_dir.exists():
            raise FileExistsError(f"{task_dir} exists")
        raw, labelled = BENCH / f"{organ}_raw.h5ad", BENCH / f"{organ}.h5ad"
        lab = ad.read_h5ad(labelled, backed="r")
        n_raw = ad.read_h5ad(raw, backed="r").n_obs
        if lab.n_obs != n_raw:
            raise ValueError(f"{organ}: {lab.n_obs} labelled vs {n_raw} raw cells")
        (task_dir / "public").mkdir(parents=True)
        (task_dir / "hidden").mkdir()
        os.link(raw, task_dir / "public" / "counts.h5ad")
        (task_dir / "public" / "task.json").write_text(json.dumps({
            "task_id": task_dir.name, "question": question(organ), "shape": "annotation",
            "organ": organ, "data": "counts.h5ad"}, indent=2))
        (task_dir / "hidden" / "labels.json").write_text(json.dumps({
            "organ": organ, "cell_type": [str(x) for x in lab.obs["cell_type"]]}))
        paths.append(task_dir)
    return paths


if __name__ == "__main__":
    for p in build(sys.argv[1:] or ORGANS):
        print(p)
