"""Task shape 2: evidence integration across sites that cannot pool their data.

The same Kang-2018 control PBMCs are split by donor into two sites of four donors each. Each
site's data is only accessible to that site's analysis agent (it lives in its own file and the
workflow gives no other component a path to it). The question asks which treatment effects
*replicate across both sites*; the hidden answer contains

* ``reproducible`` programs, planted in all eight donors — they should be reported;
* ``site_specific`` programs, planted in all donors of one site only — a real effect at that
  site, but not a replicated one, so reporting it as replicated is an error.

With four donors per site, a single site has less power than the pooled data, so a program can
be missed at one site even though it is present there: replication then depends on *how* the
sites collaborate (e.g. whether one site tests the other's findings), not only on what each finds
alone. That is the collaboration question this shape is built to ask.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from agentcoop.tasks.planted import (
    CELL_TYPES, FAMILIES, MIN_CORE, TASKS, PlantedProgram, TaskTruth, _expressed, _thin,
    core_gene_sets, load_base, load_gene_sets,
)

QUESTION = (
    "Two sites (A and B) each profiled PBMCs from 4 donors by scRNA-seq; within each donor, cells "
    "were either untreated (control) or exposed to the same uncharacterized treatment (treated). "
    "Raw data cannot leave a site: each site's data can only be analyzed by that site's analysis "
    "agent. Which transcriptional programs does the treatment induce or repress, in which cell "
    "types, such that the effect replicates at BOTH sites? Report only replicated effects."
)


def plant_multisite(base, task_id: str, seed: int, gene_sets: dict[str, list[str]]):
    rng = np.random.default_rng(seed)
    adata = base.copy()
    X = adata.X.tocsr().astype(np.float32)
    obs = adata.obs
    donors = sorted(obs["donor"].unique())
    order = list(rng.permutation(donors))
    sites = {"A": sorted(order[:4]), "B": sorted(order[4:])}
    obs["site"] = obs["donor"].map({d: s for s, ds in sites.items() for d in ds})

    condition = np.empty(adata.n_obs, dtype=object)
    for _, idx in obs.groupby(["donor", "cell_type"], observed=True).indices.items():
        idx = rng.permutation(idx)
        half = len(idx) // 2
        condition[idx[:half]] = "control"
        condition[idx[half:]] = "treated"
    obs["condition"] = condition

    gene_index = {g: i for i, g in enumerate(adata.var_names)}
    used: set[str] = set()
    specs = ["reproducible"] * int(rng.choice([1, 2, 2, 3]))
    if rng.random() < 0.7:
        specs.append("site_specific")
    truth = TaskTruth(task_id=task_id, seed=seed)
    for kind in specs:
        for _ in range(200):
            family = str(rng.choice(list(FAMILIES)))
            if any(p.family == family for p in truth.programs):
                continue
            program = str(rng.choice(FAMILIES[family]["programs"]))
            n_ct = 1 if (kind == "site_specific" or rng.random() < 0.6) else 2
            cts = list(rng.choice(CELL_TYPES, size=n_ct, replace=False))
            mask = obs["cell_type"].isin(cts).to_numpy()
            ok = _expressed(X[mask], min_mean=0.08, min_frac=0.05)
            cands = [g for g in gene_sets[program]
                     if g in gene_index and ok[gene_index[g]] and g not in used]
            if len(cands) < MIN_CORE:
                continue
            n = int(np.clip(round(rng.uniform(0.7, 1.0) * len(cands)), MIN_CORE, 30))
            genes = sorted(rng.choice(cands, size=n, replace=False).tolist())
            direction = str(rng.choice(["up", "down"]))
            target = donors if kind == "reproducible" else sites[str(rng.choice(["A", "B"]))]
            used.update(genes)
            truth.programs.append(PlantedProgram(
                program=program, family=family, direction=direction,
                cell_types=[str(c) for c in cts], genes=genes,
                log2fc=round(float(rng.uniform(0.8, 1.4)), 3), donors=[str(d) for d in target],
                kind=kind))
            break

    Xc = X.tocsc()
    for prog in truth.programs:
        lower = "control" if prog.direction == "up" else "treated"
        scope = (obs["cell_type"].isin(prog.cell_types) & obs["donor"].isin(prog.donors)
                 & (obs["condition"] == lower)).to_numpy()
        for g in prog.genes:
            _thin(Xc, scope, gene_index[g], 2.0 ** (-max(0.3, prog.log2fc + rng.normal(0, 0.2))),
                  rng)
    Xc.eliminate_zeros()
    adata.X = Xc.tocsr()
    adata.obs = obs
    return adata, truth, sites


def build_suite(name: str, seeds: list[int], out: Path = TASKS) -> list[Path]:
    base = load_base()
    gene_sets = core_gene_sets(load_gene_sets())
    paths = []
    for seed in seeds:
        task_id = f"{name}-{seed:03d}"
        if (out / name / task_id).exists():
            # Node caches key on the task id, not the data: rebuilding a task in place would
            # replay analyses of the old data against the new answer. Use a new suite name.
            raise FileExistsError(f"{out / name / task_id} exists; build a new suite instead")
        adata, truth, sites = plant_multisite(base, task_id, seed, gene_sets)
        public = out / name / task_id / "public"
        hidden = out / name / task_id / "hidden"
        public.mkdir(parents=True, exist_ok=True)
        hidden.mkdir(parents=True, exist_ok=True)
        files = {}
        for site in sites:
            sub = adata[adata.obs["site"] == site].copy()
            sub.obs = sub.obs[["cell_type", "donor", "condition"]].copy()
            files[site] = f"site_{site}.h5ad"
            sub.write_h5ad(public / files[site], compression="gzip")
        (public / "task.json").write_text(json.dumps({
            "task_id": task_id, "question": QUESTION, "shape": "multisite",
            "sites": files, "cell_types": CELL_TYPES,
            "data": files["A"],  # nominal; components must use their own site's file
        }, indent=2))
        (hidden / "truth.json").write_text(json.dumps(asdict(truth), indent=2))
        paths.append(out / name / task_id)
    return paths


if __name__ == "__main__":  # pragma: no cover
    import sys

    suite, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    for p in build_suite(suite, list(range(lo, hi))):
        print(p)
