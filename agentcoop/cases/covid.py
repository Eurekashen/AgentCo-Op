"""Case 1 (spec §5.1): severe COVID-19 PBMC across four institutions. scRNA, real data.

    python -m agentcoop.cases.covid       # build data/tasks/covid/covid-4 from data/raw/covid ($0, once)

Sites (per-study files of the Tian et al. 2021 COVID-19 Immune Atlas collection on CZ CELLxGENE
Discover, collection b9fc3d70-5a72-4479-a046-c2cc1ab19efc): Berlin (Schulte-Schrepping 2020
cohort 1; COVID-19 on 10x 3' v3, controls from Reyes 2020 on 10x 3' v2 and 10x public PBMC on
v3 - the confounded controls), Stanford (Wilk 2020, Seq-Well), Emory (Arunachalam 2020, 10x v3),
Seoul (Lee 2020, 10x v3). All four propose.

Site files are rebuilt from the raw counts (``.raw.X``) with a column whitelist: opaque sample and
donor tokens, the site's native severity code (meaning in the site codebook), the authors' own
cell labels where the study published them (Berlin, Stanford; none at Emory and Seoul, whose
agents annotate their cells with the marker resources), assay, source cohort (Berlin), sex, age.
Harmonized columns (Tian's Cell.group, disease_general, ...), ontology columns, obsm/obsp/uns
and every expression layer but the counts are dropped. One sample per donor: the sample with the
highest severity code; ties -> earliest. Gene ids are the files' Ensembl ids with a symbol column
(``var["feature_name"]``): heterogeneity in this case lies in labels, severity scales, assays and
confounded controls, not in gene ids.

``TOOL_GROUPS`` / ``TOOL_POPULATIONS`` are the committed crosswalks fixed tools use (headroom
gate); agents never see them. This module reads and writes no held-out data (eval/covid.py).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from agentcoop.coop.claims import Claim, Vocabulary
from agentcoop.coop.protocol import SiteHandle
from agentcoop.coop.site import CaseSpec, file_digest
from agentcoop.coop.tester import TesterParams, TestResult

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "covid" / "h5ad"
TASK = ROOT / "data" / "tasks" / "covid" / "covid-4"
GENESETS = ROOT / "data" / "genesets"
MARKERS = ROOT / "data" / "markers"
LIBRARIES = ("GO_Biological_Process_2023.txt", "hallmark_enrichr.txt")
MARKER_LIBRARIES = ("PanglaoDB_Augmented_2021.txt", "CellMarker_2024.txt")
#: The one CellTypist model sites get (v2, 2022-07-16; sha256 290874d3...). Healthy_COVID19_PBMC
#: (trained on Stephenson = held out) and Adult_COVID19_PBMC (Wilk = a site) are never given.
CELLTYPIST_MODEL = MARKERS / "celltypist" / "Immune_All_Low.pkl"

DE_POPULATIONS = ("CD14 Mono", "CD4 T", "CD8 T", "NK", "B")
COMPOSITION_ONLY = ("CD16 Mono", "pDC", "Plasmablast")
CASE_VOCAB = Vocabulary(populations=DE_POPULATIONS + COMPOSITION_ONLY,
                        groups=("severe", "mild", "healthy"),
                        contrasts=(("severe", "healthy"), ("severe", "mild")),
                        composition_only=COMPOSITION_ONLY)
QUESTION = ("Which cell-type-specific transcriptional and compositional changes in peripheral "
            "blood separate severe or critical COVID-19 from health, and severe from mild "
            "COVID-19 - and which apparent changes are artefacts of how a site collected or "
            "processed its samples?")
DEFAULT_SPEC = {"unit": "donor", "normalization": "CPM", "one_sample_per_donor": True,
                "exclusions": "none beyond the case groups", "covariates": "none"}
SALT = "agentcoop-covid-v1"


@dataclass(frozen=True)
class SiteDef:
    name: str
    file: str                          # raw file under RAW
    condition: Callable[[pd.Series], str]        # obs row -> native severity code
    codes: dict[str, str]              # native code -> meaning (the site codebook)
    severity: dict[str, int]           # native code -> rank (one sample per donor: highest)
    label: str | None                  # obs column with the authors' own cell labels
    order: Callable[[str], tuple]      # sample name -> time order (ties -> earliest)
    cohort: str | None = None          # obs column naming the source cohort


def _day(sample: str) -> tuple:
    m = re.search(r"_d(\d+)$", sample)
    return (int(m.group(1)) if m else 0, sample)


def _number(sample: str) -> tuple:
    m = re.search(r"(\d+)\D*$", sample)
    return (int(m.group(1)) if m else 0, sample)


SITES = (
    SiteDef("Berlin", "berlin_schulte_schrepping.h5ad", lambda r: str(r["COVID-19 Condition"]),
            {"control": "control donor (no COVID-19)",
             "mild_early": "mild COVID-19 (WHO 2-4), sampled early (up to day 10 after symptom "
                           "onset)",
             "mild_late": "mild COVID-19 (WHO 2-4), sampled late (after day 10)",
             "severe_early": "severe COVID-19 (WHO 5-7), sampled early (up to day 10)",
             "severe_late": "severe COVID-19 (WHO 5-7), sampled late (after day 10)"},
            {"control": 0, "mild_early": 1, "mild_late": 1, "severe_early": 2, "severe_late": 2},
            "cluster_labels_res.0.4", _day, cohort="Cohort"),
    SiteDef("Stanford", "stanford_wilk.h5ad", lambda r: str(r["Ventilated"]),
            {"Healthy": "healthy donor",
             "NonVent": "hospitalized COVID-19, not mechanically ventilated",
             "Vent": "hospitalized COVID-19 with ARDS, mechanically ventilated (ICU)"},
            {"Healthy": 0, "NonVent": 1, "Vent": 2}, "cell_type_coarse", _number),
    SiteDef("Emory", "emory_arunachalam.h5ad", lambda r: str(r["COVID-19 Condition"]),
            {"Healthy": "healthy donor", "COVID-19 Moderate": "moderate COVID-19 (hospitalized)",
             "COVID-19 Severe": "severe COVID-19 (ICU)"},
            {"Healthy": 0, "COVID-19 Moderate": 1, "COVID-19 Severe": 2}, None, _number),
    SiteDef("Seoul", "seoul_lee.h5ad",
            lambda r: str(r["COVID-19 Condition"]).replace("\n", " "),
            {"Healthy donor": "healthy donor", "mild COVID-19": "mild COVID-19",
             "mild COVID-19 (asymptomatic)": "mild COVID-19 without symptoms",
             "severe COVID-19": "severe COVID-19 (oxygen support or ventilation)"},
            {"Healthy donor": 0, "mild COVID-19 (asymptomatic)": 1, "mild COVID-19": 1,
             "severe COVID-19": 2}, None, _number),
)

#: Committed crosswalks for fixed tools (headroom gate) only.
TOOL_GROUPS = {
    "Berlin": {"control": "healthy", "mild_early": "mild", "mild_late": "mild",
               "severe_early": "severe", "severe_late": "severe"},
    "Stanford": {"Healthy": "healthy", "NonVent": "mild", "Vent": "severe"},
    "Emory": {"Healthy": "healthy", "COVID-19 Moderate": "mild", "COVID-19 Severe": "severe"},
    "Seoul": {"Healthy donor": "healthy", "mild COVID-19": "mild",
              "mild COVID-19 (asymptomatic)": "mild", "severe COVID-19": "severe"},
}
#: Tian's harmonized Cell.group (in the raw files, never in site files) -> case population.
TOOL_POPULATIONS = {"CD14+ Monocyte": "CD14 Mono", "CD16+ Monocyte": "CD16 Mono",
                    "CD4+ T cell": "CD4 T", "CD8+ T cell": "CD8 T", "NK cell": "NK",
                    "B cell": "B", "pDC": "pDC", "Plasmablast": "Plasmablast"}


def _token(kind: str, site: str, name: str, width: int = 8) -> str:
    digest = hashlib.sha256(f"{SALT}\x00{site}\x00{kind}\x00{name}".encode()).hexdigest()
    return kind[0] + digest[:width]


def chosen_samples(obs: pd.DataFrame, site: SiteDef) -> dict[str, str]:
    """donor -> the one sample kept: highest severity code, ties -> earliest."""
    per = obs.groupby(["donor_id", "sample"], observed=True).first().reset_index()
    per["code"] = per.apply(site.condition, axis=1)
    out = {}
    for donor, rows in per.groupby("donor_id", observed=True):
        rows = rows.assign(rank=rows["code"].map(site.severity))
        if rows["rank"].isna().any():
            raise ValueError(f"{site.name}: severity code outside the codebook: "
                             f"{sorted(set(rows['code']) - set(site.severity))}")
        best = rows[rows["rank"] == rows["rank"].max()]
        out[str(donor)] = min(best["sample"].astype(str), key=site.order)
    return out


def build_site(site: SiteDef, raw: Path = RAW) -> tuple[ad.AnnData, dict]:
    """(site file, card): raw counts of the chosen samples' cells, whitelisted obs columns."""
    src = ad.read_h5ad(raw / site.file, backed="r")
    obs = src.obs
    keep_sample = set(chosen_samples(obs, site).values())
    rows = np.flatnonzero(obs["sample"].astype(str).isin(keep_sample).to_numpy())
    counts = src.raw.X[rows] if src.raw is not None else src.X[rows]
    counts = sp.csr_matrix(counts, dtype=np.float32)
    if not np.allclose(counts.data, np.round(counts.data)):
        raise ValueError(f"{site.name}: .raw.X is not integer counts")
    o = obs.iloc[rows]
    var_names = src.raw.var_names if src.raw is not None else src.var_names
    symbols = src.var["feature_name"].reindex(var_names) if "feature_name" in src.var else None
    src.file.close()
    columns = {
        "sample": [_token("sample", site.name, s) for s in o["sample"].astype(str)],
        "donor": [_token("donor", site.name, d) for d in o["donor_id"].astype(str)],
        "condition": o.apply(site.condition, axis=1).to_numpy(),
        "assay": o["assay"].astype(str).to_numpy(),
        "sex": o["sex"].astype(str).to_numpy(),
        "age": o["development_stage"].astype(str).to_numpy(),
    }
    if site.label:
        columns["cell_label"] = o[site.label].astype(str).to_numpy()
    if site.cohort:
        columns["cohort"] = o[site.cohort].astype(str).to_numpy()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(site.name.encode()).digest()[:4], "big"))
    order = rng.permutation(len(rows))
    cells = [_token("cell", site.name, c, width=16) for c in o.index.astype(str)]
    if len(set(cells)) != len(cells) or len(set(columns["donor"])) != o["donor_id"].nunique():
        raise ValueError(f"{site.name}: token collision")
    var = pd.DataFrame(index=pd.Index([str(v) for v in var_names]))
    if symbols is not None:
        var["feature_name"] = symbols.astype(str).to_numpy()
    adata = ad.AnnData(X=counts[order],
                       obs=pd.DataFrame({k: np.asarray(v)[order] for k, v in columns.items()},
                                        index=[cells[i] for i in order]).astype("category"),
                       var=var)
    codes = sorted(set(columns["condition"]))
    card = {
        "site": site.name,
        "assay": sorted(set(columns["assay"])),
        "values": "raw UMI counts (cells x genes)",
        "features": "Ensembl gene ids; var['feature_name'] holds the gene symbol",
        "columns": {"sample": "sample id (one sample per donor)", "donor": "donor id",
                    "condition": "the site's own severity code; see codes",
                    "assay": "library chemistry", "sex": "sex", "age": "age or age band",
                    **({"cell_label": "cell-type label assigned by the study's authors"}
                       if site.label else {}),
                    **({"cohort": "source cohort of the sample"} if site.cohort else {})},
        "codes": {c: site.codes[c] for c in codes},
        "donors_per_code": {c: int(n) for c, n in
                            pd.Series(columns["condition"]).groupby(columns["donor"]).first()
                            .value_counts().sort_index().items()},
        "cell_labels": (sorted(set(columns["cell_label"])) if site.label else
                        "none: annotate cells yourself (marker libraries under markers/, the "
                        "CellTypist model celltypist/Immune_All_Low.pkl; python package "
                        "celltypist is installed)"),
    }
    return adata, card


def build(task_dir: Path = TASK, raw: Path = RAW) -> Path:
    task_dir = Path(task_dir)
    if task_dir.exists():
        raise FileExistsError(f"{task_dir} exists; build a new task instead")
    (task_dir / "sites").mkdir(parents=True)
    (task_dir / "public" / "cards").mkdir(parents=True)
    listed = []
    for site in SITES:
        adata, card = build_site(site, raw)
        adata.write_h5ad(task_dir / "sites" / f"{site.name}.h5ad", compression="gzip")
        (task_dir / "public" / "cards" / f"{site.name}.json").write_text(json.dumps(card, indent=2))
        listed.append({"name": site.name, "data": f"sites/{site.name}.h5ad",
                       "card": f"public/cards/{site.name}.json", "proposes": True})
    meta = {"task_id": task_dir.name, "question": QUESTION, "sites": listed,
            "sources": {s.name: s.file for s in SITES},
            "sha256": {p.name: file_digest(p) for p in sorted((task_dir / "sites").iterdir())}}
    (task_dir / "public" / "task.json").write_text(json.dumps(meta, indent=2))
    return task_dir


def load_case(task_dir: Path = TASK) -> tuple[CaseSpec, list[dict]]:
    """The COVID case spec and its site specs; real-data agent calls get 30 minutes (§10)."""
    task_dir = Path(task_dir)
    meta = json.loads((task_dir / "public" / "task.json").read_text())
    case = CaseSpec(name="covid", kind="scrna", vocab=CASE_VOCAB, question=meta["question"],
                    default_spec=dict(DEFAULT_SPEC), tester=TesterParams(),
                    resources={**{f"genesets/{n}": GENESETS / n for n in LIBRARIES},
                               **{f"markers/{n}": MARKERS / n for n in MARKER_LIBRARIES},
                               "celltypist/Immune_All_Low.pkl": CELLTYPIST_MODEL})
    sites = [{"name": s["name"], "data_path": task_dir / s["data"],
              "card": json.loads((task_dir / s["card"]).read_text()), "proposes": s["proposes"],
              "timeout": 1800}
             for s in meta["sites"]]
    return case, sites


if __name__ == "__main__":  # pragma: no cover
    out = build()
    print(json.dumps(json.loads((out / "public" / "task.json").read_text())["sha256"], indent=1))
