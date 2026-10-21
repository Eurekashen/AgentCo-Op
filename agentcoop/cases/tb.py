"""Case 2 (spec §5.2): active TB whole-blood signature across hospital cohorts. Bulk, real data.

    python -m agentcoop.cases.tb          # build data/tasks/tb/tb-min from data/raw/tb ($0, once)

Sites (minimal federation): S1 London (GSE19439 + GSE19444, Illumina HT-12 v3, raw linear),
S2 South Africa (GSE37250 region South Africa, HT-12 v4), S6 The Gambia (GSE28623, Agilent
feature numbers) propose; S4 (GSE22098: adult autoimmune and paediatric bacterial strata, each
with its own controls; paediatric SLE dropped) is tester-only. S1 and S4 share a lab
(SuperSeries GSE19491), so S4 can only demote (scope), never corroborate.

Each site file is a TSV matrix (first column = the platform's native feature id, one column per
sample, opaque sample tokens instead of GSM ids), with a sample table (opaque token, the site's
native condition code, whitelisted covariates) and the platform's GEO annotation table as the
mapping resource. The card states the platform, value scale, files and what each native code
means; translating codes to case groups and probes to symbols is the site agent's job.
``TOOL_GROUPS`` is the committed crosswalk fixed tools use (headroom gate), never shown to agents.

GEO downloads are read as data (gzip text); nothing in them is executed. This module writes no
held-out data and reads none: held-out cohorts are agentcoop/eval/tb.py's.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from agentcoop.coop.claims import Claim, Vocabulary
from agentcoop.coop.protocol import SiteHandle
from agentcoop.coop.site import CaseSpec, file_digest
from agentcoop.coop.tester import TesterParams, TestResult

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "tb"
TASK = ROOT / "data" / "tasks" / "tb" / "tb-min"
GENESETS = ROOT / "data" / "genesets"
LIBRARIES = ("GO_Biological_Process_2023.txt", "hallmark_enrichr.txt")

GROUPS = ("TB", "LTBI", "HC", "OD", "autoimmune", "bacterial", "paediatric_control")
CASE_VOCAB = Vocabulary(populations=("all",), groups=GROUPS,
                        contrasts=(("TB", "LTBI"), ("TB", "HC"), ("TB", "OD")))
MIMICS = {"autoimmune": ("autoimmune", "HC"), "bacterial": ("bacterial", "paediatric_control")}
OD = ("TB", "OD")
TB_HC = ("TB", "HC")
QUESTION = ("Which whole-blood transcriptional signatures distinguish active tuberculosis from "
            "latent infection, from healthy people and from other diseases, and which of them are "
            "specific to tuberculosis rather than shared with other inflammatory or infectious "
            "conditions?")
DEFAULT_SPEC = {"unit": "sample (one per patient)", "normalization": "within-site rank",
                "exclusions": "none beyond the case groups", "covariates": "none"}
# Scope thresholds (spec §5.2; a priori conventions, equal to the grader's).
SPECIFIC_AUC, NOT_SPECIFIC_AUC, MIMIC_MARGIN = 0.70, 0.65, 0.15
SALT = "agentcoop-tb-v1"     # opaque sample tokens: sha256(SALT, site, GSM), 8 hex


# ---------------------------------------------------------------------------
# GEO series matrices
# ---------------------------------------------------------------------------

def read_series(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sample metadata indexed by GSM: title, source, and one column per characteristics key;
    expression: features x GSM, feature ids as strings). Characteristics rows hold "key: value";
    the key may differ between samples of one row (GSE22098: illness / healthy control)."""
    fields: list[tuple[str, list[str]]] = []
    table: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        inside = False
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                inside = True
            elif line.startswith("!series_matrix_table_end"):
                break
            elif inside:
                table.append(line)
            elif line.startswith("!Sample_"):
                key, *values = [v.strip().strip('"') for v in line.rstrip("\n").split("\t")]
                fields.append((key, values))
    gsms = next(values for key, values in fields if key == "!Sample_geo_accession")
    rows: list[dict[str, str]] = [{} for _ in gsms]
    for key, values in fields:
        for row, v in zip(rows, values):
            if key == "!Sample_characteristics_ch1" and ": " in v:
                k, val = v.split(": ", 1)
                row[k.strip().lower()] = val.strip()
            elif key in ("!Sample_title", "!Sample_source_name_ch1"):
                row[key.removeprefix("!Sample_").removesuffix("_ch1")] = v
    expr = pd.read_csv(io.StringIO("".join(table)), sep="\t", index_col=0,
                       dtype={"ID_REF": str})
    expr.index = expr.index.astype(str)
    return pd.DataFrame(rows, index=gsms), expr


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteDef:
    name: str
    series: tuple[str, ...]
    platform: str                     # GEO platform id; its annot table is the mapping resource
    platform_text: str
    values: str                       # what the matrix values are
    features: str                     # what the first column holds
    proposes: bool
    code: Callable[[pd.Series], str | None]      # native condition code; None = not in the site
    codes: dict[str, str]             # native code -> what it means (the site's own codebook)
    covariates: dict[str, str]        # sample-table column -> metadata column


def _s1(m: pd.Series) -> str | None:
    illness = str(m.get("illness", ""))
    return illness if illness in ("PTB", "Latent") else (
        "Control" if illness.startswith("Control") else None)


def _s2(m: pd.Series) -> str | None:
    if m.get("geographical region") != "South Africa":
        return None
    return {"active tuberculosis": "ATB", "latent TB infection": "LTBI",
            "other disease": "OD"}.get(m.get("disease state"))


def _s6(m: pd.Series) -> str | None:
    return {"tuberculosis patients": "TB", "healthy donors latently infected with M. tuberculosis":
            "LTBI", "healthy non-infected donors": "NID"}.get(m.get("group"))


def _s4(m: pd.Series) -> str | None:
    if m.get("illness") in ("Still", "ASLE", "Staph", "Strep"):
        return m["illness"]
    return {"Still": "HC-Still", "ASLE": "HC-ASLE",
            "Strep and Staph": "HC-Bact"}.get(m.get("healthy control"))


SITES = (
    SiteDef("S1", ("GSE19439", "GSE19444"), "GPL6947", "Illumina HumanHT-12 v3 BeadChip",
            "background-subtracted linear intensities (not log-transformed)",
            "Illumina probe ids (ILMN_...)", True, _s1,
            {"PTB": "active pulmonary tuberculosis, before treatment",
             "Latent": "latent TB infection (positive tuberculin skin test / IGRA, no disease)",
             "Control": "healthy control (BCG vaccinated or not)"},
            {"sex": "gender", "age": "age"}),
    SiteDef("S2", ("GSE37250",), "GPL10558", "Illumina HumanHT-12 v4 BeadChip",
            "background-subtracted linear intensities (not log-transformed)",
            "Illumina probe ids (ILMN_...)", True, _s2,
            {"ATB": "active tuberculosis", "LTBI": "latent TB infection",
             "OD": "other disease: symptomatic patients investigated for TB in whom TB was "
                   "excluded (e.g. pneumonia, other infections, malignancy)"},
            {"hiv": "hiv status"}),
    SiteDef("S6", ("GSE28623",), "GPL4133", "Agilent Whole Human Genome 4x44K (one colour)",
            "linear signal intensities (not log-transformed)",
            "Agilent feature numbers (the platform table's ID column)", True, _s6,
            {"TB": "active tuberculosis patients",
             "LTBI": "healthy donors latently infected with M. tuberculosis",
             "NID": "healthy non-infected donors"},
            {"sex": "gender"}),
    SiteDef("S4", ("GSE22098",), "GPL6947", "Illumina HumanHT-12 v3 BeadChip",
            "background-subtracted linear intensities (not log-transformed)",
            "Illumina probe ids (ILMN_...)", False, _s4,
            {"Still": "adult-onset Still's disease (adult autoimmune)",
             "ASLE": "adult systemic lupus erythematosus (adult autoimmune)",
             "HC-Still": "adult healthy control (recruited with the Still's disease patients)",
             "HC-ASLE": "adult healthy control (recruited with the adult SLE patients)",
             "Staph": "paediatric Staphylococcus aureus infection (acute bacterial)",
             "Strep": "paediatric Streptococcus pneumoniae infection (acute bacterial)",
             "HC-Bact": "paediatric healthy control (recruited with the bacterial infections)"},
            {"sex": "gender", "age": "age"}),
)

#: Committed crosswalk native code -> case group, for fixed tools (headroom gate) only.
TOOL_GROUPS = {
    "S1": {"PTB": "TB", "Latent": "LTBI", "Control": "HC"},
    "S2": {"ATB": "TB", "LTBI": "LTBI", "OD": "OD"},
    "S6": {"TB": "TB", "LTBI": "LTBI", "NID": "HC"},
    "S4": {"Still": "autoimmune", "ASLE": "autoimmune", "HC-Still": "HC", "HC-ASLE": "HC",
           "Staph": "bacterial", "Strep": "bacterial", "HC-Bact": "paediatric_control"},
}


def _token(site: str, gsm: str) -> str:
    return "s" + hashlib.sha256(f"{SALT}\x00{site}\x00{gsm}".encode()).hexdigest()[:8]


def build_site(site: SiteDef, raw: Path = RAW) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(expression: native feature id x token, sample table, card) of one site. Series of one
    site are joined on the features they share; samples outside the site's codes are dropped."""
    metas, exprs = zip(*(read_series(raw / "geo" / f"{g}_series_matrix.txt.gz")
                         for g in site.series))
    meta = pd.concat(metas)
    expr = pd.concat(exprs, axis=1, join="inner")
    codes = meta.apply(site.code, axis=1).dropna()
    gsms = [g for g in expr.columns if g in codes.index]
    tokens = {g: _token(site.name, g) for g in gsms}
    if len(set(tokens.values())) != len(tokens):
        raise ValueError(f"{site.name}: sample token collision")
    matrix = expr[gsms].rename(columns=tokens)
    matrix.index.name = "ID_REF"
    samples = pd.DataFrame({"sample": [tokens[g] for g in gsms],
                            "condition": [codes[g] for g in gsms],
                            **{col: [meta.at[g, src] if src in meta.columns else None for g in gsms]
                               for col, src in site.covariates.items()}})
    card = {
        "site": site.name,
        "platform": f"{site.platform_text} (GEO {site.platform})",
        "values": site.values,
        "features": site.features,
        "files": {"expression": f"{site.name}_expression.tsv",
                  "samples": f"{site.name}_samples.tsv",
                  "platform annotation": f"{site.platform}.annot.txt (GEO annotation table: "
                                         "ID = feature id, 'Gene symbol' = mapped symbol(s))"},
        "columns": {"sample": "sample id (= an expression column header); one sample per patient",
                    "condition": "the site's own condition code; see codes",
                    **{c: f"{c} as recorded at the site" for c in site.covariates}},
        "codes": {c: site.codes[c] for c in sorted(set(samples["condition"]))},
        "samples_per_code": {c: int(n) for c, n in samples["condition"].value_counts()
                             .sort_index().items()},
    }
    return matrix, samples, card


def _annotation(platform: str, raw: Path = RAW) -> Path:
    return raw / "gpl" / f"{platform}.annot.gz"


def build(task_dir: Path = TASK, raw: Path = RAW) -> Path:
    """Write the case's site files, cards and public/task.json. Refuses an existing task (store
    keys and results are keyed by task id)."""
    task_dir = Path(task_dir)
    if task_dir.exists():
        raise FileExistsError(f"{task_dir} exists; build a new task instead")
    (task_dir / "sites").mkdir(parents=True)
    (task_dir / "public" / "cards").mkdir(parents=True)
    listed = []
    for site in SITES:
        matrix, samples, card = build_site(site, raw)
        matrix.to_csv(task_dir / "sites" / f"{site.name}_expression.tsv", sep="\t")
        samples.to_csv(task_dir / "sites" / f"{site.name}_samples.tsv", sep="\t", index=False)
        annot = task_dir / "sites" / f"{site.platform}.annot.txt"
        if not annot.exists():
            with gzip.open(_annotation(site.platform, raw), "rb") as src, open(annot, "wb") as dst:
                shutil.copyfileobj(src, dst)
        (task_dir / "public" / "cards" / f"{site.name}.json").write_text(json.dumps(card, indent=2))
        listed.append({"name": site.name, "data": f"sites/{site.name}_expression.tsv",
                       "files": {f"{site.name}_samples.tsv": f"sites/{site.name}_samples.tsv",
                                 annot.name: f"sites/{annot.name}"},
                       "card": f"public/cards/{site.name}.json", "proposes": site.proposes})
    meta = {"task_id": task_dir.name, "question": QUESTION, "sites": listed,
            "sources": {s.name: list(s.series) for s in SITES},
            "sha256": {p.name: file_digest(p) for p in sorted((task_dir / "sites").iterdir())}}
    (task_dir / "public" / "task.json").write_text(json.dumps(meta, indent=2))
    return task_dir


# ---------------------------------------------------------------------------
# Case spec: scope table and rule (§5.2)
# ---------------------------------------------------------------------------

def scope_tests(claim: Claim, sites: list[SiteHandle]) -> list[tuple[SiteHandle, tuple[str, str]]]:
    """Fixed scope table: a TB vs LTBI / HC / OD claim gets a TB-vs-OD test at every site holding
    OD (the proposer's own may only demote: scope_rule), an m-vs-control test at every other site
    holding mimic class m, and a TB-vs-HC test at the proposer (the reference the mimic margin is
    measured against)."""
    if claim.target != "TB":
        return []
    out = [(s, OD) for s in sites if s.holds("all", OD)]
    for contrast in MIMICS.values():
        out += [(s, contrast) for s in sites if s.name != claim.site and s.holds("all", contrast)]
    out += [(s, TB_HC) for s in sites if s.name == claim.site and s.holds("all", TB_HC)]
    return out


def _auc(r: TestResult) -> tuple[float, float, float] | None:
    """(AUC, CI low, CI high) in the claimed direction (testers report AUC - 0.5)."""
    if r.effect is None or r.ci95 is None:
        return None
    return r.effect + 0.5, r.ci95[0] + 0.5, r.ci95[1] + 0.5


def scope_rule(claim: Claim, results: list[TestResult]) -> str:
    """§5.2, demote-only first: `not specific vs <class>` when any TB-vs-OD test - the
    proposer's own included - has CI upper < 0.65 (mixed_OD), or a mimic test m-vs-control has CI
    lower > 0.5 and the proposer's TB-vs-HC AUC (the gate when that is the claim's contrast)
    exceeds it by < 0.15; `specific vs mixed_OD` when a non-proposer TB-vs-OD test has AUC >= 0.70
    with CI lower > 0.5 and nothing demotes; else `inconclusive` (also when only in-sample
    evidence exists). In-sample evidence may demote but never promote: it carries no optimistic
    bias against specificity, and S2, the minimal federation's only OD holder, would otherwise
    never have its own claims checked against other diseases (findings 2026-10-22)."""
    scope = [r for r in results if r.test == "scope" and _auc(r) is not None]
    od_all = [_auc(r) for r in scope if r.contrast == OD]
    od = [_auc(r) for r in scope if r.contrast == OD and r.site != claim.site]
    reference = next((_auc(r) for r in scope if r.contrast == TB_HC and r.site == claim.site),
                     None)
    for r in scope:
        if r.contrast == OD:
            continue
        for mimic, contrast in MIMICS.items():
            auc = _auc(r)
            if (r.contrast == contrast and auc[1] > 0.5 and reference is not None
                    and reference[0] - auc[0] < MIMIC_MARGIN):
                return f"not specific vs {mimic}"
    if any(a[2] < NOT_SPECIFIC_AUC for a in od_all):
        return "not specific vs mixed_OD"
    if any(a[0] >= SPECIFIC_AUC and a[1] > 0.5 for a in od):
        return "specific vs mixed_OD"
    return "inconclusive"


def load_case(task_dir: Path = TASK) -> tuple[CaseSpec, list[dict]]:
    """The TB case spec and its site specs ({"name", "data_path", "card", "proposes", "files",
    "timeout"}). Real-data agent calls get the 30-minute cap (§10)."""
    task_dir = Path(task_dir)
    meta = json.loads((task_dir / "public" / "task.json").read_text())
    case = CaseSpec(name="tb", kind="bulk", vocab=CASE_VOCAB, question=meta["question"],
                    default_spec=dict(DEFAULT_SPEC), tester=TesterParams(),
                    resources={f"genesets/{n}": GENESETS / n for n in LIBRARIES},
                    scope_tests=scope_tests, scope_rule=scope_rule)
    sites = [{"name": s["name"], "data_path": task_dir / s["data"],
              "files": {n: task_dir / p for n, p in s["files"].items()},
              "card": json.loads((task_dir / s["card"]).read_text()), "proposes": s["proposes"],
              "timeout": 1800}
             for s in meta["sites"]]
    return case, sites


if __name__ == "__main__":  # pragma: no cover
    out = build()
    print(json.dumps(json.loads((out / "public" / "task.json").read_text())["sha256"], indent=1))
