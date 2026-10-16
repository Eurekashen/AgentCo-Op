"""Invariants of v4 that no single module owns (spec §8). Offline and fast: a tiny OneK1K-like
source, no agent runs, no API key.

1. Nothing outside agentcoop/eval/ has a path to the answers.
2. A planted task's site files carry only whitelisted columns, and nothing a site agent or the
   coordinator can be given (site files, cards, resources) carries the answer.
3. Site workspaces are created under WS_ROOT, outside the repo.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "agentcoop"
#: Modules the scan must reach; a rename that silently drops one from it fails loudly instead.
REQUIRED = ("store.py", "proxy.py", "llm.py", "coop/claims.py", "coop/view.py",
            "coop/tester.py", "coop/protocol.py", "coop/site.py", "coop/arms.py",
            "agents/workspace.py", "agents/codex.py", "agents/react.py", "agents/finch.py",
            "agents/harness.py", "cases/planted.py")
FORBIDDEN_TEXT = ("hidden/", "truth.json", "load_truth")
#: The only obs columns a planted site file may carry (spec §7).
SITE_OBS = {"sample", "condition", "cell_label", "assay"}
#: Planting-record keys that state the answer (or, with the generator, regenerate it).
ANSWER_KEYS = {"programs", "knobs", "seed", "artefact_site", "shared_with_mimic", "composition",
               "log2fc", "up_genes", "down_genes", "fraction_removed"}
SIDE_CHANNELS = ("uns", "obsm", "obsp", "varm", "varp", "layers")
GENES = [f"G{i:03d}" for i in range(300)]


# ---------------------------------------------------------------------------
# 1. No path to the answers outside agentcoop/eval/
# ---------------------------------------------------------------------------

def _imported(source: str, package: str) -> set[str]:
    """Fully qualified names a module imports: import statements (relative ones resolved
    against ``package``) and importlib.import_module / __import__ calls on a literal."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                base = ".".join([*parts[:len(parts) - node.level + 1], *filter(None, [base])])
            names.add(base)
            names.update(f"{base}.{a.name}" for a in node.names)
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)
              and (isinstance(node.func, ast.Name) and node.func.id == "__import__"
                   or isinstance(node.func, ast.Attribute)
                   and node.func.attr == "import_module")):
            names.add(node.args[0].value)
    return names


def _violations(source: str, package: str) -> list[str]:
    out = [f"imports {n}" for n in sorted(_imported(source, package))
           if n == "agentcoop.eval" or n.startswith("agentcoop.eval.")]
    return out + [f"mentions {t!r}" for t in FORBIDDEN_TEXT if t in source]


def _answer_free_files() -> list[Path]:
    """Every module outside agentcoop/eval/: all of it runs at protocol time."""
    return sorted(p for p in PACKAGE.rglob("*.py") if p.relative_to(PACKAGE).parts[0] != "eval")


def test_protocol_side_has_no_path_to_the_answers():
    bad = {}
    for f in _answer_free_files():
        package = ".".join(f.relative_to(ROOT).parent.parts)
        if found := _violations(f.read_text(), package):
            bad[f.relative_to(ROOT).as_posix()] = found
    assert not bad, bad


def test_the_scan_reaches_every_v4_module():
    scanned = {f.relative_to(PACKAGE).as_posix() for f in _answer_free_files()}
    assert set(REQUIRED) <= scanned, sorted(set(REQUIRED) - scanned)


@pytest.mark.parametrize("source,package", [
    ("import agentcoop.eval.planted", "agentcoop.coop"),
    ("from agentcoop.eval import planted", "agentcoop.coop"),
    ("from agentcoop import eval", "agentcoop.cases"),
    ("from ..eval.planted import grade", "agentcoop.coop"),
    ("from .. import eval", "agentcoop.agents"),
    ("from .eval import planted", "agentcoop"),
    ("import importlib\nimportlib.import_module('agentcoop.eval.planted')", "agentcoop"),
    ("TRUTH = 'hidden/truth.json'", "agentcoop.cases"),
    ("from agentcoop.coop.site import load_truth", "agentcoop.coop"),
])
def test_the_scan_catches_every_route_to_the_answers(source, package):
    assert _violations(source, package)


def test_the_scan_passes_lookalikes():
    assert not _violations("import agentcoop.evaluation\nfrom agentcoop import llm\n"
                           "from ..coop import claims", "agentcoop.cases")


# ---------------------------------------------------------------------------
# 2. Planted tasks: whitelisted site files, no answer in anything a site or coordinator gets
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _keys(node: object) -> set[str]:
    if isinstance(node, dict):
        return {str(k) for k in node} | {k for v in node.values() for k in _keys(v)}
    if isinstance(node, list):
        return {k for v in node for k in _keys(v)}
    return set()


def _string_lists(node: object) -> list[set[str]]:
    """Every list of >= 3 strings in a JSON document, as a set."""
    if isinstance(node, dict):
        return [g for v in node.values() for g in _string_lists(v)]
    if isinstance(node, list):
        if len(node) >= 3 and all(isinstance(x, str) for x in node):
            return [set(node)]
        return [g for v in node for g in _string_lists(v)]
    return []


def assert_planted_task_boundary(task_dir: Path) -> None:
    """Everything a site agent or the coordinator can be given (site files, cards, the case's
    resources) carries no part of the answer, and site files carry only whitelisted obs columns
    and no side channels. Run by the tests, and on every real planted suite before any spend."""
    from agentcoop.cases.planted import load_case
    from agentcoop.eval.planted import load_truth

    task_dir = Path(task_dir)
    visible = (task_dir / "public", task_dir / "sites")
    case, sites = load_case(task_dir)
    universe: set[str] = set()
    for s in sites:
        path = Path(s["data_path"])
        assert path.is_relative_to(task_dir / "sites"), f"{path}: site file outside sites/"
        a = ad.read_h5ad(path, backed="r")
        try:
            assert set(a.obs.columns) <= SITE_OBS, f"{path}: obs columns {list(a.obs.columns)}"
            assert a.var.shape[1] == 0, f"{path}: var columns {list(a.var.columns)}"
            filled = [k for k in SIDE_CHANNELS if len(getattr(a, k))]
            assert not filled, f"{path}: side channels {filled}"
            universe |= set(a.var_names)
        finally:
            a.file.close()
    planted_genes = [g for g in _string_lists(load_truth(task_dir)) if g <= universe]
    assert planted_genes, f"{task_dir}: no planted gene list to check against"
    hidden = {_sha256(p) for p in task_dir.rglob("*")
              if p.is_file() and not any(p.is_relative_to(v) for v in visible)}
    exposed = {Path(s["data_path"]) for s in sites} | {Path(p) for p in case.resources.values()}
    exposed |= {p for v in visible for p in v.rglob("*") if p.is_file()}
    for p in sorted(exposed):
        assert not p.is_relative_to(task_dir) or any(p.is_relative_to(v) for v in visible), \
            f"{p}: exposed but outside public/ and sites/"
        assert _sha256(p) not in hidden, f"{p}: a copy of a file outside public/ and sites/"
        if p.suffix == ".json":
            doc = json.loads(p.read_text())
            leaked = _keys(doc) & ANSWER_KEYS
            assert not leaked, f"{p}: answer keys {sorted(leaked)}"
            for strings in _string_lists(doc):
                assert all(len(strings & g) < len(g) / 2 for g in planted_genes), \
                    f"{p}: lists planted genes"


def _onek1k_like(path: Path) -> Path:
    """OneK1K's CELLxGENE layout in miniature: 64 donors x 6 case populations x 30 cells plus
    platelets (not a case population), float32 Poisson counts, an Ensembl var index with a
    categorical feature_name, and the metadata a real file carries beyond the whitelist (sex,
    age, pool, Azimuth labels, an embedding, uns), so a leaked column would show."""
    from agentcoop.cases import planted

    rng = np.random.default_rng(0)
    first_label: dict[str, str] = {}
    for label, population in planted.CELL_TYPES.items():
        first_label.setdefault(population, label)
    labels = [*first_label.values(), "platelet"]
    rows = [(f"{d}_{d}", label) for d in range(64) for label in labels
            for _ in range(5 if label == "platelet" else 30)]
    obs = pd.DataFrame(rows, columns=["donor_id", "cell_type"])
    donors = obs["donor_id"].unique()
    obs["assay"] = "10x 3' v2"
    obs["sex"] = obs["donor_id"].map(dict(zip(donors, rng.choice(["female", "male"], 64))))
    obs["age"] = obs["donor_id"].map(dict(zip(donors, rng.integers(20, 90, 64)))).astype(int)
    obs["pool_number"] = obs["donor_id"].map(dict(zip(donors, rng.integers(1, 9, 64)))).astype(str)
    obs["predicted.celltype.l2"] = obs["cell_type"].map(planted.CELL_TYPES).fillna("Platelet")
    obs = obs.astype({c: "category" for c in obs.columns if c != "age"})
    obs.index = [f"bc{i}" for i in range(len(obs))]
    var = pd.DataFrame({"feature_name": pd.Categorical(GENES)},
                       index=[f"ENSG{i:011d}" for i in range(len(GENES))])
    X = sp.csr_matrix(rng.poisson(rng.uniform(1, 4, len(GENES)), (len(obs), len(GENES))),
                      dtype=np.float32)
    ad.AnnData(X=X, obs=obs, var=var, obsm={"X_umap": rng.normal(size=(len(obs), 2))},
               uns={"title": "synthetic OneK1K", "schema_version": "5.0.0"}).write_h5ad(path)
    return path


def _libraries(root: Path) -> tuple[Path, Path]:
    """Hallmark and GO BP in Enrichr text format over GENES: 12 genes per program, all annotated
    to the family's first defining GO term, so every gene of a program is a core gene."""
    from agentcoop.cases import planted

    genes = iter(GENES)
    hallmark, go = [], []
    for i, spec in enumerate(planted.FAMILIES.values()):
        members = []
        for program in spec["programs"]:
            core = [next(genes) for _ in range(12)]
            hallmark.append("\t".join([program, "", *core]))
            members += core
        go.append("\t".join([f"{spec['go_terms'][0]} (GO:{i:07d})", "", *members]))
    (root / "hallmark.txt").write_text("\n".join(hallmark))
    (root / "go.txt").write_text("\n".join(go))
    return root / "hallmark.txt", root / "go.txt"


@pytest.fixture(scope="module")
def planted_suite(tmp_path_factory) -> list[Path]:
    """One task per knob cell, built the way real suites are (prepare_base on a OneK1K-like
    source, then the evaluation-side build) under a temporary TASKS root."""
    from agentcoop.cases import planted
    from agentcoop.eval import planted as grader

    root = tmp_path_factory.mktemp("planted")
    hallmark, go = _libraries(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(planted, "MIN_DONOR_CELLS", 100)    # the source has 180 case cells per donor
        mp.setattr(planted, "BASE", root / "base.h5ad")
        mp.setattr(planted, "HALLMARK", hallmark)
        mp.setattr(planted, "GO_BP", go)
        mp.setattr(planted, "TASKS", root / "tasks")
        planted.prepare_base(_onek1k_like(root / "onek1k.h5ad"), root / "base.h5ad", n_donors=64)
        yield grader.build("inv", [0, 1, 2, 3], pool="dev")


def test_planted_tasks_expose_only_whitelisted_columns_and_no_answer(planted_suite):
    assert len(planted_suite) == 4
    for task_dir in planted_suite:
        assert_planted_task_boundary(task_dir)


def test_the_boundary_check_catches_a_leaked_column_and_a_copied_answer(planted_suite, tmp_path):
    leaked = shutil.copytree(planted_suite[0], tmp_path / "leaked")
    site = leaked / "sites" / "A.h5ad"
    a = ad.read_h5ad(site)
    a.obs["donor"] = "D00"
    a.write_h5ad(site)
    with pytest.raises(AssertionError, match="obs columns"):
        assert_planted_task_boundary(leaked)

    copied = shutil.copytree(planted_suite[0], tmp_path / "copied")
    shutil.copy(copied / "hidden" / "truth.json", copied / "public" / "notes.json")
    with pytest.raises(AssertionError, match="notes.json"):
        assert_planted_task_boundary(copied)


# ---------------------------------------------------------------------------
# 3. Workspaces under WS_ROOT, never inside the repo
# ---------------------------------------------------------------------------

def test_workspace_root_is_outside_the_repo(tmp_path):
    from agentcoop.agents import workspace

    assert not workspace.WS_ROOT.resolve().is_relative_to(ROOT.resolve()), workspace.WS_ROOT
    home = tmp_path / "home"
    env = {k: v for k, v in os.environ.items() if not k.startswith("AGENTCOOP_")}
    printed = subprocess.run(
        [sys.executable, "-c", "from agentcoop.agents.workspace import WS_ROOT; print(WS_ROOT)"],
        cwd=ROOT, env={**env, "HOME": str(home)}, capture_output=True, text=True, check=True)
    assert Path(printed.stdout.strip()) == home / "agentcoop_ws"


def test_site_workspaces_are_created_under_ws_root(tmp_path, monkeypatch):
    from agentcoop.agents import workspace
    from agentcoop.cases.planted import CASE_VOCAB
    from agentcoop.coop.site import CaseSpec, make_sites
    from agentcoop.coop.tester import TesterParams

    root = tmp_path / "ws_root"
    monkeypatch.setattr(workspace, "WS_ROOT", root)
    silo = tmp_path / "silo"
    silo.mkdir()
    (silo / "symbols.tsv").write_text("ensembl_id\tsymbol\n")
    case = CaseSpec(name="planted", kind="scrna", vocab=CASE_VOCAB, question="q", default_spec={},
                    tester=TesterParams(), resources={"symbols.tsv": silo / "symbols.tsv"})
    specs = []
    for name in ("A", "M"):
        (silo / f"{name}.h5ad").write_bytes(name.encode())
        specs.append({"name": name, "data_path": silo / f"{name}.h5ad", "card": {"site": name},
                      "proposes": name != "M", "agent": "codex"})
    for site in make_sites(specs, case, task_id="t1", run_id="t1/ours/s0"):
        assert site.ws.is_dir() and site.ws.is_relative_to(root), site.ws
        assert not site.ws.resolve().is_relative_to(ROOT.resolve()), site.ws
