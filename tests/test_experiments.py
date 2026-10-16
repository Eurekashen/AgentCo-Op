"""experiments/ and the code they add to the arms and the grader: recorded views, runner
bookkeeping, grading on every labelled claim, paired statistics, view-error rates and the
headroom gate.
Offline: fakes, synthetic records and one planted task built by Task 8's generator on its fake
base (no API key, no agent, no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from agentcoop import store
from agentcoop.cases import planted
from agentcoop.coop import arms
from agentcoop.coop.arms import ArmResult
from agentcoop.coop.claims import claim_to_dict
from agentcoop.coop.protocol import ProtocolRun
from agentcoop.coop.tester import TestResult
from agentcoop.coop.view import load_view, write_view
from agentcoop.eval import planted as grader
from tests.fakes import (PROGRAMS, FakeMemo, SiteFactory, case_spec, claim, fake_workspace,
                         site_specs, standard_sites)
from tests.test_planted import _fake_base, _fake_libraries

ROOT = Path(__file__).resolve().parents[1]
IFN = sorted(PROGRAMS["IFN"])


def _script(name: str):
    """experiments/ is not a package: load a script as a module under a unique name (registered
    in sys.modules, which dataclasses need to resolve the script's annotations)."""
    spec = importlib.util.spec_from_file_location(f"experiments_{name}",
                                                  ROOT / "experiments" / f"{name}.py")
    module = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------- arms: recorded views
# (MA-genome's min_sites=1 single-site mode, which the headroom gate uses, is Task 9's and is
# tested in tests/test_arms.py.)

def test_arms_record_the_view_each_site_used(tmp_path, monkeypatch):
    """single / single_codex: the site's step-1 view; ma_genome: the self-checked views; ours:
    the views after the protocol. A site without a view id (here M) records None."""
    factory = SiteFactory(standard_sites())

    def make_sites(specs, case, *, task_id, run_id, agents=None):
        sites = factory(specs, case, task_id=task_id, run_id=run_id, agents=agents)
        for s in sites[:3]:
            s.view_blob = f"{run_id.split('/')[1]}-{s.name}"
        return sites

    monkeypatch.setattr(arms, "make_sites", make_sites)
    monkeypatch.setattr(arms.store, "memo", FakeMemo())
    monkeypatch.setattr(arms, "new_workspace", fake_workspace(tmp_path / "ws"))
    monkeypatch.setattr(arms, "Coordinator", lambda case: None)
    monkeypatch.setattr(arms, "run_protocol",
                        lambda sites, case, **kw: ProtocolRun([], [], 0, 0.0))
    out = arms.run_arms(["single", "single_codex", "ma_genome", "ours"], site_specs(tmp_path),
                        case_spec(), task_id="t0", sample=0)
    assert out["single:B"].meta["views"] == {"B": "base-B"}
    assert out["single_codex:C"].meta["views"] == {"C": "single_codex-C"}
    assert out["ma_genome"].meta["views"] == {"A": "ma-A", "B": "ma-B", "C": "ma-C", "M": None}
    assert "tables" in out["ma_genome"].meta
    assert out["ours"].meta["views"] == {"A": "ours-A", "B": "ours-B", "C": "ours-C", "M": None}


# ---------------------------------------------------------------------------- run.py

run = _script("run")

SITES = [{"name": "A", "data_path": "a.h5ad", "card": {}, "proposes": True},
         {"name": "B", "data_path": "b.h5ad", "card": {}, "proposes": True},
         {"name": "M", "data_path": "m.h5ad", "card": {}, "proposes": False}]


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """run.py on a fake suite of two tasks; the fake run_arms records every call."""
    for task in ("fx-000", "fx-001"):
        (tmp_path / "tasks" / "fx" / task).mkdir(parents=True)
    calls = []

    def fake_run_arms(arm_names, specs, case, *, task_id, sample):
        calls.append((task_id, sample, list(arm_names), {s["name"]: s["agent"] for s in specs}))
        out = {}
        for arm in arm_names:
            keys = ([f"{arm}:{s['name']}" for s in specs if s["proposes"]]
                    if arm in ("single", "single_codex") else [arm])     # Task 9's key scheme
            for key in keys:
                out[key] = ArmResult(arm=key, claims=[claim(f"{key}:1", "A", tuple(IFN[:3]))],
                                     labels={}, causes={}, scopes={}, calls=2, nominal_usd=0.01,
                                     meta={"k": 1})
        return out

    monkeypatch.setattr(run, "TASKS", tmp_path / "tasks")
    monkeypatch.setattr(run, "OUT", tmp_path / "out")
    monkeypatch.setattr(run, "load_case", lambda d: (case_spec(), [dict(s) for s in SITES]))
    monkeypatch.setattr(run, "run_arms", fake_run_arms)
    monkeypatch.setenv("AGENTCOOP_BUDGET_USD", "1")
    monkeypatch.setenv("AGENTCOOP_LLM_DB", str(tmp_path / "llm.sqlite"))
    return calls


def test_run_writes_one_file_per_arm_and_skips_existing(runner, tmp_path):
    out = tmp_path / "out" / "fx"
    (out / "fx-000").mkdir(parents=True)
    (out / "fx-000" / "ours__s0.json").write_text('{"old": true}')
    run.main(["--suite", "fx", "--arms", "single", "ours", "--parallel", "1"])

    assert sorted(p.name for p in (out / "fx-000").iterdir()) == [
        "ours__s0.json", "single-A__s0.json", "single-B__s0.json"]
    assert json.loads((out / "fx-000" / "ours__s0.json").read_text()) == {"old": True}
    rec = json.loads((out / "fx-001" / "single-A__s0.json").read_text())
    assert (rec["arm"], rec["task"], rec["sample"], rec["calls"]) == ("single:A", "fx-001", 0, 2)
    assert rec["claims"][0]["id"] == "single:A:1" and rec["claims"][0]["up_genes"] == IFN[:3]
    assert rec["meta"] == {"k": 1}
    assert rec["agents"] == {"A": "codex", "B": "finch", "M": "codex"}
    assert sorted((t, a) for t, _, a, _ in runner) == [
        ("fx-000", ["single"]), ("fx-001", ["single", "ours"])]
    assert json.loads((out / "agents.json").read_text()) == rec["agents"]

    runner.clear()
    run.main(["--suite", "fx", "--arms", "single", "ours"])
    assert runner == []                                   # everything exists: nothing re-run


def test_site_agents_default_alternates_proposers_and_gives_testers_codex():
    """load_case's site specs carry no "agent" key; the runner adds it before run_arms."""
    specs = [{"name": n, "proposes": n != "M"} for n in ("A", "B", "C", "M")]
    assert run.site_agents(specs, {}) == {"A": "codex", "B": "finch", "C": "codex", "M": "codex"}
    assert run.site_agents(specs, {"M": "finch"})["M"] == "finch"


def test_run_draws_agent_overrides_and_guards(runner, tmp_path, monkeypatch):
    run.main(["--suite", "fx", "--tasks", "fx-001", "--arms", "vote", "--draws", "0", "1",
              "--agents", "B=codex"])
    assert [(t, s) for t, s, _, _ in runner] == [("fx-001", 0), ("fx-001", 1)]
    assert runner[0][3] == {"A": "codex", "B": "codex", "M": "codex"}
    assert (tmp_path / "out" / "fx" / "fx-001" / "vote__s1.json").exists()

    with pytest.raises(SystemExit, match="fixed per suite"):
        run.main(["--suite", "fx", "--arms", "vote"])      # default assignment differs
    with pytest.raises(SystemExit):
        run.main(["--suite", "fx", "--arms", "vote", "--agents", "Z=codex"])
    with pytest.raises(SystemExit):
        run.main(["--suite", "fx", "--tasks", "fx-999", "--arms", "vote"])
    monkeypatch.delenv("AGENTCOOP_BUDGET_USD")
    with pytest.raises(SystemExit, match="AGENTCOOP_BUDGET_USD"):
        run.main(["--suite", "fx", "--arms", "vote", "--agents", "B=codex"])


def test_run_failed_task_stores_nothing_and_the_others_finish(runner, tmp_path, monkeypatch):
    """A harness failure inside run_arms (Review Focus 3) must not become a stored result."""
    fake = run.run_arms

    def flaky(arm_names, specs, case, *, task_id, sample):
        if task_id == "fx-000":
            raise RuntimeError("codex timed out")
        return fake(arm_names, specs, case, task_id=task_id, sample=sample)

    monkeypatch.setattr(run, "run_arms", flaky)
    with pytest.raises(SystemExit, match="1 task"):
        run.main(["--suite", "fx", "--arms", "vote"])
    assert not (tmp_path / "out" / "fx" / "fx-000").exists()
    assert (tmp_path / "out" / "fx" / "fx-001" / "vote__s0.json").exists()


# ------------------------------------------------- analyze.py and eval.planted.view_errors

analyze = _script("analyze")

OURS = [0.8, 0.7, 0.6, 0.9, 0.5, 0.75]
MA = [0.6, 0.7 + 1e-12, 0.65, 0.5, 0.5, 0.45]      # task 1: tie within 1e-9; task 4: exact tie
ORACLE = [0.9, 0.9, 0.8, 1.0, 0.7, 0.85]
SINGLE_A = [0.5, 0.6, 0.3, 0.4, 0.2, 0.5]
SINGLE_B = [0.4, 0.65, 0.5, 0.6, 0.3, 0.35]


def test_analyze_paired_wilcoxon_and_silo_cost_recovery(tmp_path, monkeypatch):
    """Each stored result carries its F1 in its claim's effect, so a fake grader returns known
    scores and the statistics can be checked by hand."""
    scores = {"ours": OURS, "ma_genome": MA, "oracle": ORACLE, "single:A": SINGLE_A,
              "single:B": SINGLE_B}
    for i in range(6):
        task_dir = tmp_path / "out" / "fx" / f"fx-{i:03d}"
        task_dir.mkdir(parents=True)
        for arm, f1 in scores.items():
            rec = {"arm": arm, "task": task_dir.name, "sample": 0,
                   "claims": [claim_to_dict(claim("c", "A", tuple(IFN[:3]), effect=f1[i]))],
                   "labels": {}, "causes": {}, "scopes": {}, "calls": 3, "nominal_usd": 0.02,
                   "meta": {}}
            (task_dir / f"{arm.replace(':', '-')}__s0.json").write_text(json.dumps(rec))

    def fake_grade(claims, truth, *, labels=None, causes=None, scopes=None):
        f1 = claims[0].effect
        return {"precision": f1, "recall": f1, "f1": f1, "n_claims": len(claims),
                "artefact_rejected": None, "scope_accuracy": None, "cause_accuracy": None}

    monkeypatch.setattr(analyze, "OUT", tmp_path / "out")
    monkeypatch.setattr(analyze, "load_truth", lambda d: {})
    monkeypatch.setattr(analyze, "grade", fake_grade)
    analyze.main(["--suite", "fx"])
    res = json.loads((tmp_path / "out" / "fx_analysis.json").read_text())

    vs_ma = res["paired_f1_vs_ours"]["ma_genome"]
    assert (vs_ma["n"], vs_ma["wins"], vs_ma["losses"], vs_ma["ties"]) == (6, 3, 1, 2)
    nonzero = [0.8 - 0.6, 0.6 - 0.65, 0.9 - 0.5, 0.75 - 0.45]
    assert vs_ma["p"] == pytest.approx(stats.wilcoxon(nonzero).pvalue)
    assert vs_ma["p"] == pytest.approx(0.25)              # exact: W- = 1 of 16 sign patterns x 2

    best = np.maximum(SINGLE_A, SINGLE_B)                 # per task, not the better site's mean
    assert res["arms"]["best_single"]["f1"] == pytest.approx(best.mean())
    rec = res["silo_cost_recovery_f1"]
    assert rec["value"] == pytest.approx((np.mean(OURS) - best.mean())
                                         / (np.mean(ORACLE) - best.mean()))
    assert rec["value"] == pytest.approx(0.2 / 0.35)
    assert len(rec["ci95"]) == 2 and rec["n"] == 6
    assert [m["arm"] for m in res["mechanism"]] == ["ours", "ma_genome"]
    assert res["mechanism"][1]["vs_ours"]["wins"] == 3
    assert res["view_errors"] == [] and res["views"] == []   # no arm recorded a view

    same = {t: {"ours": {"f1": 0.5}, "vote": {"f1": 0.5}} for t in ("a", "b")}
    assert analyze.paired(same, "ours", "vote") == {"n": 2, "wins": 0, "losses": 0, "ties": 2,
                                                    "mean_diff": 0.0, "p": None}


def test_analyze_grades_causes_over_every_claim_ours_labelled(monkeypatch):
    """Ours' findings are its corroborated claims only; the cause it gave a demoted claim is
    scored from meta["all_claims"], with the real grader, while precision and recall stay over
    the findings."""
    p1, art = [f"G{i:03d}" for i in range(10)], [f"G{i:03d}" for i in range(30, 40)]
    truth = {"programs": [
        {"id": "P1", "kind": "reproducible", "cell_types": ["NK"], "up_genes": p1,
         "down_genes": [], "sites": ["A", "B", "C"], "shared_with_mimic": True},
        {"id": "P4", "kind": "artefact", "cell_types": ["CD4 T"], "up_genes": art,
         "down_genes": [], "sites": ["C"], "shared_with_mimic": False}],
        "composition": {"id": "C1", "kind": "composition", "population": "CD16 Mono",
                        "sign": -1, "sites": ["A", "B", "C"]},
        "artefact_site": "C"}
    found = claim("A:1", "A", tuple(p1), population="NK")
    demoted = claim("C:1", "C", tuple(art), population="CD4 T")
    cause = {"cause": "site-artefact", "site": "C", "defect": "assay differs by group"}
    rec = {"arm": "ours", "claims": [claim_to_dict(found)],
           "labels": {"A:1": "corroborated", "C:1": "inconclusive"}, "causes": {"C:1": cause},
           "scopes": {}, "calls": 5, "nominal_usd": 0.1,
           "meta": {"errors": [], "all_claims": [
               {"claim": claim_to_dict(found), "label": "corroborated", "cause": None,
                "scope": None},
               {"claim": claim_to_dict(demoted), "label": "inconclusive", "cause": cause,
                "scope": None}]}}
    monkeypatch.setattr(analyze, "load_truth", lambda d: truth)
    ours = analyze.grade_task(Path("fx-000"), [rec])["ours"]
    assert ours["cause_accuracy"] == 1.0                  # the demoted claim's cause counts
    assert (ours["precision"], ours["recall"], ours["n_claims"]) == (1.0, 0.5, 1)
    assert ours["artefact_rejected"] is True
    findings_only = {**rec, "meta": {}}                  # the old reading: findings only
    assert analyze.grade_task(Path("fx-000"), [findings_only])["ours"]["cause_accuracy"] is None


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One planted task with both knobs on, built by Task 8's generator on its fake base and
    libraries (seed 22 is the (True, True) seed of Task 8's own suite). Yielded inside the patch,
    so load_case still finds the fake gene-set library."""
    root = tmp_path_factory.mktemp("built")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(planted, "BASE", _fake_base(root))
        hallmark, go = _fake_libraries(root)
        mp.setattr(planted, "HALLMARK", hallmark)
        mp.setattr(planted, "GO_BP", go)
        mp.setattr(planted, "TASKS", root / "tasks")
        [task_dir] = planted.build_suite("hx", [22], pool="dev", knobs=[(True, True)],
                                         seal=grader.write_truth)
        yield task_dir


def _true_view(task_dir: Path, site: str, root: Path) -> None:
    """The correct view of a planted site, written from the answer key (the test's oracle)."""
    import anndata as ad

    m = grader.load_truth(task_dir)["sites"][site]
    a = ad.read_h5ad(task_dir / "sites" / f"{site}.h5ad")
    table = pd.read_csv(task_dir / "public" / "symbols.tsv", sep="\t", dtype=str)
    column = {"symbol": "symbol", "ensembl": "ensembl_id", "alias": "alias"}[m["scheme"]]
    symbol_of = dict(zip(table[column], table["symbol"]))
    group_of = {d: g for g, donors in m["donors"].items() for d in donors}
    samples = pd.DataFrame([{"sample": s, "donor": d, "group": group_of[d]}
                            for s, d in sorted(m["samples"].items())])
    cells = pd.DataFrame({"cell": a.obs_names, "sample": a.obs["sample"].astype(str).to_numpy(),
                          "population": a.obs["cell_label"].astype(str).map(m["labels"])})
    write_view(root, genes={v: symbol_of[v] for v in a.var_names}, samples=samples,
               cells=cells.reset_index(drop=True), spec={"unit": "donor"})


def _load(root: Path):
    view, errors = load_view(root, kind="scrna", vocab=planted.CASE_VOCAB)
    assert view is not None, errors
    return view


def test_view_errors_count_wrong_and_missing_items(built, tmp_path):
    truth = grader.load_truth(built)
    site = next(s for s in "ABC" if truth["sites"][s]["scheme"] == "ensembl")
    _true_view(built, site, tmp_path / "good")
    good = _load(tmp_path / "good")
    assert set(grader.view_errors(good, built, site).values()) == {0.0}

    ids = list(good.genes)
    genes = {i: good.genes[i] for i in ids[30:]}                     # 30 ids left out
    genes.update({i: "WRONG" + good.genes[i] for i in ids[30:45]})   # 15 ids misnamed
    samples = good.samples.copy()
    samples.loc[0, "group"] = "treated" if samples.loc[0, "group"] == "control" else "control"
    cells = good.cells.copy()
    nk = cells["population"] == "NK"
    cells.loc[nk, "population"] = "B"                                # NK cells called B
    write_view(tmp_path / "bad", genes=genes, samples=samples, cells=cells.iloc[10:],
               spec={"unit": "donor"})                               # 10 cells left out
    e = grader.view_errors(_load(tmp_path / "bad"), built, site)
    n_genes, n_cells, relabelled = len(ids), len(cells), int(nk.iloc[10:].sum())
    assert (e["genes_missing"], e["genes_wrong"]) == (30 / n_genes, 15 / n_genes)
    assert (e["groups_wrong"], e["groups_missing"]) == (1 / 8, 0.0)
    assert (e["populations_missing"], e["populations_wrong"]) == (10 / n_cells,
                                                                  relabelled / n_cells)
    assert e["error_rate"] == pytest.approx(max(45 / n_genes, 1 / 8,
                                                (10 + relabelled) / n_cells))


def test_analyze_restores_recorded_views_and_tabulates_their_errors(built, tmp_path,
                                                                    monkeypatch):
    monkeypatch.setattr(store, "STORE", tmp_path / "store")
    _true_view(built, "A", tmp_path / "good")
    good = _load(tmp_path / "good")
    samples = good.samples.copy()
    samples.loc[0, "group"] = "treated" if samples.loc[0, "group"] == "control" else "control"
    write_view(tmp_path / "bad", genes=good.genes, samples=samples, cells=good.cells,
               spec={"unit": "donor"})
    (tmp_path / "invalid").mkdir()
    (tmp_path / "invalid" / "spec.json").write_text("{}")             # no view files
    good_id, bad_id, invalid_id = (store.put_blob(tmp_path / d) for d in ("good", "bad", "invalid"))
    agents = {"A": "finch", "B": "codex", "C": "finch", "M": "codex"}
    results = [{"arm": arm, "agents": agents, "meta": meta} for arm, meta in [
        ("single:A", {"views": {"A": good_id}}),
        ("single:B", {"views": {"B": invalid_id}}),
        ("single_codex:A", {"views": {"A": good_id}}),
        ("ma_genome", {"views": {"A": good_id, "B": None}}),
        ("ours", {"views": {"A": bad_id}}),
        ("ours", {"views": {"A": bad_id}}),                             # a second draw, same view
        ("union", {})]]

    rows = analyze.view_rows(built, results)
    assert len(rows) == 6 and {(r["stage"], r["site"], r["agent"], r["valid"],
                                r.get("error_rate")) for r in rows} == {
        ("step 1", "A", "finch", True, 0.0),
        ("step 1", "B", "codex", False, None),
        ("step 1 (all Codex)", "A", "codex", True, 0.0),
        ("self-checked", "A", "finch", True, 0.0),
        ("self-checked", "B", "codex", False, None),
        ("repaired", "A", "finch", True, 0.125)}
    assert all(r["ids_labels"] and r["artefact"] and r["task"] == built.name for r in rows)
    summary = {(s["stage"], s["agent"]): (s["n_views"], s["n_invalid"], s["error_rate"],
                                          s["any_error"])
               for s in analyze.view_error_summary(rows)}
    assert summary == {("step 1", "finch"): (1, 0, 0.0, 0.0),
                       ("step 1", "codex"): (1, 1, None, None),
                       ("step 1 (all Codex)", "codex"): (1, 0, 0.0, 0.0),
                       ("self-checked", "finch"): (1, 0, 0.0, 0.0),
                       ("self-checked", "codex"): (1, 1, None, None),
                       ("repaired", "finch"): (1, 0, 0.125, 1.0)}
    assert analyze.view_rows(built, results[-1:]) == []
