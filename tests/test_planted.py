"""Planted suite: base preparation, task generation, case spec and the grader. Offline: tiny
synthetic data and fake libraries stand in for OneK1K, HGNC, Hallmark and GO."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from agentcoop.cases import planted
from agentcoop.coop import tester
from agentcoop.coop.claims import Claim
from agentcoop.eval import planted as grader

ROOT = Path(__file__).resolve().parents[1]
GENES = [f"G{i:03d}" for i in range(300)]


# ---------------------------------------------------------------------------
# Base preparation
# ---------------------------------------------------------------------------

def _onek1k_like(path: Path) -> None:
    """OneK1K's layout in miniature: float32 CSR counts, Ensembl var index plus a categorical
    feature_name (one ambiguous symbol, one gene never detected), categorical donor_id /
    cell_type with labels outside the case populations. Gene MARK holds row index + 1."""
    rng = np.random.default_rng(0)
    labels = [*planted.CELL_TYPES, "platelet", "plasmablast"]
    rows = []
    for d in range(12):
        for label in labels:
            if d == 1 and planted.CELL_TYPES.get(label) == "CD16 Mono":
                continue                      # D01 lacks a population
            rows += [(f"D{d:02d}", label)] * (3 if d == 0 else 12)   # D00 has too few cells
    obs = pd.DataFrame(rows, columns=["donor_id", "cell_type"]).astype("category")
    obs.index = [f"bc{i}" for i in range(len(obs))]
    symbols = ["MARK", "DUP", "DUP", "NEVER", *[f"G{i:02d}" for i in range(1, 17)]]
    counts = rng.poisson(1.0, size=(len(obs), len(symbols))).astype(np.float32)
    counts[:, 0] = np.arange(len(obs)) + 1
    counts[:, 3] = 0
    var = pd.DataFrame({"feature_name": pd.Categorical(symbols)},
                       index=[f"ENSG{i:011d}" for i in range(len(symbols))])
    ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var).write_h5ad(path)


def _hgnc(path: Path) -> Path:
    """HGNC's complete-set layout (pipe lists quoted, empty fields empty) for genes G01-G06 of
    _onek1k_like (Ensembl index = number + 3), plus one approved gene outside the base."""
    rows = [("HGNC:1", "G01", "AKA1", '"OLD1|OLD1B"', "ENSG00000000004"),   # previous symbol first
            ("HGNC:2", "G02", '"AKA2|AKA2B"', "", "ENSG00000000005"),       # else first alias
            ("HGNC:3", "G03", "SHARED", "", "ENSG00000000006"),             # alias of two genes
            ("HGNC:4", "G04", '"shared|AKA4"', "", "ENSG00000000007"),
            ("HGNC:5", "G05", "AKA5", "g06", "ENSG00000000008"),            # a base symbol
            ("HGNC:6", "G06", "APPROVED", "", "ENSG00000000009"),           # an approved symbol
            ("HGNC:7", "APPROVED", "", "", "ENSG99999999999")]
    lines = ["hgnc_id\tsymbol\talias_symbol\tprev_symbol\tensembl_gene_id",
             *["\t".join(r) for r in rows]]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_prepare_base_streams_eligible_donors_and_freezes_the_subset(tmp_path, monkeypatch):
    src, dst = tmp_path / "onek1k.h5ad", tmp_path / "base.h5ad"
    _onek1k_like(src)
    monkeypatch.setattr(planted, "HGNC", _hgnc(tmp_path / "hgnc.txt"))
    monkeypatch.setattr(planted, "MIN_DONOR_CELLS", 100)
    monkeypatch.setattr(planted, "MIN_POP_CELLS", 10)
    monkeypatch.setattr(planted, "MAX_DONOR_CELLS", 150)
    monkeypatch.setattr(planted, "ROW_BLOCK", 37)          # many blocks, selections straddle them
    meta = planted.prepare_base(src, dst, n_donors=8, seed=0)

    assert len(meta["dev_donors"]) == len(meta["test_donors"]) == 4
    assert not set(meta["dev_donors"]) & set(meta["test_donors"])
    assert not {"D00", "D01"} & set(meta["dev_donors"] + meta["test_donors"])
    assert json.loads(dst.with_suffix(".json").read_text()) == meta
    assert meta["sha256"] == hashlib.sha256(dst.read_bytes()).hexdigest()
    assert meta["hgnc_sha256"] == hashlib.sha256((tmp_path / "hgnc.txt").read_bytes()).hexdigest()

    base = ad.read_h5ad(dst)
    assert list(base.obs.columns) == ["donor", "population"]
    assert sp.issparse(base.X) and base.X.dtype == np.int32
    assert not {"DUP", "NEVER"} & set(base.var_names)
    assert base.var.loc["G01", "gene_ids"] == "ENSG00000000004"
    expected = {"G01": "OLD1", "G02": "AKA2", "G04": "AKA4", "G05": "AKA5"}
    assert dict(base.var["alias"]) == {g: expected.get(g, g) for g in base.var_names}
    assert meta["n_aliased"] == 4 and base.var["alias"].str.upper().is_unique
    per = base.obs.groupby(["donor", "population"], observed=True).size().unstack()
    assert sorted(per.index) == sorted(meta["dev_donors"] + meta["test_donors"])
    assert set(per.columns) == set(planted.POPULATIONS)
    assert (per.sum(1) <= 150).all() and (per.min(1) >= 10).all()

    # every kept cell is an intact copy of its source row (MARK identifies the row)
    source = ad.read_h5ad(src)
    rows = base[:, "MARK"].X.toarray().ravel() - 1
    np.testing.assert_array_equal(base.X.toarray(),
                                  source[rows, list(base.var["gene_ids"])].X.toarray())
    np.testing.assert_array_equal(base.obs["donor"].astype(str),
                                  source.obs["donor_id"].astype(str).to_numpy()[rows])
    np.testing.assert_array_equal(
        base.obs["population"].astype(str),
        source.obs["cell_type"].astype(str).map(planted.CELL_TYPES).to_numpy()[rows])

    with pytest.raises(FileExistsError):
        planted.prepare_base(src, dst, n_donors=8, seed=0)


# ---------------------------------------------------------------------------
# Scope table and rule
# ---------------------------------------------------------------------------

def _claim(cid: str = "c1", population: str = "NK", up=(), down=(), *, kind: str = "de",
           target: str = "treated", effect: float | None = None) -> Claim:
    return Claim(id=cid, site="A", kind=kind, population=population, target=target,
                 reference="control", up_genes=tuple(up), down_genes=tuple(down), effect=effect,
                 n_units=8, rank=1)


def _result(status: str, test: str = "scope") -> tester.TestResult:
    return tester.TestResult(claim_id="c1", site="M", test=test, status=status, effect=None,
                             se=None, ci95=None, n_per_group=None, mde=None, p=None,
                             competitive_ok=None, reason="")


class _Site:
    def __init__(self, name: str, mimic: bool) -> None:
        self.name, self._mimic = name, mimic

    def holds(self, population: str, contrast: tuple[str, str]) -> bool:
        return self._mimic or contrast == ("treated", "control")


def test_scope_tests_send_treated_claims_to_sites_holding_the_mimic_contrast():
    a, b, m = _Site("A", False), _Site("B", False), _Site("M", True)
    assert planted.scope_tests(_claim(up=GENES[:3]), [a, b, m]) == [(m, ("mimic", "control"))]
    assert planted.scope_tests(_claim(target="mimic", up=GENES[:3]), [a, b, m]) == []


def test_scope_rule_reads_only_scope_tests_and_never_counts_untestable():
    claim = _claim(up=GENES[:3])

    def rule(*statuses: str) -> str:
        return planted.scope_rule(claim, [_result(s) for s in statuses])

    assert rule("supported", "absent") == "shared"
    assert rule("absent") == rule("contradicted", "untestable") == "specific"
    assert rule("absent", "inconclusive") == "inconclusive"
    assert rule("untestable") == rule() == "inconclusive"
    assert planted.scope_rule(claim, [_result("supported", test="replication"),
                                      _result("absent")]) == "specific"


def test_the_case_module_has_no_path_to_answers():
    text = (ROOT / "agentcoop" / "cases" / "planted.py").read_text()
    for needle in ("agentcoop.eval", "load_truth", "truth.json", "hidden/", '"hidden"'):
        assert needle not in text, needle


# ---------------------------------------------------------------------------
# Task generation (one small suite, built once for the module)
# ---------------------------------------------------------------------------

SEEDS = [19, 27, 21, 22]   # KNOB_CELLS in this order; together they plant every program kind
SITES = ("A", "B", "C", "M")
ALIASES = [f"OLD{i:03d}" if i % 2 else g for i, g in enumerate(GENES)]   # base var["alias"]


def _fake_base(root: Path) -> Path:
    """40 donors x 6 populations x 30 cells x 300 genes of Poisson counts, in base layout."""
    rng = np.random.default_rng(1)
    donors = [f"D{i:02d}" for i in range(40)]
    obs = pd.DataFrame([(d, p) for d in donors for p in planted.POPULATIONS for _ in range(30)],
                       columns=["donor", "population"]).astype("category")
    obs.index = obs.index.astype(str)
    X = sp.csr_matrix(rng.poisson(rng.uniform(1, 4, len(GENES)), (len(obs), len(GENES))),
                      dtype=np.int32)
    var = pd.DataFrame({"gene_ids": [f"ENSG{i:011d}" for i in range(len(GENES))],
                        "alias": ALIASES}, index=GENES)
    path = root / "base.h5ad"
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(path)
    path.with_suffix(".json").write_text(json.dumps({"dev_donors": donors, "test_donors": [],
                                                     "sha256": "f" * 64}))
    return path


def _fake_libraries(root: Path) -> tuple[Path, Path]:
    """Hallmark and GO in Enrichr text format: 12 genes per program, all annotated to the
    family's first defining GO term, so every gene is a core gene."""
    genes = iter(GENES)
    hallmark, go = [], []
    for spec in planted.FAMILIES.values():
        members = []
        for program in spec["programs"]:
            core = [next(genes) for _ in range(12)]
            hallmark.append("\t".join([program, "", *core]))
            members += core
        go.append("\t".join([f"{spec['go_terms'][0]} (GO:{len(go):07d})", "", *members]))
    (root / "hallmark.txt").write_text("\n".join(hallmark))
    (root / "go.txt").write_text("\n".join(go))
    return root / "hallmark.txt", root / "go.txt"


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("planted")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(planted, "BASE", _fake_base(root))
        hallmark, go = _fake_libraries(root)
        mp.setattr(planted, "HALLMARK", hallmark)
        mp.setattr(planted, "GO_BP", go)
        mp.setattr(planted, "TASKS", root / "tasks")
        yield dict(zip(grader.KNOB_CELLS, grader.build("fx", SEEDS, pool="dev")))


def _site(task_dir: Path, site: str) -> tuple[ad.AnnData, dict]:
    return (ad.read_h5ad(task_dir / "sites" / f"{site}.h5ad"),
            json.loads((task_dir / "public" / "cards" / f"{site}.json").read_text()))


def _effect(task_dir: Path, truth: dict, site: str, cell_types: list[str],
            genes: list[str]) -> float:
    """Donor-pseudobulk log2 CPM of a gene set, target group minus control, read from the site
    file through the site's own codebook (as a correct site view would)."""
    a, _ = _site(task_dir, site)
    m = truth["sites"][site]
    if m["scheme"] != "symbol":
        table = pd.read_csv(task_dir / "public" / "symbols.tsv", sep="\t").set_index("symbol")
        genes = list(table.loc[genes, "ensembl_id" if m["scheme"] == "ensembl" else "alias"])
    cols = [a.var_names.get_loc(g) for g in genes]
    group = a.obs["condition"].astype(str).map(m["codes"]).to_numpy()
    in_pop = a.obs["cell_label"].astype(str).map(m["labels"]).isin(cell_types).to_numpy()
    scores = {"control": [], "target": []}
    for sample in a.obs["sample"].unique():
        rows = (a.obs["sample"] == sample).to_numpy() & in_pop
        X = a.X[rows]
        scores["control" if group[rows][0] == "control" else "target"].append(
            np.log2(X[:, cols].sum() / X.sum() * 1e6 + 1))
    return float(np.mean(scores["target"]) - np.mean(scores["control"]))


def test_planted_programs_show_their_sign_and_size_exactly_where_planted(suite):
    """Reproducible programs at A, B, C; site-specific and artefact programs at one site; the
    mimic site only for programs shared with the mimic."""
    shared = set()
    for task_dir in suite.values():
        truth = grader.load_truth(task_dir)
        for prog in truth["programs"]:
            sign = 1 if prog["up_genes"] else -1
            genes = prog["up_genes"] + prog["down_genes"]
            carriers = prog["sites"] + (["M"] if prog["shared_with_mimic"] else [])
            for site in SITES:
                effect = _effect(task_dir, truth, site, prog["cell_types"], genes)
                if site in carriers:
                    assert abs(sign * effect - prog["log2fc"]) < 0.4, (prog["id"], site, effect)
                else:
                    assert abs(effect) < 0.3, (prog["id"], site, effect)
            if prog["kind"] == "reproducible":
                shared.add(prog["shared_with_mimic"])
    kinds = {p["kind"] for d in suite.values() for p in grader.load_truth(d)["programs"]}
    assert shared == {True, False} and kinds == {"reproducible", "site_specific", "artefact"}


def test_composition_shift_drops_cells_in_treated_donors_of_proposing_sites_only(suite):
    task_dir = suite[(False, False)]
    truth = grader.load_truth(task_dir)
    population = truth["composition"]["population"]
    for site in SITES:
        a, _ = _site(task_dir, site)
        m = truth["sites"][site]
        obs = a.obs.astype(str)
        frac = (obs["cell_label"].map(m["labels"]) == population).groupby(obs["sample"]).mean()
        group = obs.groupby("sample")["condition"].first().map(m["codes"])
        diff = frac[group != "control"].mean() - frac[group == "control"].mean()
        assert diff < -0.03 if site != "M" else abs(diff) < 1e-9


def test_programs_avoid_small_populations_and_the_composition_population(suite):
    for task_dir in suite.values():
        truth = grader.load_truth(task_dir)
        comp = truth["composition"]
        assert comp["population"] in planted.COMPOSITION_POPULATIONS
        assert comp["fraction_removed"] == planted.DROP
        for prog in truth["programs"]:
            assert set(prog["cell_types"]) <= set(planted.PROGRAM_POPULATIONS)
            assert comp["population"] not in prog["cell_types"]


def test_artefact_sits_at_one_proposing_site_whose_groups_differ_in_assay(suite):
    task_dir = suite[(True, False)]
    truth = grader.load_truth(task_dir)
    artefacts = [p for p in truth["programs"] if p["kind"] == "artefact"]
    assert len(artefacts) == 1 and artefacts[0]["up_genes"]
    assert artefacts[0]["sites"] == [truth["artefact_site"]] and truth["artefact_site"] in "ABC"
    for site in SITES:
        a, card = _site(task_dir, site)
        m = truth["sites"][site]
        group = a.obs["condition"].astype(str).map(m["codes"])
        assays = {g: set(a.obs["assay"].astype(str)[group == g]) for g in m["codes"].values()}
        if site == truth["artefact_site"]:
            assert assays == {"control": {"10x 3' v2"}, "treated": {"10x 3' v3"}}
        else:
            assert len(set(a.obs["assay"])) == 1
        assert card["assay_values"] == {c: sorted(assays[g]) for c, g in m["codes"].items()}
    clean = grader.load_truth(suite[(False, False)])
    assert clean["artefact_site"] is None
    assert all(p["kind"] != "artefact" for p in clean["programs"])
    pairs = [(t["artefact_site"], p["sites"][0]) for t in map(grader.load_truth, suite.values())
             for p in t["programs"] if p["kind"] == "site_specific" and t["artefact_site"]]
    assert pairs and all(a != s for a, s in pairs)      # site biology never sits at the artefact


def test_ids_labels_knob_gives_each_site_its_own_ids_names_and_codes(suite):
    task_dir = suite[(False, True)]
    truth = grader.load_truth(task_dir)
    assert sorted(truth["sites"][s]["scheme"] for s in "ABC") == sorted(planted.SCHEMES)
    table = pd.read_csv(task_dir / "public" / "symbols.tsv", sep="\t")
    assert list(table.columns) == ["ensembl_id", "symbol", "alias"]
    assert list(table["symbol"]) == GENES and list(table["alias"]) == ALIASES
    native = {"symbol": GENES, "ensembl": list(table["ensembl_id"]), "alias": ALIASES}
    for site in SITES:
        a, card = _site(task_dir, site)
        m = truth["sites"][site]
        assert list(a.var_names) == native[m["scheme"]]
        assert card["features"] == planted.FEATURES[m["scheme"]]
        assert set(a.obs["condition"]) == set(card["codes"]) == set(m["codes"])
        assert set(m["codes"]) == {"arm A", "arm B"}
        assert {card["codes"][c] for c in m["codes"]} == {planted.GROUP_TEXT[g]
                                                         for g in m["codes"].values()}
        assert set(a.obs["cell_label"]) == set(card["cell_labels"]) == set(m["labels"])
        assert sorted(m["labels"].values()) == sorted(planted.POPULATIONS)
        assert not set(m["labels"]) & set(planted.POPULATIONS)
    plain = suite[(False, False)]
    for site in SITES:
        a, _ = _site(plain, site)
        assert list(a.var_names) == GENES
        assert set(a.obs["condition"]) == {"control", "mimic" if site == "M" else "treated"}
        assert set(a.obs["cell_label"]) <= set(planted.POPULATIONS)


def test_answers_are_written_only_under_hidden(suite):
    visible = {"public/task.json", "public/symbols.tsv",
               *[f"public/cards/{s}.json" for s in SITES], *[f"sites/{s}.h5ad" for s in SITES]}
    for task_dir in suite.values():
        truth = grader.load_truth(task_dir)
        assert (truth["task_id"], truth["pool"], truth["base_sha256"]) == (task_dir.name, "dev",
                                                                          "f" * 64)
        files = {p.relative_to(task_dir).as_posix() for p in task_dir.rglob("*") if p.is_file()}
        assert files - {"hidden/truth.json"} == visible
        for path in visible:
            if path.endswith(".json"):
                text = (task_dir / path).read_text()
                for word in ('"programs"', '"seed"', '"knobs"', "shared_with_mimic",
                             "reproducible", "site_specific", "artefact"):
                    assert word not in text, (path, word)
        for site in SITES:
            a, _ = _site(task_dir, site)
            assert list(a.obs.columns) == ["sample", "condition", "cell_label", "assay"]
            assert a.var.shape[1] == 0 and not a.uns and not a.obsm and not a.layers
            assert a.obs["sample"].nunique() == 8


def test_build_refuses_an_existing_task(suite):
    with pytest.raises(FileExistsError):
        grader.build("fx", [SEEDS[0]], pool="dev")


def test_load_case_returns_the_case_spec_and_site_specs(suite):
    case, sites = planted.load_case(suite[(True, True)])
    assert (case.name, case.kind, case.vocab) == ("planted", "scrna", planted.CASE_VOCAB)
    assert case.vocab.contrasts == (("treated", "control"), ("mimic", "control"))
    assert case.default_spec == {"unit": "donor", "normalization": "CPM",
                                 "one_sample_per_donor": True,
                                 "groups": "map site codes per the codebook"}
    assert case.tester == tester.TesterParams() and case.max_claims == 10
    assert set(case.resources) == {"symbols.tsv", f"genesets/{planted.GO_BP.name}"}
    assert all(p.exists() for p in case.resources.values())
    assert case.scope_tests is planted.scope_tests and case.scope_rule is planted.scope_rule
    assert [(s["name"], s["proposes"]) for s in sites] == [("A", True), ("B", True), ("C", True),
                                                           ("M", False)]
    assert all(s["data_path"].exists() and set(s["card"]["codes"]) == {"arm A", "arm B"}
               for s in sites)


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

TRUTH = {
    "programs": [
        {"id": "P1", "kind": "reproducible", "cell_types": ["NK", "CD8 T"], "up_genes": GENES[0:10],
         "down_genes": [], "sites": ["A", "B", "C"], "shared_with_mimic": True},
        {"id": "P2", "kind": "reproducible", "cell_types": ["B"], "up_genes": [],
         "down_genes": GENES[10:20], "sites": ["A", "B", "C"], "shared_with_mimic": False},
        {"id": "P3", "kind": "site_specific", "cell_types": ["CD14 Mono"],
         "up_genes": GENES[20:30], "down_genes": [], "sites": ["B"], "shared_with_mimic": False},
        {"id": "P4", "kind": "artefact", "cell_types": ["CD4 T"], "up_genes": GENES[30:40],
         "down_genes": [], "sites": ["C"], "shared_with_mimic": False},
    ],
    "composition": {"id": "C1", "kind": "composition", "population": "CD16 Mono", "sign": -1,
                    "fraction_removed": 0.4, "sites": ["A", "B", "C"]},
    "artefact_site": "C",
}


def test_an_exact_claim_is_precise_and_recalls_one_of_four_items():
    g = grader.grade([_claim(up=GENES[0:10])], TRUTH)
    assert g == {"precision": 1.0, "recall": 0.25, "f1": pytest.approx(0.4), "n_claims": 1,
                 "artefact_rejected": True, "scope_accuracy": None, "cause_accuracy": None}


def test_partial_overlap_matches_from_jaccard_point_two():
    hit = _claim("c1", up=GENES[0:3] + GENES[50:52])      # 3 / 12 = 0.25
    miss = _claim("c2", up=GENES[0:2] + GENES[50:53])     # 2 / 13 = 0.15
    assert grader.match(hit, TRUTH)["id"] == "P1" and grader.match(miss, TRUTH) is None
    assert grader.grade([hit, miss], TRUTH)["precision"] == 0.5


def test_wrong_direction_population_or_contrast_never_matches():
    claims = [_claim("c1", down=GENES[0:10]), _claim("c2", population="B", up=GENES[0:10]),
              _claim("c3", target="mimic", up=GENES[0:10]), _claim("c4", population="B",
                                                                    up=GENES[10:20])]
    g = grader.grade(claims, TRUTH)
    assert (g["precision"], g["recall"], g["f1"]) == (0.0, 0.0, 0.0)


def test_composition_claims_match_by_population_and_sign():
    def comp(population: str, effect: float | None) -> Claim:
        return _claim(population=population, kind="composition", effect=effect)

    assert grader.match(comp("CD16 Mono", -0.4), TRUTH)["id"] == "C1"
    for claim in (comp("CD16 Mono", 0.4), comp("CD16 Mono", None), comp("NK", -0.4)):
        assert grader.match(claim, TRUTH) is None


def test_no_claims_has_no_precision_and_zero_recall():
    assert grader.grade([], TRUTH) == {"precision": None, "recall": 0.0, "f1": 0.0, "n_claims": 0,
                                       "artefact_rejected": True, "scope_accuracy": None,
                                       "cause_accuracy": None}


def test_artefact_counts_as_claimed_only_when_it_is_a_finding():
    art = _claim("a", population="CD4 T", up=GENES[30:40])
    g = grader.grade([art], TRUTH)
    assert g["artefact_rejected"] is False and g["precision"] == 0.0
    assert grader.grade([art], TRUTH, labels={"a": "site-specific"})["artefact_rejected"] is True
    no_artefact = {**TRUTH, "programs": TRUTH["programs"][:3], "artefact_site": None}
    assert grader.grade([art], no_artefact)["artefact_rejected"] is None


def test_a_claim_counts_once_for_its_best_match():
    truth = {**TRUTH, "programs": [
        {**TRUTH["programs"][0], "cell_types": ["NK"]},
        {**TRUTH["programs"][2], "cell_types": ["NK"]}]}
    claim = _claim(up=GENES[0:4] + GENES[20:28])           # P1: 4/18, P3: 8/14
    assert grader.match(claim, truth)["id"] == "P3"
    assert grader.grade([claim], truth)["precision"] == 0.0


def test_scope_accuracy_is_over_findings_matching_reproducible_programs():
    claims = [_claim("s1", up=GENES[0:10]), _claim("s2", population="B", down=GENES[10:20]),
              _claim("s3", population="CD8 T", up=GENES[0:10]), _claim("s4", up=GENES[0:10])]
    scopes = {"s1": "shared", "s2": "shared", "s3": "inconclusive", "s4": "specific"}
    g = grader.grade(claims, TRUTH, scopes=scopes, labels={"s4": "unsupported"})
    assert g["scope_accuracy"] == 0.5 and g["n_claims"] == 3


def test_cause_accuracy_needs_the_planted_category_and_site():
    claims = [_claim("k1", population="CD4 T", up=GENES[30:40]),
              _claim("k2", population="CD14 Mono", up=GENES[20:30]),
              _claim("k3", population="CD14 Mono", up=GENES[20:30]),
              _claim("k4", up=GENES[0:10])]
    causes = {"k1": {"cause": "site-artefact", "site": "C", "defect": "assay differs"},
              "k2": {"cause": "site-artefact", "site": "B", "defect": ""},
              "k3": {"cause": "site-biology", "site": "A", "defect": ""},
              "k4": {"cause": "handoff-loss", "site": "B", "defect": "gene ids"}}
    labels = {"k1": "site-specific", "k2": "site-specific", "k3": "site-specific",
              "k4": "corroborated"}
    g = grader.grade(claims, TRUTH, labels=labels, causes=causes)
    assert g["cause_accuracy"] == pytest.approx(1 / 3) and g["n_claims"] == 1
    assert grader.grade(claims, TRUTH)["cause_accuracy"] is None
