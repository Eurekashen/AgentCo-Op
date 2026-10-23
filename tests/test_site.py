"""Real sites (coop/site.py) with a fake agent: offline, no API key, no agent process."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from agentcoop import llm, store
from agentcoop.agents import harness, workspace
from agentcoop.agents.harness import AgentRun
from agentcoop.coop import site as site_mod
from agentcoop.coop import tester
from agentcoop.coop.claims import Claim, Vocabulary
from agentcoop.coop.site import (CaseSpec, Site, file_digest, make_sites, propose_prompt,
                                 repair_prompt, serve_mailbox)
from agentcoop.coop.view import load_view, site_table, write_view

VOCAB = Vocabulary(populations=("B", "NK"), groups=("control", "treated"),
                   contrasts=(("treated", "control"),))
CARD = {"codebook": {"cond": {"ctl": "control", "trt": "treated"}, "ct": {"b": "B", "nk": "NK"}},
        "assay": "10x 3' v3", "donors_per_group": {"control": 4, "treated": 4}}
PLANTED = ("SYM0", "SYM1", "SYM2", "SYM3", "SYM4", "SYM5")
CLAIM = Claim(id="Z:1", site="Z", kind="de", population="B", target="treated",
              reference="control", up_genes=PLANTED, down_genes=(), effect=1.6, n_units=8, rank=1)
DEFECT = "genes.tsv keeps Ensembl ids as symbols; map them with symbols.tsv"


@pytest.fixture(autouse=True)
def _clean_agentcoop_env(monkeypatch):
    """Tests must not depend on AGENTCOOP_* settings exported in the developer's shell."""
    for k in [k for k in os.environ if k.startswith("AGENTCOOP_")]:
        monkeypatch.delenv(k)


def toy_case(**kw) -> CaseSpec:
    """A two-population case; ``kw`` sets further fields (e.g. resources)."""
    return CaseSpec(name="toy", kind="scrna", vocab=VOCAB,
                    question="Which programs change in treated donors?",
                    default_spec={"sample_unit": "donor"},
                    tester=tester.TesterParams(null_sets=20, n_boot=50), **kw)


def test_prompts_state_question_boundary_vocabulary_formats_and_cap():
    case = toy_case(resources={"symbols.tsv": Path("symbols.tsv")})
    p = propose_prompt(case, CARD, "site_A.h5ad", proposes=True)
    for s in (case.question, "site_A.h5ad", "codebook.json", "symbols.tsv", CARD["assay"],
              "Only the files in this directory", "cannot leave the site", "treated vs control",
              "view/genes.tsv", "view/samples.tsv", "view/cells.tsv", "view/spec.json",
              "claims.json", "at most 10 claims", '"up_genes"', '"rank"', "composition",
              *VOCAB.populations, *VOCAB.groups):
        assert s in p, s
    view_only = propose_prompt(case, CARD, "site_A.h5ad", proposes=False)
    assert "builds its view only" in view_only and "up_genes" not in view_only
    bulk = propose_prompt(dataclasses.replace(case, kind="bulk"), CARD, "x.tsv",
                          proposes=True)
    assert "cells.tsv" not in bulk and "composition" not in bulk
    r = repair_prompt(case, CLAIM, DEFECT)
    for s in (DEFECT, "SYM0", "in B, treated vs control", "Change only files under view/",
              "view/genes.tsv"):
        assert s in r, s
    assert "claims.json" not in r


def test_proposing_sites_are_told_how_claims_are_tested():
    """Sites discover, independent sites confirm: the prompt states the fixed test and its
    thresholds (from case.tester), asks for programs and liberal candidates; view-only sites
    are not asked to propose."""
    case = toy_case()
    p = propose_prompt(case, CARD, "site_A.h5ad", proposes=True)
    for s in ("separates discovery from confirmation", "coherent gene programs",
              f"mean CPM >= {case.tester.min_cpm:g}", "genome-wide correction",
              "each direction holds on its own"):
        assert s in p, s
    assert "discovery from confirmation" not in propose_prompt(case, CARD, "x", proposes=False)


def test_prompt_view_format_is_what_load_view_reads(tmp_path):
    """Files written by following the prompt's format must load: pins the prompt to view.py."""
    p = propose_prompt(toy_case(), CARD, "site_A.h5ad", proposes=True)
    for s in ("columns native_id, symbol", "columns sample, donor, group",
              "columns cell, sample, population", "or exclude for a sample",
              "or other for a cell", ", ".join(site_mod.SPEC_KEYS)):
        assert s in p, s
    root = tmp_path / "handmade"
    root.mkdir()
    (root / "genes.tsv").write_text("native_id\tsymbol\nENSG1\tSYM0\nENSG2\tSYM1\n")
    (root / "samples.tsv").write_text("sample\tdonor\tgroup\tassay\nS1\tD1\tcontrol\tv3\n"
                                      "S2\tD2\ttreated\tv3\nS3\tD3\texclude\tv2\n")
    (root / "cells.tsv").write_text("cell\tsample\tpopulation\nc1\tS1\tB\nc2\tS2\tNK\n"
                                    "c3\tS3\tother\n")
    (root / "spec.json").write_text(json.dumps({k: "x" for k in site_mod.SPEC_KEYS}))
    view, errors = load_view(root, kind="scrna", vocab=VOCAB)
    assert view is not None, errors
    assert view.genes == {"ENSG1": "SYM0", "ENSG2": "SYM1"}
    assert list(view.cells["population"]) == ["B", "NK", "other"]


def test_bulk_prompt_describes_the_matrix_that_site_table_reads(tmp_path):
    """A bulk site's data file is a TSV matrix (first column = feature id, one column per
    sample): a view written by the bulk prompt's rules must load and give a site table."""
    p = propose_prompt(dataclasses.replace(toy_case(), kind="bulk"), CARD, "expr.tsv",
                       proposes=True)
    for s in ("tab-separated expression matrix", "native_id = the matrix's first-column id",
              "sample = a column header of the matrix"):
        assert s in p, s
    assert "h5ad" not in p and "obs_names" not in p
    data = tmp_path / "expr.tsv"
    data.write_text("probe\tS1\tS2\tS3\tS4\tS5\tS6\n"
                    "P1\t1\t2\t3\t7\t8\t9\nP2\t5\t4\t6\t5\t4\t6\nP3\t1\t1\t1\t1\t1\t1\n")
    root = tmp_path / "view"
    root.mkdir()
    (root / "genes.tsv").write_text("native_id\tsymbol\nP1\tSTAT1\nP2\tGBP5\n")
    (root / "samples.tsv").write_text("sample\tdonor\tgroup\n" + "".join(
        f"S{i}\tD{i}\t{'control' if i <= 3 else 'treated'}\n" for i in range(1, 7)))
    (root / "spec.json").write_text(json.dumps({k: "x" for k in site_mod.SPEC_KEYS}))
    view, errors = load_view(root, kind="bulk", vocab=VOCAB)
    assert view is not None, errors
    table = site_table(view, data, kind="bulk")
    assert table.genes == ["GBP5", "STAT1"]
    assert list(table.units["unit"]) == [f"S{i}" for i in range(1, 7)]
    assert list(table.units["group"]) == ["control"] * 3 + ["treated"] * 3


def test_file_digest_follows_the_content(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert file_digest(f) == hashlib.sha256(b"abc").hexdigest()
    f.write_bytes(b"abcd")
    assert file_digest(f) == hashlib.sha256(b"abcd").hexdigest()


def wait_for(path: Path, seconds: float = 5.0) -> Path:
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert time.monotonic() < deadline, f"no {path.name}"
        time.sleep(0.05)
    return path


def test_mailbox_answers_each_request_once_until_stopped(tmp_path):
    def boom(req: dict) -> dict:
        raise ValueError("bad request")

    stop = threading.Event()
    thread = serve_mailbox(tmp_path, {"echo": lambda req: {"echo": req["text"]}, "boom": boom},
                           stop)
    (tmp_path / "req_1.json.tmp").write_text(json.dumps({"op": "echo", "text": "hi"}))
    os.replace(tmp_path / "req_1.json.tmp", tmp_path / "req_1.json")
    (tmp_path / "req_2.json").write_text(json.dumps({"op": "nope"}))
    (tmp_path / "req_3.json").write_text(json.dumps({"op": "boom"}))
    assert json.loads(wait_for(tmp_path / "resp_1.json").read_text()) == {"echo": "hi"}
    assert "error" in json.loads(wait_for(tmp_path / "resp_2.json").read_text())
    assert "bad request" in json.loads(wait_for(tmp_path / "resp_3.json").read_text())["error"]
    assert not list(tmp_path.glob("req_*.json"))    # answered requests are removed
    stop.set()
    thread.join(timeout=2)
    assert thread.daemon and not thread.is_alive()


RUN_TOKENS, FAIL_TOKENS = 500_000, 1_000_000    # output tokens one fake run / failed run uses
RUN_USD = llm.cost("gpt-6-luna", 0, 0, RUN_TOKENS)
FAIL_USD = llm.cost("gpt-6-luna", 0, 0, FAIL_TOKENS)


def make_h5ad(path: Path) -> None:
    """8 donors (4 ctl, 4 trt) x 2 cell types x 30 cells x 60 genes under native ids and codes;
    the first 6 genes are 3x higher in treated B cells."""
    rng = np.random.default_rng(0)
    base = rng.uniform(5, 15, size=60)
    blocks, rows = [], []
    for d in range(8):
        cond = "trt" if d >= 4 else "ctl"
        for ct in ("b", "nk"):
            mu = base.copy()
            if cond == "trt" and ct == "b":
                mu[:6] *= 3
            blocks.append(rng.poisson(mu, size=(30, 60)))
            rows += [{"donor": f"D{d}", "cond": cond, "ct": ct}] * 30
    obs = pd.DataFrame(rows, index=[f"cell{i}" for i in range(len(rows))])
    var = pd.DataFrame(index=[f"ENSG{i:011d}" for i in range(60)])
    ad.AnnData(X=sp.csr_matrix(np.vstack(blocks).astype(np.float32)), obs=obs,
               var=var).write_h5ad(path)


def write_fake_view(ws: Path, *, map_genes: bool, drop_group: bool, draw: int) -> None:
    """The view a site agent builds from its data file and codebook."""
    adata = ad.read_h5ad(ws / "site_A.h5ad")
    book = json.loads((ws / "codebook.json").read_text())["codebook"]
    obs = adata.obs.astype(str)
    donors = obs.drop_duplicates("donor")
    samples = pd.DataFrame({"sample": donors["donor"].to_numpy(),
                            "donor": donors["donor"].to_numpy(),
                            "group": donors["cond"].map(book["cond"]).to_numpy()})
    cells = pd.DataFrame({"cell": adata.obs_names.to_numpy(), "sample": obs["donor"].to_numpy(),
                          "population": obs["ct"].map(book["ct"]).to_numpy()})
    genes = {f: (f"SYM{i}" if map_genes else f) for i, f in enumerate(adata.var_names)}
    (ws / "view").mkdir(exist_ok=True)
    write_view(ws / "view", genes=genes, samples=samples, cells=cells,
               spec={"contrasts": [["treated", "control"]], "sample_unit": "donor",
                     "normalization": "log2 CPM", "gene_mapping": "symbols.tsv" if map_genes
                     else "none", "covariates": [], "exclusions": [], "draw": draw})
    if drop_group:   # a malformed view: samples.tsv without its group column
        path = ws / "view" / "samples.tsv"
        pd.read_csv(path, sep="\t").drop(columns="group").to_csv(path, sep="\t", index=False)


class FakeAgent:
    """Stands in for ``harness.run_agent``: does what the prompt asks, in the workspace.

    Like ``run_codex`` it ledgers its usage under the caller's tag *and* reports the same amount
    as ``nominal_usd_external``, so a site that charged both would show twice the cost.
    ``fail``: the run times out after writing half a view. ``map_genes``: whether step 1 maps
    native ids to symbols (repair and self-check always do). ``barrier``: wait for a second
    concurrent call before writing, so two runs race on one store key. ``fixes_view``: whether a
    format repair (step 1's fix_view call) writes a valid view or repeats step 1's mistake.
    """

    def __init__(self, *, fail: bool = False, map_genes: bool = True, drop_group: bool = False,
                 barrier: threading.Barrier | None = None, fixes_view: bool = True) -> None:
        self.fail, self.map_genes, self.drop_group = fail, map_genes, drop_group
        self.fixes_view = fixes_view
        self.barrier = barrier
        self.prompts: list[str] = []
        self.lock = threading.Lock()

    def __call__(self, agent: str, ws: Path, prompt: str, *, sample: int, timeout: int) -> AgentRun:
        with self.lock:
            self.prompts.append(prompt)
            n = len(self.prompts)
        if self.barrier is not None:
            try:
                self.barrier.wait(timeout=1)
            except threading.BrokenBarrierError:   # the store serialized the calls
                pass
        if self.fail:
            (ws / "view").mkdir(exist_ok=True)
            (ws / "view" / "genes.tsv").write_text("native_id\tsym")
            usd = llm.record_external("gpt-6-luna", 0, 0, 0, FAIL_TOKENS, billed=False)
            return AgentRun(agent, False, [], "timeout", usd, float(timeout))
        if "answer.md" in prompt:
            (ws / "answer.md").write_text(f"site answer {n}")
            (ws / "view" / "spec.json").write_text("{}")   # a careless agent touches its view
        elif "could not read your site view" in prompt and not self.fixes_view:
            write_fake_view(ws, map_genes=self.map_genes, drop_group=self.drop_group, draw=n)
        elif "Defect:" in prompt or "Checklist" in prompt:
            write_fake_view(ws, map_genes=True, drop_group=False, draw=n)
        else:
            write_fake_view(ws, map_genes=self.map_genes, drop_group=self.drop_group, draw=n)
            if "up_genes" in prompt:   # a proposing site's prompt carries the claim schema
                (ws / "claims.json").write_text(json.dumps([
                    {"kind": "de", "population": "B", "contrast": ["treated", "control"],
                     "up_genes": list(PLANTED), "down_genes": [], "effect": 1.6, "n_units": 8,
                     "rank": 1, "rationale": f"draw {n}"},
                    {"kind": "de", "population": "Platelet", "contrast": ["treated", "control"],
                     "up_genes": ["SYM6", "SYM7", "SYM8"], "down_genes": [], "effect": 1.0,
                     "n_units": 8, "rank": 2, "rationale": "not a case population"}]))
        usd = llm.record_external("gpt-6-luna", 0, 0, 0, RUN_TOKENS, billed=False)
        return AgentRun(agent, True, [{"code": "...", "output": "..."}], None, usd, 1.0)


def install(monkeypatch, **kw) -> FakeAgent:
    fake = FakeAgent(**kw)
    monkeypatch.setattr(harness, "run_agent", fake)
    return fake


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Store, workspaces and ledger under tmp_path; one site file and a toy case."""
    monkeypatch.setattr(store, "STORE", tmp_path / "store")
    monkeypatch.setattr(workspace, "WS_ROOT", tmp_path / "ws")
    monkeypatch.setenv("AGENTCOOP_LLM_DB", str(tmp_path / "llm.sqlite"))
    (tmp_path / "data").mkdir()
    data = tmp_path / "data" / "site_A.h5ad"
    make_h5ad(data)
    symbols = tmp_path / "data" / "symbols.tsv"
    symbols.write_text("".join(f"ENSG{i:011d}\tSYM{i}\n" for i in range(60)))
    return SimpleNamespace(tmp=tmp_path, data=data,
                           case=toy_case(resources={"symbols.tsv": symbols}))


def new_site(env, run_id: str = "r1") -> Site:
    return Site("A", env.data, CARD, "codex", env.case, task_id="t1", run_id=run_id)


def view_files(site: Site) -> dict[str, str]:
    return {f: (site.ws / "view" / f).read_text()
            for f in ("genes.tsv", "samples.tsv", "cells.tsv", "spec.json")}


def test_propose_builds_the_site_table_and_keeps_valid_claims(env, monkeypatch):
    fake = install(monkeypatch)
    site = new_site(env)
    claims = site.propose(0)
    assert [c.population for c in claims] == ["B"]   # the off-vocabulary claim is dropped ...
    assert any(e.startswith("claims:") for e in site.errors)   # ... and its error kept
    assert claims[0].site == "A" and claims[0].contrast == ("treated", "control")
    assert claims[0].up_genes == PLANTED
    assert site.table is not None and "SYM0" in site.table.genes
    assert site.holds("B", ("treated", "control")) and not site.holds("B", ("treated", "mimic"))
    assert site.test(claims[0]).status != "untestable"
    assert site.spec()["sample_unit"] == "donor" and site.card() == CARD
    assert site.ws == env.tmp / "ws" / "r1" / "A"
    assert (site.ws / "site_A.h5ad").exists() and (site.ws / "symbols.tsv").exists()
    assert json.loads((site.ws / "codebook.json").read_text()) == CARD
    assert "site_A.h5ad" in fake.prompts[0]
    # charged once, from the ledger: the run's own nominal_usd_external is not added again
    assert site.calls == 1 and site.nominal_usd == pytest.approx(RUN_USD)


def test_same_request_from_another_run_replays_claims_and_view(env, monkeypatch):
    fake = install(monkeypatch)
    first = new_site(env, run_id="t1/union/s0")
    claims = first.propose(0)
    second = new_site(env, run_id="t1/ours/s0")
    assert second.propose(0) == claims and len(fake.prompts) == 1   # served from the store
    assert view_files(second) == view_files(first)                  # with the same view
    assert second.table.genes == first.table.genes
    assert second.calls == 1 and second.nominal_usd == pytest.approx(RUN_USD)   # nominal on hits
    second.propose(1)                       # a new sample index is an honest re-draw
    assert len(fake.prompts) == 2


def test_concurrent_requests_for_one_proposal_converge_on_one_value(env, monkeypatch):
    install(monkeypatch, barrier=threading.Barrier(2))
    sites = [new_site(env, run_id=f"r{i}") for i in range(2)]
    llm.spent()   # create the ledger first: two threads creating a fresh sqlite file can lock
    out: list[list[Claim] | None] = [None, None]
    threads = [threading.Thread(target=lambda i=i: out.__setitem__(i, sites[i].propose(0)))
               for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out[0] is not None and out[0] == out[1]
    assert view_files(sites[0]) == view_files(sites[1])
    assert sites[0].spec() == sites[1].spec()


def test_failed_call_is_charged_and_recorded_but_never_cached(env, monkeypatch):
    install(monkeypatch, fail=True)
    site = new_site(env)
    assert site.propose(0) == []
    assert site.calls == 1 and site.nominal_usd == pytest.approx(FAIL_USD)   # charged once
    assert [f["op"] for f in site.failures] == ["propose"]
    assert not (site.ws / "view").exists()          # the half-written view is not kept
    r = site.test(CLAIM)
    assert (r.status, r.reason) == ("untestable", "view")
    fake = install(monkeypatch)
    retry = new_site(env, run_id="r2")
    assert len(retry.propose(0)) == 1 and len(fake.prompts) == 1   # the failure was not cached


def test_malformed_view_goes_back_once_with_the_validator_messages(env, monkeypatch):
    fake = install(monkeypatch, drop_group=True)
    site = new_site(env)
    claims = site.propose(0)
    assert site.table is not None and site.test(claims[0]).status != "untestable"
    issues = [e.removeprefix("propose: ") for e in site.errors if e.startswith("propose:")]
    assert len(fake.prompts) == 2 and issues and all(e in fake.prompts[1] for e in issues)
    assert "up_genes" not in fake.prompts[1] and "Change only files under view/" in fake.prompts[1]
    assert site.calls == 2 and site.nominal_usd == pytest.approx(2 * RUN_USD)
    replay = new_site(env, run_id="r2")              # another run replays both calls
    assert replay.propose(0) == claims and len(fake.prompts) == 2
    assert view_files(replay) == view_files(site)


def test_malformed_view_makes_tests_untestable_instead_of_crashing(env, monkeypatch):
    fake = install(monkeypatch, drop_group=True, fixes_view=False)
    site = new_site(env)
    claims = site.propose(0)
    assert len(fake.prompts) == 2                    # one format repair, then no more
    assert site.table is None and any(e.startswith("fix_view:") for e in site.errors)
    r = site.test(claims[0])
    assert (r.status, r.reason, r.site) == ("untestable", "view", "A")
    assert site.holds("B", ("treated", "control"))   # routed, so the broken view is on record
    assert site.spec() == {}
    assert list(site.export_stats("B", ("treated", "control")).columns) == site_mod.STATS_COLUMNS


def test_make_sites_applies_agent_overrides(env, monkeypatch):
    specs = [{"name": "A", "data_path": env.data, "card": CARD, "proposes": True,
              "agent": "codex"},
             {"name": "M", "data_path": env.data, "card": CARD, "proposes": False,
              "agent": "finch"}]
    sites = make_sites(specs, env.case, task_id="t1", run_id="t1/ours/s0", agents={"M": "react"})
    assert [(s.name, s.agent, s.proposes, s.timeout) for s in sites] == [
        ("A", "codex", True, 900), ("M", "react", False, 900)]
    assert sites[1].ws == env.tmp / "ws" / "t1" / "ours" / "s0" / "M"
    fake = install(monkeypatch)
    assert sites[1].propose(0) == [] and sites[1].table is not None   # tester-only: view only
    assert "builds its view only" in fake.prompts[0]


def test_repair_changes_the_view_and_the_next_test_sees_it(env, monkeypatch):
    install(monkeypatch, map_genes=False)            # step 1 leaves Ensembl ids unmapped
    site = new_site(env)
    claim = site.propose(0)[0]
    assert site.test(claim).status == "untestable"   # its genes are not in the view
    before = view_files(site)
    install(monkeypatch, fail=True)
    assert not site.repair(claim, DEFECT, 0)         # a failed repair leaves the view as it was
    assert view_files(site) == before
    fake = install(monkeypatch)
    assert site.repair(claim, DEFECT, 0)
    assert "SYM0" in site.table.genes and site.test(claim).status != "untestable"
    assert DEFECT in fake.prompts[0] and "up_genes" not in fake.prompts[0]
    assert site.calls == 3 and site.nominal_usd == pytest.approx(RUN_USD + FAIL_USD + RUN_USD)
    replay = new_site(env, run_id="r2")              # another run replays both calls
    replay.propose(0)
    assert replay.repair(claim, DEFECT, 0) and len(fake.prompts) == 1
    assert view_files(replay) == view_files(site)


def test_self_check_and_ask_are_charged_memoized_follow_ups(env, monkeypatch):
    fake = install(monkeypatch, map_genes=False)
    site = new_site(env)
    site.propose(0)
    assert site.self_check(0) and "SYM0" in site.table.genes   # the checklist fixed the ids
    assert "Checklist" in fake.prompts[-1]
    before = view_files(site)
    answer = site.ask("How many donors per group?", 0)
    assert answer.startswith("site answer")
    assert view_files(site) == before               # answering never changes the view
    assert site.ask("How many donors per group?", 0) == answer and len(fake.prompts) == 3
    assert site.calls == 4 and site.nominal_usd == pytest.approx(4 * RUN_USD)


def test_a_non_default_site_model_never_shares_a_stored_call(env, monkeypatch):
    """Robustness runs on another model must not replay the main model's step-1 calls (the
    store key had no model: a gpt-4.1-mini run reused gpt-6-luna proposals, findings
    2026-10-23); the default model's keys stay as they were."""
    fake = install(monkeypatch)
    main = new_site(env, run_id="r1")
    main.propose(0)
    assert "model" not in main._key
    monkeypatch.setenv("AGENTCOOP_EXTERNAL_MODEL", "gpt-4.1-mini")
    other = new_site(env, run_id="r2")
    other.propose(0)
    assert other._key["model"] == "gpt-4.1-mini" and len(fake.prompts) == 2   # ran again
