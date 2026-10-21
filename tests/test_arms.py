"""Arms of spec §4 on scripted sites: selection semantics, MA math, budgets and costs. Offline."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
from types import SimpleNamespace

import pandas as pd
import pytest
from scipy.stats import chi2, false_discovery_control, norm

from agentcoop import llm
from agentcoop.agents import harness
from agentcoop.agents.harness import AgentRun
from agentcoop.coop import arms
from agentcoop.coop.claims import claim_to_dict
from agentcoop.coop.protocol import ClaimRecord, ProtocolRun
from agentcoop.coop.site import serve_mailbox
from agentcoop.llm import ChatResult
from tests.fakes import (CCL_C, HB_A, IFN_A, IFN_B, PROGRAMS, TREATED, FakeMemo, FakeSite,
                         SiteFactory, case_spec, charge_ledger, claim, fake_workspace, gene_stats,
                         site_specs, standard_sites)

IFN6 = set(PROGRAMS["IFN"])


@pytest.fixture
def env(tmp_path, monkeypatch):
    factory, memo = SiteFactory(standard_sites()), FakeMemo()
    monkeypatch.setattr(arms, "make_sites", factory)
    monkeypatch.setattr(arms.store, "memo", memo)
    monkeypatch.setattr(arms, "new_workspace", fake_workspace(tmp_path / "ws"))
    return SimpleNamespace(factory=factory, memo=memo, specs=site_specs(tmp_path),
                           case=case_spec(), tmp=tmp_path)


def run(env, *names: str, sample: int = 0) -> dict[str, arms.ArmResult]:
    return arms.run_arms(list(names), env.specs, env.case, task_id="t0", sample=sample)


def ids(result: arms.ArmResult) -> list[str]:
    return [c.id for c in result.claims]


def genes(result: arms.ArmResult) -> list[set[str]]:
    return [set(c.up_genes) | set(c.down_genes) for c in result.claims]


def test_single_union_vote_and_no_routing_share_one_step1(env):
    out = run(env, "single", "union", "vote", "no_routing")
    assert ids(out["single:A"]) == ["IFN_a", "HB_a"]
    assert ids(out["single:B"]) == ["IFN_b"]               # NOISE_b fails its own tester
    assert ids(out["single:C"]) == ["CCL_c"]
    assert "single:M" not in out                           # tester-only sites propose nothing
    # union: every claim, matched claims merged; more proposing sites first, then own p
    assert ids(out["union"]) == ["IFN_a", "CCL_c", "HB_a", "NOISE_b"]
    assert ids(out["vote"]) == ["IFN_a"]                   # IFN is the only claim of >= 2 sites
    assert ids(out["no_routing"]) == ["IFN_a", "CCL_c", "HB_a"]
    assert [c.rank for c in out["union"].claims] == [1, 2, 3, 4]
    assert [rid for rid, _ in env.factory.made] == ["t0/base/s0"]
    assert [s.calls for s in env.factory.sites("t0/base/s0")] == [1, 1, 1, 1]   # one propose each
    # nominal: one step-1 call per site, charged to every arm that uses it
    assert (out["single:A"].calls, out["single:A"].nominal_usd) == (1, pytest.approx(0.01))
    assert (out["union"].calls, out["union"].nominal_usd) == (3, pytest.approx(0.03))
    assert (out["vote"].calls, out["no_routing"].calls) == (3, 3)


def test_single_codex_runs_every_site_with_codex(env):
    out = run(env, "single_codex")
    [(run_id, sites)] = env.factory.made
    assert run_id == "t0/single_codex/s0" and {s.agent for s in sites} == {"codex"}
    assert sorted(out) == ["single_codex:A", "single_codex:B", "single_codex:C"]
    assert ids(out["single_codex:A"]) == ["IFN_a", "HB_a"]
    assert out["single_codex:B"].calls == 1


def test_fixed_effect_meta_math():
    stats = pd.DataFrame({"site": ["A", "B", "A", "B", "C", "A"],
                          "gene": ["G", "G", "H", "H", "H", "solo"],
                          "effect": [1.0, 2.0, 3.0, -0.05, -0.05, 5.0],
                          "se": [0.5, 1.0, 0.1, 0.5, 0.5, 0.1]})
    meta = arms.fixed_effect(stats).set_index("gene")
    assert "solo" not in meta.index                        # one site: nothing to pool
    g = meta.loc["G"]                                      # weights 4 and 1
    assert g.effect == pytest.approx(1.2)
    assert g.se == pytest.approx(1 / math.sqrt(5))
    assert g.Q == pytest.approx(4 * 0.2 ** 2 + 1 * 0.8 ** 2)
    assert g.Q_p == pytest.approx(chi2.sf(0.8, 1))
    assert g.p == pytest.approx(2 * norm.sf(1.2 * math.sqrt(5)))
    assert (g.n_sites, g.n_agree) == (2, 2)
    h = meta.loc["H"]                                      # one strong site drives the pool
    assert h.effect == pytest.approx((300 - 0.2 - 0.2) / 108)
    assert (h.n_sites, h.n_agree) == (3, 1)
    assert list(meta.fdr) == pytest.approx(list(false_discovery_control(meta.p.to_numpy())))


def _library(path):
    """T1 and T2 share their overlap with IFN6 (T2 is redundant); T3 holds no IFN gene."""
    noise = [f"N{i:02d}" for i in range(34)]
    terms = [("T1", sorted(IFN6) + noise[:4]), ("T2", sorted(IFN6) + noise[4:9]),
             ("T3", noise[10:20])]
    path.write_text("".join(f"{name}\t\t" + "\t".join(g) + "\n" for name, g in terms))
    return noise


def _resources(tmp):
    """Task 8's layout: a symbol table and an Enrichr-text library keyed under genesets/ (the
    resource-key convention); a .txt file outside genesets/ is not a library."""
    (tmp / "symbols.tsv").write_text("ENSG00000126709\tIFI6\n")
    (tmp / "notes.txt").write_text("T9\t\t" + "\t".join(f"X{i}" for i in range(12)) + "\n")
    return {"symbols.tsv": tmp / "symbols.tsv", "notes.txt": tmp / "notes.txt",
            "genesets/GO_Test_2023.txt": tmp / "GO_Test_2023.txt"}


def test_enrich_keeps_one_term_per_overlap_set(tmp_path):
    resources = _resources(tmp_path)
    noise = _library(resources["genesets/GO_Test_2023.txt"])
    library = arms.gene_set_library(case_spec(resources=resources))
    assert sorted(library) == ["T1", "T2", "T3"]
    terms = arms.enrich(IFN6, IFN6 | set(noise), library)
    assert [t["term"] for t in terms] == ["T1"]            # T2 has the same overlap: redundant
    assert terms[0]["genes"] == IFN6 and terms[0]["fdr"] < 0.05


def test_ma_genome_meta_enrichment_and_composition(env):
    env.case = case_spec(resources=_resources(env.tmp))
    noise = _library(env.tmp / "GO_Test_2023.txt")
    effects = {**{g: (1.0, 0.1) for g in IFN6},
               **{g: (0.02 * (-1) ** i, 0.5) for i, g in enumerate(noise)}}
    for name in "ABC":
        env.factory.templates[name]["stats"] = {("CD14 Mono", TREATED): gene_stats(effects)}
        env.factory.templates[name]["effects"]["comp:B"] = (0.8, 0.1)
    ma = run(env, "ma_genome")["ma_genome"]
    assert [(c.kind, c.population) for c in ma.claims] == [("composition", "B"),
                                                           ("de", "CD14 Mono")]
    comp, de = ma.claims
    assert comp.effect == pytest.approx(0.8) and comp.n_units == 24
    assert set(de.up_genes) == IFN6 and de.down_genes == () and "T1" in de.rationale
    assert de.n_units == 24 and de.effect == pytest.approx(1.0)
    assert ma.meta["tables"]["CD14 Mono|treated>control"]["n_kept"] == 6
    [(run_id, sites)] = env.factory.made
    assert run_id == "t0/ma/s0"
    assert sites[3].exported == []                         # the mimic table has one site only
    assert len(ma.meta["self_checked"]) == arms.MA_SELF_CHECKS == 1   # frozen at v4-freeze
    assert (ma.calls, ma.nominal_usd) == (5, pytest.approx(0.05))   # propose + self-check


def test_ma_genome_self_checks_the_frozen_number_of_sites(env, monkeypatch):
    monkeypatch.setattr(arms, "MA_SELF_CHECKS", 2)
    first = run(env, "ma_genome")["ma_genome"]
    [(_, sites)] = env.factory.made
    assert sorted(s.checks for s in sites) == [0, 0, 1, 1]  # one call per checked site
    assert first.calls == 4 + 2 and len(first.meta["self_checked"]) == 2
    again = run(env, "ma_genome")["ma_genome"]
    assert again.meta["self_checked"] == first.meta["self_checked"]   # fixed per task


def test_ma_genome_min_sites_one_is_the_single_site_analysis(tmp_path):
    """Task 10's headroom gate runs MA-genome's own code with min_sites=1 on one site."""
    resources = _resources(tmp_path)
    noise = _library(resources["genesets/GO_Test_2023.txt"])
    effects = {**{g: (1.0, 0.1) for g in IFN6},
               **{g: (0.02 * (-1) ** i, 0.5) for i, g in enumerate(noise)}}
    site = FakeSite("A", stats={("CD14 Mono", TREATED): gene_stats(effects)})
    case = case_spec(resources=resources)
    assert arms.ma_genome([site], case, "ma") == ([], {"tables": {}})   # one site: nothing to pool
    claims, info = arms.ma_genome([site], case, "single:A", min_sites=1)
    assert [(c.kind, c.population, set(c.up_genes), c.down_genes) for c in claims] == [
        ("de", "CD14 Mono", IFN6, ())]
    assert claims[0].site == "single:A" and claims[0].n_units == 8 and "T1" in claims[0].rationale
    assert info["tables"]["CD14 Mono|treated>control"]["n_kept"] == 6


OURS = {  # claim id -> (label_step3, label, cause); every other step-1 claim stays corroborated
    "HB_a": ("disputed", "corroborated",
             {"cause": "handoff-loss", "site": "B", "defect": "gene ids"}),
    "NOISE_b": ("disputed", "inconclusive",
                {"cause": "site-artefact", "site": "A", "defect": "assay differs by group"}),
}


def _fake_protocol(seen: dict):
    """run_protocol stand-in: step 1, each claim tested at its proposer and at every other site
    holding its contrast, labels from OURS, and one repair call (HB_a at B)."""
    def run_protocol(sites, case, *, sample, followup_budget, coordinator=None):
        records = []
        for s in sites:
            for c in s.propose(sample):
                tests = [s.test(c)] + [o.test(c) for o in sites
                                       if o is not s and o.holds(c.population, c.contrast)]
                step3, label, cause = OURS.get(c.id, ("corroborated", "corroborated", None))
                records.append(ClaimRecord(c, tests, step3, label, cause, None,
                                           results_step3=tests))
        next(s for s in sites if s.name == "B").repair(HB_A, "gene ids", sample)
        run = ProtocolRun(records, [{"claim_id": "HB_a", "site": "B", "cause": "handoff-loss",
                                     "defect": "gene ids", "ok": True}],
                          sum(s.calls for s in sites), sum(s.nominal_usd for s in sites))
        seen.update(sites=sites, budget=followup_budget, coordinator=coordinator, run=run)
        return run
    return run_protocol


class _Coordinator:
    def __init__(self, case):
        self.case = case


def test_ma_claims_stouffer_over_ours_step1_tests(env, monkeypatch):
    """Test effects arrive oriented (positive = the claimed direction), so a decrease claim that
    replicates as a decrease has a positive z; flipping its sign again would reject it."""
    comp_c = claim("COMP_c", "C", kind="composition", population="B", effect=-0.8, rank=2)
    env.factory.templates["C"]["claims"] = [CCL_C, comp_c]
    for name in "ABC":
        env.factory.templates[name]["effects"]["comp:B"] = (-0.6, 0.2)   # B cells drop
    monkeypatch.setattr(arms, "run_protocol", _fake_protocol({}))
    monkeypatch.setattr(arms, "Coordinator", _Coordinator)
    out = run(env, "ma_claims")["ma_claims"]
    z = out.meta["z"]
    assert z["IFN_a"] == pytest.approx((2.5 + 1.5) / math.sqrt(2))   # B and C, not A or M
    assert z["HB_a"] == pytest.approx((0.5 - 0.2) / math.sqrt(2))
    assert z["CCL_c"] == pytest.approx(8 / math.sqrt(2))
    assert z["COMP_c"] == pytest.approx(6 / math.sqrt(2))
    assert ids(out) == ["CCL_c", "IFN_b", "COMP_c", "IFN_a"]   # BH over 6 claims; by z
    assert (out.calls, out.nominal_usd) == (4, pytest.approx(0.04))   # ours' step 1


def test_ours_and_mechanism_arms_read_one_protocol_run(env, monkeypatch):
    seen = {}
    monkeypatch.setattr(arms, "run_protocol", _fake_protocol(seen))
    monkeypatch.setattr(arms, "Coordinator", _Coordinator)
    out = run(env, "ours", "no_resolution", "ma_genome_repaired")
    pr = seen["run"]
    assert ids(out["ours"]) == [c.id for c in pr.findings()]
    assert ids(out["no_resolution"]) == [c.id for c in pr.findings_step3()]
    assert "HB_a" in ids(out["ours"]) and "HB_a" not in ids(out["no_resolution"])
    assert "NOISE_b" not in ids(out["ours"]) + ids(out["no_resolution"])
    assert out["ours"].labels["HB_a"] == "corroborated"
    assert out["no_resolution"].labels["HB_a"] == "disputed"
    assert out["ours"].causes == {cid: cause for cid, (_, _, cause) in OURS.items()}
    assert out["no_resolution"].causes == {}
    # cause accuracy is graded over demoted claims too, so ours reports every record's claim
    # with its label, cause and scope; no-resolution reports the same claims with step-3 labels
    every = {d["claim"]["id"]: d for d in out["ours"].meta["all_claims"]}
    assert list(every) == ["IFN_a", "HB_a", "IFN_b", "NOISE_b", "CCL_c"]
    assert every["IFN_a"] == {"claim": claim_to_dict(IFN_A), "label": "corroborated",
                              "cause": None, "scope": None}
    assert (every["NOISE_b"]["label"], every["NOISE_b"]["cause"]) == ("inconclusive",
                                                                      OURS["NOISE_b"][2])
    assert out["ours"].meta["errors"] == []
    step3 = {d["claim"]["id"]: d for d in out["no_resolution"].meta["all_claims"]}
    assert (step3["HB_a"]["label"], step3["HB_a"]["cause"]) == ("disputed", None)
    assert seen["budget"] == 3 and isinstance(seen["coordinator"], _Coordinator)
    assert out["ours"].calls == out["ma_genome_repaired"].calls == 5   # 4 step-1 + 1 repair
    assert out["no_resolution"].calls == 4                             # step 1 only
    assert [rid for rid, _ in env.factory.made].count("t0/ours/s0") == 1
    assert ("CD14 Mono", TREATED) in seen["sites"][0].exported         # MA on ours' views


def test_codex_coord_tools_enforce_the_budget_and_replay(env, monkeypatch):
    handlers, replies, runs = {}, [], []

    def serve(ws, h, stop):
        handlers.update(h)
        thread = threading.Thread(target=stop.wait)
        thread.start()
        return thread

    def run_agent(agent, ws, prompt, *, sample, timeout):
        runs.append(prompt)
        assert agent == "codex"
        assert (ws / "ask_site.py").exists() and (ws / "test_claim.py").exists()
        matrix = json.loads((ws / "claims_matrix.json").read_text())
        assert [row["claim"]["id"] for row in matrix] == ["IFN_a", "HB_a", "IFN_b", "NOISE_b",
                                                          "CCL_c"]
        for site in ["A", "B", "C", "A", "B", "Z"]:
            replies.append(handlers["ask_site"]({"site": site, "request": "check ISG15"}))
        replies.append(handlers["test_claim"]({"site": "B", "claim": claim_to_dict(IFN_A)}))
        replies.append(handlers["test_claim"]({"site": "B", "claim": {"kind": "de"}}))
        (ws / "claims.json").write_text(json.dumps({"claims": [claim_to_dict(IFN_A)]}))
        charge_ledger(0.5)                     # run_codex ledgers the usage it also reports
        return AgentRun("codex", True, [], None, 0.5, 1.0)

    monkeypatch.setattr(arms, "serve_mailbox", serve)
    monkeypatch.setattr(harness, "run_agent", run_agent)
    out = run(env, "codex_coord")["codex_coord"]
    assert ["answer" in r for r in replies[:5]] == [True, True, True, False, False]
    assert "unknown site" in replies[5]["error"]
    assert replies[6]["status"] == "supported" and "error" in replies[7]
    sites = env.factory.sites("t0/base/s0")
    assert sum(len(s.asked) for s in sites) == 3           # B = 2S, S spent in step 1
    assert out.calls == 4 + 1 + 3                          # step 1 + coordinator + asks
    # coordinator once (from the ledger, not again from AgentRun) + the asks' site counters
    assert out.nominal_usd == pytest.approx(0.04 + 0.5 + 0.03)
    assert genes(out) == [{"IFI6", "ISG15", "MX1"}] and out.claims[0].site == "codex_coord"
    again = run(env, "codex_coord")["codex_coord"]         # same sample: replayed, not re-run
    assert len(runs) == 1 and ids(again) == ids(out) and again.nominal_usd == out.nominal_usd


def test_failed_oracle_run_is_charged_but_not_cached(env, monkeypatch):
    outcomes = [AgentRun("codex", False, [], "timeout", 0.2, 900.0),
                AgentRun("codex", True, [], None, 0.3, 60.0)]

    def run_agent(agent, ws, prompt, *, sample, timeout):
        for spec in env.specs:
            assert (ws / spec["name"] / spec["data_path"].name).exists()
            assert json.loads((ws / spec["name"] / "card.json").read_text()) == spec["card"]
        (ws / "claims.json").write_text(json.dumps({"claims": [claim_to_dict(CCL_C)]}))
        outcome = outcomes.pop(0)
        charge_ledger(outcome.nominal_usd_external)   # run_codex ledgers the usage it reports
        return outcome

    monkeypatch.setattr(harness, "run_agent", run_agent)
    failed = run(env, "oracle")["oracle"]
    assert failed.claims == [] and failed.meta["error"] == "timeout"
    assert (failed.calls, failed.nominal_usd) == (1, pytest.approx(0.2))   # charged once
    ok = run(env, "oracle")["oracle"]
    assert genes(ok) == [{"CCL3", "CCL4", "CXCL8"}] and ok.claims[0].site == "oracle"
    assert (ok.calls, ok.nominal_usd) == (1, pytest.approx(0.3))
    assert env.memo.computes == 2 and env.factory.made == []


def test_group_chat_round_robin_stays_within_budget(env, monkeypatch):
    turns = []

    def chat_json(messages, **kwargs):
        turns.append(messages)
        final = "FINAL TURN" in messages[-1]["content"]
        reply = {"message": "please check ISG15", "final": final,
                 "tests": [] if len(turns) > 1 else [{"site": "B", "claim": claim_to_dict(IFN_A)}]}
        if final:
            reply["claims"] = [claim_to_dict(IFN_A)]
        return reply, ChatResult(content=json.dumps(reply), nominal_usd=0.001)

    monkeypatch.setattr(llm, "chat_json", chat_json)
    out = run(env, "group_chat")["group_chat"]
    sites = env.factory.sites("t0/base/s0")
    assert [len(s.asked) for s in sites] == [1, 1, 1, 0]   # budget S = 3: M never gets a turn
    assert len(turns) == 2 and "FINAL TURN" not in turns[0][-1]["content"]
    assert "[tester B]" in turns[1][-1]["content"]         # test results enter the transcript
    assert "please check ISG15" in sites[2].asked[0] and "[A]" in sites[2].asked[0]
    assert genes(out) == [{"IFI6", "ISG15", "MX1"}]
    assert out.calls == 4 + 3 and out.nominal_usd == pytest.approx(0.04 + 0.03 + 0.002)


def test_debate_keeps_claims_revised_by_two_sites(env):
    ccl_from_a = claim("CCL_a", "A", ("CCL3", "CCL4", "IL1B"), rank=2)
    env.factory.templates["A"]["replies"] = [
        json.dumps({"claims": [claim_to_dict(IFN_A), claim_to_dict(ccl_from_a)]})]
    env.factory.templates["B"]["replies"] = [
        "Revised:\n" + json.dumps({"claims": [claim_to_dict(IFN_B)]})]
    env.factory.templates["C"]["replies"] = ["I keep my claims."]
    out = run(env, "debate")["debate"]
    assert genes(out) == [{"IFI6", "ISG15", "MX1"}, {"CCL3", "CCL4", "CXCL8"}]
    assert out.meta["kept_round0"] == ["C"]
    sites = env.factory.sites("t0/base/s0")
    assert [len(s.asked) for s in sites] == [1, 1, 1, 0]
    assert '"NOISE_b"' in sites[0].asked[0]                # every site's claims and tests
    assert out.calls == 4 + 3 and out.nominal_usd == pytest.approx(0.07)


def test_prior_only_gets_the_package_and_no_data(env, monkeypatch):
    seen = []

    def chat_json(messages, **kwargs):
        seen.extend(messages)
        return {"claims": [claim_to_dict(IFN_A)]}, ChatResult(content="", nominal_usd=0.002)

    monkeypatch.setattr(llm, "chat_json", chat_json)
    out = run(env, "prior_only")["prior_only"]
    text = "\n".join(m["content"] for m in seen)
    assert env.case.question in text and "CD14 Mono" in text and '"site": "M"' in text
    assert not any(str(s["data_path"]) in text for s in env.specs)
    assert "At most 30 claims" in text                     # K x proposing sites
    assert out.claims[0].site == "prior" and (out.calls, out.nominal_usd) == (0, 0.002)
    assert env.factory.made == []


def test_every_requested_arm_returns_an_arm_result(env, monkeypatch):
    seen = {}
    final = {"final": True, "message": "done", "claims": [claim_to_dict(IFN_A)]}

    def run_agent(agent, ws, prompt, *, sample, timeout):
        (ws / "claims.json").write_text(json.dumps(final))
        charge_ledger(0.1)
        return AgentRun("codex", True, [], None, 0.1, 1.0)

    def serve(ws, h, stop):
        thread = threading.Thread(target=stop.wait)
        thread.start()
        return thread

    monkeypatch.setattr(arms, "run_protocol", _fake_protocol(seen))
    monkeypatch.setattr(arms, "Coordinator", _Coordinator)
    monkeypatch.setattr(arms, "serve_mailbox", serve)
    monkeypatch.setattr(harness, "run_agent", run_agent)
    monkeypatch.setattr(llm, "chat_json", lambda m, **k: (final, ChatResult(content="")))
    out = run(env, *arms.ARMS)
    per_site = {f"{arm}:{s}" for arm in ("single", "single_codex") for s in "ABC"}
    assert set(out) == per_site | set(arms.ARMS) - {"single", "single_codex"}
    for name, result in out.items():
        assert isinstance(result, arms.ArmResult) and result.arm == name
        assert result.calls >= 0 and result.nominal_usd >= 0
        assert [c.rank for c in result.claims] == list(range(1, len(result.claims) + 1))
    assert sorted(rid for rid, _ in env.factory.made) == [
        "t0/base/s0", "t0/ma/s0", "t0/ours/s0", "t0/single_codex/s0"]
    with pytest.raises(ValueError, match="unknown arms"):
        run(env, "best_of_n")


def test_mailbox_clients_speak_serve_mailbox(tmp_path):
    ws = tmp_path / "coord"
    ws.mkdir()
    (ws / "ask_site.py").write_text(arms.ASK_SITE)
    (ws / "test_claim.py").write_text(arms.TEST_CLAIM)
    (ws / "claim.json").write_text(json.dumps(claim_to_dict(IFN_A)))
    seen = []
    handlers = {"ask_site": lambda req: seen.append(req) or {"answer": "hi " + req["site"]},
                "test_claim": lambda req: seen.append(req) or {"status": "supported"}}
    stop = threading.Event()
    thread = serve_mailbox(ws, handlers, stop)
    try:
        ask = subprocess.run([sys.executable, "ask_site.py", "A", "check ISG15"], cwd=ws,
                             capture_output=True, text=True, timeout=30)
        test = subprocess.run([sys.executable, "test_claim.py", "B", "claim.json"], cwd=ws,
                              capture_output=True, text=True, timeout=30)
    finally:
        stop.set()
        thread.join()
    assert json.loads(ask.stdout) == {"answer": "hi A"}
    assert json.loads(test.stdout) == {"status": "supported"}
    assert not list(ws.glob("req_*.json")) and not list(ws.glob("resp_*.json"))
    assert seen[0]["request"] == "check ISG15" and seen[1]["claim"]["population"] == "CD14 Mono"
