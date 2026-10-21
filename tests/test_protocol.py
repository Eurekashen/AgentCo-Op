"""Protocol tests with fake sites and a fake coordinator: offline, no agents, no API."""

from __future__ import annotations

import json
from dataclasses import replace as dataclasses_replace
from types import SimpleNamespace

import pytest

from agentcoop import llm
from agentcoop.coop import protocol, tester
from agentcoop.coop.claims import Claim
from agentcoop.coop.protocol import ARTEFACT_REASON, Coordinator, label_claim, run_protocol

PARAMS = tester.TesterParams()

# (status, effect, se, reason) of a fake site's answer. Reasons follow Task 5: a computed result
# names its statistic ("module" | "logit" | "auc"), a blocked one names the obstacle.
SUP = ("supported", 0.8, 0.1, "module")
CON = ("contradicted", -0.8, 0.1, "module")
ABS = ("absent", 0.02, 0.05, "module")
INC = ("inconclusive", 0.3, 0.3, "module")
GENES = ("untestable", None, None, "genes")
GROUPS = ("untestable", None, None, "groups")
POP = ("untestable", None, None, "population")
VIEW = ("untestable", None, None, "view")


def claim(cid: str, site: str = "P", *, rank: int = 1, kind: str = "de") -> Claim:
    genes = ("ISG15", "IFI6", "MX1") if kind == "de" else ()
    return Claim(id=cid, site=site, kind=kind, population="CD14 Mono", target="treated",
                 reference="control", up_genes=genes, down_genes=(), effect=1.0, n_units=8,
                 rank=rank)


def result(site: str, answer: tuple, *, cid: str = "c1",
           test: str = "replication") -> tester.TestResult:
    status, effect, se, reason = answer
    ci = None if effect is None else (effect - 1.96 * se, effect + 1.96 * se)
    return tester.TestResult(claim_id=cid, site=site, test=test, status=status, effect=effect,
                             se=se, ci95=ci, n_per_group=(4, 4), mde=None,
                             p=0.001 if status == "supported" else 0.5, competitive_ok=None,
                             reason=reason)


# ---------------------------------------------------------------------------
# label_claim: the one shared decision rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("proposer, others, expected", [
    (ABS, [SUP, SUP], "unsupported"),
    (GENES, [SUP], "unsupported"),
    (SUP, [SUP, CON], "disputed"),
    (SUP, [SUP, SUP, ABS], "corroborated"),
    (SUP, [ABS, ABS], "site-specific"),
    (SUP, [ABS, INC], "site-specific"),
    (SUP, [GENES, POP], "unresolved"),
    (SUP, [VIEW], "unresolved"),
    (SUP, [], "unresolved"),
    (SUP, [INC], "inconclusive"),
    (SUP, [INC, GENES], "inconclusive"),
    # the band follows the statistic each result names: 0.30 ± 0.10 is outside the module band
    # (±0.25) but inside the logit band (±0.5)
    (SUP, [("inconclusive", 0.3, 0.05, "module")], "inconclusive"),
    (SUP, [("inconclusive", 0.3, 0.05, "logit")], "site-specific"),
    # bulk effects are AUC − 0.5, centred on 0 like the others (band ±0.05)
    (SUP, [("inconclusive", 0.02, 0.01, "auc")], "site-specific"),
    (SUP, [("inconclusive", 0.04, 0.01, "auc")], "inconclusive"),
    # effects on different scales are never pooled into an absence
    (SUP, [("absent", 0.0, 0.01, "module"), ("absent", 0.0, 0.01, "auc")], "inconclusive"),
    # a result explained as a site artefact keeps its numbers but is not pooled
    (SUP, [ABS, ("inconclusive", -0.8, 0.1, protocol.ARTEFACT_REASON)], "site-specific"),
])
def test_label_table(proposer, others, expected):
    got = label_claim(result("P", proposer), [result(f"S{i}", a) for i, a in enumerate(others)],
                      PARAMS)
    assert got == expected


def test_label_ignores_scope_tests():
    scope_only = [result("M", SUP, test="scope"), result("M2", CON, test="scope")]
    assert label_claim(result("P", SUP), scope_only, PARAMS) == "unresolved"
    mixed = [result("B", SUP), result("M", CON, test="scope")]
    assert label_claim(result("P", SUP), mixed, PARAMS) == "corroborated"


def marginal(site: str, p: float, *, effect: float = 0.5, competitive_ok=None,
             cid: str = "c1") -> tester.TestResult:
    """A replication result that does not reach q on its own (status inconclusive), raw p kept."""
    return dataclasses_replace(result(site, INC, cid=cid), effect=effect, p=p,
                               competitive_ok=competitive_ok)


def test_replication_pools_the_non_proposer_sites():
    """Two sites each at p 0.03 (inconclusive alone) replicate together (fsdev2-813: real
    programs lost when each 4-vs-4 site had to pass BH alone)."""
    reps = [marginal("A", 0.03), marginal("B", 0.03)]
    assert protocol.replication_p(result("P", SUP), reps) < 0.01
    assert label_claim(result("P", SUP), reps, PARAMS) == "corroborated"
    assert label_claim(result("P", SUP), [marginal("A", 0.03)], PARAMS) == "corroborated"
    assert label_claim(result("P", SUP), [marginal("A", 0.3)], PARAMS) == "inconclusive"
    # a significant effect inside the equivalence band (module: 0.25) is not corroborated
    # (fsdev2-813 headroom: a GO term moving 0.11 log2 at two sites)
    small = [marginal("A", 0.001, effect=0.11), marginal("B", 0.001, effect=0.11)]
    assert label_claim(result("P", SUP), small, PARAMS) == "inconclusive"
    # opposite-direction evidence pulls the pooled z down
    assert label_claim(result("P", SUP), [marginal("A", 0.03), marginal("B", 0.03, effect=-0.5)],
                       PARAMS) == "inconclusive"


def test_pooled_replication_never_counts_the_proposer_artefacts_or_failed_competitive_nulls():
    gate = result("P", SUP)
    assert protocol.replication_p(gate, [result("P", SUP)]) is None
    explained = dataclasses_replace(result("A", SUP), status="inconclusive",
                                    reason=ARTEFACT_REASON)
    assert protocol.replication_p(gate, [explained]) is None
    assert protocol.replication_p(gate, [result("A", GENES)]) is None
    # genes that do not beat random sets cannot raise the pooled z, but can lower it
    assert protocol.replication_p(gate, [marginal("A", 0.001, competitive_ok=False)]) == 0.5
    assert protocol.replication_p(gate, [marginal("A", 0.001, effect=-0.5,
                                                   competitive_ok=False)]) > 0.99


def test_label_claims_applies_bh_over_the_gated_claims():
    """One pooled test per claim; BH over the claims whose gate passed."""
    one = [(result("P", SUP), [marginal("A", 0.04)])]
    assert protocol.label_claims(one, PARAMS) == ["corroborated"]
    many = one + [(result("P", SUP, cid=f"n{i}"), [marginal("A", 0.6, cid=f"n{i}")])
                  for i in range(9)]
    assert protocol.label_claims(many, PARAMS)[0] == "inconclusive"   # 0.02 x 10 > q
    ungated = one + [(result("P", ABS, cid=f"u{i}"), [marginal("A", 0.6, cid=f"u{i}")])
                     for i in range(9)]
    assert protocol.label_claims(ungated, PARAMS)[0] == "corroborated"   # not in the family


class FakeSite:
    """SiteHandle double: canned claims and answers; every agent call costs 1 call and $0.01."""

    def __init__(self, name: str, *, claims: list[Claim] | None = None, answers: dict | None = None,
                 repaired: dict | None = None, scope: dict | None = None, holds: bool = True,
                 proposes: bool = True, propose_error: Exception | None = None,
                 repair_ok: bool = True, repair_error: Exception | None = None) -> None:
        self.name, self.proposes, self.calls, self.nominal_usd = name, proposes, 0, 0.0
        self.claims, self.answers = list(claims or []), dict(answers or {})
        self.repaired_answers, self.scope_answers = dict(repaired or {}), dict(scope or {})
        self._holds, self.propose_error = holds, propose_error
        self.repair_ok, self.repair_error = repair_ok, repair_error
        self.repairs: list[tuple[str, str]] = []
        self.tests: list[tuple[str, str]] = []

    def _charge(self) -> None:
        self.calls += 1
        self.nominal_usd += 0.01

    def propose(self, sample: int) -> list[Claim]:
        self._charge()
        if self.propose_error:
            raise self.propose_error
        return list(self.claims)

    def card(self) -> dict:
        return {"assay": f"{self.name}-assay"}

    def spec(self) -> dict:
        return {"unit": "donor", "site": self.name}

    def holds(self, population: str, contrast: tuple[str, str]) -> bool:
        return self._holds

    def test(self, claim: Claim, *, test: str = "replication",
             contrast: tuple[str, str] | None = None) -> tester.TestResult:
        self.tests.append((claim.id, test))
        if test == "scope":
            answer = self.scope_answers[claim.id]
        elif any(cid == claim.id for cid, _ in self.repairs) and claim.id in self.repaired_answers:
            answer = self.repaired_answers[claim.id]
        else:
            answer = self.answers.get(claim.id, POP)
        return result(self.name, answer, cid=claim.id, test=test)

    def repair(self, claim: Claim, defect: str, sample: int) -> bool:
        self._charge()
        self.repairs.append((claim.id, defect))
        if self.repair_error:
            raise self.repair_error
        return self.repair_ok

    def self_check(self, sample: int) -> bool:
        raise AssertionError("the protocol never runs an untargeted self-check")

    def export_stats(self, population: str, contrast: tuple[str, str]):
        raise AssertionError("the protocol never exports per-gene statistics")

    def ask(self, request: str, sample: int) -> str:
        raise AssertionError("the protocol never asks free-form questions")


class QuietSite(FakeSite):
    """Fails like Task 7's Site: the call is charged, appended to ``failures`` and answered with
    [] / False instead of an exception."""

    def __init__(self, name: str, *, fail: tuple[str, ...], **kw) -> None:
        super().__init__(name, **kw)
        self.fail = fail
        self.failures: list[dict] = []

    def propose(self, sample: int) -> list[Claim]:
        if "propose" not in self.fail:
            return super().propose(sample)
        self._charge()
        self.failures.append({"op": "propose", "error": "codex: timed out after 900 s"})
        return []

    def repair(self, claim: Claim, defect: str, sample: int) -> bool:
        if "repair" not in self.fail:
            return super().repair(claim, defect, sample)
        self._charge()
        self.repairs.append((claim.id, defect))
        self.failures.append({"op": "repair", "error": "finch: harness crashed"})
        return False


class FakeCoordinator:
    """Coordinator double: a fixed verdict per (claim id, triggering site); records each call."""

    def __init__(self, verdicts: dict | None = None) -> None:
        self.verdicts = dict(verdicts or {})
        self.seen: list[tuple[str, str, list[str], list[str]]] = []
        self.nominal_usd = 0.0

    def classify(self, claim, results, specs, cards, *, sample):
        trigger = next(s for s in specs if s != claim.site)
        self.seen.append((claim.id, trigger, sorted(specs), [r.test for r in results]))
        self.nominal_usd += 0.001
        cause, site = self.verdicts.get((claim.id, trigger), ("site-biology", trigger))
        return {"cause": cause, "site": site, "defect": f"{cause} at {site}"}


def case(**kw) -> SimpleNamespace:
    base = dict(name="fake", kind="scrna", question="What changes in treated donors?",
                default_spec={"unit": "donor", "normalization": "log-CPM"}, tester=PARAMS,
                max_claims=10, scope_tests=None, scope_rule=None)
    return SimpleNamespace(**{**base, **kw})


@pytest.fixture(autouse=True)
def bh_families(monkeypatch):
    """Identity BH that records every family it is given (BH itself is tested in test_tester)."""
    families: list[list[tester.TestResult]] = []

    def identity(results, params):
        families.append(list(results))
        return list(results)

    monkeypatch.setattr(protocol, "apply_bh", identity)
    return families


def mimic_scope(index: int):
    """A scope table that sends every claim to ``sites[index]`` for a mimic-vs-control test."""
    def scope_tests(cl: Claim, sites: list) -> list:
        return [(sites[index], ("mimic", "control"))]
    return scope_tests


def by_id(run) -> dict:
    return {r.claim.id: r for r in run.records}


# ---------------------------------------------------------------------------
# run_protocol
# ---------------------------------------------------------------------------

def test_artefact_explained_contradiction_becomes_corroborated():
    c = claim("c1")
    sites = [FakeSite("P", claims=[c], answers={"c1": SUP}), FakeSite("B", answers={"c1": SUP}),
             FakeSite("C", answers={"c1": SUP}), FakeSite("D", answers={"c1": CON})]
    coord = FakeCoordinator({("c1", "D"): ("site-artefact", "D")})
    run = run_protocol(sites, case(), sample=0, followup_budget=4, coordinator=coord)
    rec = run.records[0]
    assert (rec.label_step3, rec.label) == ("disputed", "corroborated")
    assert rec.cause == {"cause": "site-artefact", "site": "D", "defect": "site-artefact at D"}
    final_d = next(r for r in rec.results if r.site == "D")
    assert (final_d.status, final_d.reason) == ("inconclusive", ARTEFACT_REASON)
    assert next(r for r in rec.results_step3 if r.site == "D").status == "contradicted"
    assert run.followups == [] and sites[3].repairs == []
    assert coord.seen == [("c1", "D", ["D", "P"], ["replication"] * 4)]
    assert run.findings() == [c] and run.findings_step3() == []


def test_artefact_at_the_proposer_makes_the_claim_unsupported():
    sites = [FakeSite("P", claims=[claim("c1")], answers={"c1": SUP}),
             FakeSite("B", answers={"c1": SUP}), FakeSite("D", answers={"c1": CON})]
    coord = FakeCoordinator({("c1", "D"): ("site-artefact", "P")})
    run = run_protocol(sites, case(), sample=0, followup_budget=2, coordinator=coord)
    rec = run.records[0]
    assert (rec.label_step3, rec.label) == ("disputed", "unsupported")
    assert (rec.results[0].site, rec.results[0].reason) == ("P", ARTEFACT_REASON)


def test_routing_and_record_layout():
    c = claim("c1")
    sites = [FakeSite("P", claims=[c], answers={"c1": SUP}), FakeSite("B", answers={"c1": SUP}),
             FakeSite("X", holds=False, answers={"c1": SUP}),
             FakeSite("M", proposes=False, holds=False, scope={"c1": SUP})]
    run = run_protocol(sites, case(scope_tests=mimic_scope(3)), sample=0, followup_budget=4,
                       coordinator=FakeCoordinator())
    rec = run.records[0]
    assert [(r.site, r.test) for r in rec.results] == [("P", "replication"), ("B", "replication"),
                                                       ("M", "scope")]
    assert sites[2].tests == []                      # X does not hold population and groups
    assert run.calls == 4                            # one step-1 call per site, tester-only too


def test_unresolved_when_no_other_site_holds_the_claim():
    sites = [FakeSite("P", claims=[claim("c1")], answers={"c1": SUP}),
             FakeSite("B", holds=False), FakeSite("C", holds=False)]
    run = run_protocol(sites, case(), sample=0, followup_budget=3, coordinator=FakeCoordinator())
    assert run.records[0].label == "unresolved"


@pytest.mark.parametrize("answer", [VIEW, POP])
def test_view_and_population_untestables_are_not_resolved(answer):
    """Review focus 1: a site whose view failed validation answers untestable(view); like a
    missing population, that is recorded but sent to no coordinator and no follow-up."""
    sites = [FakeSite("P", claims=[claim("c1")], answers={"c1": SUP}),
             FakeSite("X", answers={"c1": answer}, repaired={"c1": SUP})]
    coord = FakeCoordinator({("c1", "X"): ("handoff-loss", "X")})
    run = run_protocol(sites, case(), sample=0, followup_budget=2, coordinator=coord)
    assert run.records[0].label == "unresolved"
    assert coord.seen == [] and run.followups == []


def test_tester_only_site_claims_are_ignored():
    sites = [FakeSite("P", claims=[claim("c1")], answers={"c1": SUP}),
             FakeSite("M", proposes=False, claims=[claim("m1", "M")])]
    run = run_protocol(sites, case(), sample=0, followup_budget=2, coordinator=FakeCoordinator())
    assert [r.claim.id for r in run.records] == ["c1"]


def test_followup_budget_is_respected():
    claims = [claim(f"c{i}", rank=i) for i in (1, 2, 3)]
    sites = [FakeSite("P", claims=claims, answers={c.id: SUP for c in claims}),
             FakeSite("B", answers={c.id: SUP for c in claims}),
             FakeSite("D", answers={c.id: CON for c in claims},
                      repaired={c.id: SUP for c in claims})]
    coord = FakeCoordinator({(c.id, "D"): ("handoff-loss", "D") for c in claims})
    run = run_protocol(sites, case(), sample=0, followup_budget=2, coordinator=coord)
    assert [f["claim_id"] for f in run.followups] == ["c1", "c2"]
    assert [cid for cid, _ in sites[2].repairs] == ["c1", "c2"]
    labels = {cid: rec.label for cid, rec in by_id(run).items()}
    assert labels == {"c1": "corroborated", "c2": "corroborated", "c3": "disputed"}
    assert by_id(run)["c3"].cause["cause"] == "handoff-loss"   # classified, not followed up
    assert run.calls == 3 + 2


def test_one_followup_per_claim_and_site():
    c = claim("c1")
    sites = [FakeSite("P", claims=[c], answers={"c1": SUP}), FakeSite("C", answers={"c1": CON}),
             FakeSite("D", answers={"c1": CON})]
    coord = FakeCoordinator({("c1", "C"): ("handoff-loss", "P"),
                             ("c1", "D"): ("handoff-loss", "P")})
    run = run_protocol(sites, case(), sample=0, followup_budget=5, coordinator=coord)
    assert [t for _, t, _, _ in coord.seen] == ["C", "D"]
    assert len(sites[0].repairs) == 1 and len(run.followups) == 1
    assert sites[0].tests == [("c1", "replication")] * 2     # gate, then one re-test after repair


@pytest.mark.parametrize("budget, repaired", [(3, ["D", "P"]), (1, ["D"])])
def test_analysis_choice_retests_both_sites_under_the_default_spec(budget, repaired):
    c = claim("c1")
    sites = [FakeSite("P", claims=[c], answers={"c1": SUP}), FakeSite("B", answers={"c1": SUP}),
             FakeSite("D", answers={"c1": CON}, repaired={"c1": SUP})]
    coord = FakeCoordinator({("c1", "D"): ("analysis-choice", "D")})
    case_ = case()
    run = run_protocol(sites, case_, sample=0, followup_budget=budget, coordinator=coord)
    assert [(f["site"], f["cause"], f["ok"]) for f in run.followups] == \
        [(s, "analysis-choice", True) for s in repaired]       # the named site first
    defects = {f["site"]: f["defect"] for f in run.followups}
    spec = json.dumps(case_.default_spec, sort_keys=True)
    assert all(spec in d for d in defects.values())
    assert "analysis-choice at D" in defects["D"]            # the diagnosis goes to D only
    assert all("analysis-choice at D" not in d for s, d in defects.items() if s != "D")
    assert sites[0].tests == [("c1", "replication")] * (1 + ("P" in repaired))
    assert sites[2].tests == [("c1", "replication")] * 2
    assert run.records[0].label == "corroborated"
    assert next(r for r in run.records[0].results if r.site == "D").status == "supported"


def test_proposer_failing_its_default_spec_retest_ends_the_resolution():
    sites = [FakeSite("P", claims=[claim("c1")], answers={"c1": SUP}, repaired={"c1": ABS}),
             FakeSite("D", answers={"c1": CON}, repaired={"c1": SUP})]
    coord = FakeCoordinator({("c1", "D"): ("analysis-choice", "P")})
    run = run_protocol(sites, case(), sample=0, followup_budget=2, coordinator=coord)
    assert [f["site"] for f in run.followups] == ["P"] and sites[1].repairs == []
    rec = run.records[0]
    assert (rec.label_step3, rec.label) == ("disputed", "unsupported")
    assert rec.results[0].status == "absent"                  # the re-test replaced the gate


@pytest.mark.parametrize("answer", [GENES, GROUPS])
def test_view_caused_untestable_triggers_repair(answer):
    c = claim("c1")
    sites = [FakeSite("P", claims=[c], answers={"c1": SUP}),
             FakeSite("B", answers={"c1": answer}, repaired={"c1": SUP})]
    coord = FakeCoordinator({("c1", "B"): ("handoff-loss", "B")})
    run = run_protocol(sites, case(), sample=0, followup_budget=2, coordinator=coord)
    rec = run.records[0]
    assert (rec.label_step3, rec.label) == ("unresolved", "corroborated")
    assert [(f["site"], f["defect"]) for f in run.followups] == [("B", "handoff-loss at B")]


def test_unsupported_claims_are_not_resolved():
    sites = [FakeSite("P", claims=[claim("c1")], answers={"c1": ABS}),
             FakeSite("D", answers={"c1": CON})]
    coord = FakeCoordinator({("c1", "D"): ("handoff-loss", "D")})
    run = run_protocol(sites, case(), sample=0, followup_budget=2, coordinator=coord)
    assert run.records[0].label == "unsupported" and coord.seen == [] and run.followups == []


def test_failed_repair_is_charged_and_not_retested():
    c = claim("c1")
    sites = [FakeSite("P", claims=[c], answers={"c1": SUP}),
             FakeSite("D", answers={"c1": CON}, repaired={"c1": SUP},
                      repair_error=TimeoutError("codex timed out")),
             FakeSite("E", answers={"c1": CON}, repaired={"c1": SUP}, repair_ok=False)]
    coord = FakeCoordinator({("c1", "D"): ("handoff-loss", "D"),
                             ("c1", "E"): ("handoff-loss", "E")})
    run = run_protocol(sites, case(), sample=0, followup_budget=2, coordinator=coord)
    assert [(f["site"], f["ok"]) for f in run.followups] == [("D", False), ("E", False)]
    assert run.errors == [{"site": "D", "op": "repair", "error": "TimeoutError: codex timed out"}]
    assert sites[1].tests == sites[2].tests == [("c1", "replication")]   # no re-test
    assert run.records[0].label == "disputed"
    assert run.calls == 3 + 2


def test_failed_propose_does_not_stop_the_run():
    sites = [FakeSite("P", claims=[claim("c1")], answers={"c1": SUP}),
             FakeSite("X", claims=[claim("x1", "X")], holds=False,
                      propose_error=TimeoutError("15 min wall clock")),
             FakeSite("B", answers={"c1": SUP})]
    run = run_protocol(sites, case(), sample=0, followup_budget=3, coordinator=FakeCoordinator())
    assert [r.claim.id for r in run.records] == ["c1"]
    assert run.records[0].label == "corroborated"
    assert run.errors == [{"site": "X", "op": "propose",
                           "error": "TimeoutError: 15 min wall clock"}]
    assert run.calls == 3 and run.nominal_usd == pytest.approx(0.03)


def test_failures_recorded_by_the_site_reach_the_run():
    """Review focus 3 with Task 7's Site, which records a failed call instead of raising."""
    sites = [FakeSite("P", claims=[claim("c1")], answers={"c1": SUP}),
             QuietSite("X", fail=("propose",), claims=[claim("x1", "X")], answers={"c1": VIEW}),
             QuietSite("D", fail=("repair",), answers={"c1": CON}, repaired={"c1": SUP})]
    sites[2].failures.append({"op": "ask", "error": "from an earlier run"})
    coord = FakeCoordinator({("c1", "D"): ("handoff-loss", "D")})
    run = run_protocol(sites, case(), sample=0, followup_budget=3, coordinator=coord)
    assert run.errors == [{"site": "X", "op": "propose", "error": "codex: timed out after 900 s"},
                          {"site": "D", "op": "repair", "error": "finch: harness crashed"}]
    assert [r.claim.id for r in run.records] == ["c1"]
    assert [(f["site"], f["ok"]) for f in run.followups] == [("D", False)]
    assert sites[2].tests == [("c1", "replication")]            # counted, not re-tested
    assert [(r.site, r.status, r.reason) for r in run.records[0].results][1] == \
        ("X", "untestable", "view")
    assert run.records[0].label == "disputed"
    assert run.calls == 3 + 1


def test_spending_cap_is_not_swallowed():
    sites = [FakeSite("P", claims=[claim("c1")], answers={"c1": SUP}),
             FakeSite("X", propose_error=llm.BudgetExceeded("cap reached"))]
    with pytest.raises(llm.BudgetExceeded):
        run_protocol(sites, case(), sample=0, followup_budget=2, coordinator=FakeCoordinator())


def test_bh_family_is_every_nonproposer_replication_test(bh_families):
    c = claim("c1")
    sites = [FakeSite("P", claims=[c], answers={"c1": SUP}),
             FakeSite("D", answers={"c1": CON}, repaired={"c1": SUP}),
             FakeSite("M", proposes=False, holds=False, scope={"c1": ABS})]
    coord = FakeCoordinator({("c1", "D"): ("handoff-loss", "D")})
    run_protocol(sites, case(scope_tests=mimic_scope(2)), sample=0, followup_budget=1,
                 coordinator=coord)
    step3, final = bh_families
    assert [(r.site, r.test, r.status) for r in step3] == [("D", "replication", "contradicted")]
    assert [(r.site, r.status) for r in final] == [("D", "contradicted"), ("D", "supported")]


def test_scope_results_never_change_labels():
    def federation():
        claims = [claim("c1", rank=1), claim("c2", rank=2), claim("c3", rank=3)]
        return [FakeSite("P", claims=claims, answers={"c1": SUP, "c2": SUP, "c3": SUP}),
                FakeSite("B", answers={"c1": SUP, "c2": CON, "c3": ABS}),
                FakeSite("M", proposes=False, holds=False,
                         scope={"c1": CON, "c2": SUP, "c3": SUP})]

    def shared_if_supported(cl: Claim, results: list[tester.TestResult]) -> str:
        return "shared" if results[0].status == "supported" else "specific"

    plain = run_protocol(federation(), case(), sample=0, followup_budget=0,
                         coordinator=FakeCoordinator())
    scoped_coord = FakeCoordinator()
    scoped = run_protocol(federation(),
                          case(scope_tests=mimic_scope(2), scope_rule=shared_if_supported),
                          sample=0, followup_budget=0, coordinator=scoped_coord)
    assert [(r.label_step3, r.label) for r in scoped.records] == \
        [(r.label_step3, r.label) for r in plain.records]
    assert [r.scope for r in scoped.records] == ["specific", "shared", "shared"]
    assert [r.scope for r in plain.records] == [None, None, None]
    assert all("scope" not in tests for *_, tests in scoped_coord.seen)


def test_findings_order():
    def strong(z: float) -> tuple:
        return ("supported", z * 0.1, 0.1, "module")

    claims = [claim("one_weak", rank=1), claim("two_sites", rank=5), claim("one_strong", rank=4),
              claim("one_weak_late", rank=3), claim("disputed", rank=0)]
    answers_b = {"one_weak": strong(3), "two_sites": strong(3), "one_strong": strong(8),
                 "one_weak_late": strong(3), "disputed": CON}
    # C cannot tell (an absent C would pool the effect inside the band: not corroborated)
    answers_c = {"two_sites": strong(3), "one_weak": INC, "one_strong": INC,
                 "one_weak_late": INC, "disputed": SUP}
    sites = [FakeSite("P", claims=claims, answers={c.id: SUP for c in claims}),
             FakeSite("B", answers=answers_b), FakeSite("C", answers=answers_c)]
    run = run_protocol(sites, case(), sample=0, followup_budget=0, coordinator=FakeCoordinator())
    assert [c.id for c in run.findings()] == ["two_sites", "one_strong", "one_weak",
                                              "one_weak_late"]
    assert [c.id for c in run.findings_step3()] == [c.id for c in run.findings()]


def test_cost_includes_followups_and_coordinator():
    c = claim("c1")
    sites = [FakeSite("P", claims=[c], answers={"c1": SUP}),
             FakeSite("D", answers={"c1": CON}, repaired={"c1": SUP})]
    sites[0].calls, sites[0].nominal_usd = 7, 1.0        # counters from earlier use of the site
    coord = FakeCoordinator({("c1", "D"): ("handoff-loss", "D")})
    run = run_protocol(sites, case(), sample=0, followup_budget=1, coordinator=coord)
    assert run.calls == 3
    assert run.nominal_usd == pytest.approx(0.03 + 0.001)


def test_deterministic():
    def once():
        claims = [claim("c1", rank=1), claim("c2", rank=2)]
        sites = [FakeSite("P", claims=claims, answers={"c1": SUP, "c2": SUP}),
                 FakeSite("B", answers={"c1": SUP, "c2": GENES}, repaired={"c2": SUP}),
                 FakeSite("D", answers={"c1": CON, "c2": SUP}, repaired={"c1": SUP})]
        coord = FakeCoordinator({("c1", "D"): ("analysis-choice", "D"),
                                 ("c2", "B"): ("handoff-loss", "B")})
        run = run_protocol(sites, case(), sample=3, followup_budget=3, coordinator=coord)
        return run.records, run.followups, run.errors, run.calls, [c.id for c in run.findings()]

    assert once() == once()


# ---------------------------------------------------------------------------
# Coordinator: fixed prompt, validated output, every model call charged
# ---------------------------------------------------------------------------

def _classify(monkeypatch, reply) -> tuple[dict, list, Coordinator]:
    """Classify one disagreement with llm.chat_json faked. Each model attempt is ledgered as a
    $0.002 call (chat_json re-asks twice before giving up on non-JSON output)."""
    calls = []

    def fake_chat_json(messages, **kw):
        calls.append((messages, kw))
        for _ in range(3 if isinstance(reply, Exception) else 1):
            llm.record_replay(llm.backbone(), 0.002)
        if isinstance(reply, Exception):
            raise reply
        return reply, llm.ChatResult(content=json.dumps(reply), nominal_usd=0.002)

    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    results = [result("P", SUP), result("D", CON), result("B", SUP)]
    specs = {"P": {"unit": "donor"}, "D": {"unit": "cell"}}
    cards = {"P": {"assay": "10x v3"}, "D": {"assay": "Seq-Well"}}
    coordinator = Coordinator(case())
    verdict = coordinator.classify(claim("c1"), results, specs, cards, sample=4)
    return verdict, calls, coordinator


def test_coordinator_accepts_valid_verdict(monkeypatch):
    reply = {"cause": "analysis-choice", "site": "D", "defect": "cells used as the sample unit"}
    verdict, calls, coordinator = _classify(monkeypatch, reply)
    assert verdict == reply
    (messages, kw), = calls
    assert kw["sample"] == 4 and messages[0]["content"] == protocol.COORDINATOR_PROMPT
    payload = json.loads(messages[-1]["content"])
    assert set(payload["sites"]) == {"P", "D"}
    assert payload["sites"]["D"] == {"spec": {"unit": "cell"}, "card": {"assay": "Seq-Well"}}
    assert [r["site"] for r in payload["results"]] == ["P", "D", "B"]
    assert coordinator.nominal_usd == pytest.approx(0.002)


@pytest.mark.parametrize("reply, usd", [
    ({"cause": "bad luck", "site": "D", "defect": "x"}, 0.002),
    ({"cause": "handoff-loss", "site": "B", "defect": "B is not involved"}, 0.002),
    ({"cause": "handoff-loss", "site": "D"}, 0.002),
    (["handoff-loss", "D"], 0.002),
    (ValueError("model did not return JSON"), 0.006),
])
def test_coordinator_invalid_output_falls_back_and_is_charged(monkeypatch, reply, usd):
    verdict, _, coordinator = _classify(monkeypatch, reply)
    assert verdict == {"cause": "site-biology", "site": "D", "defect": "unclassified"}
    assert coordinator.nominal_usd == pytest.approx(usd)
