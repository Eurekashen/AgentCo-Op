"""Cross-examination protocol (spec §2.5): propose, route tests, label, resolve, re-label.

Objective-free by construction: a claim is kept by agreement between independently held
resources, never by comparing alternatives on a score, and no reference answer is read. Step 4
resolves a disagreement at most once per (claim, site), in one pass, with a re-test spec fixed
in advance and a fixed follow-up budget.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol

from scipy.stats import false_discovery_control, norm

from agentcoop import llm
from agentcoop.coop.claims import Claim, claim_to_dict
# _BANDS (statistic named in TestResult.reason -> TesterParams field) is the table apply_bh uses;
# reading the same one keeps a pooled "absent" and a per-site "absent" on one band.
from agentcoop.coop.tester import _BANDS, TesterParams, TestResult, apply_bh

if TYPE_CHECKING:  # site.py implements SiteHandle on top of this module
    import pandas as pd

    from agentcoop.coop.site import CaseSpec

LABELS = ("unsupported", "disputed", "corroborated", "site-specific", "unresolved", "inconclusive")
CAUSES = ("analysis-choice", "handoff-loss", "site-artefact", "site-biology")
#: Untestable reasons that step 4 resolves (spec §2.5 step 4; Task 5's reason tokens). A missing
#: population is a routing matter (``holds``), and ``view`` marks a site whose own step-1 call
#: failed, not a claim-level handoff loss.
VIEW_CAUSES = ("genes", "groups")
ARTEFACT_REASON = "explained: site artefact"
_Z95 = 1.959963984540054


def _has_estimate(r: TestResult) -> bool:
    return (r.effect is not None and r.se is not None and math.isfinite(r.effect)
            and math.isfinite(r.se) and r.se > 0)


def _pooled(results: list[TestResult], params: TesterParams) -> tuple[float, float, float] | None:
    """Inverse-variance fixed-effect (mean, 95% half-width, band) of the computed results. Every
    tester effect is centred on 0 with positive = the claimed direction (Task 5; bulk = AUC −
    0.5), and the band is the one of the statistic the results name in ``reason``. Results that
    name no statistic (an explained artefact) are left out, and effects on different scales are
    never pooled (None)."""
    usable = [r for r in results if r.reason in _BANDS and _has_estimate(r)]
    scales = {r.reason for r in usable}
    if len(scales) != 1:
        return None
    weights = [1.0 / r.se ** 2 for r in usable]
    mean = sum(w * r.effect for w, r in zip(weights, usable)) / sum(weights)
    return mean, _Z95 / math.sqrt(sum(weights)), getattr(params, _BANDS[scales.pop()])


def _pooled_absent(results: list[TestResult], params: TesterParams) -> bool:
    """Is the pooled 95% CI inside the equivalence band?"""
    pooled = _pooled(results, params)
    if pooled is None:
        return False
    mean, half, delta = pooled
    return -delta < mean - half and mean + half < delta     # strict, as the testers' band


def _beyond_band(results: list[TestResult], params: TesterParams) -> bool:
    """Is the pooled effect larger than the equivalence band? An effect inside the band is what
    the testers call absent, so it is never corroborated however small its p."""
    pooled = _pooled(results, params)
    return pooled is not None and pooled[0] > pooled[2]


def replication_p(proposer: TestResult, others: list[TestResult]) -> float | None:
    """One-sided p that the claim replicates, pooled over the non-proposer replication results:
    Stouffer's z of each computed result's one-sided p in the claimed direction (each from its
    own site's test, so small sites keep their t tails). A result that fails the competitive
    null (its genes do not move more than random sets) may lower the pooled z but never raise
    it; an explained artefact or a result without a statistic does not count. None: nothing to
    pool. The proposer never counts, so selecting claims by their gate does not bias it."""
    zs = []
    for r in others:
        if r.test != "replication" or r.site == proposer.site or r.reason not in _BANDS:
            continue
        if r.p is None or r.effect is None or not math.isfinite(r.effect):
            continue
        one_sided = r.p / 2 if r.effect > 0 else 1 - r.p / 2
        z = float(norm.isf(min(max(one_sided, 1e-300), 1 - 1e-16)))
        zs.append(min(z, 0.0) if r.competitive_ok is False else z)
    return float(norm.sf(sum(zs) / math.sqrt(len(zs)))) if zs else None


def label_claim(proposer: TestResult, others: list[TestResult], params: TesterParams, *,
                replicated: bool | None = None) -> str:
    """The one shared decision rule (spec §2.5 step 3; ordered, first match wins).

    ``others`` are the non-proposer results; anything that is not a non-proposer replication test
    is dropped here, so scope evidence can never move a label. ``inconclusive`` and ``untestable``
    are never support nor contradiction. A claim is corroborated when its pooled replication
    evidence (``replication_p``) passes BH over the run's gated claims - ``label_claims`` decides
    ``replicated``; alone (None), the claim is its own family - and its pooled effect is larger
    than the equivalence band. Contradiction is per site.
    """
    if proposer.status != "supported":
        return "unsupported"
    others = [r for r in others if r.test == "replication" and r.site != proposer.site]
    if any(r.status == "contradicted" for r in others):
        return "disputed"
    if replicated is None:
        p = replication_p(proposer, others)
        replicated = p is not None and p <= params.q
    if replicated and _beyond_band(others, params):
        return "corroborated"
    if _pooled_absent(others, params):
        return "site-specific"
    if all(r.status == "untestable" for r in others):
        return "unresolved"
    return "inconclusive"


def label_claims(items: list[tuple[TestResult, list[TestResult]]],
                 params: TesterParams) -> list[str]:
    """Labels of a run's claims, (proposer gate, non-proposer results) each: BH at q over the
    pooled replication p of the claims whose gate passed (one test per claim, so a replication
    family grows with claims, not with claims x sites), then ``label_claim``."""
    pooled = {i: replication_p(gate, others) for i, (gate, others) in enumerate(items)
              if gate.status == "supported"}
    pooled = {i: p for i, p in pooled.items() if p is not None}
    adjusted = dict(zip(pooled, false_discovery_control(list(pooled.values())))) if pooled else {}
    return [label_claim(gate, others, params, replicated=adjusted.get(i, 1.0) <= params.q)
            for i, (gate, others) in enumerate(items)]


@dataclass
class ClaimRecord:
    claim: Claim
    results: list[TestResult]       # final: proposer gate, then replication tests, then scope tests
    label_step3: str
    label: str
    cause: dict | None              # first resolution of this claim: {"cause", "site", "defect"}
    scope: str | None
    results_step3: list[TestResult] = field(default_factory=list)   # same layout, before step 4


def _ranked(items: list[tuple[Claim, list[TestResult]]]) -> list[Claim]:
    """More supporting non-proposer sites first, then larger |Stouffer z| (z = effect / se over
    the supporting results; effects are centred on 0), then the proposer's own rank."""
    def key(item: tuple[Claim, list[TestResult]]) -> tuple[int, float, int]:
        claim, results = item
        support = [r for r in results if r.test == "replication" and r.site != claim.site
                   and r.status == "supported"]
        z = [r.effect / r.se for r in support if _has_estimate(r)]
        stouffer = abs(sum(z)) / math.sqrt(len(z)) if z else 0.0
        return -len(support), -stouffer, claim.rank

    return [claim for claim, _ in sorted(items, key=key)]


@dataclass
class ProtocolRun:
    records: list[ClaimRecord]
    followups: list[dict]           # {"claim_id", "site", "cause", "defect", "ok"}
    calls: int                      # site-agent calls (step 1 + follow-ups)
    nominal_usd: float
    errors: list[dict] = field(default_factory=list)   # failed agent calls: {"site", "op", "error"}

    def findings(self) -> list[Claim]:
        return _ranked([(r.claim, r.results) for r in self.records if r.label == "corroborated"])

    def findings_step3(self) -> list[Claim]:
        return _ranked([(r.claim, r.results_step3) for r in self.records
                        if r.label_step3 == "corroborated"])


class SiteHandle(Protocol):
    """Everything the protocol and the arms may do with a site: summaries, never cell-level data."""

    name: str
    proposes: bool
    calls: int
    nominal_usd: float

    def propose(self, sample: int) -> list[Claim]: ...
    def card(self) -> dict: ...
    def spec(self) -> dict: ...
    def holds(self, population: str, contrast: tuple[str, str]) -> bool: ...
    def test(self, claim: Claim, *, test: str = "replication",
             contrast: tuple[str, str] | None = None) -> TestResult: ...
    def repair(self, claim: Claim, defect: str, sample: int) -> bool: ...
    def self_check(self, sample: int) -> bool: ...
    def export_stats(self, population: str, contrast: tuple[str, str]) -> pd.DataFrame: ...
    def ask(self, request: str, sample: int) -> str: ...


COORDINATOR_PROMPT = """\
You coordinate a federation of independent analysis sites. Each site holds private data, built \
its own machine-readable view of it (gene mapping, sample groups, cell populations; described by \
its analysis spec) and ran a fixed statistical test of one claim on that view. The sites disagree \
about the claim, or a site could not test it. Using only the claim, the test results, the sites' \
analysis specs and their metadata cards, classify the most likely cause:

- "analysis-choice": the sites made different analysis choices (sample unit, normalization, \
exclusions, covariates, contrast definition) that can explain the disagreement.
- "handoff-loss": a site's view lost or garbled information the test needs (gene identifiers not \
mapped to symbols, cell-type or condition codes mapped to the wrong vocabulary label, groups \
swapped or merged).
- "site-artefact": the compared groups at one site differ in something technical recorded in its \
metadata card (assay, chemistry, source lab, batch), so that site's result cannot be trusted.
- "site-biology": none of the above is evident; the site's population may genuinely differ.

Name the one site whose view or analysis is at fault ("site", one of the sites given) and describe \
the defect in one or two sentences ("defect"). The defect is shown to that site's agent so it can \
fix its own view: describe only that site's own view or spec, never another site's results or \
statistics. Answer with a JSON object {"cause": ..., "site": ..., "defect": ...}."""


class Coordinator:
    """Classifies why sites disagree about a claim (spec §2.5 step 4).

    It sees claims, summary results, analysis specs and metadata cards — never sample- or
    cell-level data — and it does not choose the re-test spec: an analysis-choice verdict is acted
    on with the case's pre-registered default spec. Output that fails validation becomes
    "site-biology / unclassified", which triggers no follow-up. ``nominal_usd`` is read from the
    ledger, so chat_json's re-asks after non-JSON replies are charged too.
    """

    def __init__(self, case: "CaseSpec") -> None:
        self.case = case
        self.nominal_usd = 0.0

    def classify(self, claim: Claim, results: list[TestResult], specs: dict[str, dict],
                 cards: dict[str, dict], *, sample: int) -> dict:
        involved = list(specs)
        fallback = {"cause": "site-biology", "defect": "unclassified",
                    "site": next((s for s in involved if s != claim.site), claim.site)}
        payload = {
            "question": self.case.question,
            "claim": claim_to_dict(claim),
            "results": [{k: v for k, v in dataclasses.asdict(r).items() if k != "claim_id"}
                        for r in results],
            "sites": {s: {"spec": specs[s], "card": cards.get(s, {})} for s in involved},
        }
        messages = [{"role": "system", "content": COORDINATOR_PROMPT},
                    {"role": "user", "content": json.dumps(payload, sort_keys=True, default=str)}]
        with llm.tag(f"coordinator#{claim.id}.{time.time_ns()}"):
            tag = llm.current_tag()
            try:
                verdict, _ = llm.chat_json(messages, sample=sample)
            except ValueError:      # no JSON object after chat_json's re-asks
                verdict = None
        self.nominal_usd += llm.spent(tag)["nominal_usd"]
        if (not isinstance(verdict, dict) or verdict.get("cause") not in CAUSES
                or verdict.get("site") not in involved
                or not isinstance(verdict.get("defect"), str)):
            return fallback
        return {"cause": verdict["cause"], "site": verdict["site"], "defect": verdict["defect"]}


@dataclass
class _Track:
    """Mutable state of one claim while a run is in progress."""

    claim: Claim
    gate: TestResult
    reps: dict[str, int]            # site -> index of its current test in the run's BH family
    scope: list[TestResult]
    explained: set[str] = field(default_factory=set)   # sites whose result is a site artefact

    def replication(self, adjusted: list[TestResult]) -> list[TestResult]:
        """Current replication results. Explanations are applied after BH, so an adjustment can
        never revive a result that was explained away."""
        return [dataclasses.replace(adjusted[i], status="inconclusive", reason=ARTEFACT_REASON)
                if site in self.explained else adjusted[i] for site, i in self.reps.items()]

    def explain(self, site: str) -> None:
        if site == self.claim.site:
            self.gate = dataclasses.replace(self.gate, status="inconclusive",
                                            reason=ARTEFACT_REASON)
        else:
            self.explained.add(site)

    def retest(self, site: SiteHandle, family: list[TestResult]) -> None:
        if site.name == self.claim.site:
            self.gate = site.test(self.claim)
        else:
            self.reps[site.name] = len(family)
            family.append(site.test(self.claim))


def _bh(family: list[TestResult], params: TesterParams) -> list[TestResult]:
    return apply_bh(family, params) if family else []


def _needs_resolution(result: TestResult) -> bool:
    return result.status == "contradicted" or (
        result.status == "untestable" and result.reason in VIEW_CAUSES)


def _followup_sites(verdict: dict, involved: list[str]) -> list[str]:
    """Who gets a follow-up (spec §2.5 step 4). Handoff loss: the site whose view lost
    information. Analysis choice: both involved sites, the one the coordinator named first, so a
    budget that runs out mid-pair still serves the site it blamed. Artefact, biology: nobody."""
    if verdict["cause"] == "handoff-loss":
        return [verdict["site"]]
    if verdict["cause"] == "analysis-choice":
        return [verdict["site"], *(s for s in involved if s != verdict["site"])]
    return []


def _defect_text(verdict: dict, site: str, case: "CaseSpec") -> str:
    """What a site agent is told. Analysis choice: re-run under the pre-registered default spec
    (never one the coordinator chose); the coordinator's diagnosis is added only for the site it
    describes, so no site reads about another site's analysis."""
    if verdict["cause"] == "handoff-loss":
        return verdict["defect"]
    spec = json.dumps(case.default_spec, sort_keys=True)
    rerun = ("Re-run the analysis of this claim under the pre-registered default spec and update "
             f"your view and spec.json to follow it: {spec}")
    return f"{verdict['defect']}\n{rerun}" if site == verdict["site"] else rerun


def _attempt(site: SiteHandle, op: str, call: Callable[[], Any], errors: list[dict]) -> Any:
    """One site-agent call, which the site has charged whatever happens. A failure is recorded in
    ``errors`` and the run goes on: Task 7's Site appends it to ``site.failures`` and returns
    [] / False, another SiteHandle may raise. The spending cap propagates, because a run cut
    short by it is not a scientific result."""
    seen = len(getattr(site, "failures", []))
    try:
        return call()
    except llm.BudgetExceeded:
        raise
    except Exception as exc:  # timeout or harness crash: one site must not stop the federation
        errors.append({"site": site.name, "op": op, "error": f"{type(exc).__name__}: {exc}"})
        return None
    finally:
        errors.extend({"site": site.name, **f} for f in getattr(site, "failures", [])[seen:])


def run_protocol(sites: list[SiteHandle], case: "CaseSpec", *, sample: int, followup_budget: int,
                 coordinator: Coordinator | None = None) -> ProtocolRun:
    """Steps 1–5 of spec §2.5 over one federation, for draw ``sample``.

    ``calls`` and ``nominal_usd`` count what this run added to the sites' counters (plus the
    coordinator's LLM cost), so Site objects may be reused across runs.
    """
    coordinator = coordinator or Coordinator(case)
    params = case.tester
    by_name = {site.name: site for site in sites}
    calls0 = sum(site.calls for site in sites)
    usd0 = sum(site.nominal_usd for site in sites) + coordinator.nominal_usd
    errors: list[dict] = []

    # Step 1: one budgeted call per site; a tester-only site's call builds its view only.
    claims: list[Claim] = []
    for site in sites:
        proposed = _attempt(site, "propose", partial(site.propose, sample), errors)
        if site.proposes and proposed:
            claims.extend(proposed)

    # Step 2: proposer gate; replication at every other site holding the population and both
    # groups; scope tests from the case's scope table (they feed only the scope rule).
    family: list[TestResult] = []   # every non-proposer replication test, re-tests included
    tracks: list[_Track] = []
    for claim in claims:
        gate = by_name[claim.site].test(claim)
        reps: dict[str, int] = {}
        for site in sites:
            if site.name != claim.site and site.holds(claim.population, claim.contrast):
                reps[site.name] = len(family)
                family.append(site.test(claim))
        routed = case.scope_tests(claim, sites) if case.scope_tests else []
        scope = [site.test(claim, test="scope", contrast=contrast) for site, contrast in routed]
        tracks.append(_Track(claim, gate, reps, scope))

    # Step 3: the shared label function over BH-adjusted results.
    adjusted = _bh(family, params)
    step3 = [(t.gate, t.replication(adjusted)) for t in tracks]
    labels3 = label_claims(step3, params)

    # Step 4: one pass over contradictions and view-caused untestables of claims whose proposer
    # gate passed; at most one follow-up per (claim, site), none once the budget is spent. The
    # coordinator sees the step-3 record.
    followups: list[dict] = []
    causes: list[dict | None] = [None] * len(tracks)
    followed: set[tuple[int, str]] = set()
    for i, (track, (gate3, reps3)) in enumerate(zip(tracks, step3)):
        for trigger in reps3:
            if track.gate.status != "supported":
                break               # unsupported, or the proposer's evidence fell after a repair
            if not _needs_resolution(trigger):
                continue
            involved = [track.claim.site, trigger.site]
            verdict = coordinator.classify(
                track.claim, [gate3, *reps3], {s: by_name[s].spec() for s in involved},
                {s: by_name[s].card() for s in involved}, sample=sample)
            causes[i] = causes[i] or verdict
            if verdict["cause"] == "site-artefact":
                track.explain(verdict["site"])
            for name in _followup_sites(verdict, involved):
                if track.gate.status != "supported":
                    break           # the proposer's own re-test failed: nothing left to resolve
                if (i, name) in followed or len(followups) >= followup_budget:
                    continue
                followed.add((i, name))
                site, defect = by_name[name], _defect_text(verdict, name, case)
                repair = partial(site.repair, track.claim, defect, sample)
                ok = bool(_attempt(site, "repair", repair, errors))
                followups.append({"claim_id": track.claim.id, "site": name,
                                  "cause": verdict["cause"], "defect": defect, "ok": ok})
                if ok:
                    track.retest(site, family)

    # Step 5: BH over the final family, the same label function, the case's scope rule.
    adjusted = _bh(family, params)
    final = [(t.gate, t.replication(adjusted)) for t in tracks]
    labels = label_claims(final, params)
    records = []
    for i, (track, (gate3, reps3)) in enumerate(zip(tracks, step3)):
        reps = final[i][1]
        records.append(ClaimRecord(
            claim=track.claim, results=[track.gate, *reps, *track.scope],
            label_step3=labels3[i], label=labels[i], cause=causes[i],
            scope=case.scope_rule(track.claim, track.scope) if case.scope_rule else None,
            results_step3=[gate3, *reps3, *track.scope]))
    return ProtocolRun(
        records=records, followups=followups,
        calls=sum(site.calls for site in sites) - calls0,
        nominal_usd=sum(site.nominal_usd for site in sites) + coordinator.nominal_usd - usd0,
        errors=errors)
