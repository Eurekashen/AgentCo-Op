"""Offline stand-ins shared by the coop tests: scripted sites, a make_sites factory, a memo store
and a workspace maker. No agent, network or API key involved."""

from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
from scipy.stats import norm

from agentcoop import llm
from agentcoop.coop.claims import Claim, Vocabulary
from agentcoop.coop.site import CaseSpec
from agentcoop.coop.tester import TesterParams, TestResult

VOCAB = Vocabulary(populations=("CD14 Mono", "B"), groups=("control", "treated", "mimic"),
                   contrasts=(("treated", "control"), ("mimic", "control")))
TREATED = ("treated", "control")
MIMIC = ("mimic", "control")
PROGRAMS = {
    "IFN": frozenset({"IFI6", "ISG15", "MX1", "OAS1", "IFIT1", "IFIT3"}),
    "HB": frozenset({"HBA1", "HBA2", "HBB", "HBD"}),
    "CCL": frozenset({"CCL3", "CCL4", "CXCL8", "IL1B"}),
    "NOISE": frozenset({"ACTB", "GAPDH", "B2M"}),
}
STATS_COLUMNS = ["gene", "effect", "se", "n1", "n2"]


def case_spec(**overrides) -> CaseSpec:
    fields = dict(name="fake", kind="scrna", vocab=VOCAB,
                  question="Which cell-type programs change with treatment?",
                  default_spec={"unit": "donor"}, tester=TesterParams(), max_claims=10)
    return CaseSpec(**{**fields, **overrides})


def claim(cid: str, site: str, up: tuple[str, ...] = (), *, down: tuple[str, ...] = (),
          kind: str = "de", population: str = "CD14 Mono", contrast: tuple[str, str] = TREATED,
          effect: float = 1.0, rank: int = 1) -> Claim:
    return Claim(id=cid, site=site, kind=kind, population=population, target=contrast[0],
                 reference=contrast[1], up_genes=tuple(up), down_genes=tuple(down), effect=effect,
                 n_units=8, rank=rank)


def program_of(c: Claim) -> str | None:
    """The planted program a claim is about: its population for composition, else the program
    sharing most genes with it."""
    if c.kind == "composition":
        return f"comp:{c.population}"
    genes = set(c.up_genes) | set(c.down_genes)
    best = max(PROGRAMS, key=lambda p: len(PROGRAMS[p] & genes))
    return best if PROGRAMS[best] & genes else None


def result(c: Claim, site: str, raw: float | None, se: float | None, *,
           test: str = "replication") -> TestResult:
    """A test result in Task 5's convention from a raw effect (target minus reference) and its
    SE: ``effect`` is oriented so that positive = the claimed direction (a composition claim with
    a negative effect flips it), centred on 0; supported or contradicted when |z| >= 1.96 with or
    against the claim; untestable without data."""
    if raw is None or se is None:
        return TestResult(claim_id=c.id, site=site, test=test, status="untestable", effect=None,
                          se=None, ci95=None, n_per_group=None, mde=None, p=None,
                          competitive_ok=None, reason="fake: not measured")
    effect = -raw if c.kind == "composition" and (c.effect or 0.0) < 0 else raw
    z = effect / se
    status = "supported" if z >= 1.96 else "contradicted" if z <= -1.96 else "inconclusive"
    return TestResult(claim_id=c.id, site=site, test=test, status=status, effect=effect, se=se,
                      ci95=(effect - 1.96 * se, effect + 1.96 * se), n_per_group=(4, 4), mde=None,
                      p=float(2 * norm.sf(abs(z))), competitive_ok=True, reason="fake")


def gene_stats(effects: dict[str, tuple[float, float]],
               n: tuple[int, int] = (4, 4)) -> pd.DataFrame:
    return pd.DataFrame([{"gene": g, "effect": e, "se": s, "n1": n[0], "n2": n[1]}
                         for g, (e, s) in effects.items()], columns=STATS_COLUMNS)


@dataclass
class FakeSite:
    """A SiteHandle with scripted claims, per-program effects, gene stats and replies. Every
    agent call (propose, repair, self_check, ask) costs ``usd`` nominal, like a real Site."""
    name: str
    claims: list[Claim] = field(default_factory=list)
    effects: dict[str, tuple[float, float]] = field(default_factory=dict)  # program -> (raw, se)
    held: set = field(default_factory=lambda: {(p, TREATED) for p in VOCAB.populations})
    stats: dict = field(default_factory=dict)          # (population, contrast) -> gene stats
    replies: list[str] = field(default_factory=list)   # ask() answers in order, then "ok"
    proposes: bool = True
    agent: str = "codex"
    usd: float = 0.01
    calls: int = 0
    nominal_usd: float = 0.0
    asked: list[str] = field(default_factory=list)
    exported: list = field(default_factory=list)
    checks: int = 0

    def _call(self) -> None:
        self.calls += 1
        self.nominal_usd += self.usd

    def propose(self, sample: int) -> list[Claim]:
        self._call()
        return list(self.claims) if self.proposes else []

    def card(self) -> dict:
        return {"site": self.name}

    def spec(self) -> dict:
        return {"unit": "donor"}

    def holds(self, population: str, contrast: tuple[str, str]) -> bool:
        return (population, tuple(contrast)) in self.held

    def test(self, claim: Claim, *, test: str = "replication",
             contrast: tuple[str, str] | None = None) -> TestResult:
        program = program_of(claim)
        if program not in self.effects or not self.holds(claim.population, claim.contrast):
            return result(claim, self.name, None, None, test=test)
        return result(claim, self.name, *self.effects[program], test=test)

    def repair(self, claim: Claim, defect: str, sample: int) -> bool:
        self._call()
        return True

    def self_check(self, sample: int) -> bool:
        self._call()
        self.checks += 1
        return True

    def export_stats(self, population: str, contrast: tuple[str, str]) -> pd.DataFrame:
        self.exported.append((population, tuple(contrast)))
        return self.stats.get((population, tuple(contrast)), gene_stats({}))

    def ask(self, request: str, sample: int) -> str:
        self._call()
        self.asked.append(request)
        return self.replies.pop(0) if self.replies else "ok"


class SiteFactory:
    """make_sites stand-in: fresh FakeSites per call from ``templates`` (site name -> FakeSite
    kwargs), with ``proposes`` and the agent taken from the site spec unless ``agents`` overrides
    it; ``made`` logs (run_id, sites) per call."""

    def __init__(self, templates: dict[str, dict]) -> None:
        self.templates = templates
        self.made: list[tuple[str, list[FakeSite]]] = []

    def __call__(self, site_specs: list[dict], case: CaseSpec, *, task_id: str, run_id: str,
                 agents: dict[str, str] | None = None) -> list[FakeSite]:
        sites = [FakeSite(**{**copy.deepcopy(self.templates[s["name"]]), "name": s["name"],
                             "proposes": s["proposes"],
                             "agent": (agents or {}).get(s["name"], s["agent"])})
                 for s in site_specs]
        self.made.append((run_id, sites))
        return sites

    def sites(self, run_id: str) -> list[FakeSite]:
        return next(sites for rid, sites in self.made if rid == run_id)


def charge_ledger(usd: float) -> None:
    """Ledger ``usd`` nominal ($0 actual) under the current tag: what run_codex does for a Codex
    run's usage (llm.record_external(..., billed=False)), without pricing tokens. The autouse
    conftest fixture keeps the ledger in tmp_path."""
    llm.record_replay(llm.backbone(), usd)


class FakeMemo:
    """store.memo stand-in: first writer wins, exceptions are never cached (their spend stays in
    the ledger), and nominal_usd is what store.memo stores: the ledger spend under the compute's
    own nested tag plus any nominal_usd_external the compute reports."""

    def __init__(self) -> None:
        self.entries: dict[str, dict] = {}
        self.computes = 0

    def __call__(self, key: dict, compute: Callable[[], dict], *, tag: str) -> dict:
        k = json.dumps(key, sort_keys=True, default=str)
        if k not in self.entries:
            self.computes += 1
            with llm.tag(f"{tag}#{self.computes}.{time.time_ns()}"):
                exec_tag = llm.current_tag()
                out = compute()
            nominal = llm.spent(exec_tag)["nominal_usd"] + out.get("nominal_usd_external", 0.0)
            self.entries.setdefault(k, {**out, "nominal_usd": nominal})
        return dict(self.entries[k])


def fake_workspace(root: Path) -> Callable[[str, dict[str, Path]], Path]:
    """new_workspace stand-in: plain copies under ``root`` (a reused name keeps its directory)."""
    def new_workspace(name: str, files: dict[str, Path]) -> Path:
        ws = root / name
        ws.mkdir(parents=True, exist_ok=True)
        for rel, src in files.items():
            (ws / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, ws / rel)
        return ws
    return new_workspace


# The standard federation: A, B, C propose; M is a tester-only mimic site.
IFN_A = claim("IFN_a", "A", ("IFI6", "ISG15", "MX1"))
HB_A = claim("HB_a", "A", ("HBA1", "HBB", "HBD"), rank=2)
IFN_B = claim("IFN_b", "B", ("ISG15", "MX1", "OAS1"))
NOISE_B = claim("NOISE_b", "B", ("ACTB", "GAPDH", "B2M"), rank=2)
CCL_C = claim("CCL_c", "C", ("CCL3", "CCL4", "CXCL8"))


def standard_sites() -> dict[str, dict]:
    """Own-tester z: IFN_a 5, HB_a 3, IFN_b 2.5, NOISE_b 0.2 (fails), CCL_c 6."""
    return {
        "A": {"claims": [IFN_A, HB_A],
              "effects": {"IFN": (1.0, 0.2), "HB": (0.6, 0.2), "CCL": (0.8, 0.2),
                          "NOISE": (0.0, 0.5)}},
        "B": {"claims": [IFN_B, NOISE_B],
              "effects": {"IFN": (0.75, 0.3), "HB": (0.1, 0.2), "CCL": (0.8, 0.2),
                          "NOISE": (0.1, 0.5)}},
        "C": {"claims": [CCL_C],
              "effects": {"IFN": (0.45, 0.3), "HB": (-0.04, 0.2), "CCL": (1.2, 0.2),
                          "NOISE": (0.0, 0.5)}},
        "M": {"held": {(p, MIMIC) for p in VOCAB.populations}},
    }


def site_specs(root: Path) -> list[dict]:
    specs = []
    for name, agent in (("A", "codex"), ("B", "finch"), ("C", "codex"), ("M", "finch")):
        path = root / "data" / f"{name}.h5ad"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"site {name} counts")
        specs.append({"name": name, "data_path": path, "card": {"site": name},
                      "proposes": name != "M", "agent": agent})
    return specs
