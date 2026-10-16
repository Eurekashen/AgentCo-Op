"""Every arm of spec §4 against one site interface (``SiteHandle``), plus ``ArmResult``.

A call of ``run_arms`` builds each site set it needs once: the shared step-1 set S0 (run id
``<task>/base/s<sample>``), whose proposals and claim x site test matrix feed every arm that starts
from independent site analyses, and one set per view-mutating arm family (``ours`` repairs views,
``ma`` self-checks them, ``single_codex`` swaps the agents). ``Site.propose`` is memoized by
(task, site, agent, sample), so every set starts from the same step-1 claims and views (review
focus 4). The mechanism arms read ours' own ``ProtocolRun``, so they are paired with ours by
construction. Test effects follow the testers' convention: positive = the claimed direction,
centred on 0 for every kind (bulk AUC - 0.5). Costs follow nominal semantics: an arm is charged
every agent call it relies on, shared step-1 calls included, whether or not the store served
them. ``calls`` counts external-agent runs (site agents, the Codex coordinator, the oracle);
plain LLM turns are charged in ``nominal_usd``. A lead agent's Codex usage is ledgered once by
``run_codex`` under the memo's tag, so it is in the memo's ``nominal_usd``;
``AgentRun.nominal_usd_external`` is never added on top.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import chi2, false_discovery_control, hypergeom, norm

from agentcoop import llm, store
from agentcoop.agents import harness
from agentcoop.agents.workspace import new_workspace
from agentcoop.coop.claims import Claim, claim_to_dict, parse_claims, same_claim
from agentcoop.coop.protocol import (ClaimRecord, Coordinator, ProtocolRun, SiteHandle,
                                     run_protocol)
from agentcoop.coop.site import CaseSpec, file_digest, make_sites, serve_mailbox
from agentcoop.coop.tester import TestResult, apply_bh

ARMS: tuple[str, ...] = ("single", "single_codex", "union", "vote", "ma_genome", "codex_coord",
                         "group_chat", "debate", "prior_only", "oracle", "ours", "no_routing",
                         "no_resolution", "ma_claims", "ma_genome_repaired")
ALPHA = 0.05                    # MA-genome and MA-claims FDR (spec §4)
MAX_JACCARD = 0.5               # an enrichment term overlapping a chosen one more adds nothing
TERM_SIZE = (10, 300)           # library terms used for enrichment (the v3 GO convention)
MIN_GENES = 3                   # a de claim needs >= 3 genes (spec §2.3)
GENESET_PREFIX = "genesets/"     # case.resources keys of gene-set libraries (Enrichr text / GMT)
# MA-genome's untargeted self-check calls (one per site, spec §4): frozen at v4-freeze to ours'
# median follow-up count on fsdev; None, the setting before the freeze, checks every site.
MA_SELF_CHECKS: int | None = None
AGENT_TIMEOUT = 900             # seconds per agent call, planted suites (spec §10)
TRUNCATE = 6000                 # characters of a site reply kept in the group-chat transcript

CLAIM_SCHEMA = """Claims are a JSON object {"claims": [...]}; each claim is
{"id": str, "kind": "de" | "composition", "population": <population>,
 "contrast": [<target group>, <reference group>], "up_genes": [HGNC symbols higher in target],
 "down_genes": [HGNC symbols lower in target], "effect": number (de: log2 fold change of the
 program; composition: change in logit cell fraction, target minus reference),
 "n_units": number of donors behind the claim, "rank": 1 = most confident, "rationale": str}.
Population and groups come from the vocabulary; kind "de" needs >= 3 genes in total; kind
"composition" has no genes."""


@dataclass
class ArmResult:
    arm: str
    claims: list[Claim]             # final findings, in rank order
    labels: dict[str, str]          # claim id -> label (ours, no_resolution; else {})
    causes: dict[str, dict]         # claim id -> cause record (ours only)
    scopes: dict[str, str]          # claim id -> scope (cases with a scope rule)
    calls: int
    nominal_usd: float
    meta: dict


class AgentFailed(RuntimeError):
    """A lead-agent run that timed out or failed: charged to the arm, never memoized.
    ``nominal_usd`` is only the cost spent outside the call's ledger tag (codex_coord's mailbox
    site calls; 0 for the oracle): the agent's own usage is charged from the ledger."""

    def __init__(self, error: str, nominal_usd: float, calls: int) -> None:
        super().__init__(error)
        self.nominal_usd = nominal_usd
        self.calls = calls


@dataclass
class Budget:
    """Site-agent calls of a compositional arm: B = 2S, of which the S step-1 calls are spent
    before the arm starts (spec §10), so S follow-up calls remain, as for ours (F = S)."""
    limit: int
    used: int
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def for_sites(cls, sites: list[SiteHandle]) -> Budget:
        s = sum(1 for site in sites if site.proposes)
        return cls(limit=2 * s, used=s)

    @property
    def left(self) -> int:
        return self.limit - self.used

    def take(self) -> bool:
        with self.lock:
            if self.used >= self.limit:
                return False
            self.used += 1
            return True


@dataclass
class _Step1:
    sites: list[SiteHandle]
    claims: dict[str, list[Claim]]          # site name -> its step-1 claims
    cost: dict[str, tuple[int, float]]      # site name -> (calls, nominal_usd) of step 1

    @property
    def all_claims(self) -> list[Claim]:
        return [c for s in self.sites for c in self.claims[s.name]]

    def charge(self, *, proposing_only: bool) -> tuple[int, float]:
        names = [s.name for s in self.sites if s.proposes or not proposing_only]
        return sum(self.cost[n][0] for n in names), sum(self.cost[n][1] for n in names)


class _Context:
    """One task and draw: each site set, the step-1 test matrix and ours' protocol run are built
    at most once, and only when an arm needs them."""

    def __init__(self, site_specs: list[dict], case: CaseSpec, task_id: str, sample: int) -> None:
        self.site_specs, self.case, self.task_id, self.sample = site_specs, case, task_id, sample
        self.cards = {s["name"]: s["card"] for s in site_specs}
        # A single-output arm may claim as much as all proposing sites together.
        self.cap = case.max_claims * sum(1 for s in site_specs if s["proposes"])

    def make(self, family: str, agents: dict[str, str] | None = None) -> list[SiteHandle]:
        return make_sites(self.site_specs, self.case, task_id=self.task_id,
                          run_id=f"{self.task_id}/{family}/s{self.sample}", agents=agents)

    @cached_property
    def base(self) -> _Step1:
        sites = self.make("base")
        claims = {s.name: s.propose(self.sample) for s in sites}
        return _Step1(sites, claims, {s.name: (s.calls, s.nominal_usd) for s in sites})

    @cached_property
    def matrix(self) -> dict[str, list[TestResult]]:
        """claim id -> [proposer gate, replication tests at every other site holding the claim's
        population and contrast], the replications BH-adjusted as one family: ours' step 2 on
        the same views. Computed before any arm asks a site anything."""
        by_name = {s.name: s for s in self.base.sites}
        out = {c.id: [by_name[c.site].test(c)] for c in self.base.all_claims}
        pairs = [(c.id, s.test(c)) for c in self.base.all_claims for s in self.base.sites
                 if s.name != c.site and s.holds(c.population, c.contrast)]
        adjusted = apply_bh([r for _, r in pairs], self.case.tester)
        for (cid, _), r in zip(pairs, adjusted):
            out[cid].append(r)
        return out

    def matrix_json(self) -> str:
        return json.dumps([{"claim": claim_to_dict(c),
                            "tests": [asdict(r) for r in self.matrix[c.id]]}
                           for c in self.base.all_claims], indent=1)

    @cached_property
    def protocol(self) -> tuple[list[SiteHandle], ProtocolRun]:
        sites = self.make("ours")
        run = run_protocol(sites, self.case, sample=self.sample,
                           followup_budget=sum(1 for s in sites if s.proposes),
                           coordinator=Coordinator(self.case))
        return sites, run

    def package(self) -> str:
        """The shared task package (spec §3): question, vocabulary, cards, claim schema, cap."""
        vocab = {"populations": list(self.case.vocab.populations),
                 "groups": list(self.case.vocab.groups),
                 "contrasts": [list(c) for c in self.case.vocab.contrasts]}
        return (f"Research question:\n{self.case.question}\n\n"
                f"Vocabulary (use exactly these names):\n{json.dumps(vocab, indent=1)}\n\n"
                f"Site metadata cards:\n{json.dumps(self.cards, indent=1, sort_keys=True)}\n\n"
                f"{CLAIM_SCHEMA}\nAt most {self.cap} claims.")


# ---- small helpers ---------------------------------------------------------------------------

def _p(r: TestResult) -> float:
    return math.inf if r.p is None else r.p


def _spent(sites: list[SiteHandle]) -> tuple[int, float]:
    return sum(s.calls for s in sites), sum(s.nominal_usd for s in sites)


def _views(sites: list[SiteHandle]) -> dict[str, str | None]:
    """Store id of the view each site holds now, for the view-error rate (spec §6), which the
    evaluation side measures by restoring them. ``view_blob`` is a Site attribute outside the
    SiteHandle contract, so scripted sites report None."""
    return {s.name: getattr(s, "view_blob", None) for s in sites}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _ws_name(ctx: _Context, arm: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{ctx.task_id}__{arm}__s{ctx.sample}")


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _json_in(text: str) -> object | None:
    """The JSON in an agent's free-text reply: the whole text, else its outermost braces."""
    for candidate in (text, text[text.find("{"):text.rfind("}") + 1]):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _result(arm: str, claims: list[Claim], calls: int, usd: float, *, labels: dict | None = None,
            causes: dict | None = None, scopes: dict | None = None,
            meta: dict | None = None) -> ArmResult:
    """An ArmResult whose claims carry the arm's own rank order (1 = first)."""
    ranked = [replace(c, rank=i) for i, c in enumerate(claims, 1)]
    return ArmResult(arm, ranked, labels or {}, causes or {}, scopes or {}, calls, float(usd),
                     meta or {})


def _gate_pass(pairs: list[tuple[TestResult, Claim]]) -> list[Claim]:
    """Claims their proposer's own tester supports, most significant first."""
    ok = [(r, c) for r, c in pairs if r.status == "supported"]
    return [c for r, c in sorted(ok, key=lambda rc: (_p(rc[0]), rc[1].id))]


def _clusters(claims: list[Claim]) -> list[list[Claim]]:
    """Greedy clustering by the shared matching rule: each claim joins the first cluster whose
    seed it matches. Claims arrive best-first, so a seed is its cluster's best claim."""
    out: list[list[Claim]] = []
    for c in claims:
        for cluster in out:
            if same_claim(cluster[0], c):
                cluster.append(c)
                break
        else:
            out.append([c])
    return out


def _consensus(clusters: list[list[Claim]], min_sites: int) -> list[Claim]:
    """Seeds of clusters proposed by >= min_sites distinct sites, most sites first (ties keep
    the clusters' best-first order)."""
    n_sites = [len({c.site for c in cluster}) for cluster in clusters]
    keep = sorted((-n, i) for i, n in enumerate(n_sites) if n >= min_sites)
    return [clusters[i][0] for _, i in keep]


def _by_own_p(ctx: _Context, claims: list[Claim]) -> list[Claim]:
    return sorted(claims, key=lambda c: (_p(ctx.matrix[c.id][0]), c.id))


# ---- MA-genome -------------------------------------------------------------------------------

def fixed_effect(stats: pd.DataFrame, key: str = "gene", *, min_sites: int = 2) -> pd.DataFrame:
    """Inverse-variance fixed-effect meta-analysis per ``key`` over sites (columns ``site``,
    ``key``, ``effect``, ``se``), with Cochran's Q, its chi-square p, the number of sites agreeing
    in sign with the pooled effect, and BH over keys measured at >= ``min_sites`` sites (2: the
    only ones a meta-analysis pools; 1: the headroom gate's single-site analysis, where the pooled
    estimate is the site's own and Q_p is undefined)."""
    columns = [key, "effect", "se", "z", "p", "fdr", "Q", "Q_p", "n_sites", "n_agree"]
    s = stats.assign(effect=pd.to_numeric(stats["effect"], errors="coerce"),
                     se=pd.to_numeric(stats["se"], errors="coerce"))
    s = s[np.isfinite(s.effect) & np.isfinite(s.se) & (s.se > 0)]
    s = s.assign(w=1.0 / s.se ** 2, we=s.effect / s.se ** 2)
    g = s.groupby(key)
    out = pd.DataFrame({"sw": g.w.sum(), "swe": g.we.sum(), "n_sites": g.site.nunique()})
    out = out[out.n_sites >= min_sites]
    if out.empty:
        return pd.DataFrame(columns=columns)
    out["effect"] = out.swe / out.sw
    out["se"] = 1.0 / np.sqrt(out.sw)
    out["z"] = out.effect / out.se
    out["p"] = 2 * norm.sf(np.abs(out.z))
    s = s.join(out.effect.rename("pooled"), on=key, how="inner")
    s = s.assign(dev=s.w * (s.effect - s.pooled) ** 2, agree=np.sign(s.effect) == np.sign(s.pooled))
    g = s.groupby(key)
    out["Q"] = g.dev.sum()
    out["Q_p"] = chi2.sf(out.Q, out.n_sites - 1)
    out["n_agree"] = g.agree.sum()
    out["fdr"] = false_discovery_control(out.p.to_numpy())
    return out.reset_index()[columns]


def gene_set_library(case: CaseSpec) -> dict[str, frozenset[str]]:
    """Terms of the gene-set libraries the site agents are given: the case resources whose key
    (a workspace path) starts with ``genesets/``, e.g. Task 8's
    ``genesets/GO_Biological_Process_2023.txt``, in Enrichr text or GMT layout, which share the
    line format term, description, genes...; every other resource (symbols.tsv) is not a library."""
    out = {}
    for key, path in sorted(case.resources.items()):
        if key.startswith(GENESET_PREFIX):
            for line in Path(path).read_text().splitlines():
                parts = line.split("\t")
                genes = frozenset(g for g in parts[2:] if g)
                if TERM_SIZE[0] <= len(genes) <= TERM_SIZE[1]:
                    out[parts[0]] = genes
    return out


def enrich(genes: set[str], universe: set[str], library: dict[str, frozenset[str]]) -> list[dict]:
    """Hypergeometric enrichment of ``genes`` within ``universe``, BH over the library's terms;
    terms with FDR < ALPHA and >= MIN_GENES overlap genes, chosen greedily by p and skipped when
    their overlap set has Jaccard > MAX_JACCARD with an already chosen term's."""
    query = genes & universe
    if len(query) < MIN_GENES or not library:
        return []
    rows = []
    for term, members in sorted(library.items()):
        m = members & universe
        hits = frozenset(query & m)
        p = float(hypergeom.sf(len(hits) - 1, len(universe), len(m), len(query))) if hits else 1.0
        rows.append((p, term, hits))
    fdr = false_discovery_control([p for p, _, _ in rows])
    chosen: list[dict] = []
    for (p, term, hits), q in sorted(zip(rows, fdr), key=lambda x: x[0][:2]):
        if q >= ALPHA or len(hits) < MIN_GENES:
            continue
        if any(len(hits & c["genes"]) / len(hits | c["genes"]) > MAX_JACCARD for c in chosen):
            continue
        chosen.append({"term": term, "p": p, "fdr": float(q), "genes": hits})
    return chosen


def _units(stats: pd.DataFrame) -> int:
    """Donors behind a meta table: per site the largest n1 + n2 over its genes, summed."""
    per_site = stats.groupby("site")[["n1", "n2"]].max()
    return int((per_site.n1 + per_site.n2).sum())


def _ma_de(holders: list[SiteHandle], population: str, contrast: tuple[str, str],
           library: dict[str, frozenset[str]], case: CaseSpec, arm: str,
           min_sites: int) -> tuple[list[tuple[float, Claim]], dict]:
    stats = pd.concat([s.export_stats(population, contrast).assign(site=s.name) for s in holders],
                      ignore_index=True)
    meta = fixed_effect(stats, min_sites=min_sites)
    kept = meta[(meta.fdr < ALPHA) & (meta.n_agree >= min_sites)]
    terms = [(direction, t) for direction, sign in (("up", 1), ("down", -1))
             for t in enrich(set(kept.gene[np.sign(kept.effect) == sign]), set(meta.gene), library)]
    terms.sort(key=lambda dt: (dt[1]["p"], dt[1]["term"]))
    effect = meta.set_index("gene").effect
    claims = []
    for i, (direction, t) in enumerate(terms[:case.max_claims]):
        genes = tuple(sorted(t["genes"]))
        claims.append((t["p"], Claim(
            id=f"{arm}:{population}:{contrast[0]}-{contrast[1]}:{i}", site=arm, kind="de",
            population=population, target=contrast[0], reference=contrast[1],
            up_genes=genes if direction == "up" else (),
            down_genes=() if direction == "up" else genes,
            effect=float(effect[list(genes)].mean()), n_units=_units(stats), rank=0,
            rationale=f"{t['term']} ({direction}; FDR {t['fdr']:.2g})")))
    info = {"n_sites": len(holders), "n_genes": len(meta), "n_kept": len(kept),
            "n_heterogeneous": int((meta.Q_p < ALPHA).sum())}
    return claims, info


def _ma_composition(sites: list[SiteHandle], case: CaseSpec, arm: str,
                    min_sites: int) -> list[tuple[float, Claim]]:
    """Fixed-effect meta of the sites' composition tests per population (BH over populations
    within a contrast; kept like genes: FDR < ALPHA and the pooled sign at >= min_sites sites)."""
    out = []
    for target, reference in case.vocab.contrasts:
        rows = []
        for population in case.vocab.populations:
            probe = Claim(id=f"{arm}:probe", site=arm, kind="composition", population=population,
                          target=target, reference=reference, up_genes=(), down_genes=(),
                          effect=1.0, n_units=None, rank=0)
            for s in sites:
                if s.holds(population, (target, reference)):
                    r = s.test(probe)
                    if r.effect is not None and r.se is not None:
                        rows.append({"population": population, "site": s.name, "effect": r.effect,
                                     "se": r.se, "n": sum(r.n_per_group or (0, 0))})
        if not rows:
            continue
        units = pd.DataFrame(rows).groupby("population").n.sum()
        meta = fixed_effect(pd.DataFrame(rows), key="population", min_sites=min_sites)
        for row in meta[(meta.fdr < ALPHA) & (meta.n_agree >= min_sites)].itertuples():
            out.append((row.p, Claim(
                id=f"{arm}:{row.population}:{target}-{reference}:composition", site=arm,
                kind="composition", population=row.population, target=target,
                reference=reference, up_genes=(), down_genes=(), effect=float(row.effect),
                n_units=int(units[row.population]), rank=0,
                rationale=f"fixed-effect meta of {row.n_sites} sites (Q {row.Q:.2f})")))
    return out


def ma_genome(sites: list[SiteHandle], case: CaseSpec, arm: str, *,
              min_sites: int = 2) -> tuple[list[Claim], dict]:
    """MA-genome (spec §4) on the sites' current views: per (population, contrast) held by >=
    ``min_sites`` sites, gene-level fixed-effect meta, genes with FDR < ALPHA and the pooled sign
    at >= ``min_sites`` sites, enrichment per direction, one claim per non-redundant term (at most
    K per table); plus composition claims. Claims come best-first by p.

    The arms use the default (2). ``min_sites=1`` on one site is the headroom gate's scripted
    single-site analysis (Task 10, experiments/headroom.py), so the gate and the arms share
    this code."""
    library = gene_set_library(case)
    scored: list[tuple[float, Claim]] = []
    tables = {}
    for population in case.vocab.populations:
        for contrast in case.vocab.contrasts:
            holders = [s for s in sites if s.holds(population, contrast)]
            if len(holders) >= min_sites:
                claims, info = _ma_de(holders, population, contrast, library, case, arm,
                                      min_sites)
                scored += claims
                tables[f"{population}|{contrast[0]}>{contrast[1]}"] = info
    scored += _ma_composition(sites, case, arm, min_sites)
    scored.sort(key=lambda pc: (pc[0], pc[1].id))
    return [c for _, c in scored], {"tables": tables}


# ---- compositional arms: shared tools ----------------------------------------------------------

def _client(usage: str, request: str) -> str:
    """A mailbox client for the coordinator's workspace, in serve_mailbox's protocol: the request
    appears atomically as req_<id>.json next to the script (its temporary name does not match
    req_*.json, so the server never reads half a request); the server answers resp_<id>.json
    atomically, which the client prints and removes."""
    return (f"# {usage}\n"
            "import json, os, pathlib, sys, time, uuid\n"
            "box = pathlib.Path(__file__).resolve().parent\n"
            "rid = uuid.uuid4().hex\n"
            "tmp = box / ('.req_' + rid + '.tmp')\n"
            f"tmp.write_text(json.dumps({request}))\n"
            "os.replace(tmp, box / ('req_' + rid + '.json'))\n"
            "reply = box / ('resp_' + rid + '.json')\n"
            "while not reply.exists():\n"
            "    time.sleep(0.2)\n"
            "print(reply.read_text())\n"
            "reply.unlink()\n")


ASK_SITE = _client('Ask a site agent. Usage: python ask_site.py <site> "<request>"',
                   "{'op': 'ask_site', 'site': sys.argv[1], 'request': sys.argv[2]}")
TEST_CLAIM = _client("Test one claim at a site. Usage: python test_claim.py <site> <claim.json>",
                     "{'op': 'test_claim', 'site': sys.argv[1], "
                     "'claim': json.loads(pathlib.Path(sys.argv[2]).read_text())}")


def mailbox_handlers(sites: list[SiteHandle], case: CaseSpec, budget: Budget, sample: int,
                     log: list[dict], tag: str) -> dict[str, Callable[[dict], dict]]:
    """The compositional arms' tools (spec §3): ``ask_site`` is a site-agent call charged to
    ``budget`` and refused once it is spent; ``test_claim`` runs a site's deterministic tester
    and, like ours' tests, costs no call. ``tag`` keeps the asks' spend on the arm's ledger tag
    when the mailbox thread calls them."""
    by_name = {s.name: s for s in sites}

    def site_of(req: dict) -> SiteHandle | None:
        return by_name.get(str(req.get("site", "")))

    def ask_site(req: dict) -> dict:
        site = site_of(req)
        if site is None:
            return {"error": f"unknown site; sites are {sorted(by_name)}"}
        if not budget.take():
            log.append({"op": "ask_site", "site": site.name, "refused": True})
            return {"error": "budget exhausted: no site-agent calls left"}
        request = str(req.get("request", ""))
        with llm.exact_tag(tag):
            answer = site.ask(request, sample)
        log.append({"op": "ask_site", "site": site.name, "request": request[:2000]})
        return {"answer": answer}

    def test_claim(req: dict) -> dict:
        site = site_of(req)
        if site is None:
            return {"error": f"unknown site; sites are {sorted(by_name)}"}
        claims, errors = parse_claims({"claims": [req.get("claim")]}, site="coordinator",
                                      vocab=case.vocab, max_claims=1)
        if not claims:
            return {"error": "; ".join(errors) or "invalid claim"}
        result = site.test(claims[0])
        log.append({"op": "test_claim", "site": site.name, "claim": claim_to_dict(claims[0]),
                    "status": result.status})
        return asdict(result)

    return {"ask_site": ask_site, "test_claim": test_claim}


def _agent_arm(ctx: _Context, arm: str, key: dict, compute: Callable[[], dict], *,
               charge: tuple[int, float] = (0, 0.0)) -> ArmResult:
    """A memoized lead-agent run whose claims.json is the arm's output. The agent's usage is in
    the ledger under the memo's tag (run_codex ledgers Codex there), hence in the memo's
    ``nominal_usd``. A failed run is charged what the ledger holds under this call's tag (the
    amount memo would have stored, as Task 7's Site charges a failure) plus the cost it reports
    from outside that tag, reported, and never cached (review focus 3)."""
    calls, usd = charge
    with llm.tag(f"{arm}.{time.time_ns()}"):
        inside = llm.current_tag() + "/"           # store.memo nests the run's own tag here
        try:
            out = store.memo(key, compute, tag=arm)
        except AgentFailed as exc:
            return _result(arm, [], calls + exc.calls,
                           usd + llm.spent(inside)["nominal_usd"] + exc.nominal_usd,
                           meta={"error": str(exc)})
    claims, errors = parse_claims(out["raw"], site=arm, vocab=ctx.case.vocab, max_claims=ctx.cap)
    return _result(arm, claims, calls + out["calls"], usd + out["nominal_usd"],
                   meta={"errors": errors, "log": out.get("log", [])})


# ---- arms --------------------------------------------------------------------------------------

def _single(ctx: _Context) -> dict[str, ArmResult]:
    out = {}
    for s in ctx.base.sites:
        if s.proposes:
            claims = _gate_pass([(ctx.matrix[c.id][0], c) for c in ctx.base.claims[s.name]])
            out[f"single:{s.name}"] = _result(f"single:{s.name}", claims, *ctx.base.cost[s.name],
                                              meta={"views": _views([s])})
    return out


def _single_codex(ctx: _Context) -> dict[str, ArmResult]:
    sites = ctx.make("single_codex", agents={s["name"]: "codex" for s in ctx.site_specs})
    out = {}
    for s in sites:
        if s.proposes:
            claims = _gate_pass([(s.test(c), c) for c in s.propose(ctx.sample)])
            out[f"single_codex:{s.name}"] = _result(f"single_codex:{s.name}", claims, s.calls,
                                                    s.nominal_usd, meta={"views": _views([s])})
    return out


def _union(ctx: _Context) -> dict[str, ArmResult]:
    clusters = _clusters(_by_own_p(ctx, ctx.base.all_claims))
    return {"union": _result("union", _consensus(clusters, 1),
                             *ctx.base.charge(proposing_only=True),
                             meta={"clusters": [[c.id for c in cl] for cl in clusters]})}


def _vote(ctx: _Context) -> dict[str, ArmResult]:
    clusters = _clusters(_by_own_p(ctx, ctx.base.all_claims))
    return {"vote": _result("vote", _consensus(clusters, 2), *ctx.base.charge(proposing_only=True),
                            meta={"clusters": [[c.id for c in cl] for cl in clusters]})}


def _no_routing(ctx: _Context) -> dict[str, ArmResult]:
    passing = [c for s in ctx.base.sites if s.proposes
               for c in _gate_pass([(ctx.matrix[c.id][0], c) for c in ctx.base.claims[s.name]])]
    clusters = _clusters(_by_own_p(ctx, passing))
    return {"no_routing": _result("no_routing", _consensus(clusters, 1),
                                  *ctx.base.charge(proposing_only=True))}


def _self_checked(sites: list[SiteHandle], task_id: str) -> list[SiteHandle]:
    """The sites MA-genome self-checks: every site, or MA_SELF_CHECKS of them in site order from
    a start set by the task id (uninformed about which view is wrong, balanced over tasks)."""
    n = len(sites) if MA_SELF_CHECKS is None else min(MA_SELF_CHECKS, len(sites))
    start = int(_sha(task_id), 16) % len(sites)
    return [sites[(start + i) % len(sites)] for i in range(n)]


def _ma_genome(ctx: _Context) -> dict[str, ArmResult]:
    sites = ctx.make("ma")
    for s in sites:
        s.propose(ctx.sample)
    checked = _self_checked(sites, ctx.task_id)
    for s in checked:
        s.self_check(ctx.sample)
    claims, meta = ma_genome(sites, ctx.case, "ma_genome")
    return {"ma_genome": _result("ma_genome", claims, *_spent(sites),
                                 meta={**meta, "self_checked": [s.name for s in checked],
                                       "views": _views(sites)})}


def _ma_genome_repaired(ctx: _Context) -> dict[str, ArmResult]:
    sites, run = ctx.protocol
    claims, meta = ma_genome(sites, ctx.case, "ma_genome_repaired")
    return {"ma_genome_repaired": _result("ma_genome_repaired", claims, run.calls,
                                          run.nominal_usd, meta=meta)}


def _replication_z(record: ClaimRecord) -> list[float]:
    """z = effect / se of a claim's non-proposer replication tests before step 4 that produced an
    estimate. Effects are already oriented (positive = the claimed direction, centred on 0)."""
    return [r.effect / r.se for r in record.results_step3
            if r.test == "replication" and r.site != record.claim.site
            and r.effect is not None and r.se]


def _ma_claims(ctx: _Context) -> dict[str, ArmResult]:
    """Stouffer over the non-proposer replication tests of ours' step-1 claims, read from ours'
    own records before step 4, so the arm differs from no-resolution only in the decision rule;
    BH over claims; kept when FDR < ALPHA and the pooled z is positive, which needs >= 1
    non-proposer in the claimed direction. Charged ours' step 1 (the S0 entries of the same
    memoized calls)."""
    _, run = ctx.protocol
    scored = []
    for record in run.records:
        z = _replication_z(record)
        if z:
            scored.append((record.claim, sum(z) / math.sqrt(len(z))))
    fdr = false_discovery_control([2 * norm.sf(abs(z)) for _, z in scored]) if scored else []
    kept = sorted(((c, z) for (c, z), q in zip(scored, fdr) if q < ALPHA and z > 0),
                  key=lambda cz: (-cz[1], cz[0].id))
    return {"ma_claims": _result("ma_claims", [c for c, _ in kept],
                                 *ctx.base.charge(proposing_only=False),
                                 meta={"z": {c.id: z for c, z in scored}})}


def _ours(ctx: _Context) -> dict[str, ArmResult]:
    sites, run = ctx.protocol
    return {"ours": _result(
        "ours", run.findings(), run.calls, run.nominal_usd,
        labels={r.claim.id: r.label for r in run.records},
        causes={r.claim.id: r.cause for r in run.records if r.cause},
        scopes={r.claim.id: r.scope for r in run.records if r.scope},
        meta={"followups": run.followups, "errors": run.errors, "views": _views(sites),
              # every labelled claim, demoted ones included: causes mostly explain demotions
              "all_claims": _all_claims(run, step3=False)})}


def _all_claims(run: ProtocolRun, *, step3: bool) -> list[dict]:
    """Every claim the protocol labelled, demoted ones included, as {"claim": claim_to_dict,
    "label", "cause", "scope"}: what the grader needs for cause and scope accuracy. ``step3``
    gives the labels before resolution, with no cause (no-resolution never classifies)."""
    return [{"claim": claim_to_dict(r.claim), "label": r.label_step3 if step3 else r.label,
             "cause": None if step3 else r.cause, "scope": r.scope} for r in run.records]


def _no_resolution(ctx: _Context) -> dict[str, ArmResult]:
    """Ours' step-3 labels: charged ours' step 1 (the S0 entries of the same memoized calls)."""
    _, run = ctx.protocol
    return {"no_resolution": _result("no_resolution", run.findings_step3(),
                                     *ctx.base.charge(proposing_only=False),
                                     labels={r.claim.id: r.label_step3 for r in run.records},
                                     meta={"all_claims": _all_claims(run, step3=True),
                                           "errors": run.errors})}


def _prior_only(ctx: _Context) -> dict[str, ArmResult]:
    messages = [{"role": "system", "content": "You are an expert biomedical scientist."},
                {"role": "user", "content": ctx.package() + "\n\nYou have no access to any data. "
                 "From prior knowledge alone, state the claims you expect the sites' data to "
                 'support, as a JSON object {"claims": [...]}.'}]
    raw, res = llm.chat_json(messages, sample=ctx.sample)
    claims, errors = parse_claims(raw, site="prior", vocab=ctx.case.vocab, max_claims=ctx.cap)
    return {"prior_only": _result("prior_only", claims, 0, res.nominal_usd,
                                  meta={"errors": errors})}


def _oracle(ctx: _Context) -> dict[str, ArmResult]:
    """Pooled oracle (silo-violating reference): one Codex run that sees every site's data file,
    card and the case resources. Spec §4 pools the site views concatenated by the case script;
    no case script builds such a pool (Task 8 writes per-site files only), so this oracle
    harmonizes the sites itself, as each site agent does for its own file. It is therefore a
    weaker reference than a pre-harmonized pool, which favours ours in silo-cost recovery;
    ``meta["pooled_input"]`` records this for the analysis."""
    data = {f"{s['name']}/{Path(s['data_path']).name}": Path(s["data_path"])
            for s in ctx.site_specs}
    files = {**data, **ctx.case.resources}
    prompt = (ctx.package() + "\n\nEvery site's data file is in this directory: "
              f"{', '.join(sorted(data))} (each site's metadata card is <site>/card.json); "
              f"shared resources: {', '.join(sorted(ctx.case.resources)) or 'none'}. Python with "
              "scanpy, anndata, numpy, pandas, scipy and statsmodels is available (`python`). "
              "Analyze all sites' data together, then write only claims.json in this directory.")
    key = {"v": 1, "op": "oracle", "case": ctx.case.name, "task": ctx.task_id,
           "prompt": _sha(prompt), "data": {k: file_digest(p) for k, p in sorted(files.items())},
           "sample": ctx.sample}

    def compute() -> dict:
        ws = new_workspace(_ws_name(ctx, "oracle"), files)
        (ws / "claims.json").unlink(missing_ok=True)       # a failed earlier attempt's output
        for s in ctx.site_specs:
            (ws / s["name"] / "card.json").write_text(json.dumps(s["card"], indent=1))
        run = harness.run_agent("codex", ws, prompt, sample=ctx.sample, timeout=AGENT_TIMEOUT)
        if not run.ok:
            raise AgentFailed(run.error or "oracle run failed", 0.0, 1)
        # run_codex already ledgered the run's usage under this memo tag: AgentRun's copy of it
        # is kept for inspection only, never as nominal_usd_external (memo would add it again)
        return {"raw": _read_json(ws / "claims.json"), "calls": 1,
                "agent_nominal_usd": run.nominal_usd_external}

    out = _agent_arm(ctx, "oracle", key, compute)
    out.meta["pooled_input"] = "site files"
    return {"oracle": out}


def _codex_coord(ctx: _Context) -> dict[str, ArmResult]:
    """Codex CLI as lead (spec §4): no data, the step-1 test matrix, and the shared tools via a
    file mailbox; its claims.json is kept as is."""
    sites = ctx.base.sites
    budget = Budget.for_sites(sites)
    matrix = ctx.matrix_json()
    prompt = (ctx.package() + "\n\nYou coordinate this multi-site study and cannot read any "
              "site's data. claims_matrix.json lists every claim the sites proposed in their first "
              "analysis with its test results at every site that could test it (deterministic "
              "testers inside each site; statuses supported, contradicted, absent, inconclusive, "
              "untestable). Tools, run from this directory:\n"
              '- python ask_site.py <site> "<request>": asks that site\'s analysis agent, which '
              "works on its own data and replies in text (minutes per call); at most "
              f"{budget.left} calls in total, further calls are refused.\n"
              "- python test_claim.py <site> <claim.json>: runs that site's tester on one claim in "
              "the claim schema and prints the result.\n"
              f"Sites: {', '.join(s.name for s in sites)}. Decide which claims the evidence "
              "supports, then write the final claims to claims.json in this directory.")
    key = {"v": 1, "op": "codex_coord", "case": ctx.case.name, "task": ctx.task_id,
           "prompt": _sha(prompt), "matrix": _sha(matrix), "sample": ctx.sample}
    tag = llm.current_tag()

    def compute() -> dict:
        ws = new_workspace(_ws_name(ctx, "codex_coord"), {})
        (ws / "claims.json").unlink(missing_ok=True)       # a failed earlier attempt's output
        (ws / "claims_matrix.json").write_text(matrix)
        (ws / "ask_site.py").write_text(ASK_SITE)
        (ws / "test_claim.py").write_text(TEST_CLAIM)
        log: list[dict] = []
        calls0, usd0 = _spent(sites)
        stop = threading.Event()
        server = serve_mailbox(ws, mailbox_handlers(sites, ctx.case, budget, ctx.sample, log, tag),
                               stop)
        try:
            run = harness.run_agent("codex", ws, prompt, sample=ctx.sample,
                                    timeout=AGENT_TIMEOUT * (budget.left + 1))
        finally:
            stop.set()
            server.join()
        calls1, usd1 = _spent(sites)
        # The coordinator's own Codex usage is in the ledger under this memo tag. The asks ran
        # on the mailbox thread, outside it, so only their site-counter deltas are added.
        asks, usd = calls1 - calls0, usd1 - usd0
        if not run.ok:
            raise AgentFailed(run.error or "coordinator run failed", usd, 1 + asks)
        return {"raw": _read_json(ws / "claims.json"), "log": log, "calls": 1 + asks,
                "nominal_usd_external": usd, "agent_nominal_usd": run.nominal_usd_external}

    return {"codex_coord": _agent_arm(ctx, "codex_coord", key, compute,
                                      charge=ctx.base.charge(proposing_only=False))}


def _transcript_text(transcript: list[dict]) -> str:
    return "\n\n".join(f"[{e['from']}]\n{e['text']}" for e in transcript)


def _group_chat(ctx: _Context) -> dict[str, ArmResult]:
    """Round-robin group chat (in-house implementation of AutoGen's RoundRobinGroupChat turn
    protocol): the coordinator LLM, then each site agent in a fixed order, over one shared
    transcript seeded with the test matrix; site turns are budgeted ask_site calls, and the
    coordinator's turn after the budget is spent must emit the final claims."""
    sites = ctx.base.sites
    budget = Budget.for_sites(sites)
    log: list[dict] = []
    tools = mailbox_handlers(sites, ctx.case, budget, ctx.sample, log, llm.current_tag())
    transcript = [{"from": "test matrix", "text": ctx.matrix_json()}]
    names = ", ".join(s.name for s in sites)
    calls0, usd0 = _spent(sites)
    chat_usd = 0.0

    def coordinate() -> dict:
        nonlocal chat_usd
        system = (ctx.package() + f"\n\nYou coordinate a round-robin group chat with the analysis "
                  f"agents of sites {names}; each works on its own private data, you see no data. "
                  "The transcript starts with the claim x site test matrix of the sites' first "
                  'analyses. Each turn reply with a JSON object {"message": text to the sites, '
                  '"tests": [{"site": name, "claim": claim in the schema}] (optional; site '
                  'testers, results enter the transcript), "final": false}; when done reply '
                  '{"final": true, "claims": [...]} in the claim schema. '
                  f"Site turns left: {budget.left}.")
        user = _transcript_text(transcript)
        if not budget.left:
            user += ('\n\nFINAL TURN: no site turns are left; reply now with '
                     '{"final": true, "claims": [...]}.')
        reply, res = llm.chat_json([{"role": "system", "content": system},
                                    {"role": "user", "content": user}], sample=ctx.sample)
        chat_usd += res.nominal_usd
        transcript.append({"from": "coordinator", "text": str(reply.get("message", ""))})
        for t in reply.get("tests") or []:
            if isinstance(t, dict) and not reply.get("final"):
                transcript.append({"from": f"tester {t.get('site')}",
                                   "text": json.dumps(tools["test_claim"](t))})
        return reply

    reply = coordinate()
    while not reply.get("final") and budget.left:
        for s in sites:
            request = (f"You are the analysis agent of site {s.name} in a group chat with the "
                       "other sites' agents and a coordinator; use your own data. Transcript so "
                       f"far:\n\n{_transcript_text(transcript)}\n\nReply to the coordinator's "
                       "latest message.")
            answer = tools["ask_site"]({"site": s.name, "request": request})
            if "error" in answer:
                break
            transcript.append({"from": s.name, "text": answer["answer"][:TRUNCATE]})
        reply = coordinate()
    claims, errors = parse_claims({"claims": reply.get("claims") or []}, site="group_chat",
                                  vocab=ctx.case.vocab, max_claims=ctx.cap)
    calls1, usd1 = _spent(sites)
    calls, usd = ctx.base.charge(proposing_only=False)
    return {"group_chat": _result("group_chat", claims, calls + calls1 - calls0,
                                  usd + usd1 - usd0 + chat_usd,
                                  meta={"errors": errors, "log": log, "transcript": transcript})}


def _debate(ctx: _Context) -> dict[str, ArmResult]:
    """Debate (Du et al. 2023): round 0 = step-1 claims; one revision round in which each
    proposing site sees every site's claims with the test matrix; final = matched claims revised
    by >= 2 sites. A site whose reply holds no JSON keeps its round-0 claims."""
    sites = ctx.base.sites
    matrix = ctx.matrix_json()
    calls0, usd0 = _spent(sites)
    revised, kept_round0, errors = [], [], {}
    for s in sites:
        if not s.proposes:
            continue
        request = (f"Debate, revision round. Below are the claims every site proposed in its "
                   "first analysis, each with its test results at every site that could test it. "
                   f'Your own claims are those with "site": "{s.name}". Considering the other '
                   "sites' claims and the test results, and re-checking your own data where "
                   'useful, revise your claims. Reply with only a JSON object {"claims": [...]} '
                   f"(at most {ctx.case.max_claims} claims).\n\n{CLAIM_SCHEMA}\n\n{matrix}")
        raw = _json_in(s.ask(request, ctx.sample))
        if raw is None:
            revised += ctx.base.claims[s.name]
            kept_round0.append(s.name)
            continue
        claims, errors[s.name] = parse_claims(raw, site=s.name, vocab=ctx.case.vocab,
                                              max_claims=ctx.case.max_claims)
        revised += claims
    calls1, usd1 = _spent(sites)
    clusters = _clusters(sorted(revised, key=lambda c: (c.rank, c.site, c.id)))
    calls, usd = ctx.base.charge(proposing_only=False)
    return {"debate": _result("debate", _consensus(clusters, 2), calls + calls1 - calls0,
                              usd + usd1 - usd0,
                              meta={"kept_round0": kept_round0, "errors": errors,
                                    "n_revised": len(revised)})}


_RUNNERS: dict[str, Callable[[_Context], dict[str, ArmResult]]] = {
    "single": _single, "single_codex": _single_codex, "union": _union, "vote": _vote,
    "ma_genome": _ma_genome, "codex_coord": _codex_coord, "group_chat": _group_chat,
    "debate": _debate, "prior_only": _prior_only, "oracle": _oracle, "ours": _ours,
    "no_routing": _no_routing, "no_resolution": _no_resolution, "ma_claims": _ma_claims,
    "ma_genome_repaired": _ma_genome_repaired,
}


def run_arms(arms: list[str], site_specs: list[dict], case: CaseSpec, *, task_id: str,
             sample: int) -> dict[str, ArmResult]:
    """Run ``arms`` on one task and draw. "single" and "single_codex" yield one result per
    proposing site ("single:<site>"); every other arm yields one result under its name."""
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arms {unknown}; known arms: {ARMS}")
    ctx = _Context(site_specs, case, task_id, sample)
    out: dict[str, ArmResult] = {}
    for arm in arms:
        out |= _RUNNERS[arm](ctx)
    return out
