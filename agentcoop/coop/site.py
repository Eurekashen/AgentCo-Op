"""Real sites: one external agent bound to one site's private data, behind ``SiteHandle``.

A ``Site`` owns a private workspace holding its data file, the case resources and its codebook.
Its agent builds the site view there and proposes claims; deterministic testers answer for the
site from the site table computed from that view, so only summary statistics leave the site.

Every agent call goes through ``store.memo``. A step-1 proposal is shared by every arm and run
that asks for the same (task, site, agent, sample), and its view is restored from the stored
blob, so arms compare identical claims on identical views. Every call is counted and charged at
nominal cost, cache hits included. The ledger is the only source of that cost: the harness
ledgers an agent's model usage (Codex through ``llm.record_external``) under the caller's tag,
so a call costs what its tag spent. A failed call is charged and recorded but never cached; the
site is then left without a view and its tests come back ``untestable(view)``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pandas as pd

from agentcoop import llm, store
from agentcoop.agents import harness, workspace
from agentcoop.coop import tester
from agentcoop.coop.claims import Claim, Vocabulary, parse_claims
from agentcoop.coop.tester import TesterParams, TestResult
from agentcoop.coop.view import EXCLUDE, OTHER, SiteTable, SiteView, load_view, site_table

if TYPE_CHECKING:
    from agentcoop.coop.protocol import SiteHandle

SPEC_KEYS = ("contrasts", "sample_unit", "normalization", "gene_mapping", "covariates",
             "exclusions")
STATS_COLUMNS = ["gene", "effect", "se", "n1", "n2"]
TRANSCRIPT_STEPS = 40   # last steps of an agent run kept with its stored result
TABLE_CACHE = 32        # site tables kept in memory, one per (view, data file)
POLL_S = 0.2            # mailbox polling interval
BAD_POLLS = 10          # polls a request may stay unparsable (still being written) before an error


@dataclass
class CaseSpec:
    name: str
    kind: str                                   # "scrna" | "bulk"
    vocab: Vocabulary
    question: str
    default_spec: dict                          # pre-registered analysis conventions (§2.5 step 4)
    tester: TesterParams
    max_claims: int = 10
    resources: dict[str, Path] = field(default_factory=dict)   # copied into every workspace
    scope_tests: Callable[[Claim, list[SiteHandle]],
                          list[tuple[SiteHandle, tuple[str, str]]]] | None = None
    scope_rule: Callable[[Claim, list[TestResult]], str] | None = None


_DIGESTS: dict[tuple[str, int, int], str] = {}


def file_digest(path: Path) -> str:
    """sha256 of a file, cached by (path, size, mtime): site files are large and their digest
    enters every proposal key."""
    path = Path(path).resolve()
    st = path.stat()
    k = (str(path), st.st_size, st.st_mtime_ns)
    if k not in _DIGESTS:
        with open(path, "rb") as f:
            _DIGESTS[k] = hashlib.file_digest(f, "sha256").hexdigest()
    return _DIGESTS[k]


# ---------------------------------------------------------------------------
# Prompts. Each agent call is a fresh run, so every prompt restates the setting.


def _setting(case: CaseSpec) -> str:
    """What every call of a site agent is told: the question, the boundary, the vocabulary."""
    resources = "".join(f", {n}" for n in sorted(case.resources))
    contrasts = "; ".join(f"{t} vs {r}" for t, r in case.vocab.contrasts)
    return (
        f"Research question: {case.question}\n\n"
        "You are the analysis agent of one site in a multi-site study. Only the files in this "
        "directory exist for you: your site's data file, codebook.json (your site's metadata "
        f"card: what its columns and codes mean){resources}. There is no network and no other "
        "site. Your site's data cannot leave the site: the study reads only the files described "
        "below, and only summary statistics computed from them leave the site. Python "
        "(`python`) with scanpy, anndata, numpy, pandas, scipy and statsmodels is available.\n\n"
        "Case vocabulary (use these exact strings):\n"
        f"- populations: {', '.join(case.vocab.populations)}\n"
        f"- groups: {', '.join(case.vocab.groups)}\n"
        f"- contrasts (target vs reference): {contrasts}\n"
    )


def _view_format(kind: str) -> str:
    """The site-view files exactly as ``agentcoop.coop.view.load_view`` reads them, with ids
    that refer to the data file the way ``site_table`` reads it (h5ad for scRNA, a TSV matrix
    for bulk)."""
    if kind == "bulk":
        data = ("Your data file is a tab-separated expression matrix: its first column holds the "
                "feature ids and every other column is one sample, named by its column header.")
        feature, sample = "the matrix's first-column id", "a column header of the matrix"
    else:
        data = ("Your data file is an AnnData .h5ad file: var_names are its feature ids, "
                "obs_names its cell ids and obs its cell metadata.")
        feature = "the data file's feature id (a var_names entry)"
        sample = "the sample id as the data file's obs records it"
    files = [
        f"view/genes.tsv: columns native_id, symbol. One row per feature you could map: "
        f"native_id = {feature}, symbol = its HGNC gene symbol. Leave out features you cannot "
        "map.",
        "view/samples.tsv: columns sample, donor, group, then optional covariate columns (e.g. "
        f"assay, sex, age). One row per sample: sample = {sample}, donor = its donor id, group = "
        "a case group (translate the site's own codes with codebook.json) or "
        f"{EXCLUDE} for a sample outside every case group. One sample per donor: contrasts are "
        "between donors.",
    ]
    if kind == "scrna":
        files.append(
            "view/cells.tsv: columns cell, sample, population. One row per cell you keep (leave "
            "out cells you remove, e.g. in quality control): cell = its obs_names entry, sample = "
            f"a sample of view/samples.tsv, population = a case population or {OTHER} for a cell "
            "of another type.")
    files.append(
        f"view/spec.json: a JSON object with keys {', '.join(SPEC_KEYS)}: how you defined the "
        "contrasts, the sample unit, the normalization, the source of the gene mapping, the "
        "covariates and every exclusion.")
    return (data + "\nWrite these tab-separated files with a header row into the directory "
            "view/:\n" + "\n".join(f"- {f}" for f in files))


def _claims_format(case: CaseSpec) -> str:
    target, reference = case.vocab.contrasts[0]
    example = [{"kind": "de", "population": case.vocab.populations[0],
                "contrast": [target, reference], "up_genes": ["GENE_A", "GENE_B", "GENE_C"],
                "down_genes": ["GENE_D"], "effect": 1.2, "n_units": 8, "rank": 1,
                "rationale": "one sentence on the evidence"}]
    kinds = '"de" (a gene program changes in the population)'
    if case.kind == "scrna":
        kinds += (' or "composition" (the population\'s share of cells changes; no genes; '
                  "effect = log odds ratio of its cell fraction, positive = higher in the "
                  "target group)")
    return (
        f"claims.json: a JSON list of at most {case.max_claims} claims (any beyond that are "
        "dropped), each an object with\n"
        f"- kind: {kinds}\n"
        "- population: a case population; contrast: [target group, reference group], a case "
        "contrast\n"
        "- up_genes, down_genes: HGNC symbols higher / lower in the target group; a \"de\" "
        "claim names at least 3 genes in total\n"
        '- effect: your effect estimate (for "de": log2 fold change of the program); n_units: '
        "the number of donors behind the claim\n"
        "- rank: 1 for your most confident claim, then 2, 3, ...; rationale: one sentence\n"
        f"Example (placeholder genes):\n{json.dumps(example, indent=1)}\n\n"
        + _how_claims_are_tested(case)
    )


def _how_claims_are_tested(case: CaseSpec) -> str:
    """The study's division of labour, stated to every proposing site (all arms share step 1):
    sites discover, independent sites confirm. Without it, small sites either propose nothing
    (genome-wide correction at a few donors kills every single gene) or propose individually
    top-ranked, barely expressed genes that no site can test (v4 smoke, fsdev-800)."""
    t = case.tester
    if case.kind == "bulk":
        test = ("a sample score (mean rank-normalized expression of up_genes minus down_genes) "
                "compared between target and reference samples (AUC)")
    else:
        test = (f"a donor-level pseudobulk score (mean log2 CPM of up_genes minus down_genes per "
                f"donor and population; only genes with mean CPM >= {t.min_cpm:g} in that "
                f"population count, and at least {t.min_gene_frac:.0%} of a claim's genes must "
                "count) compared between target and reference donors; a composition claim by the "
                "donors' cell fractions")
    return (
        "How claims are used: this study separates discovery from confirmation. Your site "
        "discovers; every claim is then tested independently, by fixed code, at your site and at "
        f"every other site that holds the same groups: {test}. Therefore (1) propose coherent "
        "gene programs - genes that change together, for example found by testing the gene-set "
        "libraries in this directory on donor-level pseudobulk - rather than individually "
        "top-ranked genes; (2) use well-expressed genes; (3) propose your best candidates up to "
        "the cap even when they do not reach significance after genome-wide correction at your "
        "site alone: with a few donors per site a real effect is often only suggestive locally, "
        "and confirmation comes from the other sites. Rank them by your confidence."
    )


def propose_prompt(case: CaseSpec, card: dict, data_file: str, *, proposes: bool) -> str:
    """Step 1 (§2.5): build the site view and, at a proposing site, propose up to K claims."""
    task = ("Then propose claims that answer the research question from your site's data:\n"
            + _claims_format(case)) if proposes else (
        "This site builds its view only: it proposes no claims (do not write claims.json). "
        "Claims from other sites are tested on your view.")
    return (
        _setting(case)
        + f"\nYour site's data file: {data_file}. Your site's metadata card (also in "
        f"codebook.json):\n{json.dumps(card, indent=1)}\n\n"
        "First build your site view. Deterministic code at your site reads it to test claims "
        "from every site, so it must describe your data exactly.\n"
        + _view_format(case.kind) + "\n\n" + task
        + "\n\nWrite every file into this directory; nothing you print is read."
    )


def repair_prompt(case: CaseSpec, claim: Claim, defect: str) -> str:
    """Step 4 handoff-loss follow-up: the claim and the defect, never other sites' results."""
    genes = (f"up {', '.join(claim.up_genes) or '-'}; "
             f"down {', '.join(claim.down_genes) or '-'}")
    return (
        _setting(case)
        + f"\nA claim could not be tested correctly on your site view.\nClaim: {claim.kind} in "
        f"{claim.population}, {claim.target} vs {claim.reference}; genes {genes}.\n"
        f"Defect: {defect}\n\n"
        "Fix your site view so that it describes your data correctly. Change only files under "
        "view/ and keep their formats.\n" + _view_format(case.kind)
    )


def _self_check_prompt(case: CaseSpec) -> str:
    """The untargeted view check given to MA-genome (§4): a fixed checklist, no claim."""
    items = [
        "Every symbol in view/genes.tsv is the HGNC symbol of its native_id (use the mapping "
        "resources in this directory).",
        f"Every group in view/samples.tsv is the case group (or {EXCLUDE}) that codebook.json "
        "gives for the sample's own code.",
        "One sample per donor.",
    ]
    if case.kind == "scrna":
        items.insert(2, f"Every population in view/cells.tsv is a case population (or {OTHER}) "
                        "that matches the cell's own label as codebook.json describes it.")
    return (
        _setting(case)
        + "\nChecklist: check your site view in view/ against each item and fix what is wrong, "
        "in place.\n" + "".join(f"{i}. {t}\n" for i, t in enumerate(items, 1))
        + "Change only files under view/ and keep their formats.\n" + _view_format(case.kind)
    )


def _ask_prompt(case: CaseSpec, request: str) -> str:
    return (
        _setting(case)
        + f"\nRequest from the study coordinator:\n{request}\n\n"
        "Answer from your site's data and view. Write the answer to answer.md in this directory: "
        "plain text, summary statistics only, no cell- or sample-level data. Do not change files "
        "under view/."
    )


def _dispatch(msg: object, handlers: dict[str, Callable[[dict], dict]]) -> dict:
    if not isinstance(msg, dict) or msg.get("op") not in handlers:
        return {"error": f"expected a JSON object with 'op' in {sorted(handlers)}"}
    try:
        return handlers[msg["op"]](msg)
    except Exception as e:   # a bad request is answered with an error, not a dead mailbox
        return {"error": f"{type(e).__name__}: {e}"}


def serve_mailbox(ws: Path, handlers: dict[str, Callable[[dict], dict]],
                  stop: threading.Event) -> threading.Thread:
    """Answer ``req_<n>.json`` files in ``ws`` with ``resp_<n>.json`` until ``stop`` is set.

    The Codex coordinator runs sandboxed in its own workspace and reaches the sites only by
    writing requests ``{"op": <handler name>, ...}`` there. A response is written atomically and
    its request removed, so each request is answered once. Handlers run on the mailbox thread,
    outside the caller's ledger tag: a caller charges the site calls made through the mailbox
    from the sites' ``calls`` / ``nominal_usd`` counters.
    """
    def loop() -> None:
        unparsable: dict[str, int] = {}
        while not stop.is_set():
            for req in sorted(ws.glob("req_*.json")):
                n = req.name[len("req_"):-len(".json")]
                try:
                    msg = json.loads(req.read_text())
                except json.JSONDecodeError:
                    unparsable[n] = unparsable.get(n, 0) + 1
                    if unparsable[n] < BAD_POLLS:
                        continue   # probably still being written
                    msg = None
                tmp = ws / f".resp_{n}.json.tmp"
                tmp.write_text(json.dumps(_dispatch(msg, handlers), default=str))
                os.replace(tmp, ws / f"resp_{n}.json")
                req.unlink()
            stop.wait(POLL_S)

    thread = threading.Thread(target=loop, name=f"mailbox:{ws}", daemon=True)
    thread.start()
    return thread


class SiteFailed(RuntimeError):
    """The site agent's run failed as a process (timeout, harness error). It is charged and
    counted but never cached, since the same request may succeed when asked again."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _record(run: harness.AgentRun) -> dict:
    """What a stored call keeps of its run, for inspection. The run's model usage is already in
    the ledger under the memo's tag (the harness ledgers Codex usage there too), so it is part
    of the memo's ``nominal_usd``; returning it as ``nominal_usd_external`` would count it twice.
    """
    return {"agent_nominal_usd": run.nominal_usd_external, "seconds": run.seconds,
            "transcript": run.transcript[-TRANSCRIPT_STEPS:]}


_TABLES: OrderedDict[tuple[str, str, str], SiteTable] = OrderedDict()
_TABLES_LOCK = threading.Lock()


def _table(blob: str, view: SiteView, data_path: Path, kind: str) -> SiteTable:
    """The site table of a stored view, computed once per process: the Site objects of every
    arm restore the same step-1 view."""
    k = (blob, file_digest(data_path), kind)
    with _TABLES_LOCK:
        if k in _TABLES:
            _TABLES.move_to_end(k)
            return _TABLES[k]
    table = site_table(view, data_path, kind=kind)
    with _TABLES_LOCK:
        _TABLES[k] = table
        if len(_TABLES) > TABLE_CACHE:
            _TABLES.popitem(last=False)
    return table


class Site:
    """One site behind the ``SiteHandle`` interface (``agentcoop.coop.protocol``)."""

    def __init__(self, name: str, data_path: Path, card: dict, agent: str, case: CaseSpec, *,
                 task_id: str, run_id: str, proposes: bool = True, timeout: int = 900) -> None:
        self.name, self.data_path, self.agent, self.case = name, Path(data_path), agent, case
        self.task_id, self.run_id, self.proposes, self.timeout = task_id, run_id, proposes, timeout
        self.calls, self.nominal_usd = 0, 0.0
        self.view: SiteView | None = None
        self.table: SiteTable | None = None
        self.view_blob: str | None = None   # store id of the view now in ws/view
        self.errors: list[str] = []         # view and claim validation messages, "<op>: ..."
        self.failures: list[dict] = []      # failed agent calls: {"op", "error"}
        self._card = card
        self._key: dict | None = None       # store key of the step-1 proposal
        self.ws = workspace.new_workspace(
            f"{run_id}/{name}", {self.data_path.name: self.data_path, **case.resources})
        (self.ws / "codebook.json").write_text(json.dumps(card, indent=1))

    def card(self) -> dict:
        return self._card

    def spec(self) -> dict:
        return self.view.spec if self.view is not None else {}

    # -- agent calls ---------------------------------------------------------

    def _identity(self) -> dict:
        return {"v": 1, "case": self.case.name, "task": self.task_id, "site": self.name,
                "agent": self.agent, "data": file_digest(self.data_path)}

    def _call(self, op: str, key: dict, compute: Callable[[], dict]) -> dict | None:
        """One site-agent call through the store, counted and charged whether it is served from
        the store, runs or fails. A failure is charged what the ledger holds under its tag (the
        same amount ``memo`` would have stored), recorded, never cached, and returns None."""
        self.calls += 1
        with llm.tag(f"{self.run_id}/{self.name}/{op}.{time.time_ns()}"):
            calls_inside = llm.current_tag() + "/"   # store.memo nests the run's own tag here
            try:
                out = store.memo(key, compute, tag=f"{self.name}.{op}")
            except SiteFailed as e:
                self.nominal_usd += llm.spent(calls_inside)["nominal_usd"]
                self.failures.append({"op": op, "error": str(e)})
                return None
        self.nominal_usd += out["nominal_usd"]
        return out

    def _run(self, prompt: str, sample: int) -> harness.AgentRun:
        """Run the agent once in the workspace. A failed run leaves the view as it was before
        the call and raises, so it is charged but never cached."""
        run = harness.run_agent(self.agent, self.ws, prompt, sample=sample, timeout=self.timeout)
        if not run.ok:
            self._restore(self.view_blob)
            raise SiteFailed(f"{self.agent} at site {self.name}: {run.error}")
        return run

    def _capture(self) -> None:
        view = self.ws / "view"
        self.view_blob = store.put_blob(view) if view.is_dir() else None

    def _restore(self, blob: str | None) -> None:
        """Make ws/view the stored view ``blob`` (None: no view)."""
        shutil.rmtree(self.ws / "view", ignore_errors=True)
        if blob is not None:
            store.get_blob(blob, self.ws / "view")
        self.view_blob = blob

    def _load(self, op: str) -> None:
        """Validate the view in ws/view and compute its site table. A view that does not load
        leaves the site without a table, so its tests come back untestable(view)."""
        self.view = self.table = None
        if self.view_blob is None:
            self.errors.append(f"{op}: no view/ written")
            return
        self.view, errors = load_view(self.ws / "view", kind=self.case.kind, vocab=self.case.vocab)
        self.errors += [f"{op}: {e}" for e in errors]
        if self.view is None:
            return
        try:
            self.table = _table(self.view_blob, self.view, self.data_path, self.case.kind)
        except (KeyError, ValueError) as e:   # a view the data file cannot be read with
            self.errors.append(f"{op}: site table: {e}")

    def propose(self, sample: int) -> list[Claim]:
        prompt = propose_prompt(self.case, self._card, self.data_path.name,
                                proposes=self.proposes)
        self._key = self._identity() | {"op": "propose", "prompt": _sha(prompt), "sample": sample}

        def compute() -> dict:
            self._restore(None)
            (self.ws / "claims.json").unlink(missing_ok=True)
            run = self._run(prompt, sample)
            self._capture()
            raw, errors = None, []
            if self.proposes:
                try:
                    raw = json.loads((self.ws / "claims.json").read_text())
                except FileNotFoundError:
                    errors.append("no claims.json written")
                except json.JSONDecodeError as e:
                    errors.append(f"claims.json is not valid JSON: {e}")
            return {"claims": raw, "view_blob": self.view_blob, "errors": errors,
                    "ok": self.view_blob is not None and (raw is not None or not self.proposes),
                    **_record(run)}

        out = self._call("propose", self._key, compute)
        if out is not None:
            if out["view_blob"] != self.view_blob:   # served from the store: restore its view
                self._restore(out["view_blob"])
            self.errors += [f"propose: {e}" for e in out["errors"]]
        self._load("propose")
        if out is None or out["claims"] is None:
            return []
        claims, errors = parse_claims(out["claims"], site=self.name, vocab=self.case.vocab,
                                      max_claims=self.case.max_claims)
        self.errors += [f"claims: {e}" for e in errors]
        return claims

    def _followup_key(self, op: str, prompt: str, sample: int) -> dict:
        """A follow-up depends on the step-1 call it follows and on the view it starts from."""
        return self._identity() | {"op": op, "after": self._key, "view": self.view_blob,
                                   "prompt": _sha(prompt), "sample": sample}

    def _edit_view(self, op: str, key: dict, prompt: str, sample: int) -> bool:
        """A follow-up in which the agent may change only its view; the site then tests on the
        reloaded view. True iff the call succeeded and the view loads."""
        def compute() -> dict:
            run = self._run(prompt, sample)
            self._capture()
            return {"view_blob": self.view_blob, **_record(run)}

        out = self._call(op, key, compute)
        if out is None:
            return False
        if out["view_blob"] != self.view_blob:
            self._restore(out["view_blob"])
        self._load(op)
        return self.table is not None

    def repair(self, claim: Claim, defect: str, sample: int) -> bool:
        prompt = repair_prompt(self.case, claim, defect)
        key = self._followup_key("repair", prompt, sample) | {"claim": claim.id,
                                                              "defect": _sha(defect)}
        return self._edit_view("repair", key, prompt, sample)

    def self_check(self, sample: int) -> bool:
        prompt = _self_check_prompt(self.case)
        return self._edit_view("self_check", self._followup_key("self_check", prompt, sample),
                               prompt, sample)

    def ask(self, request: str, sample: int) -> str:
        prompt = _ask_prompt(self.case, request)

        def compute() -> dict:
            answer = self.ws / "answer.md"
            answer.unlink(missing_ok=True)
            run = self._run(prompt, sample)
            self._restore(self.view_blob)   # answering a request never changes the view
            return {"answer": answer.read_text() if answer.exists() else "", **_record(run)}

        out = self._call("ask", self._followup_key("ask", prompt, sample), compute)
        if out is None:
            return f"(site {self.name} did not answer: its agent call failed)"
        return out["answer"] or f"(site {self.name} wrote no answer)"

    # -- deterministic, inside the site boundary -----------------------------

    def holds(self, population: str, contrast: tuple[str, str]) -> bool:
        """Whether the view has units of ``population`` in both groups. A site without a usable
        view cannot rule anything out: the test is routed and comes back untestable(view), so a
        broken view shows in the record instead of being skipped silently."""
        if self.table is None:
            return True
        units = self.table.units
        if self.table.kind == "scrna":   # bulk units are whole samples (population "all")
            units = units[units["population"] == population]
        return all((units["group"] == g).any() for g in contrast)

    def test(self, claim: Claim, *, test: str = "replication",
             contrast: tuple[str, str] | None = None) -> TestResult:
        if self.table is None:
            return TestResult(claim_id=claim.id, site=self.name, test=test, status="untestable",
                              effect=None, se=None, ci95=None, n_per_group=None, mde=None, p=None,
                              competitive_ok=None, reason="view")
        return tester.test_claim(self.table, claim, self.case.tester, site=self.name, test=test,
                                 contrast=contrast)

    def export_stats(self, population: str, contrast: tuple[str, str]) -> pd.DataFrame:
        if self.table is None:
            return pd.DataFrame(columns=STATS_COLUMNS)
        return tester.export_gene_stats(self.table, population, contrast, self.case.tester)


def make_sites(site_specs: list[dict], case: CaseSpec, *, task_id: str, run_id: str,
               agents: dict[str, str] | None = None) -> list[Site]:
    """The Site objects of one run. ``agents`` overrides the agent per site name (e.g. the
    all-Codex robustness run); an optional ``timeout`` per spec sets the wall-clock cap in
    seconds (real cases: 1800)."""
    agents = agents or {}
    return [Site(s["name"], Path(s["data_path"]), s["card"],
                 agents[s["name"]] if s["name"] in agents else s["agent"], case,
                 task_id=task_id, run_id=run_id, proposes=s["proposes"],
                 timeout=int(s.get("timeout", 900)))
            for s in site_specs]
