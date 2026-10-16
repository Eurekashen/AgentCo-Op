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
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from agentcoop.coop.claims import Claim, Vocabulary
from agentcoop.coop.tester import TesterParams, TestResult
from agentcoop.coop.view import EXCLUDE, OTHER

if TYPE_CHECKING:
    from agentcoop.coop.protocol import SiteHandle

SPEC_KEYS = ("contrasts", "sample_unit", "normalization", "gene_mapping", "covariates",
             "exclusions")
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
        f"Example (placeholder genes):\n{json.dumps(example, indent=1)}"
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
