"""Real sites (coop/site.py) with a fake agent: offline, no API key, no agent process."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import pytest

from agentcoop.coop import site as site_mod
from agentcoop.coop import tester
from agentcoop.coop.claims import Claim, Vocabulary
from agentcoop.coop.site import CaseSpec, file_digest, propose_prompt, repair_prompt, serve_mailbox
from agentcoop.coop.view import load_view, site_table

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
