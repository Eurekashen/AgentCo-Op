"""Offline tests of the mechanisms the experiments rely on. No API key, no network."""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# The optimizer side must have no path to the answer sheet
# ---------------------------------------------------------------------------

TRUTH_FREE = ["agentcoop/evidence", "agentcoop/optimize", "agentcoop/agents",
              "agentcoop/handoffs.py", "agentcoop/workflow.py", "agentcoop/pipelines.py",
              "agentcoop/eval_free.py"]


@pytest.fixture(autouse=True)
def _clean_agentcoop_env(monkeypatch):
    """Tests must not depend on AGENTCOOP_* settings exported in the developer's shell."""
    for k in [k for k in os.environ if k.startswith("AGENTCOOP_")]:
        monkeypatch.delenv(k)


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


def test_truth_free_modules_never_import_the_grader_or_truth_loader():
    files = []
    for entry in TRUTH_FREE:
        p = ROOT / entry
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    for f in files:
        imported = _imports(f)
        assert not any(m.startswith("agentcoop.eval.") or m == "agentcoop.eval" for m in imported), f
        text = f.read_text()
        assert "load_truth" not in text and "truth.json" not in text, f
        assert '"hidden"' not in text and "hidden/" not in text, f


# ---------------------------------------------------------------------------
# Planting
# ---------------------------------------------------------------------------


def test_binomial_thinning_touches_only_selected_rows_and_column():
    from agentcoop.tasks.planted import _thin

    rng = np.random.default_rng(0)
    dense = rng.poisson(20, size=(50, 4)).astype(np.float32)
    X = sp.csc_matrix(dense)
    mask = np.zeros(50, dtype=bool)
    mask[:25] = True
    _thin(X, mask, 2, 0.25, rng)
    out = X.toarray()
    assert np.array_equal(out[:, [0, 1, 3]], dense[:, [0, 1, 3]])
    assert np.array_equal(out[25:, 2], dense[25:, 2])
    ratio = out[:25, 2].sum() / dense[:25, 2].sum()
    assert 0.18 < ratio < 0.32
    assert np.all(out == np.round(out)) and np.all(out >= 0)


def test_family_lookup_and_core_sets_come_from_two_libraries():
    from agentcoop.tasks.planted import FAMILIES, family_of

    assert family_of("Interferon Alpha Response") == "interferon"
    for fam, spec in FAMILIES.items():
        assert spec["programs"] and spec["go_terms"] and spec["description"], fam


# ---------------------------------------------------------------------------
# Normalization (grader side and truth-free side agree)
# ---------------------------------------------------------------------------

CTS = ["CD14+ Monocytes", "CD4 T cells", "B cells", "NK cells", "CD8 T cells",
       "FCGR3A+ Monocytes"]


@pytest.mark.parametrize("text,expected", [
    ("CD14+ Monocytes", "CD14+ Monocytes"),
    ("classical monocytes (CD14+)", "CD14+ Monocytes"),
    ("non-classical monocytes", "FCGR3A+ Monocytes"),
    ("FCGR3A+ Monocytes", "FCGR3A+ Monocytes"),
    ("NK cells", "NK cells"),
    ("CD8 T cells", "CD8 T cells"),
    ("B cells", "B cells"),
    ("CD4+ T cells", "CD4 T cells"),
])
def test_cell_type_normalization(text, expected):
    from agentcoop.eval.grader import canon_cell_types
    from agentcoop.eval_free import canon_cell_type

    assert canon_cell_type(text, CTS) == expected
    assert canon_cell_types(text) == [expected]


def test_direction_normalization():
    from agentcoop.eval_free import canon_direction

    assert canon_direction("up") == "up"
    assert canon_direction("Induced") == "up"
    assert canon_direction("repressed") == "down"
    assert canon_direction("mixed") == "unknown"


def test_grader_counts_duplicates_once_and_separates_error_kinds(monkeypatch):
    from agentcoop.eval import grader
    from agentcoop.tasks.planted import PlantedProgram, TaskTruth

    fam = {"IFN response": "interferon", "OXPHOS": "oxphos", "ribosome": "myc_ribosome",
           "vague": "other"}
    monkeypatch.setattr(grader, "classify_family", lambda name: fam[name])
    truth = TaskTruth("t", 0, [
        PlantedProgram("Interferon Alpha Response", "interferon", "up", ["B cells"], ["ISG15"],
                       1.0, ["D1"], "reproducible"),
        PlantedProgram("Oxidative Phosphorylation", "oxphos", "down", ["NK cells"], ["NDUFS2"],
                       1.0, ["D1"], "reproducible"),
        PlantedProgram("Myc Targets V1", "myc_ribosome", "up", ["CD4 T cells"], ["NPM1"], 2.0,
                       ["D1"], "decoy"),
    ])
    report = {"claims": [
        {"cell_type": "B cells", "direction": "up", "program": "IFN response"},
        {"cell_type": "B cells", "direction": "up", "program": "IFN response"},  # duplicate
        {"cell_type": "NK cells", "direction": "up", "program": "OXPHOS"},  # wrong direction
        {"cell_type": "CD4 T cells", "direction": "up", "program": "ribosome"},  # decoy
        {"cell_type": "CD8 T cells", "direction": "up", "program": "vague"},  # other
    ]}
    g = grader._grade(report, truth)
    assert g["n_claims"] == 4 and g["tp"] == 1
    assert g["wrong_direction"] == 1 and g["decoy_claimed"] == 1 and g["other_fp"] == 1
    assert g["recall"] == 0.5 and g["precision"] == 0.25


# ---------------------------------------------------------------------------
# Handoffs
# ---------------------------------------------------------------------------


def _table(ct="B cells"):
    from agentcoop.workflow import Artifact

    rows = [["G1", 1.2, 1e-9, 1e-6], ["G2", -1.1, 1e-8, 1e-5], ["G3", 0.9, 1e-4, 0.01],
            ["G4", -0.2, 0.3, 0.9], ["G5", 0.1, 0.5, 1.0]]
    return Artifact(kind="de_table", producer="de",
                    data={"cell_type": ct, "method": "wilcoxon",
                          "columns": ["gene", "log2fc", "pvalue", "padj"], "rows": rows})


def test_naive_genelist_handoff_mixes_directions_and_keeps_noise():
    from agentcoop.handoffs import de_to_genelists

    (lst,) = de_to_genelists([_table()], {"n": 4}, None, "interp")
    assert lst.data == {"genes": ["G1", "G2", "G3", "G4"]}
    assert lst.meta["direction"] == "mixed"


def test_split_handoff_filters_and_separates_directions():
    from agentcoop.handoffs import de_to_genelists

    out = de_to_genelists([_table()], {"n": 30, "split_direction": True, "max_padj": 0.05,
                                       "min_genes": 1}, None, "interp")
    assert {a.meta["direction"]: a.data["genes"] for a in out} == {"up": ["G1", "G3"],
                                                                   "down": ["G2"]}
    assert all(a.derived_from == [_table().id] for a in out)


def test_interpretation_view_passes_only_chosen_context():
    from agentcoop.handoffs import interpretation_view
    from agentcoop.workflow import Artifact

    interp = Artifact(kind="interpretation", producer="interp",
                      data={"process": "P", "narrative": "N", "genes": ["G1"]},
                      meta={"source": {"cell_type": "B cells", "direction": "up",
                                       "stats": [["G1", 1.2, 1e-6]]}})
    (naive,) = interpretation_view([interp], {"context": ["cell_type"]}, None, "report")
    assert naive.data == {"process": "P", "narrative": "N", "cell_type": "B cells"}
    (rich,) = interpretation_view([interp], {"context": ["cell_type", "direction", "stats"]},
                                  None, "report")
    assert rich.data["direction"] == "up" and "gene_stats(gene, log2fc, padj)" in rich.data


# ---------------------------------------------------------------------------
# Engine: lineage, node cache, per-node samples
# ---------------------------------------------------------------------------


@pytest.fixture
def toy(tmp_path, monkeypatch):
    from agentcoop import workflow as wf

    monkeypatch.setattr(wf, "NODE_CACHE", tmp_path / "cache")
    monkeypatch.setenv("AGENTCOOP_LLM_DB", str(tmp_path / "llm.sqlite"))
    calls = {"a": 0, "b": 0}

    def agent_a(inputs, config, ctx):
        calls["a"] += 1
        return [wf.Artifact(kind="num", producer=ctx.node_id, data={"x": config["x"]},
                            derived_from=[i.id for i in inputs])]

    def agent_b(inputs, config, ctx):
        calls["b"] += 1
        total = sum(i.data["x"] for i in inputs) + ctx.sample
        return [wf.Artifact(kind="num", producer=ctx.node_id, data={"x": total},
                            derived_from=[i.id for i in inputs])]

    monkeypatch.setitem(wf.AGENTS, "toy_a", agent_a)
    monkeypatch.setitem(wf.AGENTS, "toy_b", agent_b)
    public = tmp_path / "task" / "public"
    public.mkdir(parents=True)
    (public / "task.json").write_text(json.dumps({"task_id": "toy", "question": "q",
                                                  "data": "none"}))
    task = wf.Task.load(tmp_path / "task")
    w = wf.Workflow(nodes=[wf.Node("a", "toy_a", {"x": 1}), wf.Node("b", "toy_b", {})],
                    edges=[wf.Edge("task", "a", "task"), wf.Edge("a", "b", "passthrough")],
                    output="b")
    return wf, task, w, calls


def test_lineage_reaches_the_task(toy):
    wf, task, w, calls = toy
    run = wf.execute(w, task)
    out = run.output()
    assert out.data == {"x": 1}
    kinds = [run.artifacts[i].kind for i in run.lineage(out.id)]
    assert kinds[0] == "num" and "task" in kinds


def test_editing_downstream_replays_prefix_from_cache(toy):
    wf, task, w, calls = toy
    wf.execute(w, task)
    assert calls == {"a": 1, "b": 1}
    w2 = w.copy()
    w2.node("b").config = {"changed": True}
    wf.execute(w2, task)
    assert calls == {"a": 1, "b": 2}  # prefix reused, suffix re-executed


def test_per_node_sample_redraws_only_that_node(toy):
    wf, task, w, calls = toy
    base = wf.execute(w, task)
    redraw = wf.execute(w, task, samples={"b": 5})
    assert calls == {"a": 1, "b": 2}
    assert base.output().data == {"x": 1} and redraw.output().data == {"x": 6}


def test_failed_node_is_recorded_not_raised(toy, monkeypatch):
    wf, task, w, calls = toy

    def boom(inputs, config, ctx):
        raise ValueError("nope")

    monkeypatch.setitem(wf.AGENTS, "toy_b", boom)
    run = wf.execute(w, task)
    assert "b" in run.errors and run.output() is None


def test_workflow_roundtrip_and_fingerprint(toy):
    wf, task, w, calls = toy
    again = wf.Workflow.from_dict(json.loads(json.dumps(w.to_dict())))
    assert again.fingerprint() == w.fingerprint()
    again.node("a").config["x"] = 2
    assert again.fingerprint() != w.fingerprint()


# ---------------------------------------------------------------------------
# openai-0.28 shim translation
# ---------------------------------------------------------------------------


def test_shim_turns_orphan_function_results_into_paired_tool_calls():
    sys.path.insert(0, str(ROOT / "external_agents" / "shims"))
    try:
        import importlib

        shim = importlib.import_module("openai")
        assert hasattr(shim, "ChatCompletion") and shim.__file__.startswith(str(ROOT))
        msgs = shim._translate([
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "function", "name": "get_x",
             "content": "Function has been called with params {'gene_set': 'A,B'}, and returns 1."},
        ])
    finally:
        sys.path.remove(str(ROOT / "external_agents" / "shims"))
        sys.modules.pop("openai", None)
    assert msgs[2]["role"] == "assistant" and msgs[3]["role"] == "tool"
    assert msgs[2]["tool_calls"][0]["id"] == msgs[3]["tool_call_id"]
    assert json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"]) == {"gene_set": "A,B"}


# ---------------------------------------------------------------------------
# Model-agnostic capability layer
# ---------------------------------------------------------------------------


@pytest.fixture
def profiles(monkeypatch):
    from agentcoop import llm

    monkeypatch.setattr(llm, "_profiles", {
        "reasoner-no-chat-tools": {"reasoning": True, "effort_none": True,
                                   "chat_tools_with_reasoning": False,
                                   "temperature_with_none": True, "responses_reasoning": True},
        "reasoner": {"reasoning": True, "effort_none": False, "chat_tools_with_reasoning": True,
                     "temperature_with_none": False, "responses_reasoning": True},
        "plain": {"reasoning": False, "effort_none": False, "chat_tools_with_reasoning": False,
                  "temperature_with_none": False, "responses_reasoning": False},
    })
    return llm


def test_tools_on_a_reasoner_without_chat_tool_support_fall_back_to_effort_none(profiles):
    p = profiles._adapt_chat({"model": "reasoner-no-chat-tools", "messages": [], "tools": [1],
                              "reasoning_effort": "none", "temperature": 0})
    assert p["reasoning_effort"] == "none" and p["temperature"] == 0


def test_a_non_reasoning_model_gets_no_reasoning_parameter(profiles):
    p = profiles._adapt_chat({"model": "plain", "messages": [], "reasoning_effort": "low"})
    assert "reasoning_effort" not in p and "temperature" not in p
    q = profiles._adapt_chat({"model": "plain", "messages": [], "tools": [1],
                              "reasoning_effort": "none", "temperature": 0})
    assert "reasoning_effort" not in q and q["temperature"] == 0
    r = profiles._adapt_responses({"model": "plain", "input": [], "reasoning": {"effort": "low"},
                                   "include": ["reasoning.encrypted_content"]})
    assert "reasoning" not in r and "include" not in r


def test_a_reasoner_keeps_its_effort_and_drops_unsupported_temperature(profiles):
    p = profiles._adapt_chat({"model": "reasoner", "messages": [], "tools": [1],
                              "reasoning_effort": "low", "temperature": 0})
    assert p["reasoning_effort"] == "low" and "temperature" not in p
    q = profiles._adapt_chat({"model": "reasoner", "messages": [], "reasoning_effort": "none"})
    assert "reasoning_effort" not in q  # no "none" level: the model default applies


def test_grader_model_is_pinned_independently_of_the_backbone(monkeypatch):
    from agentcoop import llm

    monkeypatch.setenv("AGENTCOOP_MODEL", "some-other-model")
    monkeypatch.delenv("AGENTCOOP_GRADER_MODEL", raising=False)
    assert llm.backbone() == "some-other-model" and llm.grader_model() == llm.DEFAULT_MODEL


def test_node_cache_separates_backbones(toy, monkeypatch):
    wf, task, w, calls = toy
    wf.execute(w, task)
    assert calls["b"] == 1
    monkeypatch.setenv("AGENTCOOP_MODEL", "another-model")
    wf.execute(w, task)
    assert calls["b"] == 2  # a different backbone never reuses another model's outputs


def test_external_agents_do_not_follow_the_framework_backbone(toy, monkeypatch):
    from agentcoop import llm

    wf, task, w, calls = toy
    monkeypatch.setattr(wf, "EXTERNAL", {"toy_b"})
    wf.execute(w, task)
    monkeypatch.setenv("AGENTCOOP_MODEL", "another-model")
    wf.execute(w, task)
    assert calls["b"] == 1  # the external agent's model did not change, so it replays
    assert llm.external_model() == llm.DEFAULT_MODEL
    monkeypatch.setenv("AGENTCOOP_EXTERNAL_MODEL", "yet-another")
    wf.execute(w, task)
    assert calls["b"] == 2


def _fake_codex(tmp_path, monkeypatch, script: str):
    import stat

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "codex"
    exe.write_text("#!/bin/sh\n" + script)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    from agentcoop.agents import codex_agent

    monkeypatch.setattr(codex_agent.llm, "record_external", lambda *a, **k: 0.0)
    return codex_agent


def test_codex_failure_without_a_report_is_an_error_not_a_result(tmp_path, monkeypatch):


    ca = _fake_codex(tmp_path, monkeypatch,
                     'echo \'{"type":"turn.failed","error":{"message":"usage limit"}}\'\nexit 1\n')
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(ca.CodexFailed):
        ca.run_codex(ws, "brief", "gpt-6-luna", 30)


def test_codex_events_tolerate_a_truncated_line_and_commands_are_unique(tmp_path, monkeypatch):
    item = '{"type":"item.%s","item":{"id":"c1","type":"command_execution","command":"ls"}}'
    ca = _fake_codex(tmp_path, monkeypatch,
                     f"echo '{item % 'started'}'\necho '{item % 'completed'}'\n"
                     "echo '{\"claims\": []}' > report.json\nprintf '{\"type\":\"turn.co'\n")
    ws = tmp_path / "ws"
    ws.mkdir()
    out = ca.run_codex(ws, "brief", "gpt-6-luna", 30)
    assert out["commands"] == ["ls"] and out["report"] == {"claims": []}
    assert not out["timed_out"]


def test_shim_marks_infrastructure_errors_before_geneagent_swallows_them(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external_agents" / "shims"))
    import openai as shim  # the openai-0.28 shim, not the SDK

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(shim._llm, "chat", boom)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError):
        shim.ChatCompletion.create(messages=[{"role": "user", "content": "hi"}])
    assert "network down" in (tmp_path / shim.INFRA_MARKER).read_text()
