"""Offline tests of localization by signal tracing and of the declared edit space."""

from __future__ import annotations

from agentcoop.optimize.edits import local_edits
from agentcoop.optimize.trace import COVERED, STAGE, localize, propagation, trace_uncovered
from agentcoop.pipelines import naive_composition, pseudobulk, split_handoff
from agentcoop.workflow import Artifact, Run

SIGNAL = [f"S{i}" for i in range(8)]
NOISE = [f"N{i}" for i in range(22)]
PROBE = {"cell_type": "B cells", "direction": "up", "n_genes": 8, "genes": SIGNAL,
         "go_hint": [{"term": "t", "p": 1e-9, "overlap": 8, "genes": SIGNAL}]}


def _run(workflow, *, method="pseudobulk", sig_padj=1e-6, list_genes=None, direction="mixed",
         view_context=("cell_type",), claims=()):
    run = Run(workflow=workflow, task_id="t")

    def add(a: Artifact, node: str) -> Artifact:
        run.artifacts[a.id] = a
        run.node_outputs.setdefault(node, []).append(a.id)
        return a

    sig_p = 1e-8 if sig_padj < 0.5 else 0.6  # "missed" = not even nominally significant
    rows = [[g, 1.2, sig_p, sig_padj] for g in SIGNAL] + [[g, 0.1, 0.5, 1.0] for g in NOISE]
    table = add(Artifact(kind="de_table", producer="de",
                         data={"cell_type": "B cells", "method": method, "rows": rows}), "de")
    genes = list_genes if list_genes is not None else SIGNAL + NOISE[:4]
    lst = add(Artifact(kind="gene_list", producer="de->interp", data={"genes": genes},
                       derived_from=[table.id],
                       meta={"cell_type": "B cells", "direction": direction}), "de->interp")
    interp = add(Artifact(kind="interpretation", producer="interp",
                          data={"process": "P", "narrative": "N", "genes": genes},
                          derived_from=[lst.id], meta={"source": lst.meta}), "interp")
    view_data = {"process": "P", "narrative": "N"}
    for k in view_context:
        view_data[k] = lst.meta.get(k, method if k == "method" else None)
    add(Artifact(kind="interpretation_view", producer="interp->report", data=view_data,
                 derived_from=[interp.id]), "interp->report")
    add(Artifact(kind="report", producer="report", data={"claims": list(claims), "summary": ""}),
        "report")
    return run


def test_signal_lost_in_the_analysis_is_localized_to_the_analysis_node():
    f = trace_uncovered(_run(naive_composition(), method="wilcoxon", sig_padj=0.9), PROBE,
                        "report")
    assert (f.symptom, f.location) == ("analysis_misses_signal", "de")


def test_mixed_direction_list_is_localized_to_the_gene_list_handoff():
    f = trace_uncovered(_run(naive_composition()), PROBE, "report")
    assert (f.symptom, f.location) == ("direction_dropped", "de->interp")


def test_dropped_context_is_localized_to_the_view_edge():
    run = _run(split_handoff(naive_composition()), direction="up", view_context=("cell_type",))
    f = trace_uncovered(run, PROBE, "report")
    assert (f.symptom, f.location) == ("context_dropped", "interp->report")


def test_signal_that_arrives_with_context_is_an_integration_finding():
    run = _run(split_handoff(naive_composition()), direction="up",
               view_context=("cell_type", "direction", "method"))
    f = trace_uncovered(run, PROBE, "report")
    assert (f.symptom, f.location) == ("integration_omits_signal", "report")


def test_propagation_depth_increases_as_losses_are_removed():
    profile = {"uncovered": [PROBE], "claims": []}
    stages = [
        _run(naive_composition(), method="wilcoxon", sig_padj=0.9),
        _run(naive_composition()),
        _run(split_handoff(naive_composition()), direction="up"),
        _run(split_handoff(naive_composition()), direction="up",
             view_context=("cell_type", "direction", "method")),
    ]
    depths = [propagation(r, [PROBE], profile)[0] for r in stages]
    assert depths == [STAGE["analysis_misses_signal"], STAGE["direction_dropped"],
                      STAGE["context_dropped"], STAGE["integration_omits_signal"]]
    assert depths == sorted(depths)
    assert propagation(stages[-1], [PROBE], {"uncovered": [], "claims": []}) == [COVERED]


def test_localize_merges_findings_at_the_same_location():
    run = _run(naive_composition())
    other = dict(PROBE, cell_type="B cells", direction="up")
    findings = localize(run, {"uncovered": [PROBE, other], "claims": []}, ["B cells"])
    primary = [f for f in findings if f.symptom not in ("bypass_available", "no_claims")]
    assert len(primary) == 1 and primary[0].weight > 1.5
    # A signal lost at a handoff also offers the consumer as a bypass location, ranked lower.
    bypass = [f for f in findings if f.symptom == "bypass_available"]
    assert len(bypass) == 1 and bypass[0].location == "report"
    assert findings[0] is primary[0]


def test_edits_change_exactly_one_declared_thing():
    w = naive_composition()
    for loc in ("de", "de->interp", "interp->report", "report"):
        for e in local_edits(w, loc):
            new = e.apply(w)
            assert new.fingerprint() != w.fingerprint(), e.description
            assert w.fingerprint() == naive_composition().fingerprint()  # original untouched
    methods = [e.params["method"] for e in local_edits(pseudobulk(w), "de")]
    assert methods == ["wilcoxon"]


def test_fixing_direction_before_dilution_is_progress():
    profile = {"uncovered": [PROBE], "claims": []}
    mixed = _run(naive_composition(), list_genes=SIGNAL + NOISE)
    split_but_diluted = _run(split_handoff(naive_composition()), direction="up",
                             list_genes=SIGNAL + NOISE)
    d0 = propagation(mixed, [PROBE], profile)[0]
    d1 = propagation(split_but_diluted, [PROBE], profile)[0]
    assert d0 == STAGE["direction_dropped"] and d1 == STAGE["signal_diluted"] and d1 > d0


# ---------------------------------------------------------------------------
# Shape 2 (sites)
# ---------------------------------------------------------------------------

from agentcoop.optimize import trace_sites  # noqa: E402
from agentcoop.pipelines import naive_multisite  # noqa: E402

CTS = ["NK cells", "B cells"]
SITE_PROBE = {"cell_type": "NK cells", "direction": "up", "n_genes": 6,
              "genes": ["G1", "G2", "G3", "G4", "G5", "G6"], "go_hint": []}


def _site_run(report_a, report_b):
    run = Run(workflow=naive_multisite(), task_id="t")
    for site, rep in (("A", report_a), ("B", report_b)):
        a = Artifact(kind="site_report", producer=f"site_{site}",
                     data={"site": site, "request_kind": "analysis", "report": rep})
        run.artifacts[a.id] = a
        run.node_outputs[f"site_{site}"] = [a.id]
    rep = Artifact(kind="report", producer="lead", data={"claims": [], "summary": ""})
    run.artifacts[rep.id] = rep
    run.node_outputs["lead"] = [rep.id]
    return run


GOOD = {"findings": [{"cell_type": "NK cells", "direction": "up", "program": "OXPHOS",
                      "genes": ["G1", "G2", "G3"]}]}
VAGUE = {"findings": [{"cell_type": "NK cells", "direction": "up", "program": "OXPHOS"}]}


def test_site_trace_stages():
    none = trace_sites.trace_signal(_site_run({}, {}), SITE_PROBE, CTS)
    one = trace_sites.trace_signal(_site_run(GOOD, {}), SITE_PROBE, CTS)
    vague = trace_sites.trace_signal(_site_run(GOOD, VAGUE), SITE_PROBE, CTS)
    both = trace_sites.trace_signal(_site_run(GOOD, GOOD), SITE_PROBE, CTS)
    assert none.symptom == "site_missed_signal" and "->site_" in none.location
    assert one.symptom == "unverified_at_other_site" and one.location == "lead"
    assert vague.symptom == "not_comparable"
    assert both.symptom == "integration_omits_signal"
    profile = {"uncovered": [SITE_PROBE], "claims": []}
    depths = [trace_sites.propagation(_site_run(a, b), [SITE_PROBE], profile, CTS)[0]
              for a, b in (({}, {}), (GOOD, {}), (GOOD, VAGUE), (GOOD, GOOD))]
    assert depths == sorted(depths) and len(set(depths)) == 4


def test_site_request_edit_applies_to_every_site():
    from agentcoop.optimize.edits import local_edits

    w = naive_multisite()
    edit = next(e for e in local_edits(w, "task->site_A") if e.params.get("template") == "structured")
    new = edit.apply(w)
    assert all(e.params["template"] == "structured" for e in new.edges if e.handoff == "site_request")
    lead_edits = {e.params.get("add") or e.params.get("branch") for e in local_edits(w, "lead")}
    assert {"cross_verify", "replication_check"} <= lead_edits


def test_an_empty_report_is_a_finding_not_a_clean_bill():
    run = _run(naive_composition())
    findings = localize(run, {"uncovered": [], "claims": []}, ["B cells"])
    assert [f.symptom for f in findings] == ["no_claims"]
    empty = _site_run({}, {})
    site_findings = trace_sites.localize(empty, {"uncovered": [], "claims": []}, CTS)
    assert site_findings[0].symptom == "no_claims" and "->site_" in site_findings[0].location


def test_compound_edit_applies_both_changes_and_keys_them_together():
    from agentcoop.optimize.edits import compound, local_edits

    w = naive_multisite()
    edits = {tuple(e.params.items()): e for e in local_edits(w, "task->site_A")}
    c = compound(edits[(("scope", "local"),)], edits[(("report_weak", True),)])
    params = next(e.params for e in c.apply(w).edges if e.handoff == "site_request")
    assert params["scope"] == "local" and params["report_weak"] is True
    assert c.family == "compound" and set(c.params) == {"scope", "report_weak"}


def test_optimize_runs_end_to_end_offline(tmp_path, monkeypatch):
    """The whole loop on a toy task: stubs for execution and evidence, no LLM, no data."""
    from agentcoop.optimize import loop
    from agentcoop.workflow import Task
    import json as _json

    public = tmp_path / "t" / "public"
    public.mkdir(parents=True)
    (public / "task.json").write_text(_json.dumps({"task_id": "toy", "question": "q",
                                                   "data": "x", "cell_types": ["B cells"]}))
    task = Task.load(tmp_path / "t")
    iter(range(1000))

    def fake_execute(w, task, sample=0, samples=None):
        r = _run(w, direction="up" if w.fingerprint() != naive_composition().fingerprint()
                 else "mixed")
        r.cost = {"nominal_usd": 0.01, "actual_usd": 0.0}
        return r

    def fake_profile(run, task):
        mixed = any(a.meta.get("direction") == "mixed" for a in run.artifacts.values()
                    if a.kind == "gene_list")
        return {"claims": [], "uncovered": [PROBE] if mixed else [],
                "counts": {"supported": 0 if mixed else 1, "contradicted": 0, "unassessed": 0,
                           "ungrounded": 0, "uncovered_groups": 1 if mixed else 0}}

    monkeypatch.setattr(loop, "execute", fake_execute)
    monkeypatch.setattr(loop, "_profile", fake_profile)
    monkeypatch.setattr(loop, "_rank", lambda f, edits, sample: edits)
    monkeypatch.setattr(loop, "_prompt_edit", lambda w, n, f, s: loop.prompt_edit(n, "be careful"))
    monkeypatch.setattr(loop.judge, "evidence_judge", lambda *a, **k: {"winner": "A"})
    monkeypatch.setattr(loop.judge, "rubric_for", lambda q, **k: [])
    out = loop.optimize(task, naive_composition(),
                        budget=loop.Budget(max_usd=10, max_iterations=4, max_executions=12))
    assert out.stop_reason
    assert any(a.accepted for a in out.attempts)  # some edit removed the mixed-direction loss
    assert out.executions <= 12  # never overruns the execution budget


def test_escalation_continues_the_incumbent_under_the_same_budget(tmp_path, monkeypatch):
    """A second optimizer started from an Outcome inherits its incumbent, spend and execution
    count: the budget is shared, never renewed, and its attempts number after the first's."""
    from agentcoop.optimize import baselines, loop
    from agentcoop.workflow import Task
    import json as _json

    public = tmp_path / "t" / "public"
    public.mkdir(parents=True)
    (public / "task.json").write_text(_json.dumps({"task_id": "toy", "question": "q",
                                                   "data": "x", "cell_types": ["B cells"]}))
    task = Task.load(tmp_path / "t")

    def fake_execute(w, task, sample=0, samples=None):
        r = _run(w)
        r.cost = {"nominal_usd": 0.01, "actual_usd": 0.0}
        return r

    good = split_handoff(pseudobulk(naive_composition()))

    def fake_profile_dict(report, task):
        class P:
            def to_dict(self):
                n = len(report.get("claims", []))
                return {"claims": [], "uncovered": [],
                        "counts": {"supported": n, "contradicted": 0, "unassessed": 0,
                                   "ungrounded": 0, "uncovered_groups": 0}}
        return P()

    monkeypatch.setattr(baselines, "execute", fake_execute)
    monkeypatch.setattr(loop.checks, "profile_task", fake_profile_dict)
    start = loop.Outcome(workflow=naive_composition(), run=fake_execute(naive_composition(), task),
                         profile={"claims": [], "uncovered": [], "counts": {
                             "supported": 0, "contradicted": 0, "unassessed": 0,
                             "ungrounded": 0, "uncovered_groups": 0}},
                         nominal_usd=0.05, stop_reason="no untried edit at the implicated locations",
                         executions=5)
    start.attempts.append(loop.Attempt(3, "x", "handoff", "", "", {}, None, "B", None, None,
                                       False, 0.01))
    out = baselines._accept_loop(task, naive_composition(),
                                 lambda w, run, prof, it: [("same", good)],
                                 budget=loop.Budget(max_usd=10, max_iterations=8,
                                                    max_executions=9),
                                 sample=0, label="llm_editor", start=start)
    assert out.executions == 9  # 5 inherited + 2 iterations × (control + candidate)
    assert [a.iteration for a in out.attempts] == [3, 4, 5]
    assert out.stop_reason.startswith("no untried edit") and "llm_editor: budget" in out.stop_reason


# ---------------------------------------------------------------------------
# v3.4 fixes from the 2026-09-29 code review
# ---------------------------------------------------------------------------


def test_controls_redraw_every_node_an_edit_can_change():
    from agentcoop.optimize.loop import _affected
    from agentcoop.pipelines import ms_local, naive_multisite

    ms = naive_multisite()
    # A site-request edit rewrites both request edges: both sites and the lead are re-drawn.
    assert _affected(ms, ms_local(ms)) == {"site_A", "site_B", "lead"}
    # A compound across two locations (handoff + interpretation view) re-draws both suffixes.
    w = naive_composition()
    both = split_handoff(w)
    assert _affected(w, both) == {"interp", "report"}


def test_locks_keep_accepted_settings_but_let_lists_grow():
    from agentcoop.optimize.loop import _changes, _violates
    from agentcoop.pipelines import carry_stats

    w = naive_composition()
    accepted = split_handoff(w)
    locked = _changes(w, accepted)
    # Reverting the accepted gene-list setting is blocked ...
    assert _violates(locked, accepted, w)
    # ... adding more context to the view (a list that only grows) is not.
    assert not _violates(locked, accepted, carry_stats(accepted))


def test_parse_workflow_keeps_pooled_data_agents_out_of_multisite_tasks():
    import pytest

    from agentcoop.optimize.baselines import _parse_workflow
    from agentcoop.pipelines import naive_multisite

    ms = naive_multisite().to_dict()
    assert _parse_workflow({"workflow": ms}, "multisite")
    ms["nodes"].append({"id": "de", "agent": "de_analyst", "config": {}})
    ms["edges"].append({"src": "task", "dst": "de", "handoff": "task", "params": {}})
    with pytest.raises(ValueError):
        _parse_workflow({"workflow": ms}, "multisite")


def test_a_summary_without_the_signal_genes_does_not_count_as_carrying_it():
    from agentcoop.optimize.trace import trace_uncovered

    run = _run(naive_composition(), list_genes=NOISE[:10])  # the list drops the signal
    run.artifacts["summary"] = Artifact(
        kind="de_summary", producer="de->report", id="summary",
        data={"top_up(gene, log2fc, padj)": [[g, 1.0, 0.01] for g in NOISE[:5]]},
        derived_from=[next(a.id for a in run.artifacts.values() if a.kind == "de_table")])
    f = trace_uncovered(run, PROBE, "report")
    assert f.symptom == "signal_not_handed_off"


def test_budget_room_counts_the_control_a_candidate_needs():
    from agentcoop.optimize.loop import Budget

    b = Budget(max_usd=10, max_executions=16)
    assert b.room(0.0, 14, 2) and not b.room(0.0, 15, 2) and b.room(0.0, 15, 1)


def test_escalation_pairs_singles_that_already_failed(tmp_path, monkeypatch):
    """Every single site-request edit fails alone; the next iteration must try pairs of them."""
    from agentcoop.optimize import loop
    from agentcoop.pipelines import naive_multisite
    from agentcoop.optimize.trace import Finding
    from agentcoop.workflow import Task
    import json as _json

    public = tmp_path / "t" / "public"
    public.mkdir(parents=True)
    (public / "task.json").write_text(_json.dumps({
        "task_id": "toy", "question": "q", "data": "site_A.h5ad", "shape": "multisite",
        "cell_types": ["B cells"], "sites": {"A": "site_A.h5ad", "B": "site_B.h5ad"}}))
    task = Task.load(tmp_path / "t")
    empty = {"claims": [], "uncovered": [], "counts": {"supported": 0, "contradicted": 0,
             "unassessed": 0, "ungrounded": 0, "uncovered_groups": 0}}

    def fake_execute(w, task, sample=0, samples=None):
        r = Run(workflow=w, task_id="toy")
        r.cost = {"nominal_usd": 0.0, "actual_usd": 0.0}
        return r

    monkeypatch.setattr(loop, "execute", fake_execute)
    monkeypatch.setattr(loop, "_profile", lambda run, task: empty)
    monkeypatch.setattr(loop, "_localize", lambda run, prof, task: [
        Finding("site_missed_signal", "task->site_A", "p", "d")])
    monkeypatch.setattr(loop, "_rank", lambda f, edits, sample: edits)
    monkeypatch.setattr(loop, "_propagation", lambda *a: 0)
    monkeypatch.setattr(loop.judge, "rubric_for", lambda q, **k: [])
    out = loop.optimize(task, naive_multisite(),
                        budget=loop.Budget(max_usd=10, max_iterations=3, max_executions=40,
                                           edits_per_iteration=10))
    assert any(a.family == "compound" for a in out.attempts)


def test_bix_repair_edit_keeps_a_dag_and_pools_evidence_at_the_arbiter():
    from agentcoop.optimize.bix_loop import _decided, add_repair
    from agentcoop.pipelines import bix_evidence_team

    w = add_repair(bix_evidence_team(), "finch", 1, "audit")
    w = add_repair(w, "react", 1, "audit")
    w = add_repair(w, "finch", 2, "audit_r1")
    order = w.order()
    assert order[-1] == "arbiter" and order.index("finch_r1") < order.index("audit_r1")
    assert {e.src for e in w.incoming("arbiter")} == {"audit", "audit_r1", "audit_r2"}
    fb = [e for e in w.incoming("finch_r2") if e.handoff == "audit_feedback"]
    assert fb and fb[0].src == "audit_r1" and fb[0].params == {"agent": "finch"}
    assert w.node("finch_r1").config["replica"] == 11
    assert not _decided([])
    assert _decided([{"score": [1, 0, 1, 0]}, {"score": [0, 0, 2, -1]}])
    assert not _decided([{"score": [1, 1, 1, 0]}, {"score": [1, 1, 1, -1]}])  # evidence tie
    assert not _decided([{"score": [0, 0, 3, 0]}])  # nothing clean
