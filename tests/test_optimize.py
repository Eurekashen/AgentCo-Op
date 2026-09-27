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
    assert len(findings) == 1 and findings[0].weight > 1.5


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
