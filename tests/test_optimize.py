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

    rows = [[g, 1.2, 1e-8, sig_padj] for g in SIGNAL] + [[g, 0.1, 0.5, 1.0] for g in NOISE]
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
