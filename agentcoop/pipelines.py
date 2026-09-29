"""Reference workflows for task shape 1 (DE analysis → gene-set interpretation → integration).

``naive_composition`` is the three components connected at their native interfaces. The
``CANDIDATES`` below are the fixed diagnostic edit set of the feasibility experiment: each one
changes a single, named thing (a handoff, an analysis config, an evidence branch, a prompt), so
its effect is attributable.
"""

from __future__ import annotations

from typing import Callable

from agentcoop.workflow import Edge, Node, Workflow

NAIVE_GENELISTS = {"n": 30, "split_direction": False}
SPLIT_GENELISTS = {"n": 30, "split_direction": True, "max_padj": 0.05, "min_genes": 5}


def naive_composition() -> Workflow:
    """How the three components are wired when simply connected at their native interfaces:
    cell-level DE, the top genes per cell type handed to GeneAgent as one list, and GeneAgent's
    answer handed to the integrator labelled with its cell type."""
    return Workflow(
        name="naive",
        nodes=[
            Node("de", "de_analyst", {"method": "wilcoxon"}),
            Node("interp", "geneagent", {}),
            Node("report", "integrator", {}),
        ],
        edges=[
            Edge("task", "de", "task"),
            Edge("de", "interp", "de_to_genelists", dict(NAIVE_GENELISTS)),
            Edge("interp", "report", "interpretation_view", {"context": ["cell_type"]}),
        ],
        output="report",
    )


def _edge(w: Workflow, src: str, dst: str) -> Edge:
    return next(e for e in w.edges if e.src == src and e.dst == dst)


def split_handoff(w: Workflow) -> Workflow:
    """Handoff edit: significant genes only, one list per direction, direction travels along."""
    w = w.copy()
    _edge(w, "de", "interp").params = dict(SPLIT_GENELISTS)
    _edge(w, "interp", "report").params = {"context": ["cell_type", "direction"]}
    w.name += "+split"
    return w


def carry_stats(w: Workflow) -> Workflow:
    """Handoff edit: the per-gene statistics and selection rule travel with each interpretation."""
    w = w.copy()
    view = _edge(w, "interp", "report")
    view.params = {"context": sorted(set(view.params.get("context", ["cell_type"]))
                                     | {"direction", "selection", "stats", "method"})}
    w.name += "+stats"
    return w


def pseudobulk(w: Workflow) -> Workflow:
    """Analysis-config edit: donors, not cells, are the replicates."""
    w = w.copy()
    w.node("de").config = {"method": "pseudobulk"}
    w.name += "+pseudobulk"
    return w


def de_evidence_branch(w: Workflow) -> Workflow:
    """Evidence-branch edit: the integrator also receives the DE summary directly."""
    w = w.copy()
    w.edges.append(Edge("de", "report", "de_summary", {"k": 15, "max_padj": 0.05}))
    w.name += "+de_branch"
    return w


PROMPT_EDIT = (
    "Be careful: each gene list given to the interpretation agent may mix genes that go up and "
    "genes that go down. Infer the direction of each program from the evidence, and only report "
    "effects that are likely reproducible across donors."
)


def prompt_only(w: Workflow) -> Workflow:
    """Prompt-only edit (what a reflective prompt optimizer can change): integrator instructions."""
    w = w.copy()
    w.node("report").config = {**w.node("report").config, "instructions": PROMPT_EDIT}
    w.name += "+prompt"
    return w


def no_interpreter(w: Workflow) -> Workflow:
    """Structural edit: drop GeneAgent; the integrator reads the DE summary itself."""
    w = w.copy()
    w.nodes = [n for n in w.nodes if n.id != "interp"]
    w.edges = [e for e in w.edges if "interp" not in (e.src, e.dst)]
    if not any(e.src == "de" and e.dst == "report" for e in w.edges):
        w.edges.append(Edge("de", "report", "de_summary", {"k": 15, "max_padj": 0.05}))
    w.name += "+no_interp"
    return w


def donor_branch(w: Workflow) -> Workflow:
    """Evidence-branch edit: a donor-consistency analysis feeds the integrator directly."""
    w = w.copy()
    w.nodes.insert(len(w.nodes) - 1, Node("donor", "donor_check", {}))
    w.edges += [Edge("de", "donor", "passthrough", {"kinds": ["de_table"]}),
                Edge("donor", "report", "passthrough", {})]
    w.name += "+donor"
    return w


def compose(*edits: Callable[[Workflow], Workflow],
            base: Callable[[], Workflow] = naive_composition) -> Callable[[], Workflow]:
    """A named candidate: ``base()`` with ``edits`` applied in order."""
    def build() -> Workflow:
        w = base()
        for e in edits:
            w = e(w)
        return w
    return build


CANDIDATES: dict[str, Callable[[], Workflow]] = {
    "naive": naive_composition,
    "prompt": compose(prompt_only),
    "split": compose(split_handoff),
    "split+stats": compose(split_handoff, carry_stats),
    "pseudobulk": compose(pseudobulk),
    "pseudobulk+split+stats": compose(pseudobulk, split_handoff, carry_stats),
    "de_branch": compose(de_evidence_branch),
    "no_interp": compose(no_interpreter),
    "pseudobulk+no_interp": compose(pseudobulk, no_interpreter),
    "pseudobulk+split+stats+de_branch": compose(pseudobulk, split_handoff, carry_stats,
                                                de_evidence_branch),
    "pseudobulk+split+stats+donor": compose(pseudobulk, split_handoff, carry_stats, donor_branch),
    "pseudobulk+no_interp+donor": compose(pseudobulk, no_interpreter, donor_branch),
    "donor": compose(donor_branch),
}


def coder_single() -> Workflow:
    """Baseline: one coding agent with a Python kernel does the entire analysis."""
    return Workflow(name="coder_single", nodes=[Node("coder", "coder", {"max_steps": 30})],
                    edges=[Edge("task", "coder", "task")], output="coder")


def codex_single() -> Workflow:
    """Baseline: the Codex CLI (same backbone) does the entire analysis in a workspace."""
    return Workflow(name="codex_single",
                    nodes=[Node("codex", "codex", {"hermetic": 1, "login": "local"})],
                    edges=[Edge("task", "codex", "task")], output="codex")


#: Baselines are not edits of the naive workflow and are kept out of edit selection.
BASELINES: dict[str, Callable[[], Workflow]] = {"coder_single": coder_single,
                                                "codex_single": codex_single}


# ---------------------------------------------------------------------------
# Task shape 2: multi-site evidence integration
# ---------------------------------------------------------------------------


def naive_multisite() -> Workflow:
    """Each site analyses the question on its own data and reports in its own way; the lead
    integrates the two reports."""
    return Workflow(
        name="ms_naive",
        nodes=[Node("site_A", "site_analyst", {"site": "A"}),
               Node("site_B", "site_analyst", {"site": "B"}),
               Node("lead", "integrator", {})],
        edges=[Edge("task", "site_A", "site_request", {"template": "question"}),
               Edge("task", "site_B", "site_request", {"template": "question"}),
               Edge("site_A", "lead", "passthrough", {}),
               Edge("site_B", "lead", "passthrough", {})],
        output="lead",
    )


def ms_structured(w: Workflow) -> Workflow:
    """Request edit: both sites report comparable, structured findings (genes, effect, donors)."""
    w = w.copy()
    for e in w.edges:
        if e.handoff == "site_request":
            e.params = {**e.params, "template": "structured"}
    w.name += "+structured"
    return w


def ms_local(w: Workflow) -> Workflow:
    """Request edit (task decomposition): ask each site for its own effects, not the study
    question — replication is the lead's job."""
    w = w.copy()
    for e in w.edges:
        if e.handoff == "site_request":
            e.params = {**e.params, "scope": "local"}
    w.name += "+local"
    return w


def ms_report_weak(w: Workflow) -> Workflow:
    """Request edit: sites also report weak candidates (replication is judged jointly)."""
    w = w.copy()
    for e in w.edges:
        if e.handoff == "site_request":
            e.params = {**e.params, "report_weak": True}
    w.name += "+weak"
    return w


def ms_cross_verify(w: Workflow) -> Workflow:
    """Topology edit: each site tests the other site's findings on its own data."""
    w = w.copy()
    lead = w.output
    w.nodes[-1:-1] = [Node("verify_A", "site_analyst", {"site": "A"}),
                      Node("verify_B", "site_analyst", {"site": "B"})]
    w.edges += [Edge("site_B", "verify_A", "cross_verify", {}),
                Edge("site_A", "verify_B", "cross_verify", {}),
                Edge("verify_A", lead, "passthrough", {}),
                Edge("verify_B", lead, "passthrough", {})]
    w.name += "+xverify"
    return w


def ms_replication_branch(w: Workflow) -> Workflow:
    """Evidence-branch edit: each site runs a deterministic replication check on every reported
    finding; the per-site summaries go to the lead."""
    w = w.copy()
    lead = w.output
    w.nodes.insert(len(w.nodes) - 1, Node("repl", "replication_check", {}))
    w.edges += [Edge(s, "repl", "passthrough", {"kinds": ["site_report"]})
                for s in ("site_A", "site_B")]
    w.edges.append(Edge("repl", lead, "passthrough", {}))
    w.name += "+repl"
    return w


def _ms(*edits: Callable[[Workflow], Workflow]) -> Callable[[], Workflow]:
    return compose(*edits, base=naive_multisite)


MS_CANDIDATES: dict[str, Callable[[], Workflow]] = {
    "ms_naive": naive_multisite,
    "ms_structured": _ms(ms_structured),
    "ms_local": _ms(ms_local),
    "ms_local+structured": _ms(ms_local, ms_structured),
    "ms_local+structured+weak": _ms(ms_local, ms_structured, ms_report_weak),
    "ms_local+structured+repl": _ms(ms_local, ms_structured, ms_replication_branch),
    "ms_local+structured+xverify": _ms(ms_local, ms_structured, ms_cross_verify),
    "ms_local+structured+weak+repl": _ms(ms_local, ms_structured, ms_report_weak,
                                         ms_replication_branch),
    "ms_local+structured+weak+xverify": _ms(ms_local, ms_structured, ms_report_weak,
                                            ms_cross_verify),
}


def ms_orchestrator() -> Workflow:
    """Shape-2 baseline: one autonomous agent decides what to ask each site agent."""
    return Workflow(name="ms_orchestrator", nodes=[Node("orch", "orchestrator", {})],
                    edges=[Edge("task", "orch", "task")], output="orch")


def ms_codex_orchestrator() -> Workflow:
    """Shape-2 product-level baseline: the Codex CLI leads the study via a site-query tool."""
    return Workflow(name="ms_codex_orchestrator",
                    nodes=[Node("codex_lead", "codex_orchestrator", {"login": "local"})],
                    edges=[Edge("task", "codex_lead", "task")], output="codex_lead")


def ms_codex_native_orchestrator() -> Workflow:
    """Product-level shape-2 baseline: the user's own Codex (native config) leads the study."""
    return Workflow(name="ms_codex_native_orchestrator",
                    nodes=[Node("codex_lead", "codex_orchestrator",
                                {"model": "native", "login": "local"})],
                    edges=[Edge("task", "codex_lead", "task")], output="codex_lead")


MS_BASELINES: dict[str, Callable[[], Workflow]] = {
    "ms_orchestrator": ms_orchestrator,
    "ms_codex_orchestrator": ms_codex_orchestrator,
    "ms_codex_native_orchestrator": ms_codex_native_orchestrator,
}
