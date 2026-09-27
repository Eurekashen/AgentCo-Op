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


def compose(*edits: Callable[[Workflow], Workflow]) -> Callable[[], Workflow]:
    def build() -> Workflow:
        w = naive_composition()
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
}


def coder_single() -> Workflow:
    """Baseline: one coding agent with a Python kernel does the entire analysis."""
    return Workflow(name="coder_single", nodes=[Node("coder", "coder", {"max_steps": 30})],
                    edges=[Edge("task", "coder", "task")], output="coder")


#: Baselines are not edits of the naive workflow and are kept out of edit selection.
BASELINES: dict[str, Callable[[], Workflow]] = {"coder_single": coder_single}
