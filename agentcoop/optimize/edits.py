"""The executable edit space: each component and handoff declares its local neighbourhood.

An edit changes one thing at one location and is a pure function Workflow → Workflow. The
optimizer only ever proposes edits at a location the trace implicated, drawn from that
location's declared space — so "where" comes from evidence and "what" from a small, typed menu,
never from free-form rewriting of the workflow.

Families (for ablations and reporting):
``analysis_config`` · ``handoff`` · ``context`` · ``evidence_branch`` · ``structure`` · ``prompt``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agentcoop.workflow import Edge, Node, Workflow


@dataclass
class Edit:
    family: str
    location: str
    description: str
    apply: Callable[[Workflow], Workflow] = field(repr=False)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.family}@{self.location}:{sorted(self.params.items())}"


def _edge(w: Workflow, loc: str) -> Edge:
    src, dst = loc.split("->")
    return next(e for e in w.edges if e.src == src and e.dst == dst)


def _set_node_config(node_id: str, **changes) -> Callable[[Workflow], Workflow]:
    def f(w: Workflow) -> Workflow:
        w = w.copy()
        w.node(node_id).config = {**w.node(node_id).config, **changes}
        return w
    return f


def _set_edge_params(loc: str, **changes) -> Callable[[Workflow], Workflow]:
    def f(w: Workflow) -> Workflow:
        w = w.copy()
        e = _edge(w, loc)
        e.params = {**e.params, **changes}
        return w
    return f


# ---------------------------------------------------------------------------
# Declared local spaces
# ---------------------------------------------------------------------------

DE_METHODS = ["wilcoxon", "pseudobulk"]
GENELIST_GRID = {"split_direction": [True, False], "max_padj": [None, 0.05, 0.1],
                 "n": [15, 30, 50]}
VIEW_CONTEXT = ["direction", "method", "selection", "stats"]


def node_edits(w: Workflow, node_id: str) -> list[Edit]:
    node = w.node(node_id)
    out: list[Edit] = []
    if node.agent == "de_analyst":
        for m in DE_METHODS:
            if m != node.config.get("method", "wilcoxon"):
                out.append(Edit("analysis_config", node_id, f"set DE method to {m}",
                                _set_node_config(node_id, method=m), {"method": m}))
    if node.agent == "integrator":
        out += _site_lead_edits(w, node_id)
        analysis = next((n.id for n in w.nodes if n.agent == "de_analyst"), None)
        if analysis and not any(e.src == analysis and e.dst == node_id for e in w.edges):
            def add_branch(wf: Workflow, a=analysis, d=node_id) -> Workflow:
                wf = wf.copy()
                wf.edges.append(Edge(a, d, "de_summary", {"k": 15, "max_padj": 0.05}))
                return wf
            out.append(Edit("evidence_branch", node_id,
                            "also hand the DE summary (statistics, direction) to the integrator",
                            add_branch, {"branch": "de_summary"}))
        if analysis and not any(n.agent == "donor_check" for n in w.nodes):
            def add_donor(wf: Workflow, a=analysis, d=node_id) -> Workflow:
                wf = wf.copy()
                wf.nodes.insert(len(wf.nodes) - 1, Node("donor", "donor_check", {}))
                wf.edges += [Edge(a, "donor", "passthrough", {"kinds": ["de_table"]}),
                             Edge("donor", d, "passthrough", {})]
                return wf
            out.append(Edit("evidence_branch", node_id,
                            "add a donor-consistency analysis whose per-donor evidence goes to "
                            "the integrator", add_donor, {"branch": "donor_check"}))
        interp = [n.id for n in w.nodes if n.agent == "geneagent"]
        if interp and analysis:
            def drop(wf: Workflow, i=interp[0], a=analysis, d=node_id) -> Workflow:
                wf = wf.copy()
                wf.nodes = [n for n in wf.nodes if n.id != i]
                wf.edges = [e for e in wf.edges if i not in (e.src, e.dst)]
                if not any(e.src == a and e.dst == d for e in wf.edges):
                    wf.edges.append(Edge(a, d, "de_summary", {"k": 15, "max_padj": 0.05}))
                return wf
            out.append(Edit("structure", node_id, "remove the interpretation agent; the "
                            "integrator reads the DE summary itself", drop, {"remove": interp[0]}))
    return out


SITE_REQUEST_GRID = {"template": ["question", "structured"], "report_weak": [False, True],
                     "method": [None, "pseudobulk"]}


def _set_all_site_requests(**changes) -> Callable[[Workflow], Workflow]:
    """A request policy is shared by every site, so the edit applies to all request edges."""
    def f(w: Workflow) -> Workflow:
        w = w.copy()
        for e in w.edges:
            if e.handoff == "site_request":
                e.params = {**e.params, **changes}
        return w
    return f


def _site_lead_edits(w: Workflow, node_id: str) -> list[Edit]:
    sites = [n for n in w.nodes if n.agent == "site_analyst" and n.id.startswith("site_")]
    out: list[Edit] = []
    if not sites:
        return out
    if not any(n.id.startswith("verify_") for n in w.nodes):
        def xverify(wf: Workflow, d=node_id) -> Workflow:
            wf = wf.copy()
            ids = [n.id for n in wf.nodes if n.agent == "site_analyst" and n.id.startswith("site_")]
            new_nodes, new_edges = [], []
            for src in ids:
                for dst in ids:
                    if src == dst:
                        continue
                    site = wf.node(dst).config["site"]
                    vid = f"verify_{site}_from_{wf.node(src).config['site']}"
                    new_nodes.append(Node(vid, "site_analyst", {"site": site}))
                    new_edges += [Edge(src, vid, "cross_verify", {}), Edge(vid, d, "passthrough", {})]
            wf.nodes[-1:-1] = new_nodes
            wf.edges += new_edges
            return wf
        out.append(Edit("structure", node_id, "each site tests the other sites' findings on its "
                        "own donors (cross-site verification)", xverify, {"add": "cross_verify"}))
    if not any(n.agent == "replication_check" for n in w.nodes):
        def repl(wf: Workflow, d=node_id) -> Workflow:
            wf = wf.copy()
            ids = [n.id for n in wf.nodes if n.agent == "site_analyst" and n.id.startswith("site_")]
            wf.nodes.insert(len(wf.nodes) - 1, Node("repl", "replication_check", {}))
            wf.edges += [Edge(i, "repl", "passthrough", {"kinds": ["site_report"]}) for i in ids]
            wf.edges.append(Edge("repl", d, "passthrough", {}))
            return wf
        out.append(Edit("evidence_branch", node_id, "every site runs a replication check on "
                        "every reported finding; the per-site results go to the lead", repl,
                        {"branch": "replication_check"}))
    return out


def edge_edits(w: Workflow, loc: str) -> list[Edit]:
    e = _edge(w, loc)
    out: list[Edit] = []
    if e.handoff == "site_request":
        defaults = {"template": "question", "report_weak": False, "method": None}
        for key, values in SITE_REQUEST_GRID.items():
            cur = e.params.get(key, defaults[key])
            for v in values:
                if v != cur:
                    out.append(Edit("handoff", loc, f"site request policy: set {key}={v}",
                                    _set_all_site_requests(**{key: v}), {key: v}))
    if e.handoff == "de_to_genelists":
        for key, values in GENELIST_GRID.items():
            cur = e.params.get(key, {"split_direction": False, "max_padj": None, "n": 30}[key])
            for v in values:
                if v == cur:
                    continue
                changes = {key: v}
                if key == "split_direction" and v:
                    changes["min_genes"] = 5
                out.append(Edit("handoff", loc, f"gene-list handoff: set {key}={v}",
                                _set_edge_params(loc, **changes), changes))
    if e.handoff == "interpretation_view":
        ctx = list(e.params.get("context", ["cell_type"]))
        missing = [k for k in VIEW_CONTEXT if k not in ctx]
        if missing:
            out.append(Edit("context", loc, f"carry {', '.join(missing)} with each interpretation",
                            _set_edge_params(loc, context=ctx + missing),
                            {"add_context": missing}))
            for k in missing:
                out.append(Edit("context", loc, f"carry {k} with each interpretation",
                                _set_edge_params(loc, context=ctx + [k]), {"add_context": [k]}))
    return out


def local_edits(w: Workflow, location: str) -> list[Edit]:
    return edge_edits(w, location) if "->" in location else node_edits(w, location)


def prompt_edit(node_id: str, instructions: str) -> Edit:
    return Edit("prompt", node_id, "rewrite integrator instructions",
                _set_node_config(node_id, instructions=instructions),
                {"instructions": instructions[:80]})
