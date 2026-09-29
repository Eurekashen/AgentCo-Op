"""Workflows as data, executed with artifact lineage and node-level replay.

A workflow W = (G, {A_v}, {h_e}, {p_v}): nodes bind an agent A_v with config p_v; edges carry an
executable *handoff policy* h_e that decides what of the upstream artifacts reaches the
downstream agent. Handoffs are first-class because the research hypothesis is that collaboration
loss often lives there.

Execution records every artifact with its producer and ``derived_from`` ids, so a claim in the
final report can be traced back claim → artifact → producer → input.

Replay: a node's outputs are cached under (agent, config, input contents, sample). Editing an
edge or a downstream node therefore re-executes only the affected suffix; the prefix is reused
exactly. ``samples`` lets a caller re-draw chosen nodes (a new sample index) while pinning the
rest — the basis of the no-edit rerun control.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from agentcoop import llm

ROOT = Path(__file__).resolve().parent.parent
NODE_CACHE = ROOT / "runs" / "node_cache"


def content_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class Artifact:
    kind: str
    producer: str
    data: Any
    derived_from: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            h = content_hash({"kind": self.kind, "data": self.data, "meta": self.meta})[:10]
            self.id = f"{self.producer}::{self.kind}::{h}"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "producer": self.producer, "data": self.data,
                "derived_from": self.derived_from, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Artifact":
        return cls(kind=d["kind"], producer=d["producer"], data=d["data"],
                   derived_from=list(d.get("derived_from", [])), meta=dict(d.get("meta", {})),
                   id=d["id"])


@dataclass
class Node:
    id: str
    agent: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    handoff: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    nodes: list[Node]
    edges: list[Edge]
    output: str
    name: str = ""

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def incoming(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.dst == node_id]

    def order(self) -> list[str]:
        """Topological order; ties broken by declaration order (deterministic)."""
        ids = [n.id for n in self.nodes]
        indeg = {i: 0 for i in ids}
        for e in self.edges:
            if e.src != "task":
                indeg[e.dst] += 1
        out, ready = [], [i for i in ids if indeg[i] == 0]
        while ready:
            cur = ready.pop(0)
            out.append(cur)
            for e in self.edges:
                if e.src == cur:
                    indeg[e.dst] -= 1
                    if indeg[e.dst] == 0:
                        ready.append(e.dst)
        if len(out) != len(ids):
            raise ValueError("workflow has a cycle")
        return out

    def descendants(self, node_id: str) -> set[str]:
        out, frontier = set(), [node_id]
        while frontier:
            cur = frontier.pop()
            for e in self.edges:
                if e.src == cur and e.dst not in out:
                    out.add(e.dst)
                    frontier.append(e.dst)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "output": self.output,
                "nodes": [n.__dict__ for n in self.nodes],
                "edges": [e.__dict__ for e in self.edges]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Workflow":
        return cls(nodes=[Node(**n) for n in d["nodes"]], edges=[Edge(**e) for e in d["edges"]],
                   output=d["output"], name=d.get("name", ""))

    def copy(self) -> "Workflow":
        return Workflow.from_dict(copy.deepcopy(self.to_dict()))

    def fingerprint(self) -> str:
        return content_hash(self.to_dict() | {"name": ""})[:12]


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

#: agent(inputs, config, ctx) -> list[Artifact]
AgentFn = Callable[[list[Artifact], dict[str, Any], "RunContext"], list[Artifact]]
#: handoff(upstream artifacts, params, ctx, dst node id) -> list[Artifact]
HandoffFn = Callable[[list[Artifact], dict[str, Any], "RunContext", str], list[Artifact]]

AGENTS: dict[str, AgentFn] = {}
HANDOFFS: dict[str, HandoffFn] = {}
#: Agents that make no model calls: their output does not depend on the sample index, so a
#: re-draw reuses them instead of recomputing identical results.
DETERMINISTIC: set[str] = {"de_analyst", "donor_check", "replication_check"}
#: Given external components: they run on ``llm.external_model()``, not the framework backbone.
EXTERNAL: set[str] = {"geneagent", "site_analyst"}
#: Framework-side agents that call external agents internally (both models matter).
ORCHESTRATING: set[str] = {"orchestrator", "codex_orchestrator"}


def agent(name: str):
    def register(fn: AgentFn) -> AgentFn:
        AGENTS[name] = fn
        return fn
    return register


def handoff(name: str):
    def register(fn: HandoffFn) -> HandoffFn:
        HANDOFFS[name] = fn
        return fn
    return register


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """The public view of a task. Deliberately has no path to the hidden answer."""

    task_id: str
    question: str
    data_path: Path
    info: dict[str, Any]

    @classmethod
    def load(cls, task_dir: Path) -> "Task":
        public = Path(task_dir) / "public"
        info = json.loads((public / "task.json").read_text())
        return cls(task_id=info["task_id"], question=info["question"],
                   data_path=public / info["data"], info=info)


@dataclass
class RunContext:
    task: Task
    sample: int = 0
    node_id: str = ""


@dataclass
class Run:
    workflow: Workflow
    task_id: str
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    node_outputs: dict[str, list[str]] = field(default_factory=dict)
    node_inputs: dict[str, list[str]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    samples: dict[str, int] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0

    def output(self) -> Optional[Artifact]:
        ids = self.node_outputs.get(self.workflow.output, [])
        return self.artifacts[ids[0]] if ids else None

    def of(self, node_id: str) -> list[Artifact]:
        return [self.artifacts[i] for i in self.node_outputs.get(node_id, [])]

    def lineage(self, artifact_id: str) -> list[str]:
        """Every artifact id upstream of ``artifact_id`` (inclusive), breadth first."""
        seen, frontier = [], [artifact_id]
        while frontier:
            cur = frontier.pop(0)
            if cur in seen or cur not in self.artifacts:
                continue
            seen.append(cur)
            frontier.extend(self.artifacts[cur].derived_from)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {"workflow": self.workflow.to_dict(), "task_id": self.task_id,
                "artifacts": {k: a.to_dict() for k, a in self.artifacts.items()},
                "node_outputs": self.node_outputs, "node_inputs": self.node_inputs,
                "errors": self.errors, "samples": self.samples, "cost": self.cost,
                "seconds": self.seconds}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Run":
        return cls(workflow=Workflow.from_dict(d["workflow"]), task_id=d["task_id"],
                   artifacts={k: Artifact.from_dict(a) for k, a in d["artifacts"].items()},
                   node_outputs=d["node_outputs"], node_inputs=d["node_inputs"],
                   errors=d["errors"], samples=d.get("samples", {}), cost=d.get("cost", {}),
                   seconds=d.get("seconds", 0.0))


def _cache_path(key: str) -> Path:
    return NODE_CACHE / key[:2] / f"{key}.json"


#: Node keys for this model omit the model field (the form every early run was stored under).
LEGACY_KEY_MODEL = "gpt-6-luna"


def _publish(path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """Store a node-cache entry atomically; the first writer wins and everyone reads its value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(entry))
    try:
        os.link(tmp, path)
    except FileExistsError:
        entry = json.loads(path.read_text())
    finally:
        tmp.unlink()
    return entry


def execute(workflow: Workflow, task: Task, *, sample: int = 0,
            samples: Optional[dict[str, int]] = None) -> Run:
    """Run ``workflow`` on ``task``. ``samples`` overrides the sample index per node."""
    # Import registers the built-in agents and handoffs.
    import agentcoop.agents  # noqa: F401
    import agentcoop.handoffs  # noqa: F401

    t0 = time.time()
    samples = dict(samples or {})
    run = Run(workflow=workflow, task_id=task.task_id)
    task_artifact = Artifact(kind="task", producer="task",
                             data={"question": task.question, "data_path": str(task.data_path),
                                   "info": task.info})
    run.artifacts[task_artifact.id] = task_artifact
    run.node_outputs["task"] = [task_artifact.id]
    #: nominal = what the run would cost without any cache; actual = what was paid now.
    nominal = actual = 0.0

    for node_id in workflow.order():
        node = workflow.node(node_id)
        s = samples.get(node_id, sample)
        run.samples[node_id] = s
        ctx = RunContext(task=task, sample=s, node_id=node_id)
        inputs: list[Artifact] = []
        failed_upstream = False
        for edge in workflow.incoming(node_id):
            upstream = run.of(edge.src)
            if edge.src in run.errors:
                failed_upstream = True
            fn = HANDOFFS[edge.handoff]
            inputs.extend(fn(upstream, edge.params, ctx, node_id))
        if failed_upstream and not inputs:
            run.errors[node_id] = "upstream failed"
            run.node_outputs[node_id] = []
            continue
        for a in inputs:
            run.artifacts[a.id] = a
        run.node_inputs[node_id] = [a.id for a in inputs]

        key_fields = {"agent": node.agent, "config": node.config,
                      "sample": 0 if node.agent in DETERMINISTIC else s,
                      "task": task.task_id,
                      "inputs": [{"kind": a.kind, "data": a.data, "meta": a.meta}
                                 for a in inputs]}
        role_model = llm.external_model() if node.agent in EXTERNAL else llm.backbone()
        models = {role_model}
        if node.agent in ORCHESTRATING:
            models.add(llm.external_model())  # it runs external agents internally
        if node.agent not in DETERMINISTIC and models != {LEGACY_KEY_MODEL}:
            # A model-dependent node's output depends on the models it runs. Keys for the
            # legacy model keep their original form so earlier runs stay replayable.
            key_fields["model"] = sorted(models)
        path = _cache_path(content_hash(key_fields))
        if path.exists():
            cached = json.loads(path.read_text())
            nominal += cached.get("nominal_usd", 0.0)
        else:
            with llm.tag(f"node={node_id}#{path.stem[:8]}.{time.time_ns()}"):
                exec_tag = llm.current_tag()
                try:
                    produced = AGENTS[node.agent](inputs, node.config, ctx)
                except llm.BudgetExceeded:
                    raise
                except Exception as exc:  # a failed node is a recorded result, not a crash
                    run.errors[node_id] = f"{type(exc).__name__}: {exc}"
                    produced = None
                finally:
                    spent = llm.spent(exec_tag)  # failed attempts are charged too
                    nominal += spent["nominal_usd"]
                    actual += spent["actual_usd"]
            if produced is None:
                run.node_outputs[node_id] = []
                continue
            cached = _publish(path, {"outputs": [a.to_dict() for a in produced],
                                     "nominal_usd": spent["nominal_usd"]})
        outputs = [Artifact.from_dict(d) for d in cached["outputs"]]
        for a in outputs:
            run.artifacts[a.id] = a
        run.node_outputs[node_id] = [a.id for a in outputs]

    run.cost = {"nominal_usd": nominal, "actual_usd": actual}
    run.seconds = time.time() - t0
    return run
