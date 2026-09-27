# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is, and where it is going

Research artifact for the AgentCo-Op paper (arXiv 2605.20425, CMU, Ma lab). The paper as published
(**v1**, "Retrieval-Based Synthesis of Interoperable Multi-Agent Workflows") was criticised by
reviewers on five points. The `revision` branch is a **ground-up rebuild (v2)** meant to answer
them, and the working tree contains v2 only:

| Reviewer objection to v1 | What v1 actually did | v2 mechanism |
|---|---|---|
| Topology/roles are an LLM's imagination | LLM-proposed L0–L7 topology ladder, hand-weighted score, binding by domain-tag overlap | `ir/evidence.py` admissibility + `compile/grammar.py` applicability conditions + probe-computed certification |
| Repair is unclear and LLM-decided | gate trigger → canned action (`test_failure → retry_node`) | detect → localize (lineage slice) → diagnose (entropy) → admissible patches → shadow-validate → commit/rollback |
| No optimization, only "make it run" | completion counted as success | Pareto `UtilityVector`, six-level `CheckLevel` stack, `UNAVAILABLE ≠ PASS` |
| Why not just Codex/Claude Code? | platform features presented as contributions | `components/coding_agent.py` is a *node*; four baseline arms in `bench/baselines.py` |
| Benchmarks are old and show no multi-agent value | HotpotQA/DROP/HumanEval/MBPP/GSM8K/MATH; ablation deltas < 1 point | `bench/` with three regimes incl. `multi_harmful`, injected faults with pre-written blame |

v2 is a compiler for heterogeneous agent workflows built on one rule: **every structural decision
must rest on evidence produced by execution, never asserted by a model.** Most of the surprising
code exists to enforce that rule against the easier alternative. The most important thing it can do
is *decline* to compose.

v1 survives only in git history at `94c3750` (= `main`), the reproducibility record for the
published numbers. None of those results carry over; do not cite them as v2 results, and do not
restore v1 code into `agentcoop/`.

**Current phase: method built, no experiments.** Per the plan, do not run experiments.
[docs/experiments/feasibility.md](docs/experiments/feasibility.md) concluded that no public
benchmark tests heterogeneous composition, so the critical path is a **self-built benchmark** of
tasks where composition beats one strong agent (with a single-agent-sufficiency screen), plus
RoadmapBench / Who&When / AFlow as secondary targets. It explicitly warns against letting
Codex/Claude-Code integration work crowd out benchmark construction.

## Branches

- `revision` — v2, the active line. Clean, 474 tests.
- `main` — v1 as published. Reference only. Its deleted self-critical docs (e.g. gates rarely fired,
  back-edge repair disabled, prompts tuned on eval splits) are readable via
  `git show 'c1140a5^:report.md'` (also `findings.md`, `implement.md`). In zsh, **quote `rev:path`**
  or the `:r` modifier mangles it.
- `origin/feature/ecps-preference-search` — +9 commits on `revision` (~13.6k lines, 723 tests).
  Adds `agentcoop/optimize/` (Evidence-Constrained Preference Search: run every admissible
  candidate on the same cases, objective-gate, then resolve the observed Pareto front with an
  order-swapped two-family LLM judge panel and criterion-wise Bradley–Terry; returns a stable
  winner or a typed unresolved front) and `agentcoop/ir/preference.py`. Library-only: no CLI, no
  bench hookup. Not merged. Known issue before merge: parameter-domain probe outcomes are folded
  into `card.empirical.probes`, and `EmpiricalRecord.passed(kind)` requires *all* probes of a kind
  to pass, so one rejected config value demotes the component's base certification.
- `origin/gh-pages` — project page. Entirely v1 content and numbers; will contradict v2 if shipped
  under the same URL.

## Commands

```bash
pip install -e .[dev]                 # REQUIRED after checkout: a stale v1 editable install
                                      # (agentcoop 0.1.0, no `agentcoop` script) may be present
pytest                                # 474 tests, ~2s, fully offline
pytest tests/unit/test_probe.py -v
pytest tests/unit/test_compile.py::TestBindRule::test_binds_a_certified_matching_component
pytest tests/integration -q           # cross-subsystem properties + CLI via typer CliRunner
pytest -k "silent or regime"
python -m agentcoop.cli --help        # works even without the console script
```

`asyncio_mode = "auto"` — async tests need no marker. No linter/formatter is configured; match
surrounding style (`from __future__ import annotations`, full annotations,
`ConfigDict(extra="forbid")` on pydantic models). Every core mechanism is deterministic and needs
no API key.

```bash
agentcoop bench list                       # tasks, regimes, injected faults, coverage defects
agentcoop probe   --task syn-multi         # certify by executing probes
agentcoop compile --task syn-multi         # candidates, evidence, Pareto, choice
agentcoop run     --task syn-silent-empty --out runs/demo   # exits 1: blocking silent signal
agentcoop diagnose runs/demo               # offline, from runs/demo/run.json
agentcoop repair  runs/demo                # rebuilds live synthetic components (shadow validation)
agentcoop bench run --system agentcoop --system best_single_component
agentcoop dossier lint path/to/dossier.yaml
agentcoop explain runs/demo/workflow.json
```

The CLI only drives the built-in `synthetic` suite via `--task`. `probe <library.yaml>` deliberately
refuses (a YAML declares contracts but not how to invoke them), and `run`/`repair` have no
dossier+library path — for real components, drive `ProbeRunner` / `Compiler` / `ExecutionEngine`
from Python with your own `AdapterRegistry`.

## Architecture

```
dossier ─▶ probe ─▶ compile ─▶ execute ─▶ diagnose ─▶ repair
```

Each stage is a package; [ir/](agentcoop/ir/) is the typed IR they all compile against and has no
I/O and no LLM dependency. [docs/architecture/v2_interfaces.md](docs/architecture/v2_interfaces.md)
is the interface contract the implementation was written against (its §11 CLI sketch and §"frozen
dependencies" list are aspirational and differ from the code);
[docs/architecture/v2_method.md](docs/architecture/v2_method.md) maps each objection to its
mechanism and lists what is deliberately *not* claimed. Read both before a design change.

Data flow worth knowing: `bench/task.py::build_environment` turns a `BenchTask` (pure data) into
adapters + library + type registry; `bench/harness.py::certify_environment` probes and returns
*new* cards; `Compiler.compile` enumerates candidates (always including single-component, with a
`DECLINE_STRUCTURE` record), justifies, runs 12 static checks, Pareto-selects (lexicographic by
default); `ExecutionEngine.run` validates every handoff at the edge and records `derived_from`;
`cli.RunRecord` (`run.json`) is self-contained so diagnosis never needs live components.

### The invariants that most edits will be tempted to break

- **`UNAVAILABLE` is never `PASS`.** A check that could not run does not satisfy anything
  ([ir/checks.py](agentcoop/ir/checks.py)). `certification_report` emits an entry for every probe
  kind including the ones never run, precisely so the gap is visible. An unmeasured utility
  objective is `unavailable`, never 0.0.
- **`Compatibility.UNDERSPECIFIED` is not compatible.** A producer that does not declare a facet
  the consumer requires is a compile-time failure ([ir/artifacts.py](agentcoop/ir/artifacts.py)).
- **`certification_level` is a computed property, never assigned.** `probe/certificate.py::derived_level`
  recomputes it from the same ladder the explanation prints; a test pins the two together — change
  both or neither.
- **An LLM assertion records as `EvidenceStrength.ASSERTED` and can never make a decision
  admissible** ([ir/evidence.py](agentcoop/ir/evidence.py)). Only `DERIVED`/`OBSERVED`/`STATISTICAL`
  are load-bearing. The optional `Diagnoser(llm=...)` may only reorder within top-k.
- **`Join` is unexpressible without a registered merge** ([merges.py](agentcoop/merges.py)).
  Voting is not a default; `majority_vote` requires ≥3 branches.
- **Repair must fix the symptom *and* regress nothing** ([repair/shadow.py](agentcoop/repair/shadow.py)).
  A component whose `BehaviorContract.shadow_safe` is False is never speculatively re-run.
  Above `RepairPolicy.max_diagnosis_entropy_bits` (1.5, in `ir/workflow.py`) the loop stops
  instead of patching.
- **`FaultSpec.admissible_patches` constrains repair.** A contract fault can never be "fixed" by
  `retry_node`; an evaluator fault never patches the generator ([ir/faults.py](agentcoop/ir/faults.py)).
- **Detectors must not trust the producer's self-report.** `detect_empty_output` recomputes
  emptiness from the payload; adapter notes are only a supplement. Same in `probe/runner.py::_smoke`.
- **Determinism**: no unseeded randomness, monotonic `Trace.step` not wall-clock, artifact content
  hashes over type+facets+payload only. Bench adapters default to `zero_clock`.
- **Wire before you write.** `evaluate/`, `memory/` and the `llm/human/converter/evaluator`
  adapters were built, found to have no caller, and deleted (`cc51066`; see v2_interfaces §1–3).
  Do not add a subsystem that nothing in the pipeline calls.

### Where the non-obvious coupling lives

- `probe/suite.py::_output_contexts` emits **one schema/smoke probe per declared output**. A
  multi-capability component (any generalist) emits one artifact per call, so a single probe would
  fail it for a reason unrelated to quality. How to elicit each output is declared in
  `card.binding["probe_contexts"]`.
- `execute/engine.py::_gather_inputs` takes `required` from the **subgoal**, not the card. A
  generalist declares the union of its capabilities' inputs; treating all as mandatory would fail it.
- `bench/harness.py` certifies **before** installing faults by default
  (`certify_before_injection=True`, CLI `--certify-first`). Probing an already-broken component
  makes the compiler refuse to bind it, so localization becomes unmeasurable.
- `bench/harness.py::blame_matches` lets edge `A->B` name `A` or `B`, and artifact id
  `producer::type::hash` name its producer. `blame_exact` reports the strict rate beside it — on
  the synthetic suite the gap is large (tolerant 1.00 vs exact 0.25), so always report both.
- `RepairLoop` detects non-shadow-safe escalation by substring `"not shadow-safe"` in
  `ShadowResult.notes`; keep that string if you touch `shadow.py`.

## Adding things

- **A topology production** — rule in [compile/grammar.py](agentcoop/compile/grammar.py) with an
  explicit applicability condition, a term in [ir/workflow.py](agentcoop/ir/workflow.py), a case in
  `lower()`. A production with no stated condition is the thing this rebuild removed.
- **A component type** — implement `ComponentAdapter` in [components/](agentcoop/components/) and
  add a case to `tests/unit/test_components.py`; keep `plan()` pure so it is testable without
  Docker/CLIs. Emit observed facets via `finalize_outputs(observers=...)`; static-config facets are
  applied but excluded from `observed_facets`. Container/coding-agent adapters must degrade to
  `ok=False` (ENVIRONMENT-shaped) when the binary is absent, never fall back to the host.
- **A fault class** — `FaultClass` *and* `FAULT_TAXONOMY` with `admissible_patches`, plus a
  `LIKELIHOOD` row in [diagnose/diagnose.py](agentcoop/diagnose/diagnose.py).
- **A benchmark task** — pure data in [bench/suites/](agentcoop/bench/suites/), with `regime` and,
  for every injected fault, `expected_blame`. `BenchSuite.coverage_defects()` reports gaps. To make
  it reachable from the CLI, register the suite in `cli.SUITES`.

## Known gaps (as of `cc51066`) — do not paper over these

- **No end-to-end committed repair exists.** In `bench run` every task has `repaired=false`; the
  headline `syn-silent-empty` demo stops at entropy 1.61 > 1.5 bits (an integration test asserts
  this); no test drives `RepairLoop` to `COMMITTED`. The commit path in `repair/loop.py` is only
  exercised piecewise.
- **The synthetic suite is a harness self-test, not a result.** Multi-regime tasks have no
  `ground_truth`, so "success" there means "outputs produced + hard constraints passed".
  `same_workflow_single_agent` scores 1.00 on `multi_necessary` (above agentcoop) — the regime
  label is not yet backed by a task one agent cannot solve, which is exactly what the self-built
  benchmark must fix.
- A run that fails early uses fewer components and is then scored as a wrong regime decision, so
  `regime-decision` partly re-counts execution failures.
- Shadow validation re-runs the full inputs (the "reduced input" in docs is not implemented by the
  CLI/bench harness).
- `httpx` is a declared dependency with no importer (its only user, `components/llm.py`, was
  deleted); the README still mentions an LLM adapter and the v1 title/figure.

## Gotchas

- `bench/behaviors.py` is a **decorator-populated registry**; a behaviour is invisible until its
  module is imported.
- Node ids are `<subgoal_id>__<component>`; artifact ids are `<producer>::<type>::<hash>`. Tests
  and the blame matcher depend on those shapes.
- `Compiler.compile` refuses a dossier with specification defects unless `allow_defects=True`. An
  empty compile is often this — check `result.dossier_defects` first.
- Node failures inside the engine become failed `NodeResult`s, not exceptions.
- `agentcoop run` exits 1 when a blocking signal fires even if hard constraints pass — that is the
  point ("completed" ≠ "good"), not a CLI bug.
- `data/` (~15 GB genomics + v1 benchmark dumps) and `external/` are gitignored v1 leftovers;
  nothing in v2 reads them. `runs/` is gitignored output.
