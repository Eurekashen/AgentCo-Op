# AgentCo-Op v3 — method

> When independently developed scientific agents are composed and there is no reference answer,
> only partial and imperfect scientific feedback, how do we find where their collaboration loses
> information, obtain evidence that distinguishes between explanations, and turn a one-off fix
> into an executable, verified change to how they collaborate?

This document is the design contract for v3. Results live in `docs/results.md`; the research plan
it implements is `review.md`.

## 1. Setting and information boundary

A task is a question `q`, data `D`, and a library of external agents with native interfaces. A
workflow is `W = (G, {A_v}, {h_e}, {p_v})`: execution graph, agent per node, **executable handoff
policy per edge**, config per node (`agentcoop/workflow.py`). A handoff decides what of the
upstream artifacts the downstream agent receives; it is the object whose failures we study.

| Information | Optimizer may use it? | Role |
|---|---|---|
| task data, run artifacts, public references (GO BP, Reactome), local checks | yes | evidence for proposing and accepting edits |
| judge preferences over evidence packets | yes, as a noisy signal | tie-breaking between evidenced candidates |
| hidden answer / grader | **no** — enforced by a test (`tests/test_core.py`) | independent final evaluation only |

If two candidates are indistinguishable on all observable evidence, the optimizer keeps the
incumbent. It never manufactures certainty by repeated judging.

## 2. Tasks with a hidden, exact answer

`agentcoop/tasks/planted.py`: real PBMCs (Kang 2018, control cells, 8 donors), split within donor
× cell type into control/treated; programs planted by binomial thinning (seqgendiff). Programs are
*core* gene sets (Hallmark ∩ family-defining GO BP terms), planted only where ≥8 core genes are
expressed, so every answer is recognizable in principle. Decoys: strong effects in a single donor.
The grader (`agentcoop/eval/grader.py`) scores (cell type, direction, program family) claims;
family is assigned from the program *name* only.

## 3. Answer-free evidence (`agentcoop/evidence/checks.py`)

* **claim support** — per-donor pseudobulk log fold change of the claim's genes: supported (≥7/8
  donors in the claimed direction and ≥60% of genes agree), contradicted, unassessed, ungrounded.
  `unassessed` is never counted as support.
* **uncovered signal** — donor-consistent genes (≥7/8 donors, |median lfc| ≥ 0.4) not covered by
  any claim, reported only if they form a *coherent* program (GO BP enrichment p ≤ 1e-4, overlap
  ≥ 4). Thresholds were set on dev with empty claims and then frozen: 28/38 programs flagged, 5
  false flags, 10 misses — informative, not an oracle.

## 4. Localization by signal tracing (`agentcoop/optimize/trace.py`)

Evidence units found in the data are used as probes through artifact lineage:

* an uncovered signal is followed *forward*: significant in the analysis? handed to the
  interpreter? with its direction? not diluted by noise? passed on with direction and method?
  claimed? The first stage that loses it names a location (node or edge) and a symptom;
* a weak claim is followed *backward* to the analysis and handoffs that produced its genes.

A finding is a hypothesis about where intervening could help, not a claim of unique root cause.

## 5. Edits, replay, control, acceptance (`agentcoop/optimize/{edits,loop}.py`)

Each component declares its local edit space (DE method; gene-list handoff selection; view
context fields; evidence branch; removing the interpreter; integrator instructions). The
optimizer proposes only edits at implicated locations; a frozen LLM ranks the menu (and writes the
text of a prompt edit). Each edit is executed with **controlled replay** — prefix reused from the
node cache, suffix re-executed with the incumbent's sample index — and compared with a **no-edit
control**: the same suffix re-drawn with a new sample index.

Acceptance, in order:

1. report evidence beats both incumbent and control (`evidence_rule`: supported − 2·contradicted −
   uncovered programs), and the evidence judge does not prefer the incumbent in both orders; or
2. **signal propagation**: the report does not move, but the incumbent's uncovered probes now
   travel further through the workflow than under both incumbent and control.

(2) solves the credit-assignment problem of sequential losses: when the replicate unit, the
direction and the context are all lost in sequence, each single fix is necessary but not
sufficient, and a final-output criterion rejects every one of them.

All executions — accepted, rejected, controls — are charged to a nominal budget.

## 6. Judges (`agentcoop/evidence/judge.py`)

Adapted from Kimi K3's agentic GRM, training-free: rubric generated from the question before any
candidate is seen; per-criterion scorepad citing evidence; verdict by counting criteria won (no
weights); `unassessed` criteria count for neither side; both presentation orders, flip → tie.
A report-only judge is the control condition.

## 7. Comparisons

Method level (same backbone `gpt-6-luna`, same feedback, same budget, same acceptance):
reflective prompt optimization (GEPA-style), free-form LLM workflow editing (AFlow/ADAS-style),
random-location edits from the same space, best-of-N re-draws. Product level: a single coding
agent with a Python kernel, and the Codex CLI in a hermetic workspace, given the same data and
public resources. Costs include optimization, controls and rejected edits.
