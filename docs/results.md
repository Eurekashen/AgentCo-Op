# AgentCo-Op v3 — results

All numbers are hidden-answer F1 per task (cell type × direction × program family), averaged
over tasks, with a 95% bootstrap CI; paired comparisons are per task (win/loss, Wilcoxon).
Backbone everywhere: `gpt-6-luna`. Costs are *nominal* USD per task (what the run would cost
without caches), including rejected edits and controls. Method: [method.md](method.md).
Raw logs of every decision and finding: `findings.md`.

Suites. Shape 1 (pooled data, 8 donors, DE → GeneAgent → integrator): `dev` (development),
`test` (validation; demoted after its truth was inspected), **`heldout`** (never inspected).
Shape 2 (data held at two sites of 4 donors, site agents → lead): `msdev` (development),
**`mstest`** (never inspected).

## 1. Feasibility (dev, 16 tasks) — the three questions of the plan

**Q1 — is there collaboration loss, and room to fix it?** Same components, interface edits only:

| workflow | F1 |
|---|---|
| naive composition (native interfaces) | 0.000 |
| every cell-level-DE variant (prompt, split, stats, DE branch, drop interpreter) | 0.00–0.03 |
| pseudobulk DE alone | 0.000 |
| pseudobulk + split + stats + donor-consistency evidence branch | 0.730 |
| pseudobulk + no interpreter + donor branch | **0.757** [0.61, 0.88] (15/0 vs naive) |
| single coding agent (baseline) | 0.661 |
| Codex CLI (baseline) | 0.739 |

The loss is multi-stage (replicate unit, direction, context) and dominated by missing
*reproducibility evidence* at integration: the integrator refuses to claim an effect "reproducible
across donors" without donor-level evidence, even with pseudobulk statistics.

**Q2 — can answer-free feedback recognize improvement?** Pairwise agreement with the sign of the
hidden ΔF1 (675 pairs with |ΔF1| ≥ 0.1):

| evaluator | accuracy | error when it decides |
|---|---|---|
| report-only LLM judge | **0.50** (chance) | 0.42 |
| evidence judge (K3-style: rubric first, scorepad, both orders, data checks) | 0.85 | 0.08 |
| deterministic evidence rule (same checks) | **0.95** | 0.03 |

**Q3 — is it more than resampling?** Best-of-4 re-draws of the naive workflow: F1 0.000 under
every selector. A structural loss cannot be resampled away.

## 2. Shape 2 development (msdev, 16 tasks)

| workflow | F1 |
|---|---|
| naive (forward the study question to each site) | 0.000 — sites refuse to report local findings |
| structured template only | 0.000 |
| local scope | 0.514 |
| local + structured + weak effects + replication check | **0.766** [0.63, 0.88] |
| in-house autonomous orchestrator (same site agents) | 0.423 |
| **Codex CLI as study lead** (same site agents via a query tool) | **0.434** (P 0.73, R 0.32) |
| optimizer (final code, no answers) | 0.697; +select vs Codex lead 0.720 |

## 3. Held-out evaluation

### 3a. Shape 1 (`heldout`)

| method | F1 [95% CI] | $/task |
|---|---|---|
| naive | 0.000 | 0.044 |
| best-of-N re-draws | 0.000 | 0.764 |
| reflective prompt (GEPA-style) | 0.389 [0.19, 0.60] | 1.006 |
| random-location edits (same space) | 0.522 [0.32, 0.73] | 1.032 |
| free-form LLM editor, evidence acceptance (6 it.) | 0.683 [0.51, 0.84] | 0.267 |
| free-form LLM editor, report-judge acceptance (6 it.) | 0.719 [0.57, 0.86] | 0.340 |
| best fixed dev edit | 0.696 / 0.759 | ~0.01 |
| coder_single | 0.641 [0.44, 0.82] | 0.007 |
| **Codex CLI** | **0.787** [0.70, 0.87] | 0.014 |
| ours (frozen) | *pending* | |
| ours + answer-free select vs Codex | *pending* | |

### 3b. Shape 2 (`mstest`)

*pending — fixed references complete; optimizer runs in progress.*

## 4. What the evidence supports, and what it does not

*to be written from the final held-out numbers.*
