# AgentCo-Op v3 — results

Hidden-answer F1 per task (cell type × direction × program family), averaged over tasks; paired
comparisons are per task (win/loss, Wilcoxon signed-rank). Costs are nominal USD per task
(what the run would cost without caches), including rejected edits and no-edit controls.
Method: [method.md](method.md). Every decision, bug and negative result: `findings.md`.

Suites. Shape 1 — pooled data (8 donors; DE → GeneAgent → integrator): `dev` (development),
`test` (validation; demoted after its truth was inspected), **`heldout`** (never inspected).
Shape 2 — data held at two sites of 4 donors (site agents → lead): `msdev` (development),
**`mstest`** (never inspected before the method was frozen).

Unless stated otherwise the framework model and the external agents are `gpt-6-luna`; the grader
is pinned to `gpt-6-luna`. Method versions: v3 = tag `v3-final-eval`; **v3.1 = tag `v3.1-eval`**
(competitive claim support, 16-execution budget, lookahead off).

## 1. Collaboration loss is real, structural, and model-dependent in form

Same components, only the collaboration interface changes (dev, 16 tasks per shape):

| shape | naive composition | best interface edits | where the loss is |
|---|---|---|---|
| 1 pooled | 0.000 | 0.757 (pseudobulk + no interpreter + donor-evidence branch) | replicate unit, direction, context, *reproducibility evidence* at integration |
| 2 sites | 0.000 | 0.766 (local scope + structured + weak effects + replication check) | task decomposition: sites refuse to report local findings when handed the study question |

Resampling cannot fix it: best-of-4 re-draws of the naive workflow stay at 0.000 under every
selector. The *form* of the loss depends on the framework model: gpt-6-luna refuses to claim
without reproducibility evidence (recall loss); gpt-4.1-mini guesses directions from mixed lists
(precision loss). Interface edits repair both.

## 2. Answer-free evaluation must be grounded in the data

Pairwise agreement with the sign of the hidden ΔF1 over pairs of edit candidates (dev; 60-pair
fixed subsample for the second model):

| evaluator | gpt-6-luna (619 pairs) | gpt-6-luna (60) | gpt-4.1-mini (60) |
|---|---|---|---|
| report-only LLM judge | **0.53** | 0.52 | 0.67 |
| evidence judge (K3-style: rubric first, scorepad, both orders) | 0.86 | 0.93 | 0.80 |
| deterministic evidence rule (no model) | **0.96** | 0.95 | 0.95 |

(Corrected in the 2026-09-29 code review: the first version of this table also counted 56 re-draw
pairs, giving 0.50 / 0.85 / 0.95 over 675 pairs.)

## 3. Pooled data: a strong single agent suffices — if its model is strong (held-out `heldout`, 16 tasks, v3)

| method | F1 [95% CI] | $/task |
|---|---|---|
| naive / best-of-N re-draws | 0.000 / 0.000 | 0.04 / 0.76 |
| reflective prompt optimization (GEPA-style) | 0.389 [0.19, 0.60] | 1.01 |
| random-location edits (same space) | 0.522 [0.32, 0.73] | 1.03 |
| ours (localized evidence-driven search) | 0.678 [0.53, 0.82] | 0.67 |
| free-form LLM editor, evidence acceptance | 0.699 [0.53, 0.85] | 0.35 |
| free-form LLM editor, report-judge acceptance | 0.746 [0.59, 0.88] | 0.48 |
| best fixed dev interface | 0.759 [0.61, 0.89] | 0.01 |
| **Codex CLI alone** | **0.787** [0.70, 0.87] | 0.01 |

When one agent can see all the data, composing independently developed agents does not pay: the
right decision is to decline composition. Our answer-free select step (optimized workflow vs the
single agent, evidence rule only) reaches 0.731; it recognizes the single agent only 2/16 times.

## 4. Siloed data: evidence-optimized collaboration beats autonomous coordination

Dev (`msdev`, 8 tasks, v3.1): ours 0.732 · free-form editor 0.760 · hand-designed interface 0.755 ·
Codex lead (3 repeats, 4 tasks) 0.598 · in-house lead 0.338 · naive 0.

Held-out (`mstest`, 16 tasks, v3.1 frozen; Codex = the user's locally logged-in CLI, mean of 3
repeats per task because it cannot be seeded):

| method | F1 [95% CI] | W/L vs Codex lead | p | $/task |
|---|---|---|---|---|
| naive composition | 0.000 | 0/14 | 0.001 | 0.01 |
| Codex CLI as study lead | 0.491 [0.35, 0.63] | – | – | 0.03 |
| in-house autonomous lead | 0.526 [0.38, 0.67] | 7/3 | 0.38 | 0.03 |
| best fixed dev interface | 0.714 [0.64, 0.78] | 12/2 | 0.019 | 0.03 |
| **ours** (localized evidence-driven search) | **0.736** [0.61, 0.84] | **12/1** | **0.006** | 0.26 |
| free-form LLM editor (evidence acceptance) | 0.748 [0.65, 0.84] | 12/2 | 0.010 | 0.49 |

(W/L corrected in the code review: equal scores against a mean of equal Codex repeats had been
counted as losses by float noise.) Ours vs the free-form editor: 5/4, p 1.0 — a tie at half the
cost. Ours vs the in-house lead: 9/1, p 0.023. Codex repeats vary widely per task (e.g. 403: 0.8 / 0.5 / 0.0); an earlier single
Codex run (s0) scored 0.602, and the 4-run mean is 0.519 — the gap does not depend on which
repeats are counted. Failure case: on mstest-404 ours scores 0 — the lead will not claim anything
without cross-site replication, and no single request edit or pair of them at the request edge
improves the evidence; the fix needs a request edit and lead-side replication together.

**Decision quality** (every attempt graded after the fact; nothing flows back):

| method | tries | accepted | helpful | harmful | rejected but would have helped |
|---|---|---|---|---|---|
| ours | 120 | 46 | 21 | **1 (2%)** | 9 of 74 |
| free-form editor | 171 | 24 | 19 | 3 (12.5%) | 24 of 121 (+26 outranked by a sibling) |

The free-form editor proposes more helpful candidates, but its acceptances are less safe;
localized, typed edits change the evidence in small attributable steps. Most missed improvements
share one cause: the incumbent had more data-supported claims, one of which was wrong — support
in the data does not certify a program's name.

**Mechanism ablations** (held-out `mstest`, same budget and acceptance):

| method | F1 | vs ours (W/L) | harmful / accepted |
|---|---|---|---|
| prompt-only (GEPA-style reflective prompt optimization), 16 tasks | 0.000 | 0/15 (p 0.001) | – |
| random locations, same typed edit space, 16 tasks | 0.765 | 3/2 (p 0.38) | 2 / 35 |
| ours + rule library distilled from dev, 8 tasks | 0.655 | identical on 8/8 | 1 / 20 |
| ours, 16 tasks | 0.736 | – | 1 / 46 |

Prompt edits cannot repair a request-policy loss. In this shape the typed edit space is small, so
the order in which locations are tried does not matter under a 16-execution budget — random
locations are as accurate as localization (with more harmful acceptances); the dev-distilled rule
library reorders tries but never changes an outcome. (In shape 1, with a larger space,
localization beat random locations by +0.16.)

**Fresh held-out `mstest2`** (16 new tasks; v3.3 frozen and pre-registered before any run;
Codex = the user's local login, 3 repeats per task):

| method | F1 | W/L vs Codex lead (p) |
|---|---|---|
| naive | 0.000 | – |
| in-house autonomous lead | 0.458 | – |
| Codex CLI as lead (gpt-6-luna) | 0.522 | – |
| Codex CLI as lead, the user's own config (gpt-6-sol) | 0.542 | – |
| best fixed dev interface | 0.664 | 13/1 (0.005) |
| free-form editor v2 | 0.665 | 12/2 (0.020) |
| ours | 0.719 | 12/0 (0.0005) |
| ours + escalation to free-form editing (v3.3) | **0.731** | **12/0 (0.0005)** |

Ours+escalation vs the native Codex lead +0.19 (9/1, p 0.053); vs the free-form editor +0.07
(6/2, n.s.; precision 0.83 vs 0.67, 0 vs 2 decoy claims). The never-inspected second half gives
the same ordering (ours 0.87 · editor 0.80 · fixed 0.78 · Codex 0.66 · native Codex 0.62).
Escalation changed the outcome on 1 of 16 tasks: it adds nothing measurable.

**Decision quality pooled over every logged attempt** (all suites, three framework models):
ours accepted 156 edits on the evidence route — 80% raised hidden F1, 9% lowered it; the
free-form editor and the other baselines 75 — 73% and 17%.

## 5. Model-agnosticism (framework model swapped, external agents fixed at gpt-6-luna)

Held-out `mstest`, first 8 tasks, framework model = gpt-4.1-mini (non-reasoning):

| method | gpt-4.1-mini framework | gpt-6-luna framework (same tasks) |
|---|---|---|
| naive | 0.000 | 0.000 |
| same-model autonomous lead | 0.392 | 0.476 |
| best fixed dev interface | 0.715 | 0.683 |
| **ours** | **0.842** | 0.655 |
| free-form editor (v2: DAG documented + repair) | 0.659 | – |
| free-form editor (v1, as evaluated above) | 0.29 (n=4; 6/8 proposals invalid) | 0.733 |

With the weaker framework model ours beats free-form editing 5/0 (+0.18, p 0.06) and the
same-model lead 7/1 (p 0.04). The typed, evidence-validated search does not depend on the model
writing valid workflow JSON; free-form editing does (v1 collapsed on cyclic workflows — 6–11% of
gpt-6-luna proposals were invalid too, which motivated v2). Localization's value in this shape is
robustness to the framework model, not peak accuracy with a strong one.

Pooled data (shape 1), held-out `heldout`, first 8 tasks, framework = gpt-4.1-mini:
fixed interface 0.789 (luna 0.790) · ours 0.754 (luna 0.738) · free-form editor v2 0.579 (luna,
v1: 0.853) · naive 0.226 · same-model single agent **0.062** (luna 0.742; Codex 0.797). With a
weaker framework model the composition with a good interface keeps its accuracy and the lone
agent collapses (failed DE fits, confident wrong claims) — so "pooled data → use one strong agent"
holds only when that agent's model is strong. Third model (gpt-5-mini, reasoning, different API
profile), shape 2, 4 tasks: ours 0.864 · fixed 0.814 · editor v2 0.804 · same-model lead 0.558.

Dev (2 tasks per shape, 16 executions): shape 1 under gpt-4.1-mini is limited by program
*naming* — the optimizer removes contradicted claims (2 → 0) but the model names programs wrongly,
and no data check can see a name. External agents themselves need a capable model: gpt-4o-mini
fabricated a placeholder report; gpt-4.1-mini could not drive pydeseq2 within 30 steps.

## 6. What the evidence supports, and what it does not

Supported:
- Composition of independently developed agents incurs a structural collaboration loss that
  resampling cannot fix and interface edits can (both shapes, three framework models).
- Answer-free evaluation is reliable only when grounded in the data (evidence rule 0.96 vs
  report-only judge 0.53 over edit pairs).
- When data cannot be pooled, optimizing the explicit collaboration interface with data-grounded
  acceptance beats letting a strong agent coordinate autonomously — Codex (luna and the user's
  gpt-6-sol config) and our own lead — on two held-out suites (mstest 12/1, mstest2 12/0).
- The typed, evidence-validated search is robust to the framework model: with a weaker
  non-reasoning model it keeps its accuracy in both shapes, while free-form editing degrades and a
  single agent collapses. Its accepted edits are safer (9% harmful vs 17%, pooled).
- When data can be pooled and the single agent's model is strong, that agent wins; composition
  should then be declined.

Not supported (honest negatives):
- With a strong framework model the localized search is **not** more accurate than free-form LLM
  editing given the same interface library and evidence feedback (ties on dev and held-out); it
  is cheaper and every edit is attributable to a traced symptom.
- In shape 2, choosing *where* to edit does not matter: random locations in the same typed space
  do as well (0.765 vs 0.736, n=16) — though that ablation used one fixed random order (fixed in
  v3.4). Prompt-only optimization recovers nothing (0/16).
- Escalating to free-form editing when the typed space is exhausted (v3.3) changed 1 of 16 fresh
  held-out tasks. A dev-distilled rule library never changed an outcome. One-step lookahead: no
  benefit. Signal-propagation acceptance mattered only without a direct evidence route (dev-000).
- Program naming is invisible to answer-free checks; most harmful acceptances are a data-supported
  claim with a wrong program name, or an edit that drops an undecided but correct claim.
- The code review found typed-search bugs (controls, locks, a one-execution overrun) fixed in
  v3.4; their measured effect on stored results is small (one acceptance, +0.13 on one task), but
  v3.4 itself has not yet been evaluated.
- Public benchmarks (BixBench, scBench-Long) were not run: they need R/notebook environments or
  gated data not available here, and they evaluate single-agent analyses rather than collaboration
  between independent agents.
