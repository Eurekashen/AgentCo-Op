# AgentCo-Op v3 — results

All main-table arms run on **one backbone, `gpt-6-luna`** (framework agents, external agents,
Codex `-m gpt-6-luna` on the local login, optimizer/reflection models); the grader is a separate
pinned instrument. Other models appear only in the appendix (§8). Every baseline is defined in
[baselines.md](baselines.md). Statistics unit = task (BixBench: question, CIs clustered by
capsule). Costs are nominal USD per task, including every execution an optimizer spends.
Method: [method.md](method.md). Every decision, bug and negative result: `findings.md`.

## 1. Main result — collaboration across data silos (fresh held-out, pre-registered, n = 48)

Each task: scRNA-seq data split across two sites (4 donors each) held by independent site
agents; only the site agents read their data. Hidden answer = planted programs + a decoy;
F1 over (cell type × direction × program family). `mstest2` (16 tasks, pre-registered
2026-09-29) and `mstest3` (32 tasks, pre-registered 2026-10-02, tag `mstest3-prereg`) were
built after the method was frozen (v3.4.1) and never inspected before these runs.

| method (all gpt-6-luna) | F1 [95% CI] | $/task | ours W/L | p |
|---|---|---|---|---|
| naive composition | 0.000 | 0.01 | 45/0 | 4e-9 |
| best-of-N re-draws (self-consistency by evidence; mstest2) | 0.067 | 0.09 | 13/0 | 2e-4 |
| autonomous orchestrator lead (in-house) | 0.460 [0.36, 0.56] | 0.03 | 28/3 | 7e-6 |
| **Codex CLI as study lead** (mean of 3 repeats) | 0.472 [0.40, 0.55] | 0.03 | **37/0** | **1e-7** |
| GEPA (official library; same feedback, 16 executions) | 0.609 [0.52, 0.70] | 0.09 | **22/11** | **0.037** |
| reference: fixed interface chosen on dev *using the hidden answers* | 0.680 [0.59, 0.76] | 0.03 | 18/15 | 0.71 |
| free-form workflow editor (ADAS/AFlow-style; 16 exec.) | 0.731 [0.65, 0.81] | 0.51 | 13/14 | 0.48 |
| **ours** (localized, typed, evidence-accepted edits; 16 exec.) | **0.716** [0.64, 0.79] | 0.22 | – | – |
| ours + escalation to free-form editing when the typed space is exhausted | 0.736 [0.66, 0.81] | 0.28 | | |

Per suite: mstest2 ours 0.719 · editor 0.665 · GEPA 0.606 · Codex lead 0.522; mstest3 ours
0.715 · editor 0.764 · GEPA 0.610 · Codex lead 0.448. The mstest2 lead over the editor did not
replicate — pooled it is a tie, at 42% of the editor's cost. The escalation variant (pre-registered
on mstest2; run on mstest3 after its other outcomes were graded, so post-hoc there) changes 4 of
48 tasks (+0.020): vs GEPA 22/8 (p 0.009), vs the editor 14/10 (p 0.92), vs Codex lead 38/0.
The fixed interface is a *reference*, not a competitor: it was picked on the development suite
by the hidden grader, i.e. with the answers; ours starts from the naive composition on every
task and never sees an answer.

**Decision quality** (every attempt graded after the fact; nothing flows back), pooled 48 tasks:

| method | tries | accepted | helpful | harmful (of accepted) | rejected but would have helped |
|---|---|---|---|---|---|
| ours | 338 | 119 | 66 | **9 (7.6%)** | **18** |
| free-form editor | 528 | 70 | 57 | 8 (11.4%) | 35 (+87 outranked) |
| GEPA | 497 | 64 | 44 | 9 (14.1%) | 51 |

What this shows: making the collaboration interface explicit and accepting changes only on
data-grounded evidence beats letting strong agents coordinate on their own (Codex as lead,
+0.24, 37/0) and beats prompt optimization with identical feedback and budget (GEPA, +0.11).
A free-form editor that rewrites the whole workflow from the same documented interface library
is as accurate; ours is cheaper and its acceptances are safer, and every accepted edit is
attributed to a traced symptom (§2 shows what that buys on real data).

## 2. Case study — the real IFN-β response (Kang et al. 2018), no reference answer

Full write-up: [case_study.md](case_study.md). Agents see 4 discovery donors; validation uses 4
held-out donors that no component ever saw (pre-registered split).

| arm (gpt-6-luna) | claims | replicate in held-out donors | IFN program, cell types | reproducible programs missed | $ |
|---|---|---|---|---|---|
| naive DE → GeneAgent → integrator | 0 | 0 | 0/6 | 10 | 0.05 |
| best-of-N re-draws | 0 | 0 | 0/6 | 10 | 0.98 |
| random edit locations | 0 | 0 | 0/6 | 10 | 1.25 |
| reflective prompt optimization | 8 | 7 | 6/6 | 4 | 0.94 |
| GEPA | 6 | 6 | 6/6 | 4 | 0.92 |
| Codex CLI alone | 7 | 7 | 6/6 | 3 | 0.01 |
| ReAct coding agent alone | 8 | 8 | 6/6 | 2 | 0.01 |
| **ours** | 13 | 10 (0 contradicted) | 6/6 | 1 | 1.75 |
| free-form editor | 33 | 28 (0 contradicted) | 6/6 | 0 | 3.72 |

Every node of the naive workflow succeeds — DE finds ISG15/IFIT/MX1/CXCL10, GeneAgent correctly
names "type I interferon antiviral defense" — yet the integrator makes no claim: the handoff
dropped direction and donor support. Ours localizes the loss and repairs it in executable steps
(split lists by direction → carry direction/statistics with each interpretation → add a
donor-consistency evidence branch, the step that unlocks 9 supported claims vs 0 in the no-edit
re-draw). GEPA reaches 6 correct claims by copying the evaluator's feedback (this dataset's genes
and GO terms) into the lead's prompt — score without repair, not transferable. The free-form
editor's rewrite reaches the highest recall here, at 2× our cost.

## 3. Public benchmark — BixBench v1.5 (205 open-answer questions, 59 capsules)

Pre-registered capsule split: `bixdev` 14 capsules / 39 Qs (design), `bixtest` 45 / 166 (run once,
frozen). Official agent Finch (FutureHouse) run unmodified on our backbone through an
OpenAI-compatible proxy; all agents execute in one native replica of BixBench's environment.
Grader: BixBench's open-answer grader (minus its substring "partial match") and MCQ-with-refusal.

| arm (gpt-6-luna) | dev open | test open [capsule CI] | test MCQ | $/q |
|---|---|---|---|---|
| Finch (official BixBench agent) | **0.231** | 0.373 [0.27, 0.48] | 0.392 | 0.005 |
| Codex CLI | 0.154 | 0.500 [0.38, 0.62] | 0.518 | 0.004 |
| ReAct coding agent | 0.154 | **0.542** [0.43, 0.65] | 0.560 | 0.002 |
| self-consistency (Finch ×3) | 0.231 | 0.410 | 0.434 | 0.013 |
| majority vote (3 agents) | 0.205 | 0.506 | 0.518 | 0.011 |
| Mixture-of-Agents | 0.205 | 0.542 | 0.560 | 0.011 |
| multi-agent debate | 0.179 | 0.542 | 0.536 | 0.012 |
| LLM-as-judge selection | 0.179 | 0.536 | 0.554 | 0.011 |
| ours: rubric + code-level audit + evidence arbiter | 0.179 | 0.536 [0.42, 0.66] | 0.554 | 0.012 |
| oracle (any of the three agents) | 0.231 | 0.608 | | |

BixBench questions are answerable by one agent that sees all the data. No combination beats the
best single agent — but which agent is best flips between splits (dev: Finch; test: ReAct).
Choosing the single agent on dev costs 0.16 on test (ours vs Finch +0.157 [+0.07, +0.25], 31/5);
every team stays within 0.04 of the per-split best (ours vs ReAct −0.012 [−0.07, +0.05]; vs MoA
−0.012 [−0.06, +0.03]). Composition is insurance here, not a gain. A repair loop that re-ran
failing analysts with the auditor's findings changed 1 answer in 166 and was dropped.
BixBench-Verified-50 (curated keys) is gated on Hugging Face and was not run.

## 4. Pooled data (shape 1): a strong single agent suffices (held-out `heldout`, 16 tasks, v3)

| method | F1 [95% CI] | $/task |
|---|---|---|
| naive / best-of-N re-draws | 0.000 / 0.000 | 0.04 / 0.76 |
| reflective prompt optimization | 0.389 [0.19, 0.60] | 1.01 |
| ours | 0.678 [0.53, 0.82] | 0.67 |
| free-form LLM editor | 0.699 [0.53, 0.85] | 0.35 |
| best fixed dev interface | 0.759 [0.61, 0.89] | 0.01 |
| **Codex CLI alone** | **0.787** [0.70, 0.87] | 0.01 |

Same conclusion as BixBench: when one agent can see all the data and its model is strong,
composing independently developed agents does not pay. (With a weaker framework model the lone
agent collapses — §8.)

## 5. Answer-free evaluation must be grounded in the data

Pairwise agreement with the sign of the hidden ΔF1 over 619 pairs of edit candidates (dev):
report-only LLM judge **0.53** · K3-style evidence judge 0.86 · deterministic evidence rule
**0.96**. A judge that reads reports cannot tell a better collaboration from a worse one; checks
that go back to the data can.

## 6. Collaboration loss is structural

Same components, only the interface changes (dev, 16 tasks per shape): naive composition 0.000 in
both shapes; best interface edits 0.757 (pooled) / 0.766 (sites); best-of-4 re-draws of the naive
workflow stay at 0.000. Pooled: the loss is in the replicate unit, direction, context and
reproducibility evidence at integration; sites: in task decomposition (sites refuse to report
local findings when handed the study question).

## 7. Mechanism ablations (held-out `mstest`, 16 tasks, same budget and acceptance)

| variant | F1 | vs ours W/L (p) | harmful / accepted |
|---|---|---|---|
| ours | 0.736 | – | 1 / 46 |
| random edit locations (same typed space) | 0.765 | 3/2 (0.38) | 2 / 35 |
| prompt-only reflective optimization | 0.000 | 0/15 (0.001) | – |
| + rule library distilled from dev (8 tasks) | 0.655 | identical 8/8 | 1 / 20 |
| editor with report-only-judge acceptance (`heldout`) | 0.746 (evidence acceptance 0.699) | n.s. | – |
| escalation to free-form editing (mstest2 + mstest3, n=48) | 0.736 vs 0.716 | changed 4/48 (0.125) | |

In shape 2 the typed space is small, so *where* to edit does not matter under a 16-execution
budget (in shape 1, localization beat random locations by +0.16). Prompt edits cannot repair a
request-policy loss.

## 8. Appendix — model-agnosticism (framework model swapped, external agents fixed at gpt-6-luna)

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


## 9. What the evidence supports, and what it does not

Supported:
- Composing independently developed agents incurs a structural loss that resampling cannot fix
  and interface edits can — on planted tasks and on real data (§2: 0 → 10 replicated claims).
- When data cannot be pooled, optimizing the explicit collaboration interface with data-grounded
  acceptance beats autonomous coordination by strong agents (Codex as lead 37/0, n=48) and
  prompt optimization with the same feedback and budget (GEPA 22/11, p 0.04).
- Answer-free acceptance must be data-grounded (0.96 vs 0.53).
- The typed search is robust to the framework model; free-form editing and single agents are not.

Not supported (honest negatives):
- Ours is not more accurate than a free-form LLM editor given the same interface library and
  feedback (pooled n=48 tie; the editor wins on the real case's recall) — it is cheaper, its
  acceptances are safer, and its edits are attributable.
- On single-agent-solvable tasks (BixBench, pooled shape 1) composition does not beat the best
  single agent; its value is robustness to not knowing which agent that is.
- The BixBench repair loop did nothing; provenance checks cannot assess verbose answers.
- Program naming is invisible to answer-free checks (most harmful acceptances).
- BAIS-DPTA was rejected as a benchmark: its score is dominated by label taxonomy (cluster-majority
  oracle 0.99 vs GPTCelltype 0.42 on Trachea). scBench / scBench-Long have only 6 / 4 public tasks.
