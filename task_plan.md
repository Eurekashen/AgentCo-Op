# Task Plan: AgentCo-Op v3 — Optimizing Scientific Agent Collaboration without Reference Answers

## Goal
Make the project work: a training-free system that, without reference answers and with only
partial/unreliable scientific feedback, finds where collaboration between independently developed
external agents loses information, acquires discriminating evidence, and turns fixes into
executable, verified collaboration improvements — and show, with fair baselines (native Codex /
Claude Code, best single agent, equal-budget retry, reflective prompt optimizer), that it helps.

## Three falsifiable research questions (the paper's spine)
- RQ1: Do independently developed scientific agents incur a measurable *collaboration loss* when
  composed (not just individual-agent weakness)?
- RQ2: Without reference answers, can active evidence acquisition + controlled intervention find
  executable edits that reduce that loss?
- RQ3: Do those edits beat native coding-agent autonomy, generic reflective optimization and
  equal-budget retry, and transfer to new tasks?

## Ground rules
- Optimizer inner loop NEVER sees hidden truth / final grader. Enforce in code (separate module,
  grader only imported by the evaluation harness).
- Model: `gpt-6-luna` only ($0.10/M in, $0.50/M out). Every call goes through one client with a
  cost ledger + on-disk cache; hard budget cap per experiment.
- Never print/commit the key (`.secrets/` is gitignored).
- Report negative results honestly; statistics unit = task/study, not cells/reruns.
- Do not delete remote branches. Work on local branch `v3`.
- (user, 2026-09-27) Debug on a SMALL subset first (2–4 tasks); scale up in steps; never launch
  full suites by default.
- (user) Every framework change needs a stated reason + evidence (recorded in findings.md).
- (user) Codex CLI baselines use the user's local, logged-in Codex (no API-key login).
- (user) Model-agnostic: pipeline and conclusions must not depend on one LLM; backbone is a config
  choice with capability fallbacks; grader pinned separately; key findings checked on a 2nd model.

## Phases

### Phase 0: Setup — `complete`
- [x] Branch `v3` off `revision`; replace v2 package with new lean `agentcoop/` (v2 stays in history)
- [x] LLM client: cost ledger, cache, budget cap, retries
- [x] GeneAgent vendored as black-box external agent via openai-0.28 shim (no internal edits)
- [x] Fix CLAUDE.md drift (rewritten for v3)

### Phase 1: Feasibility experiment (decides whether the project continues) — `complete` (both shapes)
- [x] 1a Data: Kang 2018 PBMC (8 donors) + MSigDB Hallmark; planted-signal generator with hidden truth
- [x] 1b Task shape 1: DE analysis → gene-set interpretation (GeneAgent) → integrator; 12–20 dev tasks
- [x] 1c Task shape 2: data held at two sites (site agents → lead; replication question)
- [x] 1d Hidden grader (claim-level precision/recall, direction, donor-reproducibility calibration)
- [x] 1e Baselines: naive composition, best single agent, Codex CLI (gpt-6-luna), equal-budget rerun
- [x] 1f Candidate edits (handoff / analysis-config / evidence-check), executed for real
- [x] 1g Q1 room for improvement; Q2 evidence-grounded vs report-only evaluator; Q3 vs equal-budget retry
- [x] Decision gate: write findings; branch per plan (evaluator weak → Idea B first; etc.)

### Phase 2: Minimal closed loop — `complete` (loop, lineage, edit spaces, controls, acceptance all built; v3.1 frozen)
- [ ] claim→artifact→producer→input lineage; claims cite artifact ids
- [ ] small executable edit space (handoff adapter, evidence branch, agent swap, merge rule, remove branch)
- [ ] controlled suffix replay with no-edit control; Δ_j estimates over repeats
- [ ] active evidence check selection (minimal Idea B); artifact-grounded pairwise judge (rubric fixed first)
- [ ] acceptance protocol (hard constraints → evidence → judge; keep incumbent when unresolved)
- [ ] autonomous optimizer vs baselines (incl. GEPA-like prompt-only optimizer at equal budget)

### Final evaluation protocol (fixed before any held-out result is seen)
- Shape 1 held-out = `heldout` (seeds 500–515, never inspected). Shape 2 held-out = `mstest`
  (seeds 400–415, never inspected). Dev/validation = `dev`, `test` (demoted), `msdev`.
- Methods (same budget $1.0 nominal/task, 6 iterations, 3 edits/iteration, same feedback):
  ours · ours_no_propagation · ours_no_control · random_loc · reflective_prompt (GEPA-style) ·
  llm_editor (AFlow/ADAS-style free-form) · best_of_n · ours_rules (rules distilled from dev/test).
- Fixed references: naive; best fixed dev edit; coder_single; codex_single (shape 1);
  ms_orchestrator + ms_codex_orchestrator (shape 2).
- Primary metric: hidden F1 per task; paired Wilcoxon vs naive and vs best baseline; bootstrap CI;
  nominal cost incl. rejected edits and controls. Report all tasks incl. failures.

### Phase 2b: Model-agnostic + local-Codex (from user requirements) — `in_progress`
- [x] Audit model-specific assumptions (7 items, findings.md)
- [x] Capability-negotiating LLM client (intents → per-model params; probed + cached profiles)
- [x] Framework model (AGENTCOOP_MODEL) vs external-agent model (AGENTCOOP_EXTERNAL_MODEL); grader pinned; prices
- [x] Codex baselines on the user's logged-in CLI (hermetic except auth; native arm = gpt-6-sol)
- [x] Small-subset robustness check (gpt-4.1-mini framework, 4 tasks): shape 2 holds; shape 1 limited by naming
- [x] Competitive claim support adopted; 3 alternatives rejected with evidence; lookahead tested (null, off)
- [x] Shape-2 search comparison on 8 msdev tasks: ours ≈ free-form editor (correction of 4-task signal)
- [x] Codex ×3 repeats (local login) on msdev (4) and all 16 mstest tasks
- [x] mstest v3.1 steps 1+2 (ours_v31, llm_editor, Codex lead ×3) — llm_editor 414/415 finishing
- [x] mstest-404 post-hoc failure analysis (needs request edit + lead replication jointly)
- [ ] Second backbone on held-out shape 2 (gpt-4.1-mini framework): 2 tasks running → 4
- [ ] Native Codex lead (user's own config, gpt-6-sol): smoke 1 task → 4 tasks × 3
- [ ] v3.3 ours_escalate: dev msdev-303/306 → msdev 8 → fresh mstest2 (700–715) in steps

### Phase 3: Scale + mechanism ablations + transfer — `pending`
- [ ] public benchmark slice (BixBench open-ended / scBench-Long if runnable locally)
- [ ] mechanism ablations on held-out shape 2 (plan §9), v3.1 settings, steps of 4 tasks:
      random_loc (localization), reflective_prompt 413–415 (prompt-only), ours_rules (transfer)
- [ ] Idea C: verified transferable collaboration rules (msdev → mstest) = ours_rules above

### Phase 4: Write-up — `pending`
- [ ] results doc with all numbers + negative results; README/CLAUDE.md updated

## Decisions
| Decision | Why |
|---|---|
| Fresh package on branch `v3`, v2 kept in `revision` | plan: keep lineage/audit ideas, drop certification-centric design |
| Semi-synthetic planted signals on real multi-donor data as primary dev tasks | exact hidden truth needed to evaluate a no-answer optimizer; real data keeps realism |
| GeneAgent as real external agent, untouched, via LLM-transport shim | it is independently developed with a restrictive gene-list interface — the exact handoff-loss scenario |

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| `agentcoop` console script missing; stale v1 editable install | 1 | use `python -m agentcoop.cli`; reinstall on v3 |
| data/ and external/ absent in this checkout (CLAUDE.md said kept on disk) | 1 | re-download needed data into gitignored data/ |
- Final protocol amendment: max_iterations 6 → 8 for all final runs (compound escalation needs more iterations; the .0 nominal budget remains the binding limit).
- Fairness: heldout llm_editor/llm_editor_report_judge ran with 6 iterations and did not exhaust budget → re-run with 8 after the frozen ours runs (reflective_prompt, random_loc, best_of_n were budget-bound).
- Deviation: ours_no_compound not run on mstest (compute); dev evidence only.

### Phase R: Adversarial code review + simplification (user request 2026-09-29) — `in_progress`
Guard: experiments/replay_check.py (golden replay from caches, zero new spend) must stay green
for every behaviour-neutral change; behaviour changes are separate, justified, tested commits.
- [x] Stop all jobs (no contaminated outputs left); 3 independent read-only reviewers launched
- [x] Golden replay baseline: 29/32 OK; editor-v1 diffs = logging only; 1 FAIL = cache race
- [ ] A. Behaviour-neutral: dead code/options, duplication, efficiency, docstrings, cache
      first-writer-wins + atomic node-cache publish, spent() prefix, BudgetExceeded not swallowed
- [ ] B. Method bug fixes (v3.4; changes behaviour → recorded + tested; re-evaluation needed):
      compound controls vacuous (H1), site-request controls redraw only site_A (M2), lock
      semantics (H2), budget overrun by one (M1), escalation spends last execution on a control
      (MH1), unapplicable-edit fingerprint "" (M3), random_loc same permutation on every task
      (F3), data-boundary whitelist for free-form proposals, hidden-truth reachability for
      tool-using baselines, optimization LLM spend not charged (F2)
- [ ] C. Impact report on reported results + re-evaluation plan (ask before large runs)
