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
- [x] claim→artifact→producer→input lineage; claims cite artifact ids
- [x] small executable edit space (handoff, analysis config, evidence branch, structure, prompt)
- [x] controlled suffix replay with no-edit control (v3.4.1: diff-based affected set)
- [x] answer-free checks + K3-style evidence judge (rubric first, both orders) + deterministic rule
- [x] acceptance protocol (evidence vs incumbent and control → judge veto; propagation route)
- [x] optimizer vs baselines at equal execution budget (reflective prompt, free-form editor v1/v2, best-of-N, Codex/in-house leads)

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

### Phase 2b: Model-agnostic + local-Codex (from user requirements) — `complete`
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
- [x] Second backbone held-out (gpt-4.1-mini): shape 2 n=8, shape 1 n=8; third backbone gpt-5-mini shape 2 n=4
- [x] Native Codex lead (gpt-6-sol) on mstest2 16 tasks × 3 (711 s2 missing after the outage; mean of 2)
- [x] v3.3 ours_escalate on msdev 8 → fresh mstest2 16 (inert: 1/16 tasks changed)

### Phase 3: Scale + mechanism ablations + transfer — `complete` (except public benchmarks)
- [ ] public benchmark slice (BixBench / scBench-Long) — NOT run: needs R/notebook environments or gated data; single-agent benchmarks
- [x] mechanism ablations on held-out shape 2 (plan §9), v3.1 settings, n=16 / 8:
      random_loc (localization), reflective_prompt 413–415 (prompt-only), ours_rules (transfer)
- [x] Idea C: transferable rules (msdev → mstest) = ours_rules: null (identical outcomes on 8/8)

### Phase 4: Write-up — `in_progress` (results docs done; paper text not started)
- [x] results doc with all numbers + negative results; README/CLAUDE.md updated
- [ ] paper-style write-up (not started; docs/results.md is the source)

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

### Phase R: Adversarial code review + simplification (user request 2026-09-29) — `complete`
Guard: experiments/replay_check.py (golden replay from caches, zero new spend) must stay green
for every behaviour-neutral change; behaviour changes are separate, justified, tested commits.
- [x] Stop all jobs (no contaminated outputs left); 3 independent read-only reviewers launched
- [x] Golden replay baseline: 29/32 OK; editor-v1 diffs = logging only; 1 FAIL = cache race
- [x] A. Behaviour-neutral: dead code/options, duplication, efficiency, docstrings, cache
      first-writer-wins + atomic node-cache publish, spent() prefix, BudgetExceeded not swallowed
- [x] B. Method bug fixes (v3.4, tag v3.4-dev; changes behaviour → recorded + tested; re-evaluation needed):
      compound controls vacuous (H1), site-request controls redraw only site_A (M2), lock
      semantics (H2), budget overrun by one (M1), escalation spends last execution on a control
      (MH1), unapplicable-edit fingerprint "" (M3), random_loc same permutation on every task
      (F3), data-boundary whitelist for free-form proposals, hidden-truth reachability for
      tool-using baselines, optimization LLM spend not charged (F2)
- [x] C. Impact report + v3.4.1 pre-registered evaluation on mstest2 (identical to v3.1 per task)

### Phase 5: Benchmarks, well-defined baselines, case study (user feedback 2026-10-01) — `complete` (2026-10-04; results in docs/results.md §1–3)
User: (1) design not shown effective; (2) every method on the SAME model (gpt-6-luna) and real
baselines, not ablations; (3) public benchmarks, set up locally; (4) the case study from the plan.
Same-model rule: every arm in every main table runs gpt-6-luna (framework, external agents,
baselines, Codex `-m gpt-6-luna`); the grader is pinned separately. Cross-model results move to
an appendix.
- [x] P5.0 OpenAI-compatible local proxy over agentcoop.llm (ledger, cache, budget cap) so
      external harnesses (BixBench aviary/fhda, GEPA, DSPy) use luna and our accounting
- [x] P5.1 BixBench: native env replica (no Docker) + Finch venv + 205 tasks; 3 harness bugs
      fixed (findings); dev sweep of 3 single agents
- [x] P5.2 Collaboration system for BixBench (independent analyst agents + answer-free
      cross-verification evidence + optimized handoff/verification policy per capsule)
- [x] P5.3 Baselines (all luna): native BixBench agent (fhda/Finch), Codex CLI, ReAct coder,
      self-consistency/best-of-N at equal budget, multi-agent debate, GEPA (official library)
      optimizing the composed workflow with the same answer-free metric
- [x] P5.4 Runs: dev 39 → frozen test 166; capsule-clustered CIs
- [x] P5.5 BAIS-DPTA checked and rejected as main benchmark (taxonomy-dominated metric);
      scBench/scBench-Long: only 6/4 public evals (findings)
- [x] P5.7 GEPA (official) on kangreal + mstest2 + mstest3; fresh pre-registered mstest3 (n=32)
- [ ] P5.8 BixBench-Verified-50 needs user's HF login (gated) — ask user
- [x] P5.6 Case study: real Kang 2018 IFN-β stimulation (no planted answer) — naive vs optimized
      collaboration, lineage of the loss, validation against independent evidence

### Phase 6: v4 — cross-examination across resource silos — `in_progress` (from 2026-10-14)
Spec: docs/superpowers/specs/2026-10-14-cross-examination-design.md (order of work = §9).
Plan: docs/superpowers/plans/2026-10-14-v4-core.md (core); Case 1/2 plans are written after it ships.
v3 is reproducible from tag v3.4.1. Hard cap $180 actual (AGENTCOOP_BUDGET_USD=180).
- [ ] 6.1 v4 skeleton, sandbox, testers, label function, protocol, arms, `fsdev` generator, tests — $0
  - [x] 6.1a store, sandboxed harnesses, claims, site views/tables, testers, protocol, sites, planted suite + grader, arms, runner/analysis/headroom (plan Tasks 1–10)
  - [x] 6.1b invariant tests (answer boundary, site-file whitelist, workspaces outside the repo); CLAUDE.md, README, docs/method.md rewritten for v4 (plan Task 11)
  - [x] 6.1c end-to-end smoke on `fsdev-800` (9 arms, draw 0): cost, wall-clock, site calls in progress.md; stop for review before scaling
- [ ] 6.2 Headroom gates (§6) for `fsdev`, COVID, TB — $0; data prep for both real cases; held-out references built and hashed
  - [ ] pass rules written into findings.md before computing; a failing case is reframed or dropped before the freeze, with the reason
  - [ ] `fsdev` gate includes the measured view-error rate of Codex-built views (≈ 0 → drop the view-repair claim and Case 3's ID/label knob)
- [ ] 6.3 All arms on 2 `fsdev` tasks → debug → all 8 `fsdev` tasks; freeze testers, δ/q, prompts, K, B, F, matching rules, scope tables, crosswalks (incl. the MA-genome self-check count = ours' median follow-ups) → tag `v4-freeze`
- [ ] 6.4 Build `fstest` (32 tasks, unseen donors, after the freeze); run every arm once (draw 0); grade (task = unit, paired Wilcoxon; knob main effects 16 vs 16)
- [ ] 6.5 COVID (Case 1): every arm, 3 draws (Codex-heavy arms within the daily Codex quota)
- [ ] 6.6 TB (Case 2): minimal 4-site version (S1, S2, S6, S4), every arm, 3 draws; expansion (S3, S5) only if budget allows
- [ ] 6.7 Report every arm, negative results included, with the §4 interpretation rules
Rules: no method change after the freeze without a new suite; every change gets a reason in findings.md; debug on 1–2 tasks before any larger run.
