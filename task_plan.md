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

## Phases

### Phase 0: Setup — `complete`
- [x] Branch `v3` off `revision`; replace v2 package with new lean `agentcoop/` (v2 stays in history)
- [x] LLM client: cost ledger, cache, budget cap, retries
- [x] GeneAgent vendored as black-box external agent via openai-0.28 shim (no internal edits)
- [ ] Fix CLAUDE.md drift found so far (data/ absent in this checkout)

### Phase 1: Feasibility experiment (decides whether the project continues) — `complete` (shape 1); shape 2 in progress
- [x] 1a Data: Kang 2018 PBMC (8 donors) + MSigDB Hallmark; planted-signal generator with hidden truth
- [x] 1b Task shape 1: DE analysis → gene-set interpretation (GeneAgent) → integrator; 12–20 dev tasks
- [ ] 1c Task shape 2: multi-evidence integration (second modality / independent evidence source)
- [x] 1d Hidden grader (claim-level precision/recall, direction, donor-reproducibility calibration)
- [x] 1e Baselines: naive composition, best single agent, Codex CLI (gpt-6-luna), equal-budget rerun
- [x] 1f Candidate edits (handoff / analysis-config / evidence-check), executed for real
- [x] 1g Q1 room for improvement; Q2 evidence-grounded vs report-only evaluator; Q3 vs equal-budget retry
- [x] Decision gate: write findings; branch per plan (evaluator weak → Idea B first; etc.)

### Phase 2: Minimal closed loop — `in_progress` (loop built; test-suite runs going)
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

### Phase 3: Scale + mechanism ablations + transfer — `pending`
- [ ] public benchmark slice (BixBench open-ended / scBench-Long if runnable locally)
- [ ] mechanism ablations (5 claims in the plan's §9)
- [ ] Idea C: verified transferable collaboration rules, tested on held-out tasks

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
