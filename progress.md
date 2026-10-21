# Progress Log

## Session 2026-09-27
- Reviewed repo + branches; regenerated CLAUDE.md (session 1).
- User handed over v3 research plan (collaboration-interface optimization without reference answers).
- Surveyed env, verified gpt-6-luna access and network; cloned GeneAgent to scratchpad for inspection.
- Started downloads: Kang 2018 h5ad, Hallmark gene sets.
- Built v3 package: llm.py (REST, cache, ledger, budget), workflow.py (artifacts, lineage, node
  cache/replay), handoffs.py, agents (de_analyst wilcoxon|pseudobulk, geneagent, integrator),
  tasks/planted.py (binomial thinning, core genes), eval/grader.py (hidden), evidence/checks.py,
  evidence/judge.py (report-only, evidence K3-style, rule).
- GeneAgent vendored unmodified (upstream 88fd1ad) + openai-0.28 shim + HTTP record/replay.
- Launched phase-1 fixed-candidate runs on dev (10 candidates × 16 tasks), budget cap $30.
- Phase-1 runs complete on dev (13 edit candidates + coder + codex, 16 tasks). Q1 analysed (see
  findings). Added donor_check evidence component + candidates; integrator re-draws (s1–s3) for Q3.
- Codex baseline made hermetic (throwaway HOME, caches in workspace, stdin closed).
- Optimizer: propagation-depth acceptance, finer stages, visited set; 30 offline tests pass.
- Running: Q2/Q3 judge analysis (dev), optimizer debug on dev-000.
- Optimizer debug on dev-000: hidden F1 0 → 1.0 via 5 accepted edits (4 on propagation depth).
  Added param locking after acceptance; editor doc lists donor_check (fair information).
- Shape 2: scope=local request fixes the "sites refuse to report" loss (msdev-300 F1 0 → 0.4);
  evidence check flags the lead's site-specific (decoy) claim as unassessed.
- Launched on TEST suite (held out): optimizer "ours" (budget $1.0 nominal, 6 iters) and fixed
  references (naive, pb+no_interp+donor, pb+split+stats+donor, coder, codex). Cap $60 actual.
- Validation (test): fixed refs done — coder 0.808, codex 0.797, best fixed composition 0.713,
  naive 0. Optimizer on test-101/102/104 matches best fixed per task (0/0.5/1.0).
- Loop: first-improvement acceptance (cost ÷~3). Budget cap raised to $120 actual (spent $8.45).
- HELDOUT launched: fixed refs (naive, 2 best fixed, coder, codex) + ours, ours_no_propagation,
  best_of_n. Next: random_loc, reflective_prompt, llm_editor, ours_rules.
- Shape-2 dev (msdev) complete: naive 0 → best fixed interface 0.766; in-house orchestrator 0.423,
  Codex orchestrator 0.434. Optimizer trial on msdev-300/301: 0.80/0.86.
- Prompt-edit key per incumbent; free-form editor gets multisite doc; answer-free select step.
- MSTEST (shape-2 held-out) launched: fixed refs + ours, ours_no_propagation, best_of_n.
- 2026-09-27 17:03 PAUSED by user request. All experiment processes stopped (no background jobs).
  mstest done: ours 16/16, reflective_prompt 13/16, llm_editor 1/16, random_loc 1/16; not started:
  llm_editor_report_judge, best_of_n, ours_rules, ours_no_propagation. Actual spend $52.06.
  RESUME (skips finished outputs; partial runs replay from node + LLM caches):
    AGENTCOOP_BUDGET_USD=120 nohup python -u experiments/run_optimizers.py --suite mstest \
      --methods llm_editor llm_editor_report_judge best_of_n --budget 1.0 --iterations 8 \
      --parallel 4 > runs/phase2/mstest_p1.log 2>&1 &
    then: --methods random_loc ours_rules ours_no_propagation reflective_prompt (same flags) > mstest_p2.log
  Keep parallel ≤ 4 (each site-agent kernel ≈ 300 MB; 14-way overloaded the machine).
  After: analyze mstest-401/404 (ours 0 vs Codex lead 0.67/1.0), fill docs/results.md §3b/§4, commit.

## Session 2026-09-28
- mstest step 2 (408–415): ours_v31 done 8/8; Codex lead (local login) s1–s3 done 8/8;
  llm_editor 6/8 (414, 415 running). Partial n=16: ours_v31 0.736, llm_editor (14) 0.748,
  fixed 0.714, in-house lead 0.526, Codex lead mean-of-3 0.491, naive 0.
- phase2_analysis.py: `--mean-samples <candidate>` averages non-seedable repeats per task.
- mstest-404 post-hoc: ours 0 — every request edit rejected; the lead refuses without cross-site
  replication, and the no_claims finding implicates only the request edge (fix needs
  request edit + lead-side replication together). Not fixed on held-out (frozen); noted.
- Launching second-backbone held-out check (gpt-4.1-mini framework, luna external) on 2 mstest tasks.
- mstest n=16 FINAL (committed 13c0f56): ours_v31 0.736, editor 0.748, fixed 0.714, in-house 0.526,
  Codex lead mean3 0.491, naive 0. Decision quality: ours harmful 1/46, editor 3/24.
- Instrumentation: Attempt.report + Outcome.initial_report/executions; replays in runs/phase2_attempts.
- best_of_n now honours max_executions.
- Ablations batch A (mstest-400..403: reflective_prompt, ours_rules, random_loc) running
  → runs/mstest_ablate_a.log. reflective_prompt v3.1 = 0 on 4/4 (old 13 archived to _stale).
- gpt-4.1-mini framework on mstest-400/401: naive 0, same-model lead 0.33, fixed 0.63;
  ours_v31 + llm_editor running → runs/mstest_41mini_opt.log.
- v3.3 candidate `ours_escalate` (typed search → free-form editor with the unused budget when the
  typed space is exhausted). Offline test added. Next: dev msdev-303/306 (ours stopped there with
  "no untried edit"), then a FRESH held-out suite (mstest2, seeds 700–715) — never mstest.
- 2026-09-28 ~07:30 state (all in background, parallel ≤ ~6 kernels, memory ~50% free):
  * mstest2 step 1 (700–707): fixed refs DONE; Codex lead + native Codex lead s1–s3 running
    (runs/mstest2_codex_a.log, sequential); optimizers ours_v31 → ours_escalate → llm_editor_v2
    running (runs/mstest2_opt_a.log). Pre-registered in findings.md; code tag v3.3-eval.
  * gpt-4.1-mini framework extension on mstest 404–407 (+ editor v2 on 401):
    runs/mstest_41mini_fixed_c.log, runs/mstest_41mini_opt_c.log.
  * Ablations n=8 done (findings.md). Remaining optional: random_loc/reflective on 408–415.
  RESUME: rerun the same commands (outputs that exist are skipped; partial runs replay from caches):
    AGENTCOOP_BUDGET_USD=120 python -u experiments/run_optimizers.py --suite mstest2 --tasks mstest2-70{0..7} \
      --methods ours_v31 ours_escalate llm_editor_v2 --budget 5.0 --max-executions 16 --iterations 8 --parallel 2
    AGENTCOOP_BUDGET_USD=120 python -u experiments/run_candidates.py --suite mstest2 --tasks mstest2-70{0..7} \
      --candidates ms_codex_orchestrator ms_codex_native_orchestrator --samples 1 2 3 --parallel 1
  ANALYSIS: python experiments/phase2_analysis.py --suite mstest2 --reference ms_codex_orchestrator__mean \
      --mean-samples ms_codex_orchestrator ms_codex_native_orchestrator --out runs/phase2/mstest2_analysis.json
- 2026-09-28 15:20 (correction: step 1 of mstest2 ran ~07:40–15:10, not 07:30 as estimated).
  * mstest2 step 1 DONE (committed 5879d1f): ours_escalate 0.592, ours_v31 0.567, fixed 0.553,
    editor v2 0.529, native Codex lead 0.467, Codex lead 0.383, in-house 0.188, naive 0.
  * mstest2 step 2 (708–715) started 15:11: runs/mstest2_{fixed,codex,opt}_b.log (ETA ~22:30).
  * mstest ablations n=16 DONE (random_loc 0.765 ≈ ours 0.736; reflective 0).
  * gpt-4.1-mini framework on mstest 400–407 DONE: ours 0.842, editor v2 0.659, lead 0.392.
  * NEW: pooled-data (shape 1) gpt-4.1-mini check on heldout-500/501 (fixed refs + same-model
    single agent) → runs/heldout_41mini_fixed_a.log; then ours_v31 + llm_editor_v2 on the same 2.
- 2026-09-28 20:20 NETWORK OUTAGE recovery (user disconnected network; ledger gaps 18:34–20:16):
  * All processes survived; API calls resumed 20:16. No run finished inside the window.
  * Killed the one Codex run stuck across the window (mstest2-711 native s2) and QUARANTINED its
    result + node-cache entry (runs/_quarantine_outage_20260928/). TODO when mstest2_codex_b ends:
    rerun the same run_candidates command (it redoes only the missing 711 native s2).
  * TODO when the in-flight optimizers finish: list outcomes whose [start, end] overlaps
    18:34–20:16 (start = file mtime − seconds) and re-run them into the same path after
    quarantining the original (successful nodes replay from cache; failed ones re-execute).
    Affected candidates: mstest2 712–715 (ours_v31/ours_escalate/llm_editor_v2),
    heldout 502–507 (gpt-4.1-mini), mstest 400/401 (gpt-5-mini).
- 2026-09-29 01:22 PAUSED by user for an adversarial code review / simplification pass.
  * Outage redo done by cache replay: mstest2 706/707 editor v2, 714/715 ours_escalate,
    heldout 506/507 ours_v31 (4.1-mini) — all re-written (originals in runs/_quarantine_outage_20260928/).
  * STILL MISSING: mstest2-711 ms_codex_native_orchestrator s2 (redo was stopped before writing
    anything; rerun: run_candidates --suite mstest2 --tasks mstest2-711 --candidates
    ms_codex_native_orchestrator --samples 2).
  * Not yet analysed: mstest2 full n=16; heldout shape-1 gpt-4.1-mini n=8 optimizers.
- 2026-09-29 review DONE (commits after 775d1ae; tag v3.4-dev): 3 independent reviewers + own pass.
  Phase A behaviour-neutral (golden replay identical at every step), Phase B v3.4 method fixes,
  corrected stats in docs. 50 offline tests. No experiment running. Open: mstest2-711 native s2
  rerun; mstest2 step-2 and shape-1 4.1-mini analyses (v3.3 as-run); v3.4 re-evaluation plan.
- 2026-09-29 v3.4 targeted check (user approved): ours_v31 (v3.4 code) on heldout 500/503/506 +504
  (gpt-4.1-mini framework) → runs/phase2_v34@gpt-4.1-mini/ ; cap $79.4.
- 2026-09-29 v3.4.1 msdev-303 smoke: 0.50 → 1.00 (escalation fires). mstest2 step 1 (700–707) launched → runs/phase2_v34/, cap $79.97
- mstest2 step 1 v3.4.1: identical to v3.1 (0.567; 3/57 controls differ, 0 verdicts). step 2 launched, cap $80.05

## 2026-10-01 → 10-04 — Phase 5 (benchmarks, same-model baselines, case study)
- Infrastructure: proxy over llm (Finch/GEPA on the shared ledger), native BixBench env, Finch
  harness (3 local-mode bugs fixed), BixBench tasks/graders, real-data Kang case with held-out
  donors, GEPA baseline, BixBench team baselines + evidence team. Commits ed4eaf4, 1d84c3a.
- Runs: kangreal (9 arms), mstest2 + GEPA/best-of-N, fresh mstest3 (32 tasks, pre-registered),
  BixBench dev 39 + frozen test 166 (10 arms). Spend $78.7 → $104.7 actual.
- Docs rewritten around one-backbone main tables: results.md, baselines.md, case_study.md,
  README, CLAUDE.md. BAIS code removed after rejecting the benchmark.

## 2026-10-14 → 10-16 — Phase 6 (v4 cross-examination), plan Task 11 smoke
- Smoke on fsdev-800 (9 arms + mechanism arms, draw 0; sites A codex, B finch, C codex, M codex),
  four rounds, each cause recorded in findings.md: (1) no claim passed its own gate -> propose
  prompt states the discovery/confirmation split and the fixed test; (2) whole-GO-set scores
  diluted the planted program -> prompt names rank-then-group; (3) site A's view unreadable
  (duplicate column) -> step 1 sends an unreadable view back once with the validator messages;
  (4) final: ours F1 0.67 (P 1.0, R 0.5), MA-genome 0.50, vote 1.00 (2 claims), best single
  0.20, union 0.13, prior-only 0; no_resolution = ma_claims = ours (no resolvable conflict on a
  clean task); 1 follow-up (handoff-loss at B), 0 run errors.
- Cost: full fresh smoke 2,394 s wall-clock, 255 LLM calls (v4 ledger), $0.20 actual
  (Codex bills the local login), ~$0.79-1.02 nominal; per arm 0-9 site calls, $0-0.11 nominal.
  Replay of stored calls: 166 s, $0.
- Headroom gate (§6, $0, oracle views, 8 fsdev tasks): PASS (5/8 tasks with >= 0.05 gain over
  MA-tools; mean F1 scripted ours 0.631 vs best single 0.577); composition items unreachable in
  7/8 tasks. Fix: symbol table read without NA parsing. Commits 2324887, b2abe9d.
- STOPPED for review (plan Task 11 Step 17): no other fsdev task run, fstest not built, tester
  and planting parameters unchanged.
- 2026-10-21 planting revision (user approved): DROP 0.75 in a minor population (B, CD8 T, NK,
  CD14 Mono) with no program; programs only in populations with >= 50 median cells/donor (drops
  CD16 Mono). New dev suite fsdev2 (810-817) built, boundary ok; headroom PASS (scripted ours
  0.839 / MA-tools 0.745 / best single 0.825 / mean site 0.633). Commits 4f2e56c + this one.
  No agent run on fsdev2 yet; fsdev stays as built.
- 2026-10-21/22 fsdev2 smoke (810, 813; 15 arms): smoke 1 exposed two-direction hitchhiking
  (tester now intersection-union) and lost replications under a claims x sites BH (label rule
  now pools non-proposer sites + band condition); smoke 2 (fresh step 1, new prompt): ours F1
  0.85 mean (810 1.00, 813 0.70); headroom fsdev2 under the new rule FAILS by 0.011 vs hindsight
  best single. Stale results in runs/v4/_stale_fsdev2_smoke1, pre-band labels in
  runs/v4/_relabel_fsdev2_smoke2. Stopped for review before the other 6 fsdev2 tasks.
- 2026-10-22 fsdev2 complete (8 tasks x 15 arms): ours F1 0.91 vs MA-genome 0.75 (p 0.016), vote
  0.84 (n.s.), ma_claims 0.90 (n.s.), step 4 adds nothing, view errors ~0. TB and COVID built,
  held-out TB references hashed, three headroom gates computed (fsdev2 FAIL by 0.011 vs hindsight
  best single; TB FAIL; COVID passes vs MA 2/3 folds, FAIL vs best single). Stopped for the
  freeze / reframing decisions. v4 spend $2.11 actual.
- 2026-10-22 PAUSED by user. Frozen at tag v4-freeze (73851f8) + post-freeze infra/eval commits.
  fstest (32 tasks, artefact on/off 16/16) built; batch 1 (fstest-900..907, all 15 arms, draw 0)
  was running: fstest-900 finished, the rest were stopped mid-run. COVID step-1 smoke (`single`
  arm, draw 0; Berlin/Emory codex, Stanford/Seoul finch) was stopped mid-run. Resume: rerun the
  same commands (runs/v4/logs/fstest_b1.log, covid_smoke.log); finished results and stored agent
  calls are reused, interrupted calls re-run. Next after fstest: grade (analyze.py), COVID all
  arms x 3 draws (analyze_covid.py), TB descriptive run. v4 spend so far ~$2.1+ actual.
