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
