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
