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
