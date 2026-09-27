# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research code for AgentCo-Op (paper v1: arXiv 2605.20425). Branch `v3` (local) is a ground-up
rebuild around one research question — see [review.md](review.md) (the user's plan) and
[docs/method.md](docs/method.md) (the design contract):

> Without reference answers and with only partial scientific feedback, find where collaboration
> between independently developed agents loses information, acquire evidence that distinguishes
> explanations, and turn fixes into executable, verified changes to how they collaborate.

Training-free: nothing is fine-tuned; the optimizer edits workflows (handoff policies, analysis
configs, evidence branches, topology, instructions), never model weights.

Other branches are history, not dependencies: `main` = v1 as published (reproducibility record,
`94c3750`); `revision` = v2 evidence-certification compiler (superseded; its critique is in the
plan); `origin/feature/ecps-preference-search` = v2 preference search (not merged). `gh-pages`
still shows v1 numbers. Never delete remote branches.

Working memory for the long-running effort lives in `task_plan.md`, `findings.md`, `progress.md`
(read them first when resuming). Results so far: [docs/results.md](docs/results.md).

### Where the evidence stands (keep this honest; update from `findings.md`)

- Collaboration loss is real and structural: naive compositions score F1 0 in both task shapes;
  interface edits recover ~0.7; resampling (best-of-N) cannot.
- Answer-free *evaluation* needs data-grounded checks: report-only judge is at chance (0.50
  pairwise), evidence rule 0.95.
- Shape 1 (pooled data): a single strong agent (Codex CLI) beats every composition (0.79 held-out);
  our optimizer ≈ a free-form LLM editor given the same component library; propagation depth and
  evidence acceptance were *not* load-bearing there. Localization beats random locations (+0.16).
- Shape 2 (data at two sites): composition is necessary; the evidence-shaped interface beats
  Codex-as-study-lead (dev 0.77 vs 0.43; mstest fixed 0.71 vs 0.60). Final held-out optimizer
  numbers: see `docs/results.md`.
- The final evaluation code is tagged `v3-final-eval`; held-out suites are `heldout` and `mstest`
  (never inspected before the freeze). Superseded outputs are kept under `runs/phase2/_stale/`.

## Commands

```bash
pytest -q                                   # offline, ~0.2s; no API key needed
python -m agentcoop.tasks.planted dev 0 16  # build shape-1 tasks (needs data/raw/kang_2018.h5ad)
python -m agentcoop.tasks.multisite msdev 300 316
python experiments/run_candidates.py --suite dev --candidates naive pseudobulk+no_interp+donor
python experiments/redraw.py --suite dev --candidates naive --nodes report --samples 1 2 3
python experiments/phase1_analysis.py --suite dev [--judges]   # grades + Q1/Q2/Q3
python experiments/run_optimizers.py --suite test --methods ours best_of_n --budget 0.6
```

Environment: `AGENTCOOP_BUDGET_USD` (hard cap on *actual* spend across the ledger — always set it
for experiments), `AGENTCOOP_GENEAGENT_SLOTS` (concurrent GeneAgent processes, default 8),
`AGENTCOOP_LLM_DB` (ledger/cache sqlite, default `runs/llm.sqlite`). The API key is read from
`OPENAI_API_KEY` or `.secrets/openai` — never print or commit it. Model is `gpt-6-luna` only.

Data is not in git: `data/raw/kang_2018.h5ad` comes from `api.figshare.com/v2/file/download/34464122`
(the `figshare.com/ndownloader` URL is behind a WAF challenge and returns 0 bytes); gene-set
libraries come from Enrichr's `geneSetLibrary?mode=text&libraryName=...` (Hallmark 2020, GO BP 2023,
Reactome 2022). `runs/` (ledger, caches, run records) is gitignored.

## Architecture

```
tasks/ (planted, multisite) ─▶ workflow.execute ─▶ report ─▶ evidence.checks ─▶ optimize.loop
            │ hidden truth                                                          │
            └──────────────▶ eval.grader  (evaluation harness only) ◀───────────────┘ final report
```

- `workflow.py` — workflows are data (nodes = agent+config, edges = handoff+params). Every artifact
  records producer and `derived_from`; `Run.lineage()` walks it. Node outputs are cached under
  (agent, config, input contents, sample, task) → editing a downstream edge replays the prefix.
- `handoffs.py` — executable handoff policies; they produce *views* (new artifacts). Downstream
  agents render `data` only, so what a view carries is literally what the next agent can use.
- `agents/` — `de_analyst` (Wilcoxon | pseudobulk DESeq2), `geneagent` (external, see below),
  `integrator` (LLM → structured claims), `donor_check`, `site_analyst`/`replication_check`
  (shape 2), baselines `coder` (Responses API + persistent kernel), `codex_agent`, `orchestrator`.
- `evidence/` — `checks.py` answer-free checks (per-donor claim support; uncovered *coherent*
  signal via GO enrichment; `profile_task` dispatches pooled vs multisite); `judge.py`
  report-only judge, K3-style evidence judge (rubric fixed before candidates, scorepad, both
  orders), deterministic `evidence_rule`.
- `optimize/` — `trace.py`/`trace_sites.py` localize loss by tracing evidence probes through
  lineage; `edits.py` declared local edit spaces; `loop.py` propose → controlled replay →
  no-edit control → accept; `baselines.py` reflective-prompt, free-form LLM editor, best-of-N,
  all sharing the same feedback/budget/acceptance.

## Invariants — do not break

- **The optimizer side never sees the answer.** `tests/test_core.py` fails if anything under
  `evidence/`, `optimize/`, `agents/`, `handoffs.py`, `workflow.py`, `pipelines.py`, `eval_free.py`
  imports `agentcoop.eval` or mentions `load_truth`/`hidden/truth`. Truth-free normalization lives
  in `eval_free.py` for that reason. Grading is always a separate step (`experiments/*analysis*`).
- **`sample` semantics.** Same sample index ⇒ exact replay from cache; a new index is an honest
  re-draw (no-edit control, best-of-N). Agents in `workflow.DETERMINISTIC` ignore the sample.
  Controlled comparisons pin the prefix and re-draw only the affected suffix.
- **Nominal vs actual cost.** Method comparisons use `nominal_usd` (cost without cache, stored in
  node-cache entries); the budget cap uses `actual_usd`. Charge every execution — rejected edits
  and controls included.
- **`unassessed` is never support.** A check that could not decide counts for neither side.
- **External agents run unmodified.** GeneAgent (`external_agents/GeneAgent`, upstream `88fd1ad`,
  NLM public domain) runs in a subprocess; only its transport is swapped by the openai-0.28 shim
  (`external_agents/shims/openai`), and its web-API calls are record/replayed by `sitecustomize.py`
  (`requests-cache`). Don't "fix" GeneAgent's code; change the handoff instead.
- **Shape-2 data boundary.** Only `site_analyst`/`replication_check` touch a site's file, via a
  private copy; the lead, orchestrator and optimizer see site reports only.

## Gotchas

- gpt-6-luna on Chat Completions: function tools require `reasoning_effort="none"` (then
  `temperature`/`seed` work); with reasoning, temperature must be default. Tool use *with*
  reasoning → `llm.responses()` (stateless: `store=False` + encrypted reasoning items).
- The node cache keys on config: to force re-execution after changing an agent's code, bump a
  config field (e.g. `{"hermetic": 1}` on the Codex node) — deleting the run JSON is not enough.
- Codex baseline must be hermetic: isolated `CODEX_HOME` under `.secrets/` (API-key login, never
  the user's `~/.codex`), throwaway `HOME` (else it reads `~/.agents/skills`), `NUMBA_CACHE_DIR`/
  `MPLCONFIGDIR` inside the workspace (else scanpy fails in the sandbox), stdin closed.
- `nohup python ... > log` buffers stdout until exit; use `print(..., flush=True)` for progress.
- Planting uses *core* genes (Hallmark ∩ family GO terms, ≥8 expressed); random Hallmark members
  are unrecognizable (e.g. monocyte "cholesterol" = ANXA5/ACTG1/LGALS3). Cholesterol/heme/DNA
  repair are never plantable in PBMC but remain grader labels.
- Evidence thresholds (`min_agree = ceil(0.875·n_donors)`, GO p ≤ 1e-4) were calibrated on dev
  with empty claims and then frozen; don't retune them on test.
- Statistics unit is the task, never cells or re-draws.
