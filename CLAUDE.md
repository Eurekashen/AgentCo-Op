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

Main tables use ONE backbone for every arm (gpt-6-luna; Codex `-m gpt-6-luna`); other models
only in the appendix. Baselines defined in [docs/baselines.md](docs/baselines.md).

- Collaboration loss is real and structural: naive compositions score 0 (planted tasks) and make
  0 claims on real data (Kang IFN-β case, docs/case_study.md); resampling cannot fix it.
- Answer-free evaluation needs data-grounded checks: report-only judge 0.53 pairwise, rule 0.96.
- Main result, fresh pre-registered held-out shape 2 (`mstest2`+`mstest3`, n=48): ours 0.716 vs
  Codex-as-lead 0.472 (37/0), orchestrator lead 0.460, GEPA (official, same feedback/budget)
  0.609 (22/11, p 0.04), free-form editor 0.731 (tie, 2.4× cost), fixed interface picked on dev
  *with answers* 0.680 (tie), escalation 0.736. Safer acceptances (7.6% harmful vs 11–14%).
- Real-data case (4 held-out validation donors): ours 10/13 claims replicate, 1 program missed;
  GEPA 6/6 by copying feedback into the prompt; editor 28/33 (2× cost); single agents miss 2–3.
- BixBench v1.5 (166 frozen test Qs; Finch/Codex/ReAct + vote/MoA/judge/debate/SC + ours): no
  team beats the best single agent, but the best agent flips dev→test (Finch→ReAct); teams stay
  within 0.04 of it, the dev-chosen agent loses 0.16. Repair loop: no effect (negative result).
- Pooled shape 1: Codex alone 0.79 beats every composition (decline composing when one agent
  sees all data). BAIS-DPTA rejected (taxonomy-dominated metric). Verified-50 is HF-gated.
- Fresh suites now inspected: `heldout`, `mstest`, `mstest2`, `mstest3`, `bixtest`. Any revised
  method needs a new fresh suite. Spend so far ≈ $105 actual (self-imposed cap $120).

## Commands

```bash
pytest -q                                   # offline, ~0.2s; no API key needed
python -m agentcoop.tasks.planted dev 0 16  # build shape-1 tasks (needs data/raw/kang_2018.h5ad)
python -m agentcoop.tasks.multisite msdev 300 316
python experiments/run_candidates.py --suite dev --candidates naive pseudobulk+no_interp+donor
python experiments/redraw.py --suite dev --candidates naive --nodes report --samples 1 2 3
python experiments/phase1_analysis.py --suite dev [--judges]   # grades + Q1/Q2/Q3
python experiments/run_optimizers.py --suite msdev --tasks msdev-300 --methods ours_v31 llm_editor_v2 \
    --budget 5 --max-executions 16 --iterations 8     # the 16-execution protocol
python experiments/replay_check.py $(cat experiments/replay_golden.txt)  # refactoring guard, $0
python -m agentcoop.optimize.rules msdev ours runs/rules/ms_rules.json   # rule library
python experiments/pool_suites.py mstest2 mstest3                 # pooled held-out table
bash external_agents/bixbench_env/setup.sh && python -m agentcoop.tasks.bixbench   # BixBench
python experiments/run_candidates.py --suite bixbench --tasks bixdev --candidates bix_finch
python experiments/bix_analysis.py --split bixdev                  # open-answer + MCQ grading
python -m agentcoop.tasks.real_case && python experiments/case_analysis.py   # real-data case
```

Refactoring rule: a behaviour-neutral change must keep `replay_check.py` green (it re-executes
stored runs from the caches with the spend cap at the current total, so any changed prompt,
handoff output or config shows up as a cache miss or a different artifact). A behaviour change
is a new method version, recorded in `findings.md` with its reason, and old outputs stay
reproducible from their git tag.

Environment: `AGENTCOOP_BUDGET_USD` (hard cap on *actual* spend across the ledger — always set it
for experiments), `AGENTCOOP_MODEL` (backbone for every agent role; default `gpt-6-luna`),
`AGENTCOOP_GRADER_MODEL` (the hidden grader's classifier — pinned, default `gpt-6-luna`, never
follows the backbone), `AGENTCOOP_EXTERNAL_MODEL` (external agents — GeneAgent, site analysts; default `gpt-6-luna`,
does *not* follow the backbone),
`AGENTCOOP_GENEAGENT_SLOTS` (concurrent GeneAgent processes, default 8), `AGENTCOOP_LLM_DB`
(ledger/cache sqlite, default `runs/llm.sqlite`). The API key is read from `OPENAI_API_KEY` or
`.secrets/openai` — never print or commit it. A non-default backbone writes to
`runs/phase1@<model>/`, `runs/phase2@<model>/` (analysis: `--model <model>`); new models need a
row in `llm.PRICES`, and their capabilities are probed once into `runs/model_profiles.json`.

Working rules (from the user): debug on 2–4 tasks before any larger run and scale in steps; give
every framework change a stated reason and evidence in `findings.md`; Codex baselines use the
user's locally logged-in CLI (`~/.codex`, never an API-key login); keep the design and the
conclusions model-agnostic (verify key findings on a second backbone: `gpt-4.1-mini`;
`gpt-4o-mini` fabricated reports as an external agent).

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
- `proxy.py` — OpenAI-compatible local endpoint over `llm.chat_raw`, so litellm harnesses
  (Finch) and GEPA run on the shared backbone, cache, ledger and cap (stop sequences emulated).
- BixBench: `tasks/bixbench.py` (capsule split, truth in hidden/), `agents/bix.py` (Finch via
  proxy, Codex, ReAct analysts in the native `bixbench` conda env), `agents/bix_team.py`
  (vote/MoA/judge/debate), `agents/bix_evidence.py` (rubric, code-level audit, arbiter).
- `optimize/` — `trace.py`/`trace_sites.py` localize loss by tracing evidence probes through
  lineage; `edits.py` declared local edit spaces; `loop.py` propose → controlled replay →
  no-edit control → accept; `baselines.py` reflective-prompt, free-form LLM editor, best-of-N,
  all sharing the same feedback/budget/acceptance; `gepa_baseline.py` official GEPA with the
  same feedback and execution budget; `bix_loop.py` BixBench repair loop (negative result).

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
- Codex baselines borrow only the login from the user's `CODEX_HOME` and are otherwise hermetic:
  `--ignore-user-config`, `-c project_doc_max_bytes=0`, throwaway `HOME` (else it reads
  `~/.agents/skills`), `NUMBA_CACHE_DIR`/`MPLCONFIGDIR` inside the workspace (else scanpy fails in
  the sandbox), stdin closed. `model="native"` runs the user's own config (gpt-6-sol) as a
  product-level arm. Codex cannot be seeded: repeat each task ≥3 times before comparing.
- Callers pass *intent* (`reasoning_effort`, `temperature`); `llm._adapt_chat/_adapt_responses`
  map it per model. Never hardcode a model's quirks at a call site.
- `nohup python ... > log` buffers stdout until exit; use `print(..., flush=True)` for progress.
- Planting uses *core* genes (Hallmark ∩ family GO terms, ≥8 expressed); random Hallmark members
  are unrecognizable (e.g. monocyte "cholesterol" = ANXA5/ACTG1/LGALS3). Cholesterol/heme/DNA
  repair are never plantable in PBMC but remain grader labels.
- Evidence thresholds (`min_agree = ceil(0.875·n_donors)`, GO p ≤ 1e-4) were calibrated on dev
  with empty claims and then frozen; don't retune them on test.
- Statistics unit is the task, never cells or re-draws. Differences below 1e-9 are ties (a mean
  of equal Codex repeats is not exactly equal to one score).
- Caches are first-writer-wins (LLM: INSERT OR IGNORE + read-back; node cache: atomic link), so
  concurrent jobs converge on one value per key. Node keys contain the task artifact (with the
  absolute data path) and the task id, not a data hash: never rebuild a suite in place
  (`build_suite` refuses) and moving the repo invalidates those keys.
- Known, deliberately unchanged (cache-affecting): the integrator prompt shows artifact ids, so a
  meta-only handoff change re-draws the integrator; the grader matches cell types by substring
  (no stored report is affected); planting core genes and the uncovered-signal check both use GO
  BP — the check is answer-free but not independent of how programs were defined.
- BixBench env (`external_agents/bixbench_env/setup.sh`) is made read-only; agent installs land
  in the run's workspace. jupyter_client maps Finch's "python" kernel to `python3`, so both specs
  point into the env. Finch runs before `harness: 2` are stale (runs/phase1/bixbench_stale_*).
- Codex runs: `run_codex` strips the API key from Codex's environment (local login only), copies
  data into the workspace, kills the process group on timeout, and raises when Codex fails
  without a report (never cache an infrastructure failure as an F1 of 0).

- 给我report的时候，始终要使用中文（只有给我report的时候需要，其他所有时候不受影响）