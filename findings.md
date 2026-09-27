# Findings

## Environment (2026-09-27)
- macOS, 10 cores, 17 GB RAM, Python 3.11.5 (miniconda). scanpy 1.11.5, anndata 0.12, statsmodels,
  sklearn, openai 2.8.1 installed. Missing: gseapy, decoupler, pydeseq2, muon, R.
- docker installed, daemon not running. `codex` and `claude` CLIs installed.
- OpenAI key at `.secrets/openai` (bare `sk-...`). `gpt-6-luna` available; Responses and Chat
  Completions both work (~1–2 s trivial call). Pricing (search, to confirm on OpenAI page):
  $0.10/M input, $0.50/M output, cache read $0.01/M.
- Network OK: figshare, GEO, HF, Enrichr, PyPI.

## Review of existing branches (from session 1)
- v2 (`revision`): evidence-driven compiler; no end-to-end committed repair; synthetic regimes not
  backed (same_workflow_single_agent 1.00 on multi_necessary); localization only fires on blocking
  signals → cannot address "runs fine but result mediocre".
- ECPS branch: careful preference comparator; mutations limited to Atomic config; base-cert
  demotion bug via EmpiricalRecord.passed().
- v1 (`main`): gates rarely fired; ablation deltas <1pt; prompts tuned on eval splits.

## GeneAgent (ncbi-nlp/GeneAgent)
- ~670 lines; openai==0.28 `ChatCompletion.create(engine=..., functions=...)`, Azure config at
  module top of main_cascade.py and worker.py. Input: comma-separated gene symbols. Output:
  "Process: <name>" + narrative, after topic + analysis claim verification via 8 web APIs
  (g:Profiler, Enrichr, NCBI E-utils, PubTator agent APIs). No notion of direction, universe,
  condition, or cell type → restrictive interface (good for handoff-loss study).
- Writes to hard-coded relative paths under Outputs/ and Verification Reports/ → run in a temp cwd.

## Data
- Kang 2018 PBMC (IFN-β stim vs ctrl, 8 donors): figshare ndownloader/files/34464122 (pertpy).
- MSigDB Hallmark 2020 via Enrichr geneSetLibrary text endpoint.

## gpt-6-luna API constraints (verified)
- Chat Completions + function tools requires `reasoning_effort="none"`; with "none", `temperature=0`
  and `seed` are accepted. With low/medium reasoning, temperature must be default.
- Pricing confirmed on OpenAI page: $0.10 in / $0.01 cached / $0.125 cache-write / $0.50 out per 1M.
- GeneAgent via shim: ~54 LLM calls, ~$0.007, ~2.5 min per gene set (sleep(1)/step + web APIs).

## Task construction lessons
- figshare ndownloader is behind an AWS WAF challenge; `api.figshare.com/v2/file/download/<id>` works.
- Kang X = raw integer counts, median 1246 UMI/cell; ctrl-only has 8 donors × 6 usable cell types.
- DESIGN FLAW found & fixed: planting random expressed members of a Hallmark set yields
  unrecognizable programs (CD14 "Cholesterol Homeostasis" expressed members = ANXA5/ACTG1/LGALS3/
  PLAUR). Fix: plant only *core* genes = Hallmark ∩ family-defining GO BP terms, and only where
  ≥8 core genes are expressed. Cholesterol/heme/DNA-repair thus never planted in PBMC (not
  expressed) — they remain as grader labels. OXPHOS over-represented (19/78 programs).
- Suites: dev (seeds 0–15) and test (100–115): 63 reproducible programs + 15 decoys; 36 up / 42 down.

## First naive run (dev-000, pre-fix tasks)
- DE was fine (planted genes ranked 0–11 in CD14/CD4), but the naive handoff mixed ~20 noise genes
  into each list; GeneAgent named noise ("cell adhesion", "ribosome"); integrator made 0 claims,
  saying it had no direction and no reproducibility evidence → F1 0. A visible collaboration loss.

## Evidence-check calibration (dev, empty claims; thresholds then frozen)
- Single-gene donor consistency is noisy: 2–12 "8/8 consistent" null genes per ct×direction.
- Requiring a coherent GO BP enrichment (p≤1e-4, overlap≥4) of uncovered consistent genes:
  flags 28/38 reproducible programs, 5 false flags (1 decoy, 4 noise), misses 10 (small programs).
  => partial, imperfect feedback — informative, not an oracle.

## Single-agent baselines are strong (dev-003 trial)
- coder_single (Responses API, persistent kernel, GO/Reactome libs): 71 s, $0.004, F1 0.67, P 1.0 —
  it did donor pseudobulk DESeq2 + per-donor sign checks + enrichment on its own.
- codex_single (Codex CLI 0.155, gpt-6-luna, hermetic HOME): 130 s, $0.012, F1 0.67, same claims.
  Needed: stdin=/dev/null, throwaway HOME (else it reads ~/.agents/skills), NUMBA_CACHE_DIR and
  MPLCONFIGDIR inside the workspace (else scanpy import fails in the sandbox), and
  `-c shell_environment_policy.inherit=all`.
- Implication (as the plan warned via BioAgent Bench): routine DE+enrichment is within a single
  agent's reach; composition must earn its place against these arms.

## Early Q1 signal (5 dev tasks)
- Every Wilcoxon-based variant scores F1 0 (naive, prompt, split, split+stats, de_branch,
  no_interp): the integrator refuses to claim without evidence of donor reproducibility.
- pseudobulk is necessary but not sufficient: pb 0.0; pb+split+stats 0.13; pb+no_interp 0.28;
  pb+split+stats+de_branch 0.41. Loss is multi-stage: replicate unit, direction, context.

## Optimizer design lesson: necessary-but-not-sufficient edits
- Greedy single-edit acceptance on final report evidence stalls: on dev-000 the trace correctly
  implicated the DE node, the pseudobulk edit was tried, but the report stayed empty (downstream
  losses remained), so it was rejected and the loop stopped.
- Fix: *signal propagation depth* — fix the incumbent's uncovered signals as probes and measure how
  far each travels (analysis → handoff → interpreter → view → claimed). An edit that moves probes
  further than both the incumbent and the no-edit control, without losing report evidence, is
  accepted. Answer-free intermediate credit assignment.

## Phase-1 Q1 — full dev results (16 tasks, hidden F1, mean [95% bootstrap CI])
| candidate | F1 | W/L vs naive |
|---|---|---|
| naive | 0.000 | – |
| prompt / split / split+stats / de_branch / no_interp (all Wilcoxon) | 0.00–0.03 | – |
| pseudobulk (alone) | 0.000 | 0/0 |
| pseudobulk+split+stats (keeps GeneAgent) | 0.146 [0.00,0.31] | 3/0 |
| pseudobulk+split+stats+de_branch | 0.606 [0.40,0.79] | 12/0 |
| pseudobulk+no_interp | 0.615 [0.41,0.80] | 12/0 |
| donor (naive + donor-consistency evidence branch) | 0.667 [0.50,0.81] | 14/0 |
| pseudobulk+split+stats+donor | 0.730 [0.56,0.87] | 14/0 |
| pseudobulk+no_interp+donor | **0.757** [0.61,0.88] | 15/0 |
| coder_single (baseline) | 0.661 [0.50,0.81] | 14/0 |
| codex_single (baseline) | 0.739 [0.58,0.87] | 14/0 |
| per-task oracle over edits | 0.818 | |
- RQ1 holds for this shape: same components, interface-only edits: 0 → 0.76 F1.
- The dominant loss is missing *reproducibility evidence* at integration: the integrator refuses
  ("does not show donor-level effects") even with pseudobulk stats — stochastically (dev-006/011
  refused, dev-003/015 claimed for the same kind of input).
- GeneAgent names clean lists correctly ("Fatty acid β-oxidation", "Mitochondrial respiratory
  chain…") but adds no F1 over the integrator reading stats itself (0.730 with vs 0.757 without).
- Best composition ≈ strongest single agent (0.757 vs Codex 0.739): in this shape the external
  interpreter does not bring a capability the single agent lacks. Honest negative for "composition
  beats Codex" here.
- CAVEAT: the donor_check component was added to the edit space by me after reading dev failures
  (human-in-the-loop on dev). Its value must be shown on the held-out test suite, found by the
  optimizer autonomously.

## Optimizer debug (dev-000) with propagation acceptance
- pseudobulk accepted on "signal propagation" (depth 0→2, control 0) — mechanism works.
- Then split / max_padj each rejected (2→2): direction_dropped and signal_diluted shared a stage
  level. Fixed by finer stages (direction 2 < diluted 3 < context 4 < integration 5 < covered 6)
  and a visited-fingerprint set (it was trying to revert the accepted pseudobulk edit).
