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
