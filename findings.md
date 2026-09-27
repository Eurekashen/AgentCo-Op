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

## Shape 2 (multisite) — first observations (msdev-300/301)
- Motivation: when data cannot be pooled (IRB/privacy; consortium studies), no single agent can
  run the analysis; the collaboration interface decides the answer. Codex-style comparison =
  an autonomous orchestrator that can send *any* request to the same site agents
  (`agents/orchestrator.py`), same data boundary.
- Naive (forward the study question to each site): sites refuse to report local findings
  ("only site A available… replication not assessable → no effects reported"); lead makes 0
  claims; F1 0, while the evidence scan finds replicated OXPHOS in CD8/NK at both sites.
- Structured template alone does not fix it: sites return {"findings": []} — the request still
  carries the global replication criterion. The loss is *task decomposition* in the request, so
  added `scope: forward|local` to the site_request policy (and to its declared edit space).

## Optimizer end-to-end on dev-000 (no answers used): hidden F1 0 → 1.0
- Accepted in sequence: pseudobulk (depth 0→2, control 0), split_direction (2→3), n=15 (3→4,
  de-dilution), carry direction/method/stats (4→5), DE-summary evidence branch (report evidence:
  2 supported, judge A). 5 iterations, $0.975 nominal, 19 min.
- Without propagation depth none of the first four edits would be accepted (report unchanged).
- Waste: attempts to revert accepted params (wilcoxon, split=False) → now locked after acceptance.
- msdev-300 ms_local: both sites report detailed local findings (both chose donor pseudobulk on
  their own); lead claims UPR↓ CD14 (TP) and ribosome↑ NK (site-specific decoy) → F1 0.4. The
  evidence profile marks the NK claim unassessed ("supported at one site only") — visible to the
  optimizer without the answer.
- msdev smoke (2 tasks, hidden F1): naive 0/0; structured-only 0/0; local .40/0; local+structured
  .50/0; +weak .67/1.00; +xverify .50/.67; +repl .40/.86. Evidence counts track F1 (supported↑,
  decoy claims land in unassessed).

## Phase-1 Q2/Q3 (dev, 16 tasks) — answer-free evaluation works only when grounded in data
Q2 pairwise agreement with the sign of hidden ΔF1 (675 pairs with |ΔF1| ≥ 0.1; tie = wrong):
| evaluator | accuracy | decided | error when decided |
|---|---|---|---|
| report-only LLM judge | **0.50** (chance) | 0.87 | 0.42 |
| evidence judge (K3-style: rubric first, scorepad, both orders, data checks) | 0.85 | 0.92 | 0.08 |
| evidence rule (deterministic, same checks) | **0.95** | 0.98 | 0.03 |
Selection among the 13 fixed edits (Copeland, incumbent on ties): report-only 0.708, evidence
judge 0.700, evidence rule 0.742, oracle 0.818 (naive 0.000).
Q3 re-draws (integrator only, 4 draws):
- naive: every draw F1 0 → no amount of resampling fixes a structural loss.
- pseudobulk+no_interp: single 0.615, draws mean 0.696, per-task std 0.115; best-of-4 picked by
  evidence rule 0.784, evidence judge 0.766, report-only 0.540 (worse than one draw), oracle 0.791.
- pseudobulk+no_interp+donor: std 0.039 — the evidence branch also removes integrator variance.
Decision gate (plan §10): room exists (Q1), answer-free evidence identifies it (Q2), it is not
resampling (Q3) → proceed to the closed loop; judge must be evidence-grounded (report-only fails).

## Bug found on test and fixed (before any test result was used)
- After un-truncating uncovered gene lists, probes contained every donor-consistent gene (80–127,
  mostly chance hits); trace required ≥30% significant → pseudobulk edits scored depth 0→0 and the
  optimizer accepted nothing on test-101..103. Fix: probe = the coherent program (union of genes
  driving the GO enrichment). Calibration unchanged (28 flagged / 5 false / 10 missed); probe
  median 8 genes, 96% of them planted. Test outcomes from the buggy run were deleted and re-run.
- Note: phase-1 Q2 evidence-judge packets were built with the old example genes (first 10
  alphabetical consistent genes); evidence_rule uses counts only and is unaffected.

## Test-set hygiene
- While debugging the optimizer's early stop I printed test-101's truth and changed the trace's
  analysis-stage criterion (nominal presence instead of gene-level FDR — a set-level program can be
  present while few genes pass FDR, e.g. an 8-gene UPR↓ in NK). The change is generic, but it was
  made after looking at a test task, so `test` is demoted to a validation suite and a fresh
  `heldout` suite (seeds 500–515, never inspected) is used for all final claims.
- Also added edit grids for the DE-summary handoff (max_padj, k) and donor_check (max_padj), so
  weak-but-consistent programs have a path downstream.

## Validation (test-101/102) after probe fix
- test-102: split (depth 4→7) → pseudobulk (7→11) → carry stats (report evidence: 3 supported).
- test-101 stalled after split. Three loop defects fixed:
  1. `tried` ignored context: an edit rejected before another was accepted was never retried →
     key tried edits by (incumbent fingerprint, edit).
  2. only the top-2 findings got a menu → every located finding contributes (one menu per
     location, heaviest finding first), top-K overall.
  3. no alternative routes: a signal lost at a handoff while still present in the analysis output
     now also yields a lower-weight `bypass_available` finding at the consumer (evidence branch).
- ms_codex_orchestrator trial (msdev-300): Codex CLI as study lead via a loopback site-query
  tool: 6 site requests incl. targeted follow-ups ("evidence extract for two candidate programs",
  "exhaustive inventory"), no data-access violations, F1 0.67, $0.09 nominal. Strong baseline.

## Shape-2 optimizer trial (msdev-300/301, no answers used)
- msdev-300: accepted report_weak=True (report evidence, judge A) → hidden F1 0.80 (best fixed
  candidate 0.67; Codex orchestrator 0.67; in-house orchestrator 0.50).
- msdev-301: report_weak → method=pseudobulk → template=structured (last on propagation) → 0.86.
- It found a route I had not designed: asking sites to also report weak effects "judged jointly"
  is itself enough to make them share local findings (without scope=local).
- Waste: 5 prompt rewrites at the lead, each a "new" edit because the key included the text →
  prompt edits now keyed per incumbent (one rewrite per incumbent).

## Shape-2 Q1 (msdev, 16 tasks, hidden F1)
| candidate | F1 [95% CI] | W/L vs naive |
|---|---|---|
| ms_naive (forward study question) | 0.000 | – |
| ms_structured (format only) | 0.000 | – |
| ms_local | 0.514 [0.36,0.66] | 13/0 |
| ms_local+structured | 0.533 | 13/0 |
| +weak | 0.566 | 11/0 |
| +repl | 0.642 | 15/0 |
| +xverify | 0.678 (P 0.94) | 15/0 |
| **+weak+repl** | **0.766** [0.63,0.88] | 15/0 |
| +weak+xverify | 0.703 | 14/0 |
| ms_orchestrator (in-house autonomous lead) | 0.423 [0.30,0.54] | 12/0 |
| **ms_codex_orchestrator (Codex CLI lead)** | **0.434** [0.30,0.55] (P 0.73, R 0.32) | 12/0 |
| per-task oracle | 0.863 | |
- When data cannot be pooled, an autonomous strong agent coordinating the same site agents is
  precise but misses weak replicated effects (no cross-site replication design); the evidence-
  shaped interface (local scope + structured + weak effects + replication check) wins by ~0.33.
- Opposite of shape 1 (pooled data), where single agents beat every composition. Together: the
  value of composing is conditional on the data-access structure — which is the measurable claim.

## HELD-OUT shape 1 (heldout, 16 never-inspected tasks, hidden F1, budget $1.0 nominal)
| method | F1 [95% CI] | W/L vs naive | $nominal/task |
|---|---|---|---|
| naive | 0.000 | – | 0.044 |
| best_of_n (re-draws, evidence-picked) | 0.000 | 0/0 | 0.764 |
| reflective_prompt (GEPA-style, same feedback/acceptance) | 0.389 [0.19,0.60] | 8/0 | 1.006 |
| ours (trace + typed edits + control + propagation) | 0.678 [0.53,0.82] | 15/0 | 0.663 |
| ours_no_propagation | 0.698 [0.53,0.85] | 14/0 | 0.621 |
| llm_editor (free-form, same feedback + evidence acceptance) | 0.683 [0.51,0.84] | 14/0 | 0.267 |
| fixed pb+no_interp+donor (best dev fixed) | 0.696 | 15/0 | 0.000 |
| fixed pb+split+stats+donor | 0.759 [0.61,0.89] | 15/0 | 0.010 |
| coder_single | 0.641 [0.44,0.82] | 12/0 | 0.007 |
| codex_single | **0.787** [0.70,0.87] | 16/0 | 0.014 |
| ours + answer-free select vs codex | 0.731 [0.60,0.86] (picked codex 2/16) | 16/0 | 0.677 |
Holds: resampling cannot fix structural loss (best-of-N 0); prompt-only editing is far weaker than
structural/handoff edits (0.39 vs ~0.68); with pooled data a strong single agent (Codex) is best.
DOES NOT HOLD (honest negatives):
- Propagation depth adds nothing on held-out (0.678 vs 0.698 without). Once the edit space has a
  direct evidence branch (donor_check) the loss is bypassed in one step, so sequential repair of
  the handoff chain — where propagation is necessary (dev-000) — is not needed.
- Localized typed edits do not beat a free-form LLM editor given the *same evidence feedback and
  acceptance* (0.678 vs 0.683), and cost 2.5× more. In shape 1 the load-bearing component is the
  evidence-grounded acceptance, not localization. Testing that directly: llm_editor with a
  report-only judge (typical LLM-as-judge optimizer) — running.

## Shape-2 optimizer on msdev (16 tasks) with the first site checks: 0.403 — diagnosed & fixed
- 5 tasks stopped at iteration 0 with "no findings" and F1 0: the naive report had no claims and
  the site-level uncovered scan (4/4 donors at *each* site + both sites must find it) detected
  nothing → the optimizer treated "nothing detected" as "nothing to fix".
- Fix 1: checks pool per-donor gene-level fold changes across sites (summary statistics — what a
  federated meta-analysis would share), detect coherent programs over 8 donors, and require each
  site's donors to lean the same way (≥3/4). Claim support: pooled supported + each site ≥3/4.
  Re-calibrated on msdev with empty claims: 34 flagged / 4 false (1 site-specific) / 11 missed.
- Fix 2: an empty report is itself a finding (`no_claims`), localized by lineage (evidence reached
  the lead → lead; nothing reached it → the requests). Acceptance still needs evidence gains, so a
  genuine null result cannot be "fixed".
- mstest optimizer runs made with the old checks were stopped before any result was read and moved
  to runs/phase2/_stale/; they will be re-run.
- NOTE: heldout shape-1 'ours' ran before the no_claims finding existed; ours will be re-run on heldout with the frozen final code so both shapes report the same method.
