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

## HELD-OUT shape 1 — ablations (correction to the previous entry)
| method | F1 [95% CI] | $nominal |
|---|---|---|
| random_loc (same typed edit space + acceptance, random location) | 0.522 [0.32,0.73] | 1.03 |
| ours (trace-localized) | 0.678 | 0.66 |
| llm_editor (free-form, evidence acceptance) | 0.683 | 0.27 |
| llm_editor_report_judge (free-form, LLM-as-judge acceptance; 60 edits accepted vs 21) | **0.719** [0.57,0.86] | 0.34 |
- Localization matters vs random locations within the same space (+0.16).
- CORRECTION: "evidence-grounded acceptance is load-bearing" does NOT hold for optimization in
  shape 1. With a well-documented component/handoff library the free-form editor's proposals are
  mostly improvements, so even an acceptance filter at chance (report-only judge, phase-1 Q2 0.50)
  does no harm. The load-bearing asset in shape 1 is the explicit, executable library of handoffs
  and evidence components (incl. donor_check) — not the search or the acceptance rule.

## Shape-2 validation with compound escalation (msdev, final code)
- ours 0.697 [0.55,0.83] (0.648 without compound; 0.403 with the first site checks); +select vs
  Codex orchestrator 0.720; best fixed 0.766; orchestrators 0.42/0.43. msdev-305: 0 → 1.0 via a
  compound (scope=local + report_weak) after every single edit failed alone.
- CODE FROZEN here for the final held-out runs (tag v3-final-eval).
- BUG (ablation only, found after freeze): random_loc excluded edges leaving `task`, which in
  shape 2 are exactly the site-request edges → it saw only 2 lead edits and stopped (F1 0 on all
  mstest tasks, $0.03). Harmless in shape 1 (task->de has no edits; heldout random_loc was
  budget-bound). Fixed (all edges; compound escalation also allowed in random mode so the ablation
  differs from ours only in localization); mstest random_loc re-run. `ours` code unchanged.

## HELD-OUT shape 1 — final (frozen code; editors re-run at 8 iterations)
naive 0.000 · best_of_n 0.000 · reflective_prompt 0.389 · random_loc 0.522 · coder 0.641 ·
ours_no_propagation 0.671 · **ours 0.678** [0.53,0.82] · fixed pb+no_interp+donor 0.696 ·
llm_editor (evidence acc.) 0.699 · ours+select-vs-Codex 0.731 · **llm_editor_report_judge 0.746**
[0.59,0.88] · fixed pb+split+stats+donor 0.759 · **Codex 0.787** [0.70,0.87].
Shape-1 verdict: with a documented component/handoff library, free-form LLM editing is the best
automated optimizer; our localized search is on par or slightly below; with pooled data a strong
single agent is best. Localization > random locations (+0.16); propagation ≈ no effect.

## Ops note (mstest final runs)
- Running 14 optimizer task-runs at once (two processes, parallel 8 + 6) oversubscribed the
  machine: load 123, swap 11.7/12.3 GB (each site-agent kernel ≈ 300 MB). Killed, relaunched
  the priority group (ours, llm_editor ×2, best_of_n) at parallel 4, then random_loc, ours_rules,
  ours_no_propagation, reflective_prompt. ours_no_compound dropped from the held-out set (dev
  already shows its effect: 0.648 → 0.697); noted as a deviation from the pre-registered list.

## Model-specific assumptions audit (2026-09-27, prompted by the model-agnostic requirement)
1. backbone is a hardcoded constant (llm.DEFAULT_MODEL) — no config;
2. gpt-6-luna quirk baked into llm.chat and the GeneAgent shim (tools ⇒ reasoning_effort="none",
   temperature 0) — a non-reasoning model rejects reasoning_effort;
3. roles pass reasoning_effort low/medium directly — unsupported by non-reasoning models;
4. Responses calls always send reasoning + encrypted reasoning content;
5. price table only has gpt-6-luna (other models would be mis-costed);
6. the hidden grader's family classifier uses the same default model as the system under test —
   the evaluation instrument would change with the backbone (most serious);
7. Codex baselines pass -m gpt-6-luna via an API-key CODEX_HOME.
Plan: roles declare intents; a capability layer maps them per model (probed once, cached);
AGENTCOOP_MODEL selects the backbone; AGENTCOOP_GRADER_MODEL pins the grader (default gpt-6-luna,
the instrument used for every result so far, so all numbers stay comparable).
- Found while making backbones switchable: the node cache key did not include the model, so a
  run under another backbone would silently reuse gpt-6-luna outputs. Fixed: model-dependent nodes
  key on the backbone (default-backbone keys unchanged for replayability). Test added.
- Capability layer verified: replaying earlier gpt-6-luna work hits the LLM cache exactly
  (actual $0, identical F1) — the refactor does not change any existing result.
- gpt-4o-mini profile (probed): non-reasoning (no reasoning params; temperature supported).

## Codex on the user's local login (2026-09-27)
- Works with `CODEX_HOME=~/.codex` (auth only), `--ignore-user-config`, throwaway HOME,
  `-c project_doc_max_bytes=0`, `-m <backbone>`; asked whether any user instruction files were in
  context it answered NO (with and without the doc flag). Billed to the ChatGPT plan: ledger records
  API-equivalent nominal cost, $0 actual. `model="native"` arm = the user's own config (gpt-6-sol).
- The earlier API-key Codex home (.secrets/codex_home) was removed. Earlier Codex results used the
  same CLI version and model (gpt-6-luna) with API-key auth — auth does not change behaviour, so
  they remain valid; new runs use the local login.
- RUN-TO-RUN VARIANCE: dev-003 local-login Codex: 0 claims (found OXPHOS nominally but applied a
  stricter correction) vs 0.67 F1 in the earlier run. Every Codex comparison so far used ONE run per
  task — too noisy. Baselines that cannot be seeded need ≥3 repeats per task (feasibility doc), on
  small subsets first.

## Second backbone, first attempt: gpt-4o-mini (msdev-301/305, ms_naive only before timeout)
- Site agents' reports came back as "error": (1) OUR brittleness — the wrapper accepted only a JSON
  object and rejected a JSON array; (2) MODEL capability — its code crashed repeatedly and it then
  submitted an invented placeholder report ("cell_type1", "GeneA", p=0.001). The evidence checks
  would mark such claims ungrounded/unassessed, but no workflow can recover data an agent never
  analysed.
- Fix (justified by model-agnosticism of the interface): reports that are JSON arrays are wrapped
  ({"findings": [...]} / {"claims": [...]}) instead of discarded. Cached gpt-6-luna outputs are
  unaffected.
- Decision: gpt-4o-mini measures a capability floor (it cannot act as a coding agent here), not
  whether the design is model-agnostic. Second backbone for the robustness check → gpt-4.1-mini
  (non-reasoning, different family/generation from gpt-6-luna, reliable tool use; $0.40/$1.60).
  The 4o-mini observation is kept as a separate finding.

## Roles separated: framework model vs external-agent model (justification)
- gpt-4.1-mini as a *site coding agent* spent all 30 steps fighting the pydeseq2 API and never
  reported — a property of the external agent, not of our method. "Model-agnostic" therefore means
  (A) the framework's own LLM roles (integrator/lead, judges, optimizer ranking/prompting, and the
  single-agent baselines that compete with it) can be swapped, and (B) separately, what happens
  when the external agents' model changes. AGENTCOOP_MODEL = framework; AGENTCOOP_EXTERNAL_MODEL =
  external agents (default gpt-6-luna, fixed). Node cache keys on the model each role runs.

## Condition A, small subset (framework gpt-4.1-mini, external agents gpt-6-luna)
| task | candidate | F1 (4.1-mini framework) | F1 (all luna) |
|---|---|---|---|
| dev-003 | naive | 0.60 (6 claims) | 0.00 (0 claims) |
| dev-011 | naive | 0.00 (6 claims) | 0.00 (0 claims) |
| dev-003 / dev-011 | pb+no_interp+donor | 0.67 / 0.75 | 0.67 / 1.00 |
| msdev-301 / 305 | ms_naive | 0.00 / 0.00 | 0.00 / 0.00 |
| msdev-301 / 305 | best fixed interface | 0.80 / 1.00 | 1.00 / 1.00 |
- "naive F1 = 0" was partly a gpt-6-luna trait (it refuses without reproducibility evidence);
  gpt-4.1-mini instead guesses directions from mixed lists and over-claims (dev-011: 6 claims, 0 TP).
  The collaboration loss changes form (recall loss vs precision loss) but interface edits repair
  it under both models. The answer-free checks are deterministic, so they see both failure forms.
- Q2 on a second judge model (60 fixed pairs, seed 0, |ΔF1| ≥ 0.1): report-only judge luna 0.52 /
  gpt-4.1-mini 0.67; evidence judge luna 0.93 / 4.1-mini 0.80; deterministic evidence rule 0.95
  (model-free). The ordering evidence-grounded > report-only holds under both models; absolute
  judge accuracy is model-dependent, which is why acceptance rests on the model-free rule.

## Evidence-rule inflation under a verbose integrator (gpt-4.1-mini, dev-003) — diagnosis & decisions
- The optimizer raised the report from 6 to 10 claims; "supported" 3 → 6, hidden F1 0.60 → 0.43.
  The self-contained support check passed spurious/misnamed claims (CD4 "translation", NK
  "antigen processing", FCGR3A "IFN-γ") because many genes shift together in the data.
- ADOPTED: competitive support — claimed genes must change more (median over donors, claimed
  direction) than the 95th percentile of 200 random measurable gene sets of equal size (seeded by
  the claim; deterministic, model-free). Q2 on the 619 luna dev pairs: 0.956 → 0.956 (no loss);
  dev-003: the three spurious supports become unassessed; the three supported ones are exactly the
  planted programs.
- REJECTED (validated, not assumed): (i) dropping the supported reward: luna pair accuracy
  0.947 → 0.800; (ii) penalizing unassessed claims (λ=0.5/1): 0.956 → 0.913/0.803 — and it
  violates "unassessed counts for neither side" (many true small programs are unassessed);
  (iii) evidence-gated reporting (only supported claims reported): luna candidates F1 0.273 →
  0.246 (worse 12, better 2) while helping 4.1-mini — model-dependent, so not a design choice.
- Remaining limit (honest): with an over-claiming integrator, extra claims the data cannot decide
  lower precision and no answer-free signal can tell right from wrong among them.
- Budget: a $ budget buys a different search depth per model price (4.1-mini exhausted $1 in
  ~2 iterations). Added `max_executions` (luna held-out median ≈ 15 executions at $1); cross-model
  comparisons use max_executions=16 with a $5 safety cap.
- BUG: execution-budget counter was never initialized (text replacement missed) → every ours_v31 run crashed. Fixed; added an offline end-to-end optimize() test so loop-level crashes are caught before any API run.

## Cross-model, fixed references (4 tasks: dev-003, dev-011, msdev-301, msdev-305; external agents gpt-6-luna)
| framework model | naive | best fixed interface | same-model single agent (coder / in-house lead) |
|---|---|---|---|
| gpt-6-luna | 0.00 | 0.92 | 0.52 |
| gpt-4.1-mini | 0.15 | 0.80 | 0.33 (coder 0.00 on both dev tasks) |
- Directional (n=4): a weaker framework model hurts the single agent much more than the
  composition with a good interface (external experts carry the analysis; the framework integrates
  and coordinates) — composition is *less* sensitive to the framework model.

## Cross-model optimizer (ours_v31: competitive support + 16-execution budget), 4 tasks
| task | luna framework | 4.1-mini framework | best fixed (luna / 4.1) |
|---|---|---|---|
| dev-003 | 0.67 | 0.25 (naive 0.60) | 0.67 / 0.67 |
| dev-011 | 1.00 | 0.00 | 1.00 / 0.75 |
| msdev-301 | 0.89 | 1.00 | 1.00 / 0.80 |
| msdev-305 | 1.00 | 0.67 | 1.00 / 1.00 |
- Shape 2 works under both framework models (4.1-mini mean 0.84 vs same-model lead 0.67).
- Shape 1 under 4.1-mini fails for two different reasons:
  (a) dev-011 — greedy search: the fix needs split_direction AND carrying direction to the
      integrator; split alone makes a guessing integrator produce MORE contradicted claims (both
      up- and down-lists called "up"), so it is rejected, and compound escalation never fires
      because other single edits remain. → ADD one-step lookahead (trace the rejected edit's own
      run; if the loss surfaces at a location the incumbent's trace did not implicate, try the
      pair next). Unit test added. Evaluated as ours_v32.
  (b) dev-003 — naming: the optimizer made the report better supported (contradicted 2 → 0,
      supported 2 → 6) but gpt-4.1-mini names many programs wrongly; claim genes were selected as
      top DE genes on the same data, so they also beat random sets (double dipping). A GO-coherence
      requirement was tested and REJECTED (luna pair accuracy 0.956 → 0.937, and it does not fix
      dev-003). Program *naming* is invisible to data checks: an honest limit of answer-free
      feedback with a framework model that names poorly.
- ours_v32 (lookahead) vs ours_v31 on the 4 tasks: luna identical on all 4; 4.1-mini dev-011
  0.00→0.25, msdev-301 1.00→0.80, others equal (mean 0.48→0.49). No measurable benefit → kept as an
  option, OFF by default (default method stays v3.1). Null result recorded.
- Comparison optimizers now honour the same execution budget (fairness).

## Shape-2 search comparison (msdev-300/301/302/305, luna framework, 16-execution budget)
| method | 300 | 301 | 302 | 305 | mean | $/task |
|---|---|---|---|---|---|---|
| ours_v31 | 0.80 | 0.89 | 1.00 | 1.00 | **0.92** | 0.23 |
| llm_editor (evidence acceptance) | 0.57 | 0.89 | 1.00 | 0.80 | 0.82 | 0.47 |
| llm_editor_report_judge | 0.80 | 0.86 | 1.00 | 0.67 | 0.83 | 0.61 |
| reflective_prompt | 0 | 0 | 0 | 0 | 0.00 | 0.12 |
| best fixed interface (hand-designed) | 0.67 | 1.00 | 1.00 | 1.00 | 0.92 | – |
| Codex orchestrator (1 run, API-key auth) | 0.67 | 0.40 | 0.50 | 0.00 | 0.39 | – |
- Unlike shape 1, where composition is necessary the localized typed search beats free-form
  editing given the same feedback (2/0/2 vs each editor) at half the cost, and matches the
  hand-designed interface without answers. n=4: directional → scale to 8 msdev tasks next.

## Shape-2 search comparison scaled to 8 msdev tasks — CORRECTION of the 4-task signal
| method | 300 | 301 | 302 | 303 | 304 | 305 | 306 | 307 | mean |
|---|---|---|---|---|---|---|---|---|---|
| ours_v31 | .80 | .89 | 1.0 | .50 | .50 | 1.0 | .67 | .50 | 0.732 |
| llm_editor (evidence) | .57 | .89 | 1.0 | 1.0 | .80 | .80 | .80 | .22 | 0.760 |
| llm_editor_report_judge | .80 | .86 | 1.0 | .00 | .80 | .67 | .67 | .50 | 0.661 |
| best fixed (hand-designed) | .67 | 1.0 | 1.0 | 1.0 | .00 | 1.0 | .80 | .57 | 0.755 |
| in-house orchestrator | .50 | .40 | .50 | .00 | .80 | .00 | .00 | .50 | 0.338 |
| Codex lead, mean of 3 (local login) | .33 | .79 | .60 | – | – | .67 | – | – | 0.598 (n=4) |
- ours vs llm_editor: Δ −0.028, W/L 3/3 (p 0.84) — the 4-task advantage did NOT replicate.
  ours vs in-house lead: +0.394, 6/1, p 0.047. ours vs Codex lead (3-repeat mean): +0.324, 4/0, p 0.125.
- Codex repeats vary a lot per task (300: 0.5 / 0.0 / 0.5): single-run baselines were noisy.
- Revised claim: the contribution is NOT the search algorithm (localized ≈ free-form editing in
  accuracy; localized is cheaper and auditable). What holds: (1) making the collaboration interface
  an explicit, executable, optimizable object (handoff/request policies, evidence branches,
  cross-site verification); (2) answer-free, data-grounded acceptance; (3) when data cannot be
  pooled, evidence-optimized collaboration clearly beats autonomous orchestration (in-house, Codex).
- Method frozen for held-out as v3.1 (tag v3.1-eval): competitive support, 16-execution budget,
  lookahead off. Held-out mstest in steps of 8 tasks.

## HELD-OUT shape 2, step 1 (mstest-400..407, v3.1 frozen, 16-execution budget)
| method | mean F1 (n=8) |
|---|---|
| naive | 0.000 |
| Codex lead (user's local login, mean of 3 repeats) | 0.420 |
| in-house autonomous lead | 0.476 |
| ours_v31 | 0.655 |
| best fixed dev interface | 0.683 |
| llm_editor (evidence acceptance) | 0.733 |
- ours vs Codex lead +0.235 (6/0, p 0.031); llm_editor vs Codex lead +0.313 (7/0, p 0.016);
  llm_editor vs in-house lead +0.257 (6/1, p 0.047); ours vs llm_editor −0.078 (3/4, ns) —
  consistent with dev. Codex repeats vary widely (403: 0.8 / 0.5 / 0.0).

## HELD-OUT shape 2, all 16 tasks (mstest-400..415, v3.1 frozen) — FINAL
| method | F1 [95% CI] | W/L vs Codex lead (p) | $nominal/task |
|---|---|---|---|
| naive | 0.000 | 0/14 | 0.01 |
| Codex CLI as study lead (local login, mean of 3 repeats s1–s3) | 0.491 [0.35, 0.63] | – | 0.03 |
| (Codex lead, earlier single run s0 — API-key login, same CLI/model) | 0.602 | | |
| in-house autonomous lead | 0.526 [0.38, 0.67] | 7/4 (0.37) | 0.03 |
| best fixed dev interface | 0.714 [0.64, 0.78] | 12/3 (0.017) | 0.03 |
| **ours_v31** | **0.736** [0.61, 0.84] | **12/2 (0.005)** | 0.26 |
| free-form LLM editor (evidence acceptance) | 0.748 [0.65, 0.84] | 12/3 (0.010) | 0.49 |
- ours vs editor: 5/4, p 1.0 (tie, half the cost). ours vs in-house lead 9/1 (p 0.023).
- Step 2 alone (408–415): ours 0.818, editor 0.768, fixed 0.745, Codex mean 0.563, in-house 0.576.
- Even the 4-run Codex mean (incl. s0) is 0.519: the gap does not depend on which repeats count.
- Failure: mstest-404 ours 0 (see progress 2026-09-28): the lead refuses without cross-site
  replication; every request edit and pairwise compound at the request edge was rejected (0 or only
  unassessed claims), and the no_claims finding never implicated the lead. Needs a request edit
  AND lead-side replication together. Documented, not fixed (frozen method).

## Decision quality of the optimizers (graded post hoc; experiments/edit_quality.py)
Instrumentation (no behaviour change): every Attempt now keeps its candidate report and every
Outcome its initial report. Finished runs were replayed from the caches into runs/phase2_attempts
(identical workflows and attempt sequences for all 32 runs, ≈$0.002).
| method (mstest, 16 tasks) | tries | accepted | helpful | harmful | missed (rule rejected, F1 ↑) | outranked by sibling |
|---|---|---|---|---|---|---|
| ours_v31 | 120 | 46 | 21 | **1 (2%)** | 9 of 74 rejected | 0 |
| llm_editor | 171 | 24 | 19 | 3 (12.5%) | 24 of 121 rejected | 26 |
- The free-form editor proposes more helpful candidates (67/171 raise F1 vs 30/120), but keeps
  fewer; its missed improvements are mostly "incumbent has more data-supported claims, one of
  which is wrong" — an inherent limit of answer-free acceptance (support ≠ correct naming).
- Localized typed edits make acceptance safer (harmful 2% vs 12.5%): small, attributable steps
  change the evidence monotonically. This is the measurable advantage of localization in shape 2,
  not higher final F1.

## Fairness fix: best_of_n now honours the execution budget
best_of_n looped `while nominal < max_usd and k <= 3·iterations` — it ignored max_executions (up to
24 re-draws against 16 for every other method). Now uses Budget.exhausted(usd, executions).
Affects only future best_of_n runs (shape-1 heldout best_of_n ran under the $ budget as all v3
methods did).

## Free-form editor proposals are often invalid workflows (model-dependent) → editor v2
- Found via the gpt-4.1-mini check: on mstest-400 the editor stopped with "no proposal" after one
  acceptance — all 3 next proposals were rejected by the validator ("workflow has a cycle": it wired
  site_A <-> site_B cross-verification both ways). Invalid proposals had been dropped silently.
- Replaying the gpt-6-luna editor runs with logging (cache, $0): invalid 12/183 attempts on mstest,
  11/98 on msdev — all cycles. So the v1 editor doc ("cross_verify: site_analyst -> another
  site_analyst") under-specifies the DAG constraint for every model; weaker models hit it harder.
- Change (baseline fairness + model-agnosticism): `llm_editor_v2` = documented acyclicity
  constraint with the verifier-node pattern the typed edit uses (site_A -> verify_B ->
  lead) + one repair round that returns the validator's error. v1 kept unchanged (replay
  identical) for the as-run held-out numbers; invalid proposals are now logged for both versions.
- ours_escalate uses editor v2.

## Held-out shape-2 mechanism ablations, batch A (mstest-400..403, v3.1 settings, 16 executions)
| method | 400 | 401 | 402 | 403 | mean |
|---|---|---|---|---|---|
| reflective_prompt (prompt-only, GEPA-style) | 0 | 0 | 0 | 0 | 0.000 |
| random_loc (same edit space, random locations) | .86 | .67 | .86 | .50 | 0.720 |
| ours_rules (msdev rule library first) | .86 | .80 | .67 | .50 | 0.706 |
| ours_v31 | .86 | .80 | .67 | .50 | 0.706 |
| llm_editor | .75 | .50 | .86 | .80 | 0.727 |
- Prompt-only optimization cannot repair a request-policy loss (sites refuse to report local
  findings when handed the study question) — consistent with msdev.
- In shape 2 random locations ≈ localization (n=4): the shape-2 edit space is small (request
  params + lead structure), so 16 executions cover most of it in any order. Decision quality also
  similar (random_loc 1 harmful / 9 accepted; ours_rules 1 / 13).
- Rules transfer: ours_rules identical F1 to ours on all 4 and no cost saving (nominal 1.21 vs
  1.16 over 4 tasks) — the LLM ranker already puts the rule edits first. Null.
- Batch B (404–407) done → n=8 (400–407):
  | method | mean F1 (n=8) | vs ours W/L (p) | harmful / accepted |
  |---|---|---|---|
  | reflective_prompt | 0.000 | 0/7 (0.02) | – |
  | random_loc | 0.629 | 1/2 (0.75) | 2 / 15 |
  | ours_rules | 0.655 | identical on 8/8 | 1 / 20 |
  | ours_v31 | 0.655 | – | – |
  | llm_editor | 0.733 | 4/3 (0.61) | – |
  | Codex lead (mean of 3) | 0.420 | 0/6 (0.03) | – |
- Localization vs random in shape 2: a small, non-significant edge (0.655 vs 0.629; random has
  twice the harmful acceptances). The shape-1 held-out gap (+0.16) does not carry over: with ~10
  typed edits in the shape-2 space, order matters little under a 16-execution budget.
- ours_rules stopped at n=8 (deviation from the full-16 plan): identical final F1 to ours on all
  8 tasks — the rule library changes the order of tries but never the outcome here.

## Second framework model on held-out shape 2 (gpt-4.1-mini framework, gpt-6-luna external agents)
mstest-400..407 (first 8 by index, not selected); editor v1 stopped after 4 (invalid proposals).
| method | 400 | 401 | 402 | 403 | 404 | 405 | 406 | 407 | mean |
|---|---|---|---|---|---|---|---|---|---|
| naive | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000 |
| same-model autonomous lead | .67 | 0 | .40 | .50 | 0 | .57 | 0 | 1.0 | 0.392 |
| best fixed dev interface | .86 | .40 | .86 | .86 | .67 | .75 | .67 | .67 | 0.715 |
| **ours_v31** | .86 | .80 | .86 | 1.0 | 1.0 | .89 | .67 | .67 | **0.842** |
| llm_editor v2 | .86 | .50 | .67 | .50 | .67 | .75 | .67 | .67 | 0.659 |
| llm_editor v1 | .86 | .29 | 0 | 0 | | | | | 0.286 (n=4; 6/8 proposals invalid) |
- ours vs editor v2 +0.183 (5/0, p 0.062); ours vs same-model lead +0.450 (7/1, p 0.039).
- Same 8 tasks with the gpt-6-luna framework: ours 0.655, editor v1 0.733, lead 0.476.
- MODEL-AGNOSTICISM: the typed, evidence-validated search does not degrade with a weaker,
  non-reasoning framework model (0.655 → 0.842; on 404 the luna lead refuses to claim without
  replication, the 4.1-mini lead does not), while free-form editing does (0.733 → 0.659, and v1
  collapses to 0.29 on invalid JSON) and the autonomous lead stays poor. Localization's value in
  shape 2 is robustness to the framework model, not peak accuracy with a strong one.

## Free-form editor proposals are often invalid workflows (model-dependent) → editor v2
- Found via the gpt-4.1-mini check: on mstest-400 the editor stopped with "no proposal" after one
  acceptance — all 3 next proposals were rejected by the validator ("workflow has a cycle": it wired
  site_A <-> site_B cross-verification both ways). Invalid proposals had been dropped silently.
- Replaying the gpt-6-luna editor runs with logging (cache, $0): invalid 12/183 attempts on mstest,
  11/98 on msdev — all cycles. So the v1 editor doc ("cross_verify: site_analyst -> another
  site_analyst") under-specifies the DAG constraint for every model; weaker models hit it harder.
- Change (baseline fairness + model-agnosticism): `llm_editor_v2` = documented acyclicity
  constraint with the verifier-node pattern the typed edit uses (site_A -> verify_B ->
  lead) + one repair round that returns the validator's error. v1 kept unchanged (replay
  identical) for the as-run held-out numbers; invalid proposals are now logged for both versions.
- ours_escalate uses editor v2.

## Held-out shape-2 mechanism ablations, batch A (mstest-400..403, v3.1 settings, 16 executions)
| method | 400 | 401 | 402 | 403 | mean |
|---|---|---|---|---|---|
| reflective_prompt (prompt-only, GEPA-style) | 0 | 0 | 0 | 0 | 0.000 |
| random_loc (same edit space, random locations) | .86 | .67 | .86 | .50 | 0.720 |
| ours_rules (msdev rule library first) | .86 | .80 | .67 | .50 | 0.706 |
| ours_v31 | .86 | .80 | .67 | .50 | 0.706 |
| llm_editor | .75 | .50 | .86 | .80 | 0.727 |
- Prompt-only optimization cannot repair a request-policy loss (sites refuse to report local
  findings when handed the study question) — consistent with msdev.
- In shape 2 random locations ≈ localization (n=4): the shape-2 edit space is small (request
  params + lead structure), so 16 executions cover most of it in any order. Decision quality also
  similar (random_loc 1 harmful / 9 accepted; ours_rules 1 / 13).
- Rules transfer: ours_rules identical F1 to ours on all 4 and no cost saving (nominal 1.21 vs
  1.16 over 4 tasks) — the LLM ranker already puts the rule edits first. Null.
- Batch B (404–407) done → n=8 (400–407):
  | method | mean F1 (n=8) | vs ours W/L (p) | harmful / accepted |
  |---|---|---|---|
  | reflective_prompt | 0.000 | 0/7 (0.02) | – |
  | random_loc | 0.629 | 1/2 (0.75) | 2 / 15 |
  | ours_rules | 0.655 | identical on 8/8 | 1 / 20 |
  | ours_v31 | 0.655 | – | – |
  | llm_editor | 0.733 | 4/3 (0.61) | – |
  | Codex lead (mean of 3) | 0.420 | 0/6 (0.03) | – |
- Localization vs random in shape 2: a small, non-significant edge (0.655 vs 0.629; random has
  twice the harmful acceptances). The shape-1 held-out gap (+0.16) does not carry over: with ~10
  typed edits in the shape-2 space, order matters little under a 16-execution budget.
- ours_rules stopped at n=8 (deviation from the full-16 plan): identical final F1 to ours on all
  8 tasks — the rule library changes the order of tries but never the outcome here.

## Second framework model on held-out shape 2 (gpt-4.1-mini framework, gpt-6-luna external agents)
mstest-400..403 (first 4 by index, not selected):
| method | 400 | 401 | 402 | 403 | mean |
|---|---|---|---|---|---|
| naive | 0 | 0 | 0 | 0 | 0.000 |
| same-model autonomous lead (in-house) | .67 | 0 | .40 | .50 | 0.392 |
| best fixed dev interface | .86 | .40 | .86 | .86 | 0.743 |
| ours_v31 | .86 | .80 | .86 | 1.0 | **0.879** |
| llm_editor v1 | .86 | .29 | (running) | | 0.571 (n=2; stops early: cyclic proposals) |
| llm_editor v2 | .86 | | (running) | | smoke n=1: 0 invalid of 11 proposals |
- The shape-2 ranking (ours ≥ fixed interface ≫ autonomous lead > naive) holds under a
  non-reasoning framework model. The typed edit space makes our optimizer insensitive to whether the
  model can write valid workflow JSON; the free-form editor v1 was not (fixed in v2 by
  documentation alone).

## v3.3 dev result (msdev, 8 tasks) and PRE-REGISTRATION for fresh held-out `mstest2`
| method | 300 | 301 | 302 | 303 | 304 | 305 | 306 | 307 | mean | $/task |
|---|---|---|---|---|---|---|---|---|---|---|
| ours_v31 | .80 | .89 | 1.0 | .50 | .50 | 1.0 | .67 | .50 | 0.732 | 0.25 |
| llm_editor v1 | .57 | .89 | 1.0 | 1.0 | .80 | .80 | .80 | .22 | 0.760 | 0.50 |
| ours_escalate (v3.3) | .80 | .89 | 1.0 | 1.0 | .50 | 1.0 | .67 | .50 | 0.794 | 0.28 |
- Escalation fired on 3/8 (301, 303, 306: typed space exhausted); it helped on 303 (.50 → 1.0),
  was neutral on 301/306; never below ours. Dev evidence only (msdev is heavily used).

PRE-REGISTERED (before any run on mstest2; code frozen at tag v3.3-eval):
- Suite `mstest2` = seeds 700–715, generated 2026-09-28, never inspected. Run in steps of 8
  (700–707, then 708–715).
- Methods (16 executions, $5 cap, 8 iterations): ours_escalate (v3.3), ours_v31, llm_editor_v2
  (free-form, DAG doc + repair); fixed refs ms_naive, ms_local+structured+weak+repl (best fixed
  dev interface), ms_orchestrator (in-house lead); Codex lead (local login) samples 1–3, mean.
- Primary: ours_escalate vs Codex lead (mean of 3) and vs llm_editor_v2 — paired per task,
  Wilcoxon. Secondary: ours_escalate vs ours_v31; decision quality (edit_quality.py).
- No method change after the first mstest2 result; failures reported as they come.
- Amendment before any mstest2 result (2026-09-28): + secondary baseline `ms_codex_native_orchestrator`
  (the user's own Codex config = gpt-6-sol, product-level arm), samples 1–3, step 1 first (it uses
  the user's ChatGPT plan quota: ~840k mostly-cached input tokens per run). Smoke on msdev-300:
  asked both sites for local analyses, then ran cross-site verification itself; no data-access
  violation; F1 0.40 (luna Codex lead there: .5/.0/.5).
