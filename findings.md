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

## FRESH held-out mstest2, step 1 (700–707; v3.3-eval frozen, pre-registered)
| method | 700 | 701 | 702 | 703 | 704 | 705 | 706 | 707 | mean |
|---|---|---|---|---|---|---|---|---|---|
| naive | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000 |
| in-house autonomous lead | 0 | 0 | 1.0 | 0 | 0 | .50 | 0 | 0 | 0.188 |
| Codex lead, gpt-6-luna (mean of 3) | .70 | 0 | 1.0 | 0 | .44 | .70 | .22 | 0 | 0.383 |
| Codex lead, native (user config, gpt-6-sol; mean of 3) | .70 | 0 | 1.0 | .67 | .67 | .70 | 0 | 0 | 0.467 |
| best fixed dev interface | .86 | .67 | 1.0 | .20 | .50 | .80 | .40 | 0 | 0.553 |
| llm_editor_v2 | 1.0 | .27 | .67 | 0 | .80 | 1.0 | .50 | 0 | 0.529 |
| ours_v31 | 1.0 | .40 | 1.0 | 0 | .67 | .80 | .67 | 0 | 0.567 |
| **ours_escalate (v3.3)** | 1.0 | .40 | 1.0 | 0 | .67 | 1.0 | .67 | 0 | **0.592** |
- Primary: ours_escalate vs Codex lead +0.208 (5/0, p 0.062); vs llm_editor_v2 +0.062 (3/1, p 0.38).
- Secondary: vs native Codex lead +0.125 (4/1, p 0.56); vs ours_v31 +0.025 (1/0); vs in-house
  lead +0.404 (5/0, p 0.062).
- Escalation fired on 6/8 (typed space exhausted) and changed the outcome once (705 .80 → 1.0).
- mstest2 is harder than mstest for every method (707: all 0; 703: only native Codex scores).
- Step 1 actual API spend $4.22. Step 2 (708–715) next, same protocol.

## Held-out shape-2 ablations, all 16 tasks (random_loc, reflective_prompt)
- reflective_prompt 0.000 on 16/16 (0/15 vs ours, p 0.001): prompt-only optimization cannot repair
  a collaboration-interface loss.
- random_loc 0.765 vs ours 0.736 (+0.029, 3/2, p 0.38): NO localization advantage in shape 2
  (small typed space; 16 executions cover it in any order). Harmful acceptances: random 2/35
  (5.7%) vs ours 1/46 (2%). Honest negative — the shape-1 localization gain (+0.16) is
  shape-specific.

## Pooled data (shape 1) under a weaker framework model — held-out, small subset first
heldout-500/501, framework gpt-4.1-mini (external GeneAgent stays gpt-6-luna):
| method | 500 | 501 | luna framework (500 / 501) |
|---|---|---|---|
| naive | 0 | .22 | 0 / 0 |
| best fixed dev interface (pseudobulk+no_interp+donor) | .40 | 1.0 | .40 / 1.0 |
| same-model single agent (coder) | 0 | 0 | .80 / 1.0 |
- The 4.1-mini single agent failed on both: 500 — could not get a DE fit working (pydeseq2 API
  errors over 20 steps) and reported nothing; 501 — four plausible-sounding but wrong claims.
- Extended to 8 tasks (heldout-500..507, fixed refs):
  | method | gpt-4.1-mini framework | gpt-6-luna framework |
  |---|---|---|
  | naive | 0.226 | 0.000 |
  | best fixed dev interface | **0.789** | 0.790 |
  | same-model single agent (coder) | **0.062** | 0.742 |
  | (Codex CLI alone, gpt-6-luna) | – | 0.797 |
- The composition's accuracy is independent of the framework model (0.789 vs 0.790); the lone
  agent collapses with the weaker model (0.742 → 0.062: failed DE fits, confident wrong claims).
  REVISED shape-1 conclusion: "decline composition" holds only when the single agent's model is
  strong; an explicit interface around independently developed agents is what makes the result
  model-agnostic. Optimizers (4.1-mini) on 500/501: ours .40/1.0 (= fixed), editor v2 .13/.33;
  502–507 running.

## Infrastructure incident: network outages 2026-09-28 ~18:34–20:16
- Ledger gaps of 15–24 min between 18:34 and 20:16 (user's network was disconnected).
- Safe by design: an LLM call that fails after retries raises inside the node → the node is
  recorded as an error and NOT written to the node cache; no optimizer/fixed-ref run finished
  inside the window (no outputs written after 19:40; no FAILED lines).
- Not safe: the Codex lead talks to site agents over a loopback HTTP service, so an outage looks
  to Codex like a failed site/tool call and its (empty) report is cached as a genuine result. The
  one Codex run spanning the window (mstest2-711 native s2, started 18:29, stuck past its 90-min
  timeout) was killed; its result "(no valid report.json written)" and node-cache entry were
  QUARANTINED to runs/_quarantine_outage_20260928/ and the run will be redone.
- The BrokenPipe at 18:29 (mstest2-711 luna s2) predates the outage: Codex's ask_site client gave
  up on a slow site analysis — ordinary baseline behaviour, kept.
- Outage re-check: the two gpt-5-mini outcomes that overlapped the window were quarantined and
  re-run: identical accepted edits and final workflows (replayed in 48 s / 4 s). For the
  optimizers the outage caused delay (retries), not different results.

## Third framework model: gpt-5-mini (reasoning; different capability profile: no effort=none,
## tools allowed with reasoning), held-out shape 2, small subset first
| method | 400 | 401 |
|---|---|---|
| naive | 0 | 0 |
| same-model autonomous lead | .67 | .67 |
| best fixed dev interface | 1.0 | .40 |
| ours_v31 | 1.0 | .80 |
| llm_editor_v2 | .86 | .50 |
- Extended to 4 tasks (400–403): naive 0.000 · same-model lead 0.558 · fixed interface 0.814 ·
  llm_editor_v2 0.804 · **ours_v31 0.864** (per task: ours 1.0/.80/.86/.80).
- The capability layer handled the new profile without code changes. Across three framework
  models (gpt-6-luna reasoning, gpt-4.1-mini non-reasoning, gpt-5-mini reasoning with a different
  API profile) the shape-2 ordering is the same: ours ≥ fixed interface ≥ free-form editor >
  autonomous lead ≫ naive.

## CODE REVIEW (2026-09-29) — concurrency bug in both caches (last writer wins)
- Found by a golden replay check (experiments/replay_check.py: re-executes stored runs from the
  caches with the spend cap at the current total, so any uncached call fails): 29/32 identical;
  2 editor-v1 diffs are logging-only (invalid proposals now logged); 1 FAIL —
  phase2@gpt-4.1-mini/mstest/mstest-400/ours_v31 diverges at iteration 3.
- Root cause: two processes that miss the same key at the same time both compute it; the node
  cache is a plain file write and the LLM cache uses INSERT OR REPLACE, so the LAST writer wins.
  Site-agent node keys are shared across framework backbones by design (external model fixed),
  and the luna ablation batch and the 4.1-mini run both executed mstest-400 site agents at
  04:15–04:24. The losing run cannot be replayed, and inside it a later execution that re-reads
  the key sees the other process's output — the incumbent/candidate prefix is then not the same,
  which the controlled comparison assumes.
- Fix: first writer wins everywhere (node cache: exclusive create, on conflict read the stored
  outputs; LLM cache: INSERT OR IGNORE then read back the stored row). No change for
  non-concurrent runs; existing entries untouched. Impact audit: replay every stored outcome.

## CODE REVIEW 2026-09-29 — consolidated triage (3 independent read-only reviewers + own reading)
Verified by me against code/stored runs where marked ✓. A = behaviour-neutral (replay must stay
identical); B = changes method behaviour (v3.4; needs re-evaluation); R = reporting/analysis fix.
R1 ✓ phase2_analysis: float ties vs a __mean reference counted as losses (mstest-408 Codex
   0.8,0.8,0.8 → mean 0.8000000000000002). Corrected: ours vs Codex lead 12/1 (p .006), fixed
   12/2, editor 12/2, in-house 7/3. Conclusions unchanged; error ran against ours.
R2 phase1_analysis Q2 mixes Q3 re-draw pairs into evaluator accuracy: edit pairs only →
   report-only 0.533 / evidence judge 0.858 / rule 0.956 (published 0.50/0.85/0.95, n=675).
R3 Optimization LLM calls (rank, prompt, judge, editor/reflect proposals) not in nominal_usd;
   GeneAgent result-cache hits give node nominal 0 (205/1030 entries). $/task understated,
   more for the editor. Report as separate field; re-cost.
R4 spent() uses LIKE without trailing '/': "phase2/mstest" also counts mstest2 (printed totals).
R5 ✓ findings claim "escalation fired 6/8" overstated: 4/12 escalations on mstest2 evaluated 0
   editor candidates (only the control ran).
A1 ✓ Both caches last-writer-wins (+ non-atomic node-cache write → partial-file reads).
A2 Dead: UNCACHEABLE, use_cache, RunContext.workdir, trace._significant, chat(seed,
   max_completion_tokens), _adapt_chat deterministic branch, ChatResult unused fields, luna
   write-price special case, METHODS ours/ours_v32/ours_no_compound/ours_no_control/ours_no_judge/
   prompt_only_loc + options use_lookahead/families/control/use_judge/use_compound/
   first_improvement, handoff params min_abs_lfc/cell_types/fields, codex_native_single,
   unused imports; rules.distill/save have no caller (add a CLI instead).
A3 Duplication: chat/responses cache block; initial-Outcome setup ×3; profile helper ×5; Codex
   runner ×2; report normalization ×4; site-agent call ×3; compose/_ms; findings merge ×2.
A4 Efficiency: sqlite connection + CREATE TABLE per call; competitive null computed per site
   although per-site status is unused.
A5 Robustness (no change on success paths): BudgetExceeded swallowed by baselines' except;
   failed-node spend not charged (finally); Codex returncode/failure events unchecked; Codex-lead
   handler exceptions swallowed → degraded report cached; GeneAgent infra errors → cached
   ok:false; kernel protocol fragile; probe non-200 stored as "capability absent".
B1 ✓ Compound edits across locations ("a+b"): _affected splits on "->" → control redraws
   nothing (vacuous control, still charged). 16/18 stored cross-location compounds.
B2 ✓ site-request edits change both request edges but control redraws only site_A.
B3 Locks keyed (location, param): additive params (branch/add_context/rewrite) block further
   branches (donor_check never tried after de_summary in 6/8 4.1-mini shape-1 runs); compound
   locks never match → accepted changes can be reverted (msdev-307).
B4 ✓ ours exceeds the 16-execution budget by one (5/36 ours_v31, 5/12 random_loc).
B5 ✓ escalation: editor's control consumes the last execution → 0 candidates.
B6 ✓ random_loc never seeded per task: identical permutation on 16/16 mstest tasks; also no
   ranker/prompt family → the "random ≈ localization" ablation is one fixed ordering.
B7 _safe_fp returns "" for unappliable edits → they burn menu slots.
B8 Trace overstates propagation via de_summary (counts as handed off without the genes).
B9 Acceptance parity: ours ignores cand_run.errors; baselines require none.
B10 Free-form proposals may place shape-1 agents in shape-2 workflows → they read site A's raw
   file (task.data_path = site_A.h5ad). Hidden truth is a sibling of the data path and
   tool-using baselines (coder, Codex) can read anywhere; audited coder transcripts clean,
   shape-1 Codex commands not logged (not auditable).
- R1 applied (phase2_analysis tie tolerance): mstest n=16 vs Codex lead — ours 12/1 (p 0.006),
  fixed 12/2 (0.019), editor 12/2 (0.010), in-house 7/3 (0.38). docs/results.md, README, CLAUDE.md
  corrected.
- R2 recomputed from the stored verdicts (runs/phase1/dev_analysis.json): all pairs 0.502/0.855/0.947
  (n=675, the published numbers) → edit pairs only 0.533/0.858/0.956 (n=619). Conclusion unchanged.
- Phase A (behaviour-neutral) complete: golden replay 30/30 identical after every step.

## v3.4 — method bug fixes from the review (behaviour changes; tag v3.4-dev)
- B1/B2 controls: the no-edit control re-draws every incumbent node an edit can change, found by
  diffing candidate vs incumbent (new config or inputs, plus descendants) — fixes vacuous controls
  for cross-location compounds and single-site controls for site-request edits.
- B3 locks: value-aware and diff-based — an accepted scalar is never changed again, an accepted
  list may only grow (context fields, branches stay addable), compound parts are protected.
- B4 budget: a candidate starts only if the candidate and its (new) control fit (Budget.room);
  B5 same for the editor loop (control + one candidate) and for escalation.
- B6 random_loc seeded per task (was one fixed permutation for all tasks).
- B7 unappliable edits dropped from menus; B9 candidates whose run has node errors are rejected
  (parity with the baselines).
- B8 trace: a DE summary carries the signal only if its top list in the signal's direction has
  signal genes; a carrying list with no interpretation is `interpretation_lost` (stage 3), not
  "reached the integrator".
- B10 data boundary: free-form proposals may only use the agents of the task's shape;
  pooled-data agents (de_analyst, donor_check, coder, codex) raise in a multi-site task.
- F2: each outcome records `optimization_usd` (the method's own LLM calls, excluded from
  nominal_usd so far); report it next to $/task.
- Replay after v3.4: all 12 fixed-workflow runs and all 7 baseline-optimizer outcomes identical;
  every ours-family outcome diverges (expected: controls/locks/budget changed). The stored
  ours_v31/ours_escalate/random_loc/ours_rules results are v3.1/v3.3 as-run results (tags
  v3.1-eval, v3.3-eval); comparisons involving them need re-running with v3.4.

## Analyses of stored runs after the review (2026-09-29; no new experiments, ~$0 grading)
**mstest2 full n=16 (v3.3 as-run, pre-registered):** ours_escalate 0.731 · ours_v31 0.719 ·
llm_editor_v2 0.665 · fixed dev interface 0.664 · native Codex lead (gpt-6-sol, mean 3; 711 has 2)
0.542 · Codex lead (luna, mean 3) 0.522 · in-house lead 0.458 · naive 0.
- ours_escalate vs Codex lead +0.209 (12/0, p 0.0005); vs native Codex +0.189 (9/1, p 0.053);
  vs in-house +0.274 (7/0, p 0.016); vs editor v2 +0.067 (6/2, p 0.27); vs ours_v31 +0.012 (1/0).
- Never-inspected step 2 (708–715): ours 0.871, editor v2 0.800, fixed 0.776, in-house 0.728,
  Codex 0.660, native Codex 0.617 — same ordering as step 1.
- Precision: ours 0.83 vs editor v2 0.67; decoys claimed: ours 0, editor 2, fixed 3.
- Escalation is inert (changed 1/16 tasks; identical on all of step 2) — partly B5 (4/12
  escalations evaluated no editor candidate). v3.3 adds nothing measurable over v3.1.
**Decision quality mstest2:** ours 34 accepted, 4 harmful (12%), 3 missed; editor v2 21 / 4 (19%)
/ 15. Harmful cases: an extra data-supported but misnamed claim (701), and two acceptances that
*dropped* a correct claim (705, 714).
**Pooled decision quality (all logged attempts, all suites and three framework models,
evidence-route acceptances):** ours 156 accepted — 125 helpful (80%), 14 harmful (9%); editor and
other baselines 75 — 55 (73%), 13 (17%).
**Evidence-rule tie-break (−unassessed −ungrounded) checked:** only 11 acceptances were decided by
it alone (ours 3 helpful / 3 harmful; editor 4 helpful / 0). Too few and too mixed to justify a
change — kept (a change would also invalidate the frozen 0.96 evaluator result).
**Impact of the v3.4 bugs on stored results:** B1 (vacuous compound controls): 1 accepted
cross-location compound, ΔF1 0. B4 (overrun): one acceptance came from the 17th execution
(4.1-mini mstest-401, +0.13 → 8-task mean 0.842 would be ~0.826). B2/B3 not measurable without
re-running.
**Shape 1, gpt-4.1-mini framework, heldout 500–507 (n=8):** ours_v31 0.754 · fixed 0.789 ·
llm_editor_v2 0.579 · naive 0.226 · same-model single agent 0.062. ours vs single agent +0.69
(7/1, p 0.016); vs editor v2 +0.18 (4/2, n.s.). Luna on the same 8: ours(v3) 0.738, editor v1
0.853, Codex 0.797. → Across both shapes, the typed search keeps its accuracy with a weaker
framework model while free-form editing and single agents do not.
Ours < fixed on heldout 504/506: donor_check (part of the fixed interface) never tried after the
de_summary branch was accepted — exactly lock bug B3. A v3.4 check on these tasks tests a
specific hypothesis (cheap: most nodes cached).
- Signal-propagation acceptances (all stored runs): 91, almost all ΔF1 = 0 at the time (stepping
  stones: shape 2 scope/template/cross-verify), 2 helpful directly, 1 harmful: 4.1-mini
  heldout-506 "remove the interpretation agent" accepted at depth 3→5 after the donor branch had
  reached F1 1.00 → 0.33. That depth came from bug B8 (a DE summary counted as carrying the signal
  without its genes); re-tracing the final run with the v3.4 trace gives depth 1 — the edit would
  not be accepted. Counterfactual 8-task mean for 4.1-mini shape 1: ~0.84 instead of 0.754
  (above the fixed interface's 0.789). Offline mechanism check also confirms B3: under v3.1 locks
  donor_check was blocked after de_summary was accepted (6/8 4.1-mini shape-1 runs); v3.4 admits it.
- My earlier hypothesis that B3 explained ours < fixed on heldout 504/506 was wrong: 504 stopped
  with "no findings" at F1 0.50 (checks blind to the remaining error), 506 was B8.

## Where ours still loses F1 — and a rejected detector change (dev-only calibration, $0)
- Error decomposition of final reports: ours is conservative (mstest 2 FP / 32 claims; mstest2
  2 / 27; editor 12 / 47 and 17 / 46; 4.1-mini shape 1 editor 40 FP / 56 claims). Ours' errors are
  misses. Of the 21 planted items ours missed in shape 2, 17 were NOT flagged by the answer-free
  checks at stop — the bottleneck is detection, not search.
- Missed-and-unflagged programs have the same planted effect (median |log2FC| 1.10) but fewer genes
  (median 8 vs 10), mostly UPR / proliferation / MYC-ribosome.
- Dev diagnosis (dev + msdev, empty-claim profiles, sensitivity 64/85 = 0.75, 7 false flags): 14/21
  misses have 5–8 planted genes passing every per-gene filter but drown in a 25–116-gene
  donor-consistent group (compositional co-movement): e.g. dev-008 NK ribosome — planted genes
  alone p 5e-11, inside the 99-gene group p 3.9e-4 (> 1e-4). 7/21: too few planted genes pass
  the donor/size filter.
- Variants (dev only): stricter effect floor 0.6 → 65/85, 4 false flags; effect-ranked prefixes
  (GOrilla-style, Bonferroni over 4 cutoffs) → 65/85, 3; donor-t-ranked prefixes → 68/85 = 0.80,
  4 false flags.
- REJECTED: with the t-ranked detector the evidence rule's pairwise accuracy on dev edit pairs drops
  0.956 → 0.883 (922 pairs, stored reports and grades): the detector's probe genes no longer line up
  with claimed genes, so it penalizes reports that did claim the program. More sensitive detection
  alone makes answer-free comparison worse; the current detector stays. A detector change must be
  evaluated on comparison accuracy, not sensitivity.

## v3.4 targeted check (user-approved; 4.1-mini framework, heldout 500/503/504/506; ~$0.01 actual)
| task | reason | v3.1 | v3.4 | fixed |
|---|---|---|---|---|
| 506 | B8 (remove-interp accepted on inflated depth) | 0.33 | **1.00** | 0.67 |
| 500 | B3 (donor branch blocked) | 0.40 | 0.40 | 0.40 |
| 503 | B3 | 1.00 | 1.00 | 1.00 |
| 504 | control (checks blind; no bug involved) | 0.50 | 0.50 | 0.80 |
- 506: v3.4 keeps the donor branch and no longer removes the interpreter — the B8 fix works end
  to end, as predicted by the offline re-trace.
- 500/503: the donor branch is now admissible (tried on 503, rejected on evidence; on 500 it is not
  reached before the 16-execution budget ends) — B3 was a real restriction but not binding here.
- 504 unchanged (detection limit). The new optimization_usd field: ~$0.01/task for ours.
- heldout is an inspected suite: this confirms the mechanisms, it is not a performance claim.
- Sensitive detector as a probe source only (acceptance unchanged): on dev it would newly probe 1
  of 7 missed items — not worth a method change. Dropped.

## PRE-REGISTRATION: v3.4 on mstest2 (2026-09-29, before any v3.4 run on it)
- Method: METHODS["ours_v31"] at tag v3.4-dev (code-review fixes only; no change informed by
  mstest2 results — v3.4 was frozen before mstest2 was analysed). Disclosure: mstest2 step-1 scores
  were inspected for v3.3 (no method change followed).
- Baselines reused as stored (their code path replays identically): fixed dev interface, in-house
  lead, Codex lead luna + native (mean of 3), llm_editor_v2; comparison also with ours v3.1.
- Protocol: 16 executions, $5 cap, 8 iterations; outputs runs/phase2_v34/; steps of 8 (700–707,
  then 708–715) after a 2-task msdev smoke test.
- Primary: v3.4 vs Codex lead (mean of 3) and vs llm_editor_v2, paired per task, Wilcoxon;
  secondary: vs ours v3.1, vs native Codex; decision quality.
- Amendment (before any v3.4 run on mstest2): the msdev smoke (300, 303) revealed a regression I
  introduced in v3.4 — compound escalation filtered its single edits by "unvisited", so the
  singles that had just failed (all visited) were excluded and escalation never fired (303
  stopped at 11/16 executions). Fixed (singles need only apply and respect locks; the pair must be
  new) with a regression test that fails on v3.4. Evaluated method = v3.4.1 (tag v3.4.1).
  Smoke scores were unchanged vs v3.1 (300 0.80, 303 0.50); the stricter both-site control made 3
  request edits on 303 lose to resampling (intended).

## RESULT: v3.4.1 on mstest2 (pre-registered; 16 tasks; $0.21 actual)
ours v3.4.1 0.719 — identical to v3.1 on every task (same final workflows and execution counts;
controls differed in 3/57 step-1 attempts, no verdict changed: evidence counts are coarse
integers, so re-drawing site B as well rarely moves them).
vs Codex lead (luna, mean 3) +0.197 (12/0, p 0.0005); vs native Codex (gpt-6-sol) +0.177 (9/1,
p 0.051); vs in-house lead +0.261 (7/0, p 0.016); vs llm_editor_v2 +0.054 (6/3, p 0.44); vs fixed
dev interface +0.054 (7/4). → The review's typed-search bugs do not change any shape-2 held-out
outcome; the as-run v3.1 claims stand. The fixes matter in shape 1 (heldout-506 0.33 → 1.00,
B8) and for escalation (msdev-303 0.50 → 1.00, v3.4.1).

## Phase 5 (2026-10-01): public benchmarks, same-model baselines, case study

User critique (2026-10-01): effect not evident; baselines confusing/mixed models (→ every arm on
gpt-6-luna) and mostly ablations; no public benchmark; no case study.

### Infrastructure (no spend)
- BixBench v1.5 (`futurehouse/BixBench`, 205 open-answer Qs, 59 capsules, 5.2 GB zips → 12 GB
  unpacked). BixBench-Verified-50 is gated (HF login required) — not used.
- Execution env: native conda `bixbench` replicating the Finch notebook image (R 4.4/Bioc 3.20
  instead of 4.3/3.18: pinned set does not solve on osx-arm64; metaeuk/spades/busco missing).
  `external_agents/bixbench_env/setup.sh` rebuilds it; exact solve in explicit-osx-arm64.txt.
- `agentcoop/proxy.py`: OpenAI-compatible endpoint over `llm.chat_raw` (same cache, ledger,
  cap, capability layer) so litellm harnesses run on the shared backbone. gpt-6-luna rejects
  `stop` → emulated client-side (truncate at first stop sequence; `llm.accepts_stop` probes).
- Finch (data-analysis-crow aea66fd, BixBench config: ReActAgent, T=1, 40 steps, open prompt,
  no images) runs unmodified in its own py3.12 venv; smoke bix-18-q2 correct, $0.005, 34 s.
- Pre-registered capsule split (sha256 of capsule id): bixdev 14 capsules / 39 Qs, bixtest 45
  capsules / 166 Qs. Grader = BixBench open-answer grader minus its substring "partial match"
  (accepts "2" for "0.0002"); numeric range answers checked numerically.
- Truth hygiene: capsule reference notebooks moved to data/bench/bixbench/reference (never in a
  workspace); ideal/distractors/eval_mode/result/hypothesis in hidden/truth.json; every analysis
  is scanned for paths reaching the answers (`meta.leak_suspect`).

### First dev observations (bix-10, bix-13; s0)
- All three agents often agree on an answer different from the key (bix-10-q2: all 0.83, key
  0.74–0.77; bix-13-q2: 165 vs 166). BixBench rewards reproducing the original authors' hidden
  analysis choices; agreement-based aggregation cannot fix shared interpretations.
- Failures that collaboration *can* address: refusals/incomplete analyses (Finch "unable to
  determine" on bix-13-q5; Finch refusing AESEV as COVID severity on bix-10-q3).

### Case study: real Kang IFN-β (kangreal-0; 4 discovery donors D101/D1016/D107/D1256, 4
held-out validation donors D1015/D1039/D1244/D1488, split pre-registered by sorted donor id)
Validation = held-out-donor replication (same per-donor competitive check, new people) + missed
reproducible programs per (cell type, direction) + IFN-program coverage (known biology).
| arm (s0) | claims | replicated | IFN cell types | missed (ct,dir) |
|---|---|---|---|---|
| naive DE→GeneAgent→integrator | 0 | 0 | 0/6 | 10 |
| coder_single | 8 | 8 | 6/6 | 2 (translation↓ CD8, NK) |
| codex_single (luna) | 7 | 7 | 6/6 | 3 (translation↓) |
| fixed repaired workflow (pseudobulk+no_interp+donor) | 13 | 11 | 6/6 | 0 |
- The plan's §5.2 hypothetical happens on real data: GeneAgent names "type I IFN antiviral
  defense" correctly for every list, but the mixed-direction gene list + interpretation view
  carries no direction/donor support, and the integrator rightly refuses every claim
  ("does not establish whether the genes are higher or lower in treated cells or whether effects
  reproduce across donors"). Every node succeeded; the collaboration lost the science.
- Caveat for the optimizer on real data: with dozens of induced ISGs, gene-level uncovered-
  signal probes stay positive even when the program is claimed (a claim cites ~5–8 genes).

### Harness bugs found and fixed before any BixBench comparison (Finch arm)
1. jupyter_client maps Finch's kernel name "python" to the *native* python3 spec = Finch's own
   venv interpreter (no openpyxl/scanpy/R bridge) → many "unable to determine" answers. Fixed:
   the env interpreter is published as `python3` too (setup.sh).
2. Finch local mode shows the notebook's container path (/workspace/...) → agent reads a
   nonexistent dir. Fixed in run_finch.py (identity path mapping in local mode).
3. Shared env was writable (agent `pip install` would change later runs) → env read-only;
   installs land in the run's workspace (PYTHONUSERBASE / R_LIBS_USER).
All Finch runs before these fixes are stale (config bumped to `harness: 2`).

### BAISBench (HF EperLuo/BaisBench, apache-2.0, not gated) — candidate second public benchmark
- BAIS-DPTA: 15 organs (raw h5ad 0.27–0.86 GB each; labels in the non-raw file), per-cell
  annotation scored on the uHAF hierarchy (exact or descendant 1, parent 0.5, grandparent 0.2)
  after an LLM maps predicted names onto the organ's uHAF reference types (repo Task1.ipynb,
  external_agents/BaisBench cfab1be). Shape = analysis (QC/cluster/markers) → independently
  developed annotator (CellTypist / LLM annotators): a natural handoff benchmark.
- BAIS-SD: 193 MCQs over 41 studies (task2 h5ads 0.2–1.8 GB each).
- scBench (latchbio/scbench 0bc3403): 195 evals but only 6 public (rest withheld), data via
  latch:// nodes; scBench-Long (99da791): 4 public evals. Too few public tasks for a
  quantitative comparison (n=6 / n=4); at most extra qualitative cases.
- BAIS-DPTA rejected as a main benchmark (Trachea dev check): labelling each Leiden cluster
  with its majority expert label scores 0.986, naive GPTCelltype 0.417 — the gap is label
  *granularity/taxonomy*, not analysis: experts label fibroblasts/chondrocytes "Mesenchymal
  cell" and all lymphocytes "Myeloid cell", while uHAF puts Fibroblast under Stromal cell and
  T cell under Lymphoid cell. A biologically correct, specific annotation scores 0; no
  answer-free method can know the expert's granularity. Code kept (tasks/bais.py,
  eval/bais_grader.py, agents/annot.py) for a possible qualitative case.
- BixBench-Verified-50 (curated keys; Claude Code Opus 4.6 65%, GPT-5.2 Agents SDK 61%,
  Biomni Lab 89% per the OmicOS writeup) is the discriminative public set. Its 50 question ids
  are recoverable from omicverse/OmicOS-BixBench (data/bench/bixbench/verified50_ids_from_omicos
  .json; 6 in bixdev), but 17 questions have revised text/keys only in the gated HF dataset →
  needs the user's HF login (accept terms for phylobio/BixBench-Verified-50).

### GEPA baseline (official `gepa` 0.1.4; agentcoop/optimize/gepa_baseline.py)
Same task, same answer-free evidence (score = evidence-rule net score, feedback = the same text
the other baselines get), same backbone (reflection LM through our ledger), same budget (16
distinct workflow executions); it evolves the lead's instructions (+ a site-request note in
multi-site tasks); external agents and workflow structure fixed.
- kangreal-0: 15 proposals; score −10 → −4 at the 10th; $0.92 nominal. Final report: 6 IFN-up
  claims, all replicate in held-out donors; translation-down still missed in 4 (ct,dir).
  *How*: the winning instruction pastes the evaluator's feedback (this dataset's GO terms, cell
  types, genes, directions) into the lead's prompt; the lead asserts "up" because the prompt
  says so — its claims cite interpretation views that still carry no direction. Prompt
  optimization recovered score by moving the verifier's findings into the prompt (task-specific,
  non-transferable), not by repairing what the handoff carries.
- kangreal-0, ours_v31 (16 executions, $1.75 nominal, from the naive workflow, answer-free):
  accepted split_direction (de→interp), carry direction/method/stats (interp→report), then the
  donor-consistency evidence branch (0 → 9 supported vs control 0), then n=50, max_padj=0.1.
  Held-out validation: 13 claims, 10 replicate, 0 contradicted, 3 unassessed; IFN 6/6; missed
  1 (ct,dir). vs coder 8/8 missed 2; Codex 7/7 missed 3; GEPA 6/6 missed 4; fixed dev
  interface 11/13 missed 0. Iterations 0–1 were accepted with evidence equal to the control
  (the signal-propagation route: lineage showed the signal carried one step further) — the
  payoff came only once the evidence branch existed: a staged repair, not a single edit.

### BixBench dev, single agents (39 Qs / 14 capsules, s0, all gpt-6-luna)
| agent | open-answer | MCQ w/ refusal (random 0.2) | $/q |
|---|---|---|---|
| Finch (official BixBench agent, fixed harness) | 0.231 [0.09,0.40] | 0.205 | 0.005 |
| Codex CLI | 0.154 | 0.179 | 0.005 |
| ReAct (in-house) | 0.154 | 0.154 | 0.003 |
| oracle (any of the three) | 0.231 | | |
- Finch's correct set contains the other two's: no selection/aggregation among these three
  analyses can beat Finch on dev; only producing *new* correct analyses (repaired re-runs,
  re-draws) can. Codex and ReAct agree on 24/39 answers.
- Fixed-harness Finch still refuses on 4 Qs ("cannot be determined") and timed out on 1.

### Results 2026-10-02
**mstest2 (fresh held-out, n=16, all gpt-6-luna) — main baseline table**
| method | F1 [95% CI] | $/task | ours v3.1 W/L, p |
|---|---|---|---|
| naive | 0.000 | 0.008 | |
| best-of-N (self-consistency by evidence, 16 executions) | 0.067 [0.00,0.17] | 0.091 | 13/0, 0.0002 |
| orchestrator lead | 0.458 [0.25,0.66] | 0.027 | 7/0, 0.016 |
| Codex CLI as lead (mean of 3) | 0.522 [0.36,0.68] | 0.029 | 12/0, 0.0005 |
| GEPA (official, 16 executions) | 0.606 [0.44,0.76] | 0.096 | 7/2, 0.32 |
| best fixed dev interface | 0.664 [0.52,0.80] | 0.027 | 7/4, 0.45 |
| free-form editor (ADAS/AFlow-style, 16 exec.) | 0.665 [0.49,0.83] | 0.521 | 6/3, 0.44 |
| **ours v3.1** | **0.719** [0.55,0.87] | 0.194 | |
| ours + escalation | 0.731 [0.56,0.89] | 0.266 | vs GEPA 7/2 p 0.27 |
GEPA made 19 accepted prompt changes; decoy reported once (ours 0). vs the strong optimizers
the margin is directional but not significant at n=16 → a second fresh suite (mstest3, 32
tasks, seeds 720–751) pre-registered 2026-10-02 with the method frozen at v3.4.1, to pool n=48.

**Case study, all optimizers (held-out replication / missed programs, nominal $):**
naive 0/0, 10 missed; best-of-N 0/0, 10 ($0.98); random-location 0/0, 10 ($1.25);
reflective prompt 7/8, 4 ($0.94); GEPA 6/6, 4 ($0.92); ours 10/13, 1 ($1.75); free-form
editor 28/33, 0 ($3.72; 3 accepted rewrites); fixed dev interface 11/13, 0.
The editor's rewrite reached the highest recall on this real case (honest: it beats ours here,
at 2.1× cost). Prompt-only methods recover IFN claims by moving feedback into the prompt and
still miss the repressed programs; resampling and random edit locations recover nothing.

**BixBench dev, teams (39 Qs):** Finch 0.231; self-consistency Finch×3 0.231; MoA 0.205
(MCQ 0.256); vote3 0.205; debate 0.179 (MCQ 0.231); judge 0.179; ours evidence team 0.179;
ours + repair loop 0.179 ($0.018/q); SC ReAct×3 0.179. No combination beats the best single
agent (oracle of the three = Finch). The evidence team differs from Finch on 2 Qs, both on
bix-10 (BCG capsule) where all three analyses follow every stated requirement and the key
rests on one modeling choice (OR 0.76 Finch vs 0.83 Codex+ReAct). Provenance stays
`unassessed` for verbose answers (number_of needs a single number) — a real limitation of the
check. Interpretation: BixBench is single-agent-solvable (one agent sees all data), like
shape 1: composition does not pay; the best decision is not to compose.
- 2026-10-02: BixBench bixtest (166 Qs / 45 capsules) launched with every arm frozen exactly as
  run on dev (no method change after seeing dev): 3 single agents, vote3, MoA, judge, debate,
  evidence team, SC-Finch×3, ours + repair loop.

### mstest3 (fresh, pre-registered 2026-10-02, tag mstest3-prereg, method frozen v3.4.1), n=32
ours 0.715 [0.63,0.80] $0.23 · free-form editor v2 0.764 [0.68,0.84] $0.50 · best fixed dev
interface 0.688 · GEPA 0.610 [0.50,0.72] $0.09 · orchestrator lead 0.461 · Codex lead (mean of
3) 0.448 · naive 0. Ours vs Codex lead 25/0 (p 1e-5).

### Pooled fresh held-out (mstest2 + mstest3, n=48 tasks, all gpt-6-luna)
| method | F1 [95% CI] | $/task | ours W/L | p |
|---|---|---|---|---|
| Codex CLI as lead (mean of 3) | 0.472 [0.40,0.55] | 0.029 | 37/0 | 1e-7 |
| orchestrator lead | 0.460 [0.36,0.56] | 0.027 | 28/3 | 7e-6 |
| GEPA | 0.609 [0.52,0.70] | 0.091 | 22/11 | 0.037 |
| best fixed dev interface | 0.680 [0.59,0.76] | 0.027 | 18/15 | 0.71 |
| free-form editor (ADAS/AFlow-style) | 0.731 [0.65,0.81] | 0.507 | 13/14 | 0.48 |
| **ours** | **0.716** [0.64,0.79] | 0.215 | | |
Conclusions: ours > agent products and autonomous leads (large, significant); ours > GEPA
(+0.11, significant at n=48); ours ≈ free-form editor (tie, 42% of its cost); ours ≈ the best
fixed interface found on dev (the dev interface already transfers). The mstest2 edge over the
editor (0.719 vs 0.665) did not replicate on mstest3 (0.715 vs 0.764) — reported as a tie.
- 2026-10-03: ours_escalate (v3.3 logic, unchanged; pre-registered on mstest2) run on mstest3 AFTER its outcomes were graded — reported as post-hoc.
- Decision quality, pooled mstest2+mstest3 (48 tasks; hidden grading after the fact):
  ours 338 tries / 119 accepted / 66 helpful / 9 harmful (7.6% of accepted) / 18 missed;
  editor 528 / 70 / 57 / 8 (11.4%) / 35 missed (+87 outranked siblings); GEPA 497 / 64 / 44 /
  9 (14.1%) / 51 missed. Ours: safest acceptances, fewest missed improvements, but many
  neutral acceptances (signal-propagation route) — the editor's acceptances are more often
  helpful (81% vs 55%).

### BixBench test (166 Qs / 45 capsules, frozen arms, all gpt-6-luna) — open-answer / MCQ
ReAct 0.542/0.560 · Codex 0.500/0.518 · Finch 0.373/0.392 · SC-Finch×3 0.410/0.434 ·
vote3 0.506/0.518 · MoA 0.542/0.560 · debate 0.542/0.536 · judge 0.536/0.554 ·
ours evidence team 0.536/0.554 · ours + repair loop 0.530/0.548 · oracle(3) 0.608.
(test is far easier than dev: dev capsules happened to be the hard ones.)
Paired, capsule-clustered 95% CI: ours vs ReAct −0.012 [−0.07,+0.05] (6/8); vs Codex +0.030
[0.00,+0.06]; vs Finch +0.157 [+0.07,+0.25] (31/5); vs MoA −0.012 [−0.06,+0.03]; evidence
arbiter vs majority vote +0.030 [−0.01,+0.07] (7/2); repair loop vs evidence team 0/1.
- No combination beats the best single agent; the best agent *flips* between splits (dev:
  Finch 0.231 > ReAct 0.154; test: ReAct 0.542 > Finch 0.373). Choosing the single agent on
  dev costs −0.157 on test; every team stays within 0.04 of the per-split best. On a
  single-agent-solvable benchmark, composition is insurance, not a gain.
- The audit-feedback repair loop is a negative result: re-running failing analysts with the
  reviewer's findings changed 1 answer in 166 (for the worse). Dropped from the method.
- mstest3 ours_escalate (post-hoc on mstest3): 0.739 (v3.1 0.715); pooled n=48 0.736 vs ours_v31
  0.716 (4/48 tasks changed, 0/4 worse, p 0.125); vs GEPA 22/8 p 0.009; vs editor 14/10 p 0.92;
  vs Codex lead 38/0. Escalation is a small, consistent add-on, not a different method.
- Framing fixed: the "best fixed dev interface" was selected on msdev with the hidden grader — it
  is an answer-informed reference, not an answer-free competitor (docs updated).
- Cleanup: BAIS-DPTA code removed (rejected benchmark; recoverable from commit ed4eaf4).

## v4 Task 1 (2026-10-14): branch cleanup and `agentcoop/store.py` (no spend)
- Removed (recoverable from tag `v3.4.1`): workflow DAG executor, handoffs, pipelines,
  eval_free, `optimize/`, `evidence/`, shape-1 agents (de_analyst, donor_check, geneagent +
  shim/sitecustomize, integrator, orchestrators, site), BixBench team/evidence agents, BixBench
  and real-case task builders, v3 graders, all v3 experiment scripts incl. `replay_check.py`.
  Reason: v4 replaces the workflow optimizer with the cross-examination protocol (spec §11).
- Kept as migration sources for Tasks 2 and 8 (not imported by any live module):
  `codex_agent.py`, `coder.py`, `bix.py`, `tasks/planted.py`, `tasks/multisite.py`,
  `external_agents/run_finch.py`. Only `codex_agent.py`, `coder.py` and `bix.py` fail to import
  until Task 2 replaces them (they import the removed `workflow`/`integrator`); `planted.py` and
  `multisite.py` depend only on numpy/scipy/anndata and each other; `run_finch.py` is a
  standalone script for the Finch env.
- `store.memo` replaces the node cache: same first-writer-wins publish (atomic hard link) and
  nominal-cost accounting (per-call `llm.tag` + `llm.spent`). New: a hit ledgers its nominal
  cost via `llm.record_replay` (the v3 node cache added it to the run total, not the ledger),
  and a miss returns the stored JSON form, so a miss and a hit return equal dicts.
- Found while testing: `llm._conn` races on a brand-new ledger file. Two threads opening it at
  once fail with `database is locked` at `PRAGMA journal_mode=WAL` in 7 of 30 trials; an
  existing (already-WAL) ledger is unaffected. `llm.py` is unchanged in Task 1; the
  concurrency test creates the ledger first. Parallel runners must touch the ledger once
  before starting workers.

## v4 Task 2: sandboxed site-agent harnesses (2026-10-14/15, $0 actual)
- sandbox-exec profile: the first form ((allow default)(deny network*)(allow network* (remote ip
  "localhost:*")) + silo denies) blocks *binding* loopback ports: jupyter_client's
  find_available_port fails with EPERM, so Finch's kernel could never start (every Finch site would
  have failed for a non-scientific reason). Added (allow network-bind network-inbound (local ip
  "localhost:*")); external connects still EPERM. (allow network* (local ip "localhost:*")) was
  rejected: it re-opened outbound connections (1.1.1.1:80 connected). tests/test_sandbox.py.
- Codex: `-s workspace-write` replaced by Codex's permission profile (default_permissions="site";
  filesystem :root=read, workspace=write, data/runs/.secrets/CODEX_HOME/WS_ROOT=none;
  web_search="disabled"). Smoke (codex-cli 0.159.0): sibling workspace and repo data unreadable
  (EPERM), curl cannot resolve, ok.txt written, $0.0014 nominal / $0 actual. A prompt asking directly
  to read a denied file was refused by the model without executing anything: judge sandbox checks
  by command_execution items, not by the agent's message.
- Codex runs commands in a login zsh; /etc/zprofile's path_helper moved the analysis env to the end
  of PATH (python3 -> /usr/local/bin/python3, no scanpy). `-c allow_login_shell=false` -> /bin/zsh -c,
  python3 = bixbench env. Same reason for `bash -c` (not v3's `bash -lc`) in the ReAct shell.
- Codex items outside {agent_message, reasoning, command_execution, file_change, todo_list} (web
  search, MCP, ...) fail the call as CodexFailed (charged, never cached).
- Spec §8 harness test (each harness, sibling workspace + data/hidden/ + network): bare sandbox and
  ReAct kernel/shell offline; Finch's interpreter + Jupyter kernel under the same profile
  (AGENTCOOP_SLOW_TESTS=1); Codex live through run_codex (AGENTCOOP_LIVE_TESTS=1, 2026-10-15:
  sibling and hidden reads EPERM, curl cannot resolve, ok.txt written, $0.0018 nominal / $0 actual).
  data/hidden/ is denied as part of data/; agents/ code never names it.
- Cold scanpy import in a fresh workspace ≈ 18 s (matplotlib font cache built in the workspace's
  MPLCONFIGDIR); accepted per site call, so the real-interpreter test is opt-in
  (AGENTCOOP_SLOW_TESTS=1, ≈ 25 s).

### v4 Case 3 base (OneK1K, frozen 2026-10-14)
- `data/bench/onek1k/base.h5ad` = `python -m agentcoop.cases.planted data/bench/onek1k/onek1k.h5ad`
  (seed 0): 80 of the 205 OneK1K donors that have >= 1,000 cells in the six case populations and
  >= 20 cells in each; 40 dev (fsdev) / 40 test (fstest) donors, disjoint (lists in base.json);
  <= 1,500 cells per donor, stratified by population; 105,091 cells x 26,296 genes (genes with an
  ambiguous symbol or no counts dropped, so symbol <-> Ensembl is a bijection for the ID knob).
  sha256 437758b247c74dff3718407982d33b0ef9692026c86f6553c0ea0ebed08917d6.
- Alias gene-ID scheme (spec §7 lists symbol / Ensembl / alias): each gene's first previous HGNC
  symbol, else its first alias symbol, is used if HGNC lists that name for one gene only and it is
  no approved or base symbol; otherwise the gene keeps its symbol. Source: HGNC complete set
  2026-10-06 (dated archive, sha256 8bf6f686e640dbc864a346484626953621b0500d900d26cd085a35c2782d725d),
  frozen into base var["alias"]; 16,244 of 26,296 genes get a legacy name (e.g. ATP5F1D -> ATP5D).
  Reason for previous symbols first: outdated official symbols are what real legacy annotations
  carry. Reason for the uniqueness rules: symbol <-> alias stays a bijection, so a correct view
  exists and view errors are gradable.
- Under ids_labels, A/B/C get the three schemes in random order (M draws one). Reason: each
  scheme is then measured at the same number of sites in the §6 view-error rate.
- Resource keys: gene-set libraries go under `genesets/` in CaseSpec.resources
  (`genesets/GO_Biological_Process_2023.txt`), which is where MA-genome's `gene_set_library` reads
  them; the symbol table is `symbols.tsv` (ensembl_id, symbol, alias).
- Reason for the 20-cell floor in subsampling: the median OneK1K donor has 20 CD14 and 10 CD16
  monocytes; proportional subsampling alone would push rare populations under the tester's
  20-cell minimum (TesterParams.min_cells) and make them untestable by construction.
- Reason one-site programs never share a site: at the assay-confounded artefact site an effect seen
  only there cannot be told apart from the artefact, so its cause (site-biology vs site-artefact)
  would be ungradable.
- Answers: `cases/planted.py` hands each planting record to a `seal` callback; only
  `eval/planted.py` writes hidden/truth.json (spec §8; build suites with
  `python -m agentcoop.eval.planted <suite> <pool> <lo> <hi>`, knobs balanced 2 x 2 by seed order).
