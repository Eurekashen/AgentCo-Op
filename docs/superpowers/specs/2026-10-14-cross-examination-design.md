# AgentCo-Op v4 — cross-examination between external agents across resource silos

Status: design approved in conversation (2026-10-14); revised after a four-lens adversarial
review (77 findings, all 29 high-severity ones verified); awaiting written review.
Branch `v4` (from `v3`; v3 results reproducible from tags `v3.4.1` / `v3-final`).

## 1. Thesis and what counts as evidence

> When the evidence needed to answer a scientific question is distributed across independently
> operated agents that cannot pool their resources — each holding data in its own,
> non-harmonized form — cross-examination lets them (1) recover much of what pooling the data
> would give, without pooling; (2) do better than a fixed federated pipeline run on the same
> agent-built inputs; (3) with no reference answer or optimization target at run time.

"Cannot pool" is a real governance constraint (institutions may not share patient-level data).
The experiments simulate it with public data split by institution, as federated-learning work
does, and the boundary is **enforced by an OS sandbox and tested** (§8).

Because "a single site cannot see the other sites" is true by construction, beating single
sites is a sanity check, not evidence for the thesis. The evidence is:
- **Silo-cost recovery** = (ours − best single site) / (pooled oracle − best single site), on
  each case's primary endpoint. The pooled oracle violates the silo and is a reference only.
- **Ours vs MA-genome**, the textbook fixed federated meta-analysis on the same agent-built
  site views (§4). Clause (2) is supported only if ours beats it.
- **Mechanism decomposition at $0** (§4): how much comes from cross-testing, from the fixed
  decision rule, from view repair, and from disagreement resolution.
- **Prior-only control**: gains must exceed what the backbone asserts without data.

Not claimed (refuted by v3 and by the 2026-10-14 scouting): that collaboration beats a single
agent that can see all the data; that adding installable tools or public APIs is something a
single agent cannot do.

## 2. Method: the cross-examination protocol

### 2.1 Roles

- **Site agent** — an independently developed external agent bound to one site's private data:
  Codex CLI or FutureHouse Finch (both external products), all on `gpt-6-luna`. The in-house
  ReAct coder is used only in the all-ReAct robustness run, never as a site agent in the main
  arms. A site agent (a) builds the **site view**, (b) proposes claims, (c) answers follow-ups.
- **Site tester** — deterministic code run inside the site boundary on the site view. No LLM.
  Returns summary statistics only.
- **Coordinator** — LLM plus fixed rules. Sees claims, test results, analysis specs and site
  metadata cards; never any sample- or cell-level data.

### 2.2 Site view (machine-readable; produced in step 1, repaired in step 4)

A directory with fixed files, validated on write:
- `genes.tsv` — native feature ID → HGNC symbol (one symbol per row; unmapped rows dropped and
  counted).
- `samples.tsv` — sample → donor, case group (from the case vocabulary), covariates.
- `cells.tsv` (scRNA only) — cell → population (case vocabulary).
- `spec.json` — the site's analysis spec: contrast definitions, sample unit, normalization, gene
  mapping source, covariates, exclusions. Mandatory; reading it costs no call.
From the view, deterministic code computes once per view the **site table**: donor ×
population × gene pseudobulk (scRNA) or sample × gene matrix (bulk), plus group labels. Testers
read only site tables, never cell-level files.

### 2.3 Claims and test results

Claim (one schema for every arm, validated at submission):
`{id, site, kind: "de" | "composition", population, contrast: [target_group, reference_group],
up_genes: [...], down_genes: [...], effect, n_units, rank, rationale}` — population and groups
from the case vocabulary; `kind=de` needs ≥ 3 genes in total; `kind=composition` has no genes.
`rank` is the arm's own confidence order (§4). Final reports add `label`, `scope` (Case 2) and
`cause` (if a disagreement was classified).

Test result: `{claim_id, site, test: "replication" | "scope", status, effect, se, ci95,
n_per_group, mde, reason}`.

### 2.4 Testers (fixed per data kind; parameters frozen at `v4-freeze`)

- scRNA `de`: donor pseudobulk, ≥ 20 cells per donor × population (otherwise the donor is
  dropped; < 3 donors per group → `inconclusive`). Module score per donor = mean log-CPM of
  `up_genes` minus mean of `down_genes` over genes measurable in the site (mean CPM ≥ 20); needs
  ≥ max(3, 50%) of the claim's genes measurable, else `untestable(genes)`. Effect = Welch t
  difference of means; plus the v3 competitive null (the effect must exceed the 95th
  percentile of 200 random same-size gene sets).
- scRNA `composition`: donor-level logit proportion, Welch t.
- bulk `de` (TB): per-gene rank-to-normal transform within the site; sample score = mean(up) −
  mean(down); effect = AUC, bootstrap 95% CI.
- Status: `supported` = CI excludes the null in the claimed direction and BH-adjusted p ≤ q;
  `contradicted` = CI excludes the null in the opposite direction; `absent` = CI inside the
  equivalence band (±δ around the null); otherwise `inconclusive`. `untestable` when the site
  lacks the groups, population or genes. `inconclusive` and `untestable` are never support and
  never contradiction. All non-proposer tests in one run form one BH family (re-tests included);
  it decides per-site statuses (contradiction). Corroboration pools the non-proposer sites
  (§2.5 step 3). A claim with both up and down genes is tested as two modules and holds only if
  each does (intersection-union; the reported result is the weaker half).
  δ and q are frozen on `fsdev` (scRNA) and set a priori for AUC (δ = 0.05, q = 0.05).

### 2.5 Protocol

1. **Propose.** Every proposing site builds its view and proposes ≤ K claims (one budgeted call).
   Tester-only sites (§5.2 S4, §7 mimic site) build their view only.
2. **Test.** Each claim is first tested by its proposer's tester (a gate: failing → `unsupported`,
   dropped from findings, still reported). Then it goes to every other site whose view holds the
   claim's population and both groups (**replication test**) and, per the case's fixed **scope
   table**, to sites holding a scope contrast (**scope test**, e.g. mimic vs healthy). Scope
   tests feed only the scope rule (§5.2), never step 3 or step 4.
3. **Label** (one shared function, used by ours and by every mechanism arm; ordered, first match):
   `unsupported` (proposer gate failed) → `disputed` (any non-proposer `contradicted`) →
   `corroborated` (pooled replication: Stouffer's z over the non-proposer results' one-sided p
   in the claimed direction — a result failing the competitive null can lower it, never raise it
   — passes BH at q over the run's gated claims, one test per claim — and the inverse-variance
   pooled non-proposer effect is larger than the equivalence band δ) → `site-specific`
   (inverse-variance pooled non-proposer effect is `absent`) → `unresolved` (no non-proposer
   site could test it) → `inconclusive`. The step-3 labels are persisted (`labels_step3`).
   *Revised on `fsdev2` before the freeze (was: ≥ 1 non-proposer `supported` after BH over every
   non-proposer test): with 4 + 4 donors per site, real programs supported at two other sites at
   raw p 0.01–0.04 were each demoted by BH over ~90 tests; see `findings.md` 2026-10-21.*
4. **Resolve.** Triggered by `disputed` claims and by `untestable(genes|groups)` results. The
   coordinator reads the involved sites' `spec.json` and metadata cards and classifies the cause:
   - *analysis choice* → both sites re-test once under the case's **pre-registered default
     spec** (not chosen by the coordinator); the re-test result is final;
   - *handoff loss* (gene IDs, label vocabulary, group mapping) → one repair call to the site
     agent with the specific defect; the site re-tests;
   - *site artefact* (e.g. groups differ in assay or source lab in the metadata card) → no
     re-test; that site's result is set to `inconclusive` with the cause recorded;
   - *site biology* → no re-test; recorded.
   Site agents in follow-ups see the claim (groups, genes) and the defect, never other sites'
   results or statistics. At most one follow-up per (claim, site). Step 3 is re-applied after all
   follow-ups.
5. **Stop** when no `disputed` claim and no view-caused `untestable` remain, or the follow-up
   budget (F calls per run, §10) is spent. Findings = `corroborated` claims, ordered by `rank`
   (label tier, then number of supporting sites, then |Stouffer z|). Every other claim is
   reported with its label and cause.

Objective-free: the protocol keeps claims by agreement between independent resources and stops
on "no open disagreement". It never compares alternatives by a score, never re-tests a claim
twice, and the re-test spec is fixed in advance. Rules and thresholds are frozen on `fsdev`
before any real-case run; no quantity computed from answers or held-out data is used at run
time. The design-time freeze on `fsdev` uses hidden answers and is stated as such.

## 3. Shared task package and fairness

Every arm that analyses data receives the same task package: question, case vocabulary, claim
schema, claim cap K, site metadata cards (codebook, assay, donors per group). Arms that start
with independent site analyses (single site, union, vote, MA-genome, ours, debate round 0)
share the same step-1 runs through the cache (same request, same sample index). Every
compositional arm gets the same site interface: `ask_site(site, request)` and
`test_claim(site, claim)`, the same full claim × site test matrix of the step-1 claims, and the
same budget B site-agent calls. Codex coordinator, group chat and debate keep their own outputs
(no post-hoc filtering by our rule).

## 4. Arms (all `gpt-6-luna`; all emit the claim schema with `rank`)

Baselines:
| arm | definition |
|---|---|
| best single site | one site agent alone; output = its claims that pass its own tester. Every site is reported; "best" = the maximum over sites on each endpoint (an after-the-fact choice that favours the baseline, i.e. conservative against ours), except Case 2 where it is S2, the only site able to propose TB-vs-OD claims |
| single site, common harness | the same with every site run by Codex |
| union | all step-1 claims of all sites, no filter; rank = #proposing sites, then own p |
| majority vote | claims proposed (matched by the shared matching rule) by ≥ 2 sites |
| **MA-genome** (primary baseline) | each site tester exports per-gene effect and SE for every (population, contrast) in its view; inverse-variance fixed-effect meta-analysis + Cochran's Q; genes with BH < 0.05 and the same sign in ≥ 2 sites; per (population, contrast, direction), enrichment against the gene-set library the site agents are given, one claim per non-redundant term at FDR < 0.05 with its overlap genes; composition by meta-analysis of per-site proportion tests. Runs on the step-1 views after an untargeted self-check (one call per site, fixed checklist; number of calls = ours' median follow-up count on `fsdev`, frozen) |
| Codex coordinator | Codex CLI as lead with the shared tools and budget |
| group chat | round-robin group chat among site agents and a coordinator (in-house implementation of AutoGen's RoundRobinGroupChat turn protocol, stated as such), shared tools and budget |
| debate | Du et al. 2023: round 0 = step-1 claims, one revision round with the shared tools, majority |

References (not targets):
| arm | definition |
|---|---|
| prior-only | backbone given exactly the task package, no count data; ≥ 3 draws |
| pooled oracle | one Codex agent with all site views concatenated by the case script (silo-violating) |
| MA-tools | MA-genome on views built by fixed tools (GPL annotation / frozen HGNC table, CellTypist `Immune_All_Low` per site, codebook maps committed before runs) — the "manual harmonization" reference |

Mechanism decomposition (computed from ours' stored runs, $0, paired):
| arm | isolates |
|---|---|
| no-routing | union of single-site outputs (sanity check: cross-testing vs none) |
| no-resolution | ours' `labels_step3` (isolates step 4) |
| MA-claims | Stouffer over non-proposer tests of ours' step-1 claims, BH < 0.05, no proposer gate and no contradiction veto (since the step-3 revision, ours corroborates by the same pooled test, so this arm isolates the gate, the veto and resolution) |
| MA-genome on repaired views | MA-genome on ours' final views (isolates view repair) |

Robustness: ours with all site agents = Codex; ours with all = ReAct; Case 3 subset with the
framework model `gpt-4.1-mini` (CLAUDE.md rule).

Primary comparisons (pre-registered): ours vs MA-genome; ours vs best single site; silo-cost
recovery. Interpretation rules, fixed before any run:
- Step 4 has value iff ours − no-resolution > 0 (paired), promoted claims replicate at least as
  often as no-resolution's corroborated claims minus a margin frozen on `fsdev`, and demoted
  claims replicate less often than kept ones. Otherwise reported as "the gain comes from
  agent-built views and deterministic cross-testing".
- Clause (2) is supported only if ours > MA-genome; ours vs MA-tools is reported alongside.
- If prior-only's precision@10 is within 0.05 of ours (or inside ours' bootstrap CI), plain
  precision is not reported as a collaboration gain for that case; only the prior-adjusted
  endpoint counts.

## 5. Real-data case studies

### 5.1 Case 1 — severe COVID-19 PBMC across four institutions

Sites: Berlin (Schulte-Schrepping cohort 1; cases 10x v3, controls from other labs partly on
10x v2), Stanford (Wilk 2020, Seq-Well), Emory (Arunachalam 2020), Seoul (Lee 2020); cellxgene
per-study files of the Tian 2021 collection. Site files are rebuilt from raw counts with a
column whitelist (opaque donor/sample IDs, opaque native severity codes + site-only codebook,
native author cell labels where present, assay, source cohort, sex/age); harmonized columns,
disease columns, obsm/obsp/uns are dropped (tested). The case script keeps one sample per donor
(the sample with the highest severity code; ties → earliest), so contrasts are between donors.
Gene IDs are already Ensembl with a symbol column (cellxgene schema): heterogeneity in this case
lies in labels, severity scales, assays and confounded controls, not in gene IDs.

Question: which cell-type-specific transcriptional and compositional changes separate
severe/critical COVID-19 from health (primary) and severe from mild (secondary), and which are
site artefacts?

Site annotation: sites without native labels annotate their own cells with offline resources in
the workspace (CellTypist `Immune_All_Low` model, PanglaoDB/CellMarker libraries). Blocklisted
everywhere: CellTypist `Healthy_COVID19_PBMC` (trained on Stephenson = held-out) and
`Adult_COVID19_PBMC` (Wilk = a site).

Held-out validation (built by `eval/covid.py` only; files stored under `data/hidden/`, inside
the sandbox deny list; reference hash committed before runs):
- Stephenson 2021 (UK, 10x 3′, RNA + CITE-seq) — donor pseudobulk with pyDESeq2 (an operator
  different from the testers' module Welch test): each claim's genes scored as a module,
  replicated = same sign and BH-FDR < 0.05 (BH family = the arm's claims).
- COMBAT 2022 (Oxford, 10x 5′ v1) — same-sign check.
- Crosswalks fixed and hashed before runs: case cell vocabulary → Stephenson `author_cell_type`
  and COMBAT `major_subset`; severity: severe = Stephenson Severe + Critical, COMBAT SEV + CRIT;
  mild = Mild; healthy = Healthy / HV (LPS, non-COVID, sepsis, flu excluded).
- CITE-seq protein (CLR) for surface-marker claims (HLA-DR, CD14, CD16, CD163, CD86, CD38) —
  secondary.
- Artefact truth: a technical signature from public healthy 10x PBMC v2 vs v3 datasets computed
  by the grader; a claim is an artefact if it fails Stephenson replication and ≥ 50% of its genes
  are in that signature with the same sign.
- Literature list from studies no site holds (Stephenson, COMBAT, Silvin 2020) — secondary.

Endpoints: primary — (a) prior-adjusted replication: drop each claim's genes that any prior-only
draw named for the same population, contrast and direction; residual lists with ≥ 3 genes are
scored; report residual precision@k and replicated residual claims at k ∈ {5, 10, 20} by each
arm's `rank`; (b) artefact rejection at matched claim count. Secondary — plain precision@k,
recall vs a reference of non-redundant programs significant in both held-out cohorts (Reactome
2022, a library no arm is given), F1, protein concordance, literature hits. DE claims only for
CD14 monocytes, CD4 T, CD8 T, NK, B; pDC, non-classical monocytes and plasmablasts only as
composition claims.

Statistics: the 4-site federation, 3 draws per arm; leave-one-site-out federations computed at
$0 by masking a site in stored outputs (exact before step 4, approximate after; robustness only);
descriptive with bootstrap CIs.

### 5.2 Case 2 — active TB whole-blood signature across hospital cohorts

Sites (natively heterogeneous: platform, scale, ID scheme, label vocabulary):
- S1 London (GSE19439 + GSE19444; Illumina HT-12 v3, raw linear): TB / LTBI / healthy.
- S2 South Africa (GSE37250, region = South Africa; HT-12 v4): TB 93 / LTBI 96 / OD 111, HIV±.
- S6 The Gambia (GSE28623; Agilent GPL4133 feature numbers): TB / LTBI / uninfected.
- S4 OD-only, **tester-only** (GSE22098): adult autoimmune stratum (adult-onset Still's 31,
  adult SLE 28 vs adult controls) and paediatric bacterial stratum (Staph 40, Strep 12 vs
  paediatric controls); paediatric SLE excluded. S1 and S4 share a lab (SuperSeries GSE19491);
  stated, and S4 can only demote claims.
- Expansion only if budget allows: S3 Malawi (GSE37250, region = Malawi), S5 Bloom
  (GSE42825/26/30 merged and deduplicated).

What a single site lacks: S2 alone (as Kaforou 2013 did on GSE37250) can propose a signature and
label scope within its own population. It lacks healthy controls, non-African populations, other
platforms, and adult autoimmune / bacterial mimics. Case 2 therefore tests whether
cross-examination **corrects S2's scope labels and improves cross-population generalization**,
not that one site is structurally unable to label scope. Pre-registered prediction: on TB vs
pneumonia (GSE73408) ours ties S2-alone in AUC; gains, if any, are in scope labels and sign
stability across platforms.

Scope table and rule (frozen in `cases/tb.py` before runs): a claim with contrast TB vs
{LTBI, HC} or TB vs OD gets a TB-vs-OD scope test at every site holding OD (S2; S3/S5 if added)
and an m-vs-HC scope test at S4 per mimic class m ∈ {autoimmune, bacterial}. Scope ∈
{specific vs <class>, not specific vs <class>, inconclusive}:
- not specific vs bacterial_resp / mixed_OD: a non-proposer TB-vs-OD test has CI upper < 0.65,
  or S4's bacterial m-vs-HC AUC in the claim's direction has CI lower > 0.5 and the proposer's
  TB-vs-HC AUC minus it is < 0.15 (demote-only);
- specific: a non-proposer TB-vs-OD test has AUC ≥ 0.70 with CI lower > 0.5 and no
  not-specific condition holds;
- inconclusive: otherwise (including only in-sample proposer evidence). AUC thresholds are a
  priori conventions, equal to the grader's.

Held-out validation (hidden; exclusion table of shared patients/SuperSeries published;
GSE19442, GSE107991/2 excluded):
- Confirmatory: GSE73408 (US, Affymetrix; TB / LTBI / pneumonia), GSE89403 (South Africa,
  RNA-seq, technical replicates summed; TB vs healthy), GSE101705 (India, RNA-seq; TB vs LTBI).
- Exploratory (labelled): GSE39940 (children, HIV±, OD; same consortium as S2), GSE83456 and
  GSE34608 (sarcoid), GSE107994 (baseline, outliers removed, one sample per patient).
- Scoring rule (frozen): per cohort, per-gene rank-to-normal, score = mean(up) − mean(down),
  AUC per contrast.

Endpoints: primary — (a) among claims where ours' scope label differs from S2-alone's, the
fraction correct on GSE73408 TB vs pneumonia (correct "specific" needs held-out AUC ≥ 0.70,
correct "not specific" < 0.65; [0.65, 0.70) excluded and reported); (b) final-signature AUC on
GSE73408 TB vs pneumonia with bootstrap CI — final signature = up/down genes of the top-5 findings
by rank (fixed rule for every arm). Secondary — TB vs LTBI/HC AUC on the three confirmatory
cohorts (non-inferiority margin 0.03), claim replication rate (≥ 2/3 of confirmatory cohorts
holding the contrast), S4-driven relabels (count and genes; autoimmune mimics have no held-out
cohort, so unvalidated). Memorization controls: prior-only; published signatures scored with the
same rule (Sweeney_OD_3, Zak_RISK_16, Kaforou_27, Kaforou_OD_44, Kaforou_OD_53 — the Kaforou
ones derive from GSE37250); a paired gene-ID-blinded replicate of ours and S2-alone (consistent
hashing applied by the case script after symbol mapping; this replicate disables the gene-mapping
handoff branch and is reported as a memorization check only).

Statistics: claims and per-cohort AUCs with bootstrap CIs; 3 draws per arm; descriptive.

## 6. Headroom gate ($0, before any LLM spend on a case)

Per case, before the freeze and before any held-out file is opened, using site data only:
compare (i) best single site, (ii) MA-tools, (iii) scripted ours = MA-tools plus the correct
resolution applied by script (COVID: Berlin restricted to its v3 controls; TB: S2 with S4-based
pruning), scored by out-of-site replication (COVID: each clean-control site Stanford, Emory,
Seoul held out in turn, Berlin always in the federation; TB: the expansion sites S5 Bloom (TB vs
sarcoid / pneumonia / lung cancer) and S3 Malawi, which are not in the minimal federation, as
targets for the scope of S2's claims). Pass rule, written into `findings.md` before computing:
(iii) − (ii) ≥ 0.05 precision in ≥ 2 of 3 folds (COVID) or on both targets (TB), and (iii) > (i).
Also record whether the union of single-site claim sets covers MA's replicating hits (if not,
ours is capped below MA on recall and the case claims precision only). Scripted ours is a ceiling:
passing is necessary, not sufficient. A failing case is reframed or dropped before the freeze,
with the reason recorded. The same gate runs on `fsdev` against the hidden truth, together with
the measured view-error rate of Codex-built views; if that rate is ≈ 0, the view-repair claim is
dropped and Case 3's ID/label knob is removed.

## 7. Case 3 — controlled multi-site suite (quantitative, memorization-free)

Base: a frozen donor subset of a healthy multi-donor PBMC atlas used by no case or held-out
cohort (OneK1K via CELLxGENE Census; fallback another healthy PBMC atlas with ≥ 80 donors);
donors with ≥ 1,000 cells; disjoint donor pools for `fsdev` and `fstest` (≥ 36 donors each);
donor lists and file hash recorded. Kang 2018 is not used.

Each task: 3 proposing sites with 4 control + 4 treated donors each (condition at donor level,
unpaired, as in the real cases) and one tester-only mimic site with 4 control + 4 mimic donors.
Planted by binomial thinning (programs = core gene sets, as in v3): reproducible programs (all
proposing sites); half of them also planted in mimic donors (truth: shared vs specific);
site-specific biology (one site's treated donors); composition shifts (subsampling a cell type in
treated donors). Knobs, balanced 2 × 2 (artefact on/off × IDs+labels on/off):
- artefact: at one site a coherent program thinned in control donors only, with an `assay`
  column differing between that site's control and treated donors (as in Berlin);
- IDs+labels: per-site gene ID scheme (symbol / Ensembl / alias) and renamed cell-type and
  condition codes with a site codebook; mapping resources (symbol tables) supplied in the
  workspace, as for real sites.
Suites: `fsdev` 8 tasks (2 per cell; debugging and freezing), `fstest` 32 tasks (8 per cell;
built after the freeze from unseen donors, run once, 1 draw per arm; task = unit; paired
Wilcoxon). Knob main effects pre-registered (16 vs 16); per-cell results descriptive.

Grading (`eval/planted.py`): a claim matches a planted program if population and direction
match and its genes overlap the program's core genes with Jaccard ≥ 0.2 (frozen); a claim matching
several programs counts once for its best match. Metrics: F1 and precision vs reproducible
programs; artefact rejection; scope accuracy (shared vs specific, via the mimic site); cause
localization accuracy (artefact vs site biology vs handoff loss vs reproducible) against the
planted record. Note: majority vote and MA with ≥ 2 sites reject a one-site artefact by
construction, so artefact rejection separates ours mainly from union and single sites; vs
MA-genome the test is the IDs+labels knob.

## 8. Invariants (tested)

- Protocol, site agents, testers and baselines never import `agentcoop.eval` or read
  `data/hidden/`; graders live only in `eval/`; case builders that write hidden references live
  in `eval/` too, so `cases/` and `coop/` have no path to hidden data.
- **OS sandbox**: every site-agent process runs under `sandbox-exec` with a profile that denies
  file reads/writes to the repo's `data/`, `runs/`, `.secrets/`, `data/hidden/`, `CODEX_HOME`
  and the workspace root except its own workspace, and denies network except loopback; Codex
  site runs use Codex's own permission profile with the same deny list and web search disabled;
  the Codex coordinator talks to sites through a file mailbox in its workspace. A test runs each
  harness and asserts that reading a sibling workspace, `data/hidden/` and the network fails.
- Site files contain only whitelisted columns (test per case).
- Held-out cohorts and sites share no GSM, SuperSeries or donor (exclusion table, test).
- `inconclusive` / `untestable` are never support.
- `sample` semantics: same index ⇒ exact replay; new index ⇒ honest re-draw.
- Nominal vs actual cost; every call charged, failed follow-ups included.

## 9. Order of work and pre-registration

1. v4 skeleton (§11), sandbox, testers, label function, protocol, arms, `fsdev` generator, tests
   — $0.
2. Headroom gates (§6) for `fsdev`, COVID, TB — $0; data prep for both real cases, held-out
   references built and hashed.
3. All arms on 2 `fsdev` tasks; debug; all 8 `fsdev` tasks; freeze testers, δ/q, prompts, K, B,
   F, matching rules, case scope tables, crosswalks → tag `v4-freeze`.
4. Build `fstest`; run every arm once; grade.
5. COVID: every arm, 3 draws (Codex-heavy arms scheduled within the daily Codex quota).
6. TB: minimal 4-site version (S1, S2, S6, S4), every arm, 3 draws; expansion only if budget
   allows.
7. Report every arm, negative results included, with the interpretation rules of §4.
No method change after the freeze without a new suite; every change gets a reason in
`findings.md`.

## 10. Budget and throughput

Hard cap $180 actual (`AGENTCOOP_BUDGET_USD=180`; $104.7 spent before v4). Measured costs: a
ReAct site run ≈ $0.007 nominal (143 s), Finch ≈ $0.005; Codex runs bill the local login ($0
actual) and are capped at ≤ 250 runs/day (observed throughput, no quota errors). Budget per run:
B = 2S site-agent calls for compositional arms (S step-1 calls + ≤ S follow-ups), F = S
follow-ups for ours; wall-clock cap per call 15 min (planted) / 30 min (real). Run-count caps:
`fsdev` ≈ 600 site calls, `fstest` ≈ 1,100, COVID ≈ 170, TB ≈ 200; re-estimated after the first
2 tasks of each stage. Planted stages run 4-way parallel; real-case arms run one at a time.

## 11. Code layout (v4) — old → new

```
agentcoop/
  llm.py  proxy.py                      kept
  store.py      ← workflow.py node cache (first-writer-wins publish, nominal_usd per entry, per-call tag)
  agents/
    workspace.py ← bix.workspace/exec_env without ctx; APFS clones; sandboxed(cmd, ws)
    codex.py     ← codex_agent.run_codex, CodexFailed, _events, _native_model (permission profile, no network)
    react.py     ← coder.Kernel, TOOLS, coding_loop (sandboxed kernel)
    finch.py     ← ctx-free wrapper of external_agents/run_finch.py (kept unchanged)
  coop/
    claims.py   schema, validation, matching rule
    view.py     site view files, validation, site-table computation
    tester.py   ← evidence.checks.donor_tables / competitive null, parameterized per data kind
    site.py     boundary: private workspace, tester calls, follow-ups, file mailbox, budget B
    protocol.py steps 1–5, labels_step3 / labels_final, scope rule hook
    arms.py     every arm of §4 against the site interface
  cases/  planted.py (← tasks/multisite + planted primitives)  covid.py  tb.py   site files, codebooks, scope tables
  eval/   planted.py  covid.py  tb.py   hidden references and graders only
experiments/ run.py  analyze.py
tests/
```
Removed from `v4` (recoverable from `v3.4.1`): `workflow.py` DAG executor, `handoffs.py`,
`pipelines.py`, `optimize/`, `evidence/judge.py`, `eval_free.py`, shape-1 agents (de_analyst,
geneagent + its shim and sitecustomize, integrator, donor_check, orchestrators), BixBench agents
and tasks, v3 experiment scripts and `replay_check.py`. A store-based replay check replaces the
latter as the refactoring guard.
