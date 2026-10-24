# AgentCo-Op v4 — results

Every arm, negative results included, read with the interpretation rules fixed in the design
spec before any run (§4). Design: [spec](superpowers/specs/2026-10-14-cross-examination-design.md),
[method summary](method.md); the full log of decisions, failures and their reasons is
[`findings.md`](../findings.md). Backbone for every main arm: gpt-6-luna (Codex `-m gpt-6-luna`
on the local login; Finch through the shared proxy). Spend for all of v4: $5.55 actual
(planted test, both real cases), on top of $104.7 before v4.

**Thesis.** When the evidence for a question is spread over independently operated agents that
cannot pool their data, each holding it in its own non-harmonized form, cross-examination —
sites propose claims, every other site tests them with fixed code on its own data, a claim is
kept only when independent sites agree — finds more of what is true and less of what is not
than a fixed federated pipeline on the same agent-built inputs, with no reference answer or
optimization target at run time.

## 1. Pre-registered planted test (`fstest`)

32 tasks built after the freeze (tag `v4-freeze`) from OneK1K donors never used in development;
3 proposing sites with 4 + 4 donors each and one tester-only mimic site; planted reproducible
programs, site-specific programs, a composition shift, and in half the tasks an assay artefact
at one site. One draw per arm; the task is the unit; paired Wilcoxon against ours.

| arm | F1 [95% CI] | precision | recall | artefact rejected | ours − arm (W/L/T, p) | site calls | nominal $ |
|---|---|---|---|---|---|---|---|
| **ours** (cross-examination) | 0.88 [0.85, 0.92] | 0.97 | 0.83 | 1.00 | – | 5.4 | 0.063 |
| MA-claims (pooled rule, no gate/veto) | 0.84 [0.79, 0.88] | 0.86 | 0.83 | 1.00 | +0.04 (16/1/15, p 0.0005) | 4.0 | 0.049 |
| MA-genome (federated meta-analysis) | 0.74 [0.67, 0.81] | 0.74 | 0.79 | 0.94 | +0.14 (22/9/1, p 0.001) | 5.0 | 0.061 |
| majority vote | 0.71 [0.63, 0.79] | 0.93 | 0.63 | 1.00 | +0.17 (18/1/13, p 0.0002) | 3.0 | 0.045 |
| debate | 0.69 [0.62, 0.76] | 0.97 | 0.58 | 1.00 | +0.19 (22/1/9, p 3e-05) | 7.0 | 0.074 |
| Codex lead + tools | 0.66 [0.60, 0.73] | 0.79 | 0.62 | 1.00 | +0.22 (27/0/5, p 6e-06) | 6.8 | 0.071 |
| group chat | 0.63 [0.57, 0.69] | 0.65 | 0.67 | 1.00 | +0.25 (29/0/3, p 3e-06) | 7.0 | 0.075 |
| best single site (hindsight) | 0.58 [0.53, 0.62] | 0.56 | 0.77 | 1.00 | +0.30 (29/1/2, p 2e-06) | 1.0 | 0.015 |
| one agent on all sites' data | 0.45 [0.36, 0.53] | 0.58 | 0.47 | 1.00 | +0.44 (31/0/1, p 1e-06) | 1.0 | 0.007 |
| no cross-testing | 0.36 [0.32, 0.40] | 0.24 | 0.84 | 0.38 | +0.53 (32/0/0, p 5e-10) | 3.0 | 0.045 |
| union of site claims | 0.30 [0.26, 0.33] | 0.19 | 0.84 | 0.38 | +0.59 (32/0/0, p 8e-07) | 3.0 | 0.045 |
| prior only (no data) | 0.00 [0.00, 0.00] | – | 0.00 | 1.00 | +0.88 (32/0/0, p 7e-07) | 0.0 | 0.000 |

- **Clause (2) of the thesis holds on the test suite**: ours beats the fixed federated pipeline
  run on the same agent-built views (MA-genome) by +0.14 F1 (22/9/1, p 0.001), with precision
  0.97 vs 0.74 at similar recall.
- **Where the gain comes from.** By the pre-registered rule for step 4, it adds nothing here:
  ours and no-resolution are identical on all 32 tasks, so the gain is reported as coming from
  *agent-built views and deterministic cross-testing*. Cross-testing is the large effect (no
  cross-testing: 0.36); the proposer gate and the per-site contradiction veto add a small,
  consistent precision gain over the same pooled rule without them (MA-claims, +0.04, p 0.0005).
- **Not artefact handling.** The gain over MA-genome is +0.19 (p 0.002) on the 16 tasks
  without the artefact and +0.09 (p 0.21, n.s.) on the 16 with it. MA-genome already rejects
  most one-site artefacts (0.94), as the spec anticipated.
- **Agent-built views were nearly error-free** (error rate ≤ 0.001 for Codex and Finch), which
  is why the IDs/labels knob was dropped before the test (§6) and why "view repair" is not
  claimed as a contribution.
- **Pooling is not an upper bound here.** One Codex agent given every site's data scores 0.45,
  below a single site; the silo-cost recovery ratio is undefined and not reported.

## 2. Case 1 — severe COVID-19 PBMC across four institutions (primary real case)

Sites: Berlin (Schulte-Schrepping 2020; COVID-19 on 10x v3, controls from other cohorts on 10x
v2 and v3), Stanford (Wilk 2020, Seq-Well), Emory (Arunachalam 2020), Seoul (Lee 2020), rebuilt
from raw counts with a column whitelist. Held out: Stephenson 2021 (replication by a pyDESeq2
module test, BH over each arm's claims) and COMBAT 2022 (same-sign check). Three draws per arm;
descriptive. Primary endpoint: prior-adjusted replication — genes a data-free prior-only model
names are removed from each claim before scoring.

| arm | findings | replicated in top 10 (precision) | replicated in top 20 (precision) |
|---|---|---|---|
| **ours** | 10.0 | **6.7 (0.97)** | 7.7 (0.97) |
| MA-claims | 18.3 | 8.3 (0.92) | 11.0 (0.90) |
| MA-genome | 76.3 | 5.0 (0.50) | 10.3 (0.52) |
| union of site claims | 24.7 | 5.3 (0.58) | 8.7 (0.58) |
| no cross-testing | 20.0 | 5.0 (0.55) | 8.0 (0.59) |
| group chat | 7.0 | 4.3 (0.90) | 4.3 (0.90) |
| Codex lead + tools | 5.3 | 4.0 (1.00) | 4.0 (1.00) |
| debate | 3.7 | 2.3 (1.00) | 2.3 (1.00) |
| majority vote | 2.7 | 1.0 (0.56) | 1.0 (0.56) |
| Stanford alone | 5.3 | 3.7 (0.92) | 3.7 (0.92) |
| Seoul alone | 6.0 | 2.7 (0.62) | 2.7 (0.62) |
| Berlin alone (confounded controls) | 5.3 | 3.0 (0.43) | 3.0 (0.43) |
| Emory alone | 3.7 | 1.7 (0.50) | 1.7 (0.50) |
| one agent on all four sites' data | 10.0 | 0.0 (0.00) | 0.0 (0.00) |

- Ours has the highest replication precision of every arm that keeps more than a handful of
  claims, and replicates more claims in its top 10 than any single site, MA-genome, the union,
  or the coordinator-style teams. MA-claims replicates more claims in total (11 vs 7.7 at k = 20)
  at lower precision, and MA-genome's 76 findings contain more replicated claims at k = 20 than
  ours' 10 (10.3 vs 7.7) at half the precision: ours trades recall for precision.
- The prior-only control does not explain the result (its plain precision@10 is 0.47 vs ours'
  0.97, so by §4 plain precision may be reported). Every Stephenson-replicated finding of every
  arm also has the same sign in COMBAT.
- Berlin analysed alone, with its assay-confounded controls, has the lowest precision (0.43). No
  finding of any arm met the pre-registered artefact definition, so endpoint (b) is uninformative.
- Step 4 again changed nothing (no-resolution = ours in all three draws).

## 3. Case 2 — active TB whole-blood signature across hospital cohorts (negative result)

Sites: S1 London, S2 South Africa, S6 The Gambia (proposing; three platforms), S4 autoimmune and
paediatric bacterial mimics (tester-only). Held out: GSE73408 (TB vs pneumonia, primary),
GSE89403, GSE101705. Reported descriptively after its headroom gate failed (§6, user decision).

| arm | top-5 signature, TB vs pneumonia AUC (draws) | mean |
|---|---|---|
| ours | 0.45 / 0.60 / 0.40 | **0.48** |
| MA-claims | 0.41 / 0.60 / 0.42 | 0.48 |
| S2 alone | 0.64 / 0.67 / 0.41 | 0.57 |
| MA-genome | 0.62 / 0.62 / 0.62 | 0.62 |
| union | 0.62 / 0.68 / 0.55 | 0.62 |
| group chat | 0.57 / 0.69 / 0.59 | 0.62 |
| debate | 0.72 / 0.67 / 0.64 | 0.68 |
| one agent on all data | 0.76 / 0.74 / 0.57 | 0.69 |

- The pre-registered prediction (ours ties S2 alone) does not hold. Ours' findings are real TB
  signal — they replicate against latent infection and health in 83–90% of cases and its
  signature separates TB from LTBI with AUC 0.92–0.98 — but its most reproducible programs
  (interferon/GBP up, T-cell genes down, neutrophil genes up) are shared with bacterial
  pneumonia. Corroboration rewards reproducibility, and in this case the most reproducible
  programs are the least specific.
- The scope endpoint had no judgeable disagreement with S2 alone in any draw. An exploratory
  (post hoc) signature built only from findings not labelled "not specific" raises ours to 0.59,
  still below S2 alone filtered the same way (0.66): the information that separates TB from
  pneumonia sits at the one site that holds other-disease patients.

## 4. Headroom gates (§6, $0, computed before spending on each case)

All three failed their pre-written rule on the same criterion — beating the single site that
does best *in hindsight*, which no single analysis could pick without the answer: `fsdev2`
(scripted ours 0.814 vs 0.825), TB (0.375 vs 0.45; no gain over MA on Malawi), COVID (0.204 vs
0.234, while beating MA-tools in 2 of 3 folds). The rules were not redefined after computing.

## 5. Robustness: weaker site agents (second model, harness)

Codex on the user's ChatGPT login refuses gpt-4.1-mini, gpt-5-mini and gpt-5.4-mini, and an
API-key Codex login is ruled out, so the second-model runs use Finch at every site. gpt-4.1-mini
could not do the site agents' job at all (invalid views, no claims), so the second model is
gpt-5.4-mini. An all-Finch run on the main model separates the model from the harness. Same 8
fstest tasks (900–907), arms ours / MA-genome / MA-claims / vote, F1:

| setting | ours | MA-genome | MA-claims | vote |
|---|---|---|---|---|
| main model, Codex + Finch sites (the test run) | **0.90** | 0.82 | 0.86 | 0.76 |
| main model, Finch at every site | 0.78 | **0.82** | 0.69 | 0.51 |
| gpt-5.4-mini, Finch at every site | 0.19 | **0.43** | 0.14 | 0.04 |

**Ours' advantage over the fixed federated pipeline is conditional on site agents that propose
good claims.** With Finch at every site it ties MA-genome (−0.04, p 0.74); with the weaker model
it loses (−0.24, p 0.06): most proposed claims fail their own site's gate, so little is left to
cross-examine, while MA-genome needs only correct views, which even the weaker model builds.
Ours still beats the other claim-based arms (MA-claims, vote) in every setting. The main result
is therefore *model- and harness-dependent*, not model-agnostic; it was obtained with Codex
(gpt-6-luna) at two of three proposing sites.

## 6. Limitations and deviations

- **Step 4 (resolution) never mattered** — on fsdev2, fstest and COVID, ours equals
  no-resolution; agent-built views were nearly error-free, so there was nothing to repair.
- **Recall for precision.** Ours keeps fewer claims than MA-claims and MA-genome; on COVID they
  replicate more claims in absolute number at lower precision.
- **The headroom gates all failed** on their best-single criterion; TB failed outright and is
  reported descriptively; COVID and TB are 3 draws each, descriptive, one federation each.
- **Revisions before the freeze, all on dev data and recorded with their reasons**: the
  two-direction claim test, the pooled replication rule with the equivalence-band condition, the
  propose prompt, the view-format repair, the planting parameters (fsdev2), the TB demote-only OD
  test (chosen after seeing its gate's targets), MA-genome's self-check count, dropping the
  IDs/labels knob. After the freeze only infrastructure and evaluation code changed, plus one bug
  fix that made store keys carry a non-default site model (main results unaffected).
- **Endpoint definitions added after data**: the TB scope comparison without autoimmune labels
  (the spec calls them unvalidated) is reported next to the as-defined number; the COVID
  matched-count artefact endpoint degenerated (m = 1).
- **Real-data artefact truth** relies on a 2 vs 2 public 10x v2/v3 comparison that also carries
  donor biology; no claim met the artefact definition.
- **CellTypist** was not in the analysis environment originally; it was installed (no
  dependency changes) before any COVID run.

## 7. Conclusion

On planted multi-site data with non-harmonized, siloed sites, cross-examination by capable site
agents beats a fixed federated meta-analysis on the same agent-built inputs (+0.14 F1, precision
0.97 vs 0.74, pre-registered, n = 32), and every multi-agent team design we compared, without any
reference answer or optimization target at run time. On real COVID-19 data it gives the most
precise replicating findings. Its limits are as clear: the gain comes from claim-level
cross-testing, not from dispute resolution or view repair; it depends on site agents strong
enough to propose good claims; it trades recall for precision; and where the decisive
information sits at one site — TB versus pneumonia — it does not beat that site.
