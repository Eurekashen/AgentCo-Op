# Case study — the real IFN-β response in human PBMCs (Kang et al. 2018), no reference answer

**Question given to every system** (treatment blinded): PBMCs from 4 donors, control vs an
uncharacterized treatment within each donor; characterize induced and repressed programs per
cell type; report only effects reproducible across donors.

**Data split (pre-registered, `agentcoop/tasks/real_case.py`).** Kang 2018 has 8 donors with
matched control / IFN-β-stimulated cells. Sorted by donor id, even positions are *discovery*
(D101, D1016, D107, D1256: the only data any agent, check or optimizer ever sees), odd positions
are *validation* (D1015, D1039, D1244, D1488: held out). Six major cell types.

**Independent validation (`agentcoop/eval/case_validation.py`)** — evidence no component used:
1. held-out replication: each claim re-tested on the 4 unseen donors (per-donor sign agreement
   in all 4, ≥60% of measured genes agreeing, competitive random-gene-set null);
2. missed signal: coherent programs (GO enrichment) that change reproducibly in the unseen
   donors in a (cell type, direction) where no claim replicates;
3. known biology: IFN-β induces the type-I interferon program in every PBMC type (Kang et al.
   2018); coverage = cell types with a replicated interferon claim.

## 1. Every node succeeds; the collaboration loses the science

Naive composition — DE (Wilcoxon) → top-30 gene list per cell type → GeneAgent (the external,
independently developed interpreter; its interface is a gene list) → integrator.

- DE finds the response: ISG15, IFIT1/3, MX1, CXCL10, RSAD2 … at the top of every cell type
  (log2FC 6–10).
- GeneAgent names it correctly for every list: "Type I interferon–driven antiviral defense".
- The integrator writes **0 claims**: "The evidence identifies interferon-associated antiviral
  defense programs across the listed cell types, but does not establish whether the genes are
  higher or lower in treated cells or whether effects reproduce across donors."

The handoff forwarded a mixed-direction gene list and an interpretation with no direction, no
statistics and no donor support. The integrator's refusal is correct given what it received —
exactly the hypothetical of the plan (§5.2), on real data. Validation: 0 replicated claims,
10 reproducible programs missed.

## 2. Same backbone, three ways to recover

| arm (all gpt-6-luna) | claims | replicated in held-out donors | IFN cell types | missed programs |
|---|---|---|---|---|
| naive composition | 0 | 0 | 0/6 | 10 |
| ReAct coding agent alone | 8 | 8 | 6/6 | 2 |
| Codex CLI alone | 7 | 7 | 6/6 | 3 |
| GEPA (prompt evolution, 16 executions) | 6 | 6 | 6/6 | 4 |
| repaired interface (dev's best fixed edits) | 13 | 11 | 6/6 | 0 |
| **ours** (typed localized search, 16 executions, $1.75) | 13 | 10 (0 contradicted) | 6/6 | **1** |
| free-form workflow editor (16 executions) | *pending* | | | |

How GEPA recovered: its winning instruction copies the evaluator's feedback for this dataset
(GO terms, cell types, gene names, directions) into the lead's prompt. The lead then asserts
"up" because the prompt says so; its claims still cite interpretation views that carry no
direction, and the repressed translation program the feedback also listed stays unclaimed.
The score moved because the verifier's findings moved into the prompt, not because the
collaboration now carries the evidence — such an instruction cannot transfer to another study.

## 3. What our optimizer did (no answer, 16 executions)

| step | localized at | executable edit | evidence (supported / uncovered) | no-edit re-draw |
|---|---|---|---|---|
| 0 | DE → GeneAgent handoff | split gene lists by direction | 0 / 10 | 0 / 10 |
| 1 | GeneAgent → integrator view | carry direction, method, selection, statistics with each interpretation | 0 / 10 | 0 / 10 |
| 2 | integrator input | add a donor-consistency analysis whose per-donor evidence reaches the integrator | **9 / 9** | 0 / 10 |
| 3 | DE → GeneAgent handoff | 50 genes per list (rejected: 15 genes, padj ≤ 0.05) | 9 / 9 | 9 / 9 |
| 4 | DE → GeneAgent handoff | significance filter padj ≤ 0.1 | 10 / 9 | 9 / 9 |
| 5 | integrator input | also pass the DE summary — rejected (8 vs control 8, more unassessed) | | |

Steps 0–1 do not change the evidence yet (accepted because the lineage trace shows the
signal reaching one node further); the integrator still refuses until step 2 gives it donor
support. The repair is staged: the direction must survive the handoff *and* reproducibility
evidence must reach the integrator — the two pieces of context the naive collaboration lost.
The edits that changed the evidence (steps 2 and 4) beat a no-edit re-draw of the affected
suffix (9 vs 0, 10 vs 9), so the gain is not resampling. The final workflow is an executable artifact (DAG of agents, handoff policies and an
evidence branch), not a prompt tied to this dataset.
