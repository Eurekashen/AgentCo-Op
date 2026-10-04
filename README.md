<div align="center">
  <h1>AgentCo-Op</h1>
  <p><b>Optimizing collaboration between independently developed scientific agents — without reference answers</b></p>
  <p>
    <a href="https://ma-compbio-lab.github.io/AgentCo-Op">Project page (v1)</a> ·
    <a href="docs/method.md">Method</a> ·
    <a href="docs/results.md">Results</a>
  </p>
</div>

> **Branch `v3`.** This is a ground-up rebuild. The published v1 ("Retrieval-Based Synthesis of
> Interoperable Multi-Agent Workflows", arXiv 2605.20425) is preserved at `main` / `94c3750`; its
> results do not carry over. The project page still describes v1.

## The question

When independently developed agents are composed for a scientific analysis, there is usually no
reference answer to optimize against — only partial, imperfect feedback from the data. AgentCo-Op
v3 asks:

1. Do composed agents lose information *at their interfaces* (not just through individual weakness)?
2. Without reference answers, can we find where that happens and fix it with executable, verified
   changes to how they collaborate?
3. Does that beat a strong single agent (Codex CLI), reflective prompt optimization, free-form
   workflow editing and equal-budget re-drawing?

## How it works

- **Workflows as data** with executable **handoff policies** on every edge; every artifact keeps
  its lineage; node outputs are cached so an edit re-executes only the affected suffix.
- **Answer-free evidence** computed from the task's own data: whether each claim's genes change
  consistently across donors; whether coherent, donor-consistent signal went unreported.
- **Localization by signal tracing**: evidence units are followed through lineage to the node or
  edge where they are lost.
- **Controlled acceptance**: an edit is kept only if its evidence beats both the incumbent and a
  no-edit re-draw of the same suffix; judges follow a K3-style rubric-first, scorepad, both-orders
  protocol, and a report-only judge is the control.
- **External agents run unmodified**: GeneAgent (NCBI) is vendored as-is; only its model transport
  is swapped (openai-0.28 shim → `gpt-6-luna`) and its database calls are record/replayed.

## Benchmarks built for the question

Real PBMCs (Kang 2018, 8 donors) with treatment effects planted by binomial thinning, so the
answer is exact and hidden from the optimizer:

- **Shape 1 — pooled data**: DE analysis → gene-set interpretation (GeneAgent) → integrator.
- **Shape 2 — data held at two sites**: site analysis agents (data never leaves a site) → study
  lead; the question asks which effects *replicate* across sites.

## Headline results (all arms on one backbone, gpt-6-luna; see [docs/results.md](docs/results.md))

Baselines are defined in [docs/baselines.md](docs/baselines.md).

**Data held at two sites — fresh, pre-registered held-out suites (`mstest2` + `mstest3`, 48 tasks):**

| method | F1 | ours W/L (p) | $/task |
|---|---|---|---|
| naive composition | 0.00 | 45/0 | 0.01 |
| autonomous orchestrator lead | 0.46 | 28/3 (7e-6) | 0.03 |
| Codex CLI as study lead (×3) | 0.47 | 37/0 (1e-7) | 0.03 |
| GEPA, same feedback & budget | 0.61 | 22/11 (0.04) | 0.09 |
| reference: fixed interface picked on dev with the answers | 0.68 | 18/15 (n.s.) | 0.03 |
| free-form workflow editor, same feedback & budget | 0.73 | 13/14 (n.s.) | 0.51 |
| **ours** | **0.72** | – | 0.22 |

**Real data, no answer (Kang 2018 IFN-β; 4 held-out validation donors)** — [docs/case_study.md](docs/case_study.md):
the naive composition makes 0 claims although every agent succeeds (the handoff drops direction
and donor support); ours repairs it in executable steps → 13 claims, 10 replicate in held-out
donors, 0 contradicted, 1 reproducible program missed (single agents miss 2–3; GEPA reaches 6
claims by copying the evaluator's feedback into the prompt).

**Public benchmark — BixBench v1.5** (official Finch agent, Codex, ReAct, and five standard
team methods; 166 frozen test questions): no team beats the best single agent (ReAct 0.54;
ours 0.54, MoA 0.54), but the best single agent flips between dev and test (Finch → ReAct);
choosing it on dev costs −0.16, every team stays within 0.04 of the best. One agent sees all the
data here — composition is insurance, not a gain.

- Answer-free evaluation must be grounded in the data: report-only LLM judge 0.53 pairwise vs a
  deterministic evidence rule 0.96.
- Re-drawing never fixes a broken composition (best-of-N ≈ 0); prompt optimization reaches a
  fraction of the interface repair.
- Ours ties free-form editing in accuracy at 42% of its cost, with safer acceptances
  (7.6% harmful vs 11.4%, GEPA 14.1%) and fewer missed improvements.
- Model-agnostic (appendix): with gpt-4.1-mini the typed search keeps its accuracy while
  free-form editing and single agents degrade.

Honest negatives are listed in [docs/results.md](docs/results.md) §9 and `findings.md`.

## Quick start

```bash
pip install -e .[dev]
pytest -q                                          # offline
# data: see CLAUDE.md (Kang 2018 via the figshare API; Enrichr gene-set libraries)
python -m agentcoop.tasks.planted dev 0 16
AGENTCOOP_BUDGET_USD=5 python experiments/run_candidates.py --suite dev --tasks dev-000 \
    --candidates naive pseudobulk+no_interp+donor
AGENTCOOP_BUDGET_USD=5 python experiments/run_optimizers.py --suite dev --tasks dev-000 --methods ours
python experiments/phase2_analysis.py --suite dev
```

Requires an OpenAI API key in `OPENAI_API_KEY` or `.secrets/openai`. All model calls go through
one cached, cost-ledgered client with a hard budget cap (`AGENTCOOP_BUDGET_USD`).

## Citation (v1)

```bibtex
@article{shen2026agentcoop,
  title={AgentCo-op: Retrieval-Based Synthesis of Interoperable Multi-Agent Workflows},
  author={Shuaike Shen and Wenduo Cheng and Shike Wang and Mingqian Ma and Jian Ma},
  year={2026}, eprint={2605.20425}, archivePrefix={arXiv}, primaryClass={cs.AI}
}
```

## License

Apache-2.0. GeneAgent (`external_agents/GeneAgent`) is a U.S. Government work (NLM public domain).
