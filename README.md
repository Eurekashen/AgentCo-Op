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

## Headline results (hidden-answer F1, held-out suites; see [docs/results.md](docs/results.md))

| | naive composition | autonomous lead / single agent | best fixed interface | ours | free-form editor |
|---|---|---|---|---|---|
| shape 1 — pooled data (16 tasks) | 0.00 | **0.79** (Codex CLI alone) | 0.76 | 0.68 | 0.70 |
| shape 2 — data at two sites (16 tasks) | 0.00 | 0.49 (Codex as lead, ×3) · 0.53 (in-house lead) | 0.71 | **0.74** | 0.75 |

- When data cannot be pooled, optimizing the explicit collaboration interface with answer-free,
  data-grounded acceptance beats letting a strong agent coordinate (ours vs Codex lead 12/1 tasks,
  p 0.006). When it can be pooled, a strong single agent wins and composition should be declined.
- Answer-free evaluation must be grounded in the data: a report-only LLM judge is near chance
  (0.53 pairwise), a deterministic evidence rule reaches 0.96.
- Re-drawing a broken composition never fixes it (best-of-N stays at 0); prompt-only optimization
  recovers nothing in shape 2 (0.00).
- Localized typed search is *not* more accurate than free-form editing with the same feedback
  (tie), but it is half the cost and its acceptances are safer (1/46 harmful vs 3/24).
- Model-agnostic checks (framework model swapped to gpt-4.1-mini): the shape-2 ranking holds;
  shape 1 is limited by program naming, which no data check can see.

Honest negatives are listed in [docs/results.md](docs/results.md) §6 and `findings.md`.

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
