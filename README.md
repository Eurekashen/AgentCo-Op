<div align="center">
  <h1>AgentCo-Op</h1>
  <p><b>Cross-examination between independently developed scientific agents across resource silos</b></p>
  <p>
    <a href="https://ma-compbio-lab.github.io/AgentCo-Op">Project page (v1)</a> ·
    <a href="docs/method.md">Method</a> ·
    <a href="docs/superpowers/specs/2026-10-14-cross-examination-design.md">Design (v4)</a> ·
    <a href="docs/results.md">v3 results</a>
  </p>
</div>

> **Branch `v4`.** A ground-up rebuild. The published v1 ("Retrieval-Based Synthesis of
> Interoperable Multi-Agent Workflows", arXiv 2605.20425) is preserved at `main` / `94c3750`;
> v3 (optimizing agent collaboration without reference answers) is reproducible from tag
> `v3.4.1`. The project page still describes v1.

## The question

The evidence for a scientific question may be spread over institutions that cannot pool their
data. Each holds it in its own, non-harmonized form and is served by its own, independently
developed agent. Can those agents cross-examine each other's claims and

1. recover much of what pooling the data would give, without pooling;
2. do better than a fixed federated pipeline (meta-analysis) run on the same agent-built inputs;
3. with no reference answer and no optimization target at run time?

A single site cannot see the others by construction, so beating a single site is only a sanity
check. The evidence is silo-cost recovery against a pooled oracle, the comparison with federated
meta-analysis, a $0 mechanism decomposition and a prior-only control.

## How it works

- **Sites behind a hard boundary.** Each site's data lives in a private workspace. Its agent
  (Codex CLI or FutureHouse Finch, unmodified, all on `gpt-6-luna`) runs under an OS sandbox
  (`sandbox-exec` or Codex's permission profile) that denies other sites, the repo's data and
  the network. Only summary statistics leave a site.
- **Agent-built site views.** In one call per site, the agent maps its native data (gene IDs,
  cell labels, condition codes) onto the case vocabulary as a machine-readable view. It also
  writes its analysis spec and proposes at most 10 claims in one shared schema.
- **Deterministic cross-testing.** Each claim is first tested by its proposer's tester. It is
  then tested at every other site that holds its population and contrast: donor-pseudobulk
  module scores, Welch t, a competitive gene-set null, BH over the run, and an equivalence band
  for "absent".
- **One shared labelling rule.** unsupported → disputed → corroborated → site-specific →
  unresolved → inconclusive. Inconclusive and untestable results never count as evidence.
- **Disagreements resolved by cause, not by score.** A coordinator reads the involved sites'
  specs and metadata cards and classifies each dispute, within a fixed follow-up budget, as one
  of: analysis choice (re-test once under a pre-registered spec), handoff loss (one repair call
  naming the defect), site artefact, or site biology.
- **Every baseline on the same interface and budget.** Single sites, union, majority vote,
  MA-genome (fixed federated meta-analysis), a Codex coordinator, group chat, debate, a
  prior-only control and a silo-violating pooled oracle.

## Benchmarks

- **Controlled multi-site suite (Case 3).** Real healthy PBMCs (a frozen OneK1K donor subset) are
  split into 3 proposing sites and a mimic site. Treatment programs are planted by binomial
  thinning, so the answer is exact and hidden from every arm. Two knobs add a site artefact
  (assay confounded with condition) and heterogeneous gene IDs, labels and condition codes.
  `fsdev` (8 tasks) is for debugging and freezing; `fstest` (32 tasks) is run once after the
  freeze.
- **Real data (planned).** Severe COVID-19 PBMC across four institutions, validated on held-out
  cohorts; active TB whole-blood signatures across hospital cohorts.

## Results

**v4 results are pending.** No v4 experiment has been run beyond debugging and smoke tests.
Numbers will appear here only after the pre-registered runs (spec §9). v3 results answer a
different question (optimizing workflows without reference answers). They are in
[docs/results.md](docs/results.md) and reproducible from tag `v3.4.1`.

## Quick start

```bash
pip install -e .[dev]
pytest -q                                              # offline, no API key
# data: see CLAUDE.md (OneK1K from CZ CELLxGENE Discover; Enrichr gene-set libraries)
python -m agentcoop.cases.planted data/bench/onek1k/onek1k.h5ad   # freeze the base once
python -m agentcoop.eval.planted fsdev dev 800 808    # build the dev suite
AGENTCOOP_BUDGET_USD=180 python experiments/run.py --suite fsdev --tasks fsdev-800 \
    --arms ours single union vote ma_genome --draws 0
python experiments/analyze.py --suite fsdev --tasks fsdev-800
```

This needs an OpenAI API key in `OPENAI_API_KEY` or `.secrets/openai` (for Finch, the
coordinator and prior-only) and a locally logged-in Codex CLI (for site agents; never an API-key
login). All model calls go through one cached, cost-ledgered client with a hard budget cap
(`AGENTCOOP_BUDGET_USD`).

## Citation (v1)

```bibtex
@article{shen2026agentcoop,
  title={AgentCo-op: Retrieval-Based Synthesis of Interoperable Multi-Agent Workflows},
  author={Shuaike Shen and Wenduo Cheng and Shike Wang and Mingqian Ma and Jian Ma},
  year={2026}, eprint={2605.20425}, archivePrefix={arXiv}, primaryClass={cs.AI}
}
```

## License

Apache-2.0. Third-party agents under `external_agents/` keep their own licenses.
