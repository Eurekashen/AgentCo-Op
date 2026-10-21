# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research code for AgentCo-Op (paper v1: arXiv 2605.20425). Branch `v4` (local) rebuilds the
project around one thesis — see the design contract
[docs/superpowers/specs/2026-10-14-cross-examination-design.md](docs/superpowers/specs/2026-10-14-cross-examination-design.md)
(cited as §N), the implementation plan
[docs/superpowers/plans/2026-10-14-v4-core.md](docs/superpowers/plans/2026-10-14-v4-core.md) and
the summary in [docs/method.md](docs/method.md):

> When the evidence needed to answer a scientific question is distributed across independently
> operated agents that cannot pool their resources — each holding data in its own,
> non-harmonized form — cross-examination lets them (1) recover much of what pooling the data
> would give, without pooling; (2) do better than a fixed federated pipeline run on the same
> agent-built inputs; (3) with no reference answer or optimization target at run time.

Training-free and objective-free: nothing is fine-tuned and nothing is optimized against a score
at run time; claims are kept by agreement between independent resources (§2.5).

Other branches and tags are history, not dependencies: v3 (optimizing collaboration without
reference answers) is reproducible from tag `v3.4.1` (also `v3-final`) — its code (workflow DAG,
handoffs, `optimize/`, judges, BixBench, the GeneAgent shim, `replay_check.py`) is gone from `v4`
and its numbers stay in [docs/results.md](docs/results.md), [docs/baselines.md](docs/baselines.md)
and [docs/case_study.md](docs/case_study.md); `main` = v1 as published (reproducibility record,
`94c3750`); `revision` = v2 evidence-certification compiler; `origin/feature/ecps-preference-search`
= v2 preference search (not merged). `gh-pages` still shows v1 numbers. Never delete remote branches.

Working memory for the long-running effort lives in `task_plan.md` (Phase 6 = v4), `findings.md`,
`progress.md` (read them first when resuming).

### Where the evidence stands (keep this honest; update from `findings.md`)

Main tables use ONE backbone for every arm (gpt-6-luna; Codex `-m gpt-6-luna`); other models
only in the appendix. Arms are defined in spec §4.

- Dev only so far (`fsdev2`, 8 tasks; the method was revised on two of them, so it is not test
  evidence): ours F1 0.91 vs MA-genome 0.75 (p 0.016), vote 0.84 (n.s.), ma_claims 0.90 (n.s.),
  group_chat 0.71, debate 0.68, codex_coord 0.58, one agent on all data 0.44; step 4 adds
  nothing; agent-built views have ~0 errors (so fstest drops the IDs/labels knob). All three
  headroom gates (fsdev2, TB, COVID) fail their best-single criterion; TB is descriptive.
  Frozen at tag `v4-freeze`; fstest is the test. Details: `findings.md`.
- Why v4 exists (v3, tag `v3.4.1`): collaboration loss at agent interfaces is real, but an agent
  that sees all the data beats every composition (pooled shape: Codex alone 0.79; BixBench: no
  team beats the best single agent). v4 claims gains only where data cannot be pooled; beating
  a single site is a sanity check, not evidence (§1).
- Spend before v4 ≈ $104.7 actual; v4 so far ≈ $2.1; hard cap $180 actual (§10).

## Commands

```bash
pytest -q                                                   # offline; no API key, no agent runs
python -m agentcoop.cases.planted data/bench/onek1k/onek1k.h5ad   # freeze the Case 3 base once ($0)
python -m agentcoop.eval.planted fsdev dev 800 808          # build a planted suite (eval side writes answers)
AGENTCOOP_BUDGET_USD=180 python experiments/run.py --suite fsdev --tasks fsdev-800 \
    --arms ours single union vote ma_genome --draws 0       # -> runs/v4/<suite>/<task>/<arm>__s<k>.json
python experiments/analyze.py --suite fsdev --tasks fsdev-800   # grade + tabulate -> runs/v4/<suite>_analysis.json
python experiments/headroom.py --help                       # $0 headroom gate (§6)
python -c "from agentcoop.cases.planted import TASKS; from tests.test_invariants import assert_planted_task_boundary as check; [check(d) for d in sorted((TASKS / 'fsdev').iterdir())]"
```

Refactoring rule: a behaviour-neutral change must not change any store key or stored result.
Check it by moving a finished task's result files aside and re-running `experiments/run.py` with
the same arms and draws: every agent call must come from `runs/store` (no new store entry, $0
actual) and the results must be identical. A behaviour change is a new method version, recorded
in `findings.md` with its reason; after tag `v4-freeze` it also needs a new suite (§9).

Environment: `AGENTCOOP_BUDGET_USD` (hard cap on *actual* spend across the ledger — always set it
for experiments; 180 for v4), `AGENTCOOP_MODEL` (backbone for every LLM role; default
`gpt-6-luna`), `AGENTCOOP_GRADER_MODEL` (pinned, default `gpt-6-luna`, never follows the
backbone), `AGENTCOOP_EXTERNAL_MODEL` (external agents' model; default `gpt-6-luna`, does *not*
follow the backbone), `AGENTCOOP_WS_ROOT` (site workspaces; default `~/agentcoop_ws`; must stay
outside the repo), `AGENTCOOP_BIXBENCH_ENV` (the read-only conda env site agents analyse in;
default `~/miniconda3/envs/bixbench`, built by `external_agents/bixbench_env/setup.sh`),
`AGENTCOOP_LLM_DB` (ledger/cache sqlite, default `runs/llm.sqlite`). The API key is read from
`OPENAI_API_KEY` or `.secrets/openai` — never print or commit it. A non-default backbone writes to
`runs/v4@<model>/`; new models need a row in `llm.PRICES`, and their capabilities are probed once
into `runs/model_profiles.json`.

Working rules (from the user): debug on 2–4 tasks before any larger run and scale in steps; give
every framework change a stated reason and evidence in `findings.md`; Codex baselines use the
user's locally logged-in CLI (`~/.codex`, never an API-key login); keep the design and the
conclusions model-agnostic (verify key findings on a second backbone: `gpt-4.1-mini`;
`gpt-4o-mini` fabricated reports as an external agent).

Data is not in git. Case 3 base: OneK1K (Yazar et al. 2022) from CZ CELLxGENE Discover,
`https://datasets.cellxgene.cziscience.com/1e44db10-b572-46cc-adae-dcc7acd44ca6.h5ad`, saved as
`data/bench/onek1k/onek1k.h5ad` (4.4 GB, schema 7.1.0: 1,248,980 cells × 35,528 genes, raw counts
in X, Ensembl var index + `feature_name`, `donor_id`, Cell Ontology `cell_type`); the frozen donor
subset is `data/bench/onek1k/base.h5ad` + `base.json` (donor pools, sha256). Suites live in
`data/tasks/<suite>/<task>/` (`public/`, `sites/`, `hidden/`). v3 data: `data/raw/kang_2018.h5ad`
comes from `api.figshare.com/v2/file/download/34464122` (the `figshare.com/ndownloader` URL is
behind a WAF challenge and returns 0 bytes); v4 does not use Kang 2018 (§7). Gene-set libraries
come from Enrichr's `geneSetLibrary?mode=text&libraryName=...` (Hallmark 2020, GO BP 2023,
Reactome 2022); marker libraries (CellMarker 2024, PanglaoDB) are in `data/markers/`. `runs/`
(ledger, store, results) is gitignored.

## Architecture

```
cases/planted.py ─▶ data/tasks/<suite>/<task>/public/ + sites/     eval/planted.py ─▶ hidden/ (answers)
      │ load_case → CaseSpec + site specs                                   │ grade (analyze.py only)
      ▼                                                                     ▼
coop/site.Site: workspace under WS_ROOT ─ sandboxed site agent (Codex | Finch) ─▶ view/ + claims.json
      │ site table (coop/view.py) ─▶ deterministic testers (coop/tester.py) ─▶ summary statistics only
      ▼
coop/protocol.run_protocol: propose → test → label_claim → resolve (Coordinator) → stop
      ▼
coop/arms.run_arms: ours, baselines, mechanism arms on one SiteHandle ─▶ ArmResult
      ▼
experiments/run.py ─▶ runs/v4/<suite>/<task>/<arm>__s<k>.json ─▶ experiments/analyze.py
```

- `llm.py` — one client: cost ledger, cache, hard budget cap, per-model capability adaptation.
  `proxy.py` — OpenAI-compatible local endpoint over it, so Finch's litellm calls share the
  backbone, cache, ledger and cap.
- `store.py` — `memo(key, compute, tag=)`: first-writer-wins memo of expensive agent calls with
  the nominal USD of each entry (charged again on every replay); `put_blob`/`get_blob` hold site
  views.
- `agents/` — `workspace.py` (`WS_ROOT`, APFS clones, `exec_env`, `sandbox_profile`,
  `sandboxed`), `codex.py` (Codex CLI on the local login, permission profile), `finch.py`,
  `react.py` (robustness runs only), `harness.py` (`run_agent` → `AgentRun`).
- `coop/` — `claims.py` (schema, validation, matching rule), `view.py` (site view files,
  validation, site table), `tester.py` (replication and scope tests, BH), `protocol.py` (shared
  label function, resolution, `run_protocol`), `site.py` (`Site`, prompts, file mailbox),
  `arms.py` (every arm of §4).
- `cases/planted.py` — Case 3 (§7): frozen OneK1K base, planted generator (3 proposing sites + a
  mimic site; knobs artefact and IDs+labels), `load_case`. It writes no answers.
- `eval/planted.py` — the only writer and reader of answers (`write_truth`, `build`,
  `load_truth`, `grade`).
- `experiments/` — `run.py` (runs arms, never grades), `analyze.py` (grades and tabulates),
  `headroom.py` ($0 headroom gate).

## Invariants — do not break

- **No path to the answers (§8).** `tests/test_invariants.py` fails if any module outside
  `agentcoop/eval/` imports `agentcoop.eval` or mentions `hidden/`, `truth.json` or
  `load_truth`. Suites are built from the eval side (`python -m agentcoop.eval.planted`); grading
  is always a separate step (`experiments/analyze.py`, `experiments/headroom.py`).
- **Site files are whitelisted.** Planted site files carry only `sample`, `condition`,
  `cell_label`, `assay` (opaque IDs), no var columns and no uns/obsm/obsp/layers; nothing in
  `public/` or `sites/` carries any part of the answer. Run `assert_planted_task_boundary` on
  every real suite before spending on it.
- **OS sandbox.** Every site-agent process runs under `sandbox-exec`
  (`agents/workspace.sandbox_profile`) or Codex's permission profile with the same deny list:
  the repo's `data/`, `runs/`, `.secrets/`, `CODEX_HOME`, and `WS_ROOT` except the agent's own
  workspace; network loopback only; Codex web search off. `tests/test_sandbox.py` proves it.
- **Workspaces live under `WS_ROOT`, never inside the repo.**
- **`inconclusive` and `untestable` are never support and never contradiction.**
- **`sample` semantics.** Same index ⇒ exact replay (store and LLM cache); a new index is an
  honest re-draw. Step-1 proposal keys contain no run id, so every arm compares identical step-1
  claims on identical views.
- **Nominal vs actual cost.** Method comparisons use nominal USD (stored with each store entry,
  charged on every replay); the cap uses actual USD. Every agent call is charged, failed ones
  included; a failure is recorded and never cached.
- **External agents run unmodified** (Codex CLI; Finch via `external_agents/run_finch.py`):
  change the prompt or the view contract, not the agent.
- **Freeze (§9).** Testers, δ/q, prompts, K, B, F, matching rules and scope tables are frozen on
  `fsdev` (tag `v4-freeze`) before `fstest` or any real case; no method change after the freeze
  without a new suite.

## Gotchas

- gpt-6-luna on Chat Completions: function tools require `reasoning_effort="none"` (then
  `temperature`/`seed` work); with reasoning, temperature must be default. Tool use *with*
  reasoning → `llm.responses()` (stateless: `store=False` + encrypted reasoning items).
- Callers pass *intent* (`reasoning_effort`, `temperature`); `llm._adapt_chat/_adapt_responses`
  map it per model. Never hardcode a model's quirks at a call site.
- Codex borrows only the login from the user's `CODEX_HOME` and is otherwise hermetic:
  `--ignore-user-config`, `-c project_doc_max_bytes=0`, throwaway `HOME`, `TMPDIR`/
  `NUMBA_CACHE_DIR`/`MPLCONFIGDIR` inside the workspace (else scanpy fails in the sandbox), stdin
  closed. `run_codex` strips `OPENAI_API_KEY` and `AGENTCOOP_*`, kills the process group on
  timeout and raises `CodexFailed` when Codex fails without output. Codex cannot be seeded:
  compare over ≥ 3 draws. It bills the local login ($0 actual; the API-equivalent cost is
  recorded as nominal); observed throughput ≤ 250 runs/day.
- The sandbox profile must allow the interpreters (conda base, the bixbench env, Finch's
  `.venv`) and the agent's own workspace: a profile that blocks Python startup makes every site
  look like a scientific failure. Seatbelt matches real paths (`/tmp` is `/private/tmp`).
- To force a site-agent call to re-execute after changing its harness or prompt code, bump `"v"`
  in the site's memo key (a behaviour change: record it). Deleting a result file only re-runs the
  arm on the stored calls.
- A task directory can be built only once (`build_suite` refuses), and the base refuses to be
  overwritten (suites record its sha256): results and store keys are keyed by task id, so build a
  new suite instead.
- `experiments/run.py` fixes the site → harness assignment per suite on first use
  (`runs/v4/<suite>/agents.json`; default Codex/Finch alternating over the site order) and never
  recomputes an existing result file.
- `nohup python ... > log` buffers stdout until exit; use `python -u` / `print(..., flush=True)`.
- Planting uses *core* genes (Hallmark ∩ family GO terms, ≥8 expressed); random Hallmark members
  are unrecognizable. Cholesterol/heme/DNA repair are rarely plantable in PBMC.
- Statistics unit is the task (planted suites), never cells or re-draws. Differences below 1e-9
  are ties.
- Caches are first-writer-wins (LLM: INSERT OR IGNORE + read-back; store: atomic link), so
  concurrent arms and runs converge on one value per key.
- Store entries are `{"key": ..., "value": ...}` JSON files under `runs/store/` (blobs under
  `runs/store/blobs/`); step-1 entries have `key["op"] == "propose"`.
- OneK1K is 1.2 M cells (4.4 GB): read it with h5py or backed mode; `prepare_base` streams it.
- `tests/__init__.py` keeps `tests` importable: the boundary check is imported from
  `tests.test_invariants`.

- 给我report的时候，始终要使用中文（只有给我report的时候需要，其他所有时候不受影响）
