# Baselines — definitions

Every arm in every main table runs on the **same backbone, `gpt-6-luna`**: framework agents,
external agents (GeneAgent, site analysts, Finch), Codex (`codex exec -m gpt-6-luna` on the
user's local login), reflection/editor models. The grading model is a separate, pinned
instrument (also `gpt-6-luna`). Runs with other models (`gpt-4.1-mini`, `gpt-5-mini`, the user's
native Codex config `gpt-6-sol`) appear only in the model-agnosticism appendix, never mixed into
a main table.

Costs are nominal USD per task (what the run would cost with no cache), including every
execution an optimizer spends (rejected edits, no-edit controls). Codex usage is billed to the
user's ChatGPT plan and enters the ledger at API-equivalent price.

## A. Single agents and agent products (no composition)

| arm | what it is | sees |
|---|---|---|
| ReAct coding agent (in-house) | one LLM with a persistent Python kernel (and a shell on BixBench), ReAct-style tool loop (Yao et al. 2023) | the task and all data it is allowed to read |
| Codex CLI | OpenAI's coding agent product, headless `codex exec`, hermetic config | same |
| Finch (BixBench only) | FutureHouse's data-analysis agent, the official BixBench agent (data-analysis-crow), unmodified, BixBench's own config | same |
| Orchestrator lead (multi-site only) | an LLM lead that decides what to ask each site agent (manager/orchestrator pattern, as in Magentic-One-style systems) | site reports only |
| Codex as lead (multi-site only) | Codex CLI in the lead role, talking to site agents through a local tool server | site reports only |

## B. Fixed compositions

| arm | what it is |
|---|---|
| naive composition | the independently developed agents wired the obvious way (e.g. DE → gene lists → GeneAgent → integrator; question forwarded to each site) |
| best fixed interface | the best composition found on the development suite, applied unchanged |

## C. Combining several runs (test-time compute)

| arm | source | how |
|---|---|---|
| Self-consistency / best-of-N | Wang et al. 2023 | N re-draws; majority answer (BixBench) or best by the same answer-free evidence (collaboration suites) |
| Majority vote over heterogeneous agents | — | Finch + Codex + ReAct, answer clusters |
| Mixture-of-Agents aggregator | Wang et al. 2024 | an aggregator LLM synthesizes the proposers' responses |
| LLM-as-judge selection | Zheng et al. 2023 | a judge picks the most credible analysis from their reports |
| Multi-agent debate | Du et al. 2023 | each analysis defended by a debater that sees the others, 2 rounds, majority |

## D. Workflow / prompt optimizers — same feedback, same budget as ours

All receive the same task, the same answer-free evidence (per-claim support, contradictions,
uncovered signal; as text and as the evidence-rule score), the same backbone, and the same
budget (16 workflow executions per task); none sees an answer.

| arm | source | what it changes |
|---|---|---|
| GEPA | Agrawal et al. 2025, official `gepa` library | evolves the workflow's text: lead instructions (+ a note appended to every site request) |
| Free-form LLM workflow editor | ADAS (Hu et al. 2025) / AFlow (Zhang et al. 2025) style | rewrites the whole workflow JSON from the documented agent/handoff options |
| Best-of-N re-draws | — | nothing: keeps the best re-draw by evidence |

Ours: localize the loss by tracing evidence through artifact lineage, propose typed edits to
the collaboration interface (handoff policies, evidence branches, decomposition), accept only
with controlled replay against the incumbent and a no-edit re-draw.

## Ablations (reported separately, not as baselines)

Random edit location (same edit space), no signal-propagation route, report-only-judge
acceptance, validated rule library, escalation to the free-form editor.
