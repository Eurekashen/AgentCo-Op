# AgentCo-Op v4 — method (summary)

The design contract is the v4 spec,
[superpowers/specs/2026-10-14-cross-examination-design.md](superpowers/specs/2026-10-14-cross-examination-design.md)
(sections cited as §N); the implementation plan is
[superpowers/plans/2026-10-14-v4-core.md](superpowers/plans/2026-10-14-v4-core.md). This page
summarizes the protocol; where it and the spec differ, the spec wins. The v3 method (workflow
optimization without reference answers) is `docs/method.md` at tag `v3.4.1`.

## Setting and roles (§1, §2.1)

Evidence for a question sits at sites that cannot pool data, each in its own non-harmonized form.

- **Site agent**: an unmodified external agent (Codex CLI or FutureHouse Finch, on `gpt-6-luna`)
  bound to one site's private data in an OS-sandboxed workspace. It builds the site view,
  proposes claims and answers follow-ups.
- **Site tester**: deterministic code inside the site boundary. It reads only the site table
  computed from the view and returns summary statistics.
- **Coordinator**: an LLM plus fixed rules. It sees claims, test results, analysis specs and
  metadata cards, never sample- or cell-level data.

A site view (§2.2) is `genes.tsv` (native ID → symbol), `samples.tsv` (sample → donor, group),
`cells.tsv` (cell → population; scRNA) and `spec.json` (the site's analysis spec). A claim (§2.3)
is `{id, site, kind: de | composition, population, contrast: [target, reference], up_genes,
down_genes, effect, n_units, rank, rationale}` over the case vocabulary.

## Protocol (§2.5)

1. **Propose.** Each proposing site builds its view and proposes at most K = 10 claims in one
   call; tester-only sites build their view only.
2. **Test.** The proposer's own tester is a gate. The claim is then tested at every other site
   whose view holds its population and both groups (replication) and, per the case's fixed scope
   table, as a scope contrast (feeding only the scope rule). All non-proposer tests of a run form
   one BH family.
3. **Label** (one function shared by every arm; first match): `unsupported` (gate failed) →
   `disputed` (a non-proposer contradicts) → `corroborated` (≥ 1 non-proposer supports) →
   `site-specific` (the pooled non-proposer effect is absent) → `unresolved` (no other site could
   test it) → `inconclusive`. `inconclusive` and `untestable` never count as support or
   contradiction.
4. **Resolve.** Disputed claims and `untestable(genes|groups)` results go to the coordinator,
   which classifies the cause from the involved sites' specs and cards: analysis choice → both
   sites re-test once under the pre-registered default spec; handoff loss → one repair call to the
   site agent naming the defect, then a re-test; site artefact → that site's result becomes
   `inconclusive`; site biology → recorded. Follow-ups never show a site other sites' results; at
   most one per (claim, site); budget F = number of proposing sites. Step 3 is then re-applied.
5. **Stop** when nothing is disputed or view-caused untestable, or F is spent. Findings are the
   corroborated claims by rank (label tier, supporting sites, |Stouffer z|); every other claim is
   reported with its label and cause.

The protocol never compares alternatives by a score, never re-tests a claim twice, and uses a
re-test spec fixed in advance; thresholds are frozen on `fsdev` before any real case (§2.5, §9).

## Everything else

Testers and their parameters: §2.4. Arms (baselines, references, the $0 mechanism decomposition)
and the pre-registered comparisons: §3–§4. Cases: §5 (COVID-19, TB) and §7 (the controlled
planted suite). Headroom gate: §6. Invariants: §8, tested in `tests/test_invariants.py` and
`tests/test_sandbox.py`. Budget: §10.
