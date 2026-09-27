"""Signal tracing for task shape 2 (data held at separate sites).

The probe is a *replicated* uncovered signal — a coherent program every site's own scan finds but
the lead does not claim. Stages, in order:

0. ``site_missed_signal``      no site's analysis report mentions it
1. ``unverified_at_other_site`` only some sites report it and nobody tested it at the others
2. ``not_comparable``          every site reports it, but without cell type/direction/genes the
                               lead can match on
3. ``integration_omits_signal`` comparable reports from every site reached the lead, unclaimed
4. covered

Weak claims: a claim supported at one site only is traced to the lead's merge rule.
"""

from __future__ import annotations

from typing import Any, Optional

from agentcoop.eval_free import canon_cell_type, canon_direction
from agentcoop.optimize.trace import Finding
from agentcoop.workflow import Artifact, Run

STAGE = {"site_missed_signal": 0, "unverified_at_other_site": 1, "not_comparable": 2,
         "integration_omits_signal": 3}
COVERED = 4


def _items(report: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("findings", "claims", "results", "programs", "verifications"):
        v = report.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


def _mentions(item: dict[str, Any], ct: str, direction: str, genes: set[str],
              cell_types: list[str]) -> tuple[bool, bool]:
    """(mentions the signal, is comparable: has cell type, direction and >= 2 of its genes)."""
    ict = canon_cell_type(str(item.get("cell_type", "")), cell_types)
    idir = canon_direction(item.get("direction", ""))
    igenes = {str(g).upper() for g in (item.get("genes") or [])}
    overlap = len(igenes & genes)
    if item.get("replicated") in ("no",):
        return False, False
    if ict == ct and idir == direction:
        return True, overlap >= 2
    if overlap >= 3:
        return True, False
    return False, False


def _site_reports(run: Run) -> list[Artifact]:
    return [a for a in run.artifacts.values() if a.kind == "site_report"]


def _request_edges(run: Run) -> list[str]:
    return [f"{e.src}->{e.dst}" for e in run.workflow.edges if e.handoff == "site_request"]


def trace_signal(run: Run, probe: dict[str, Any], cell_types: list[str]) -> Optional[Finding]:
    ct, d, genes = probe["cell_type"], probe["direction"], set(probe["genes"])
    what = f"replicated {d} signal in {ct} ({probe['n_genes']} shared genes)"
    reports = _site_reports(run)
    sites = sorted({r.data["site"] for r in reports if r.data.get("request_kind") != "verification"})
    mention: dict[str, tuple[bool, bool]] = {}
    for r in reports:
        hits = [_mentions(i, ct, d, genes, cell_types) for i in _items(r.data.get("report", {}))]
        m = any(h[0] for h in hits)
        c = any(h[1] for h in hits)
        prev = mention.get(r.data["site"], (False, False))
        mention[r.data["site"]] = (prev[0] or m, prev[1] or c)
    reported = [s for s in sites if mention.get(s, (False, False))[0]]
    edges = _request_edges(run)
    lead = run.workflow.output
    if not reported:
        return Finding("site_missed_signal", edges[0] if edges else lead, what,
                       "no site's report mentions this program, although every site's own "
                       "data shows it consistently across its donors", weight=1.5)
    if len(reported) < len(sites):
        return Finding("unverified_at_other_site", lead, what,
                       f"reported only by site(s) {', '.join(reported)}; the other site never "
                       "tested it on its own donors", weight=1.5,
                       facts={"reported_by": reported})
    if not all(mention[s][1] for s in sites):
        return Finding("not_comparable", edges[0] if edges else lead, what,
                       "every site mentions it, but not all with cell type, direction and genes "
                       "the lead could match across sites", weight=1.2)
    return Finding("integration_omits_signal", lead, what,
                   "comparable reports from every site reached the lead but it did not claim "
                   "the effect; it may lack evidence that the effect replicates", weight=1.0)


def localize(run: Run, profile: dict[str, Any], cell_types: list[str]) -> list[Finding]:
    lead = run.workflow.output
    findings: list[Finding] = []
    for probe in profile["uncovered"]:
        f = trace_signal(run, probe, cell_types)
        if f:
            findings.append(f)
    for chk in profile["claims"]:
        if chk["status"] == "supported":
            continue
        what = f"claim {chk['claim_index']} ({chk['cell_type']}, {chk['direction']}, " \
               f"{chk['program']}) is {chk['status']}"
        if "site-specific" in chk.get("detail", ""):
            findings.append(Finding("site_specific_claimed", lead, what,
                                    "the lead reported as replicated an effect that holds at one "
                                    "site only", weight=1.5))
        else:
            findings.append(Finding("overclaim", lead, what, chk.get("detail", ""), weight=1.0))
    merged: dict[tuple[str, str], Finding] = {}
    for f in findings:
        key = (f.location, f.symptom)
        if key in merged:
            merged[key].weight += f.weight
            merged[key].detail += f"; {f.probe}"
        else:
            merged[key] = f
    return sorted(merged.values(), key=lambda f: -f.weight)


def propagation(run: Run, probes: list[dict[str, Any]], profile: dict[str, Any],
                cell_types: list[str]) -> list[int]:
    still = {(u["cell_type"], u["direction"]) for u in profile["uncovered"]}
    out = []
    for p in probes:
        if (p["cell_type"], p["direction"]) not in still:
            out.append(COVERED)
            continue
        f = trace_signal(run, p, cell_types)
        out.append(STAGE.get(f.symptom, 0) if f else 0)
    return out
