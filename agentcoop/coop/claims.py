"""Claims: the one schema every arm emits (spec §2.3), its validation and the matching rule.

Claims come from external agents as JSON, so `parse_claims` is the boundary where malformed
output is caught: a bad item is skipped with a readable reason instead of raising (one sloppy
claim must not void a site's whole step-1 call), ids are assigned here (an agent cannot collide
with another site's ids), and gene symbols are upper-cased once so every later comparison is
plain string equality. `same_claim` is the single matching rule shared by every arm that pairs
claims (majority vote, debate, the mechanism arms).
"""

from __future__ import annotations

import math
import numbers
from dataclasses import asdict, dataclass, fields, replace


@dataclass(frozen=True)
class Vocabulary:
    """A case's controlled names; claims and site views may use only these."""

    populations: tuple[str, ...]
    groups: tuple[str, ...]
    contrasts: tuple[tuple[str, str], ...]      # (target, reference) pairs allowed in claims
    composition_only: tuple[str, ...] = ()      # populations without de claims (too few cells)

    def __post_init__(self) -> None:
        # Case specs are read back from JSON (lists); claim validation tests tuple membership.
        object.__setattr__(self, "populations", tuple(self.populations))
        object.__setattr__(self, "composition_only", tuple(self.composition_only))
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "contrasts", tuple((t, r) for t, r in self.contrasts))
        unknown = sorted({g for pair in self.contrasts for g in pair} - set(self.groups))
        if unknown:
            raise ValueError(f"contrasts use groups outside the vocabulary: {unknown}")


@dataclass(frozen=True)
class Claim:
    """One validated claim. `rank` is the emitting arm's confidence order (lower = stronger)."""

    id: str
    site: str
    kind: str                       # "de" | "composition"
    population: str
    target: str
    reference: str
    up_genes: tuple[str, ...]
    down_genes: tuple[str, ...]
    effect: float | None            # composition: its sign is the claimed direction
    n_units: int | None
    rank: int
    rationale: str = ""

    @property
    def contrast(self) -> tuple[str, str]:
        return (self.target, self.reference)

    def signed_genes(self) -> frozenset[tuple[str, int]]:
        """{(gene, +1|-1)}: a gene's direction is part of what the claim asserts."""
        return frozenset([(g, 1) for g in self.up_genes] + [(g, -1) for g in self.down_genes])


def claim_to_dict(c: Claim) -> dict:
    """JSON-ready form (tuples → lists) for the store and reports; inverse of claim_from_dict."""
    return {**asdict(c), "up_genes": list(c.up_genes), "down_genes": list(c.down_genes)}


def claim_from_dict(d: dict) -> Claim:
    """Inverse of claim_to_dict; extra keys (label, scope, cause in final reports) are ignored."""
    kw = {f.name: d[f.name] for f in fields(Claim) if f.name in d}
    return Claim(**{**kw, "up_genes": tuple(d["up_genes"]), "down_genes": tuple(d["down_genes"])})


def _sign(x: float | None) -> int:
    return 0 if x is None or x == 0 else (1 if x > 0 else -1)


def same_claim(a: Claim, b: Claim, min_jaccard: float = 0.2) -> bool:
    """The shared matching rule: same kind, population and contrast, and the same content.

    de: Jaccard of signed genes ≥ min_jaccard (a gene up in one claim and down in the other is
    no overlap); composition: the same direction. Symmetric by construction.
    """
    if (a.kind, a.population, a.contrast) != (b.kind, b.population, b.contrast):
        return False
    if a.kind == "composition":
        return _sign(a.effect) != 0 and _sign(a.effect) == _sign(b.effect)
    sa, sb = a.signed_genes(), b.signed_genes()
    union = sa | sb
    return bool(union) and len(sa & sb) / len(union) >= min_jaccard




# ---------------------------------------------------------------------------
# Validation at submission
# ---------------------------------------------------------------------------

KINDS = ("de", "composition")
MIN_DE_GENES = 3


def parse_claims(raw: object, *, site: str, vocab: Vocabulary,
                 max_claims: int) -> tuple[list[Claim], list[str]]:
    """An agent's decoded JSON → (valid claims in rank order, one reason per dropped item).

    `raw` is a list of claim objects or {"claims": [...]}; a contrast is `contrast: [target,
    reference]` (spec schema) or `target`/`reference` (claim_to_dict's form). Never raises:
    malformed items, duplicates of a better-ranked claim (same kind, population, contrast and
    signed content) and items beyond `max_claims` are skipped and reported. Order is by `rank`
    (default: 1-based input position), ties by input position; ids are `<site>:<k>` with k the
    1-based position in the result, so any id or site the agent wrote is ignored.
    """
    if isinstance(raw, dict):
        items = raw.get("claims")
        if not isinstance(items, list):
            return [], ["expected {'claims': [...]}: 'claims' is missing or not a list"]
    elif isinstance(raw, list):
        items = raw
    else:
        return [], [f"expected a list of claims or {{'claims': [...]}}, got {type(raw).__name__}"]
    errors: list[str] = []
    valid: list[tuple[int, Claim]] = []
    for i, item in enumerate(items):
        try:
            valid.append((i, _parse_item(item, site=site, vocab=vocab, default_rank=i + 1)))
        except ValueError as e:
            errors.append(f"item[{i}]: {e}")
    valid.sort(key=lambda ic: (ic[1].rank, ic[0]))
    kept: list[Claim] = []
    first: dict[tuple, int] = {}
    for i, c in valid:
        key = _signature(c)
        if key in first:
            errors.append(f"item[{i}]: duplicate of item[{first[key]}]")
        elif len(kept) >= max_claims:
            errors.append(f"item[{i}]: beyond the cap of {max_claims} claims")
        else:
            first[key] = i
            kept.append(c)
    return [replace(c, id=f"{site}:{k}") for k, c in enumerate(kept, 1)], errors


def _parse_item(item: object, *, site: str, vocab: Vocabulary, default_rank: int) -> Claim:
    """One raw claim → Claim with an empty id; ValueError names the first problem found."""
    if not isinstance(item, dict):
        raise ValueError(f"not an object ({type(item).__name__})")
    kind = _text(item.get("kind"))
    kind = kind.lower() if isinstance(kind, str) else kind
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {list(KINDS)}, got {item.get('kind')!r}")
    population = _text(item.get("population"))
    if population not in vocab.populations:
        raise ValueError(f"population {population!r} not in {list(vocab.populations)}")
    if kind == "de" and population in vocab.composition_only:
        raise ValueError(f"population {population!r} takes composition claims only")
    pair = item["contrast"] if "contrast" in item else [item.get("target"), item.get("reference")]
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError(f"contrast must be [target, reference], got {pair!r}")
    target, reference = (_text(g) for g in pair)
    if (target, reference) not in vocab.contrasts:
        raise ValueError(f"contrast {[target, reference]!r} not in "
                         f"{[list(c) for c in vocab.contrasts]}")
    up, down = _genes(item, "up_genes"), _genes(item, "down_genes")
    effect = _number(item.get("effect"), "effect")
    n_units = None if item.get("n_units") is None else _integer(item["n_units"], "n_units")
    if n_units is not None and n_units < 0:
        raise ValueError(f"n_units must be >= 0, got {n_units}")
    rank = default_rank if item.get("rank") is None else _integer(item["rank"], "rank")
    if kind == "de":
        if len(up) + len(down) < MIN_DE_GENES:
            raise ValueError(f"a de claim needs >= {MIN_DE_GENES} genes, got {len(up) + len(down)}")
        both = sorted(set(up) & set(down))
        if both:
            raise ValueError(f"genes in both up_genes and down_genes: {both}")
    elif up or down:
        raise ValueError("a composition claim takes no genes")
    elif not effect:
        raise ValueError("a composition claim needs a nonzero effect (its sign is the direction)")
    rationale = item.get("rationale")
    return Claim(id="", site=site, kind=kind, population=population, target=target,
                 reference=reference, up_genes=up, down_genes=down, effect=effect,
                 n_units=n_units, rank=rank,
                 rationale="" if rationale is None else str(rationale).strip())


def _text(value: object) -> object:
    return value.strip() if isinstance(value, str) else value


def _genes(item: dict, key: str) -> tuple[str, ...]:
    """Stripped, upper-cased, de-duplicated (first occurrence kept) symbols; missing = none."""
    value = item.get(key)
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not all(isinstance(g, str) and g.strip()
                                                       for g in value):
        raise ValueError(f"{key} must be a list of non-empty strings, got {value!r}")
    return tuple(dict.fromkeys(g.strip().upper() for g in value))


def _number(value: object, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number, got {value!r}")
    return float(value)


def _integer(value: object, key: str) -> int:
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"{key} must be an integer, got {value!r}")


def _signature(c: Claim) -> tuple:
    """Duplicate key within one site: kind, population, contrast and signed content (signed
    genes for de, direction for composition)."""
    direction = _sign(c.effect) if c.kind == "composition" else 0
    return (c.kind, c.population, c.contrast, c.signed_genes(), direction)
