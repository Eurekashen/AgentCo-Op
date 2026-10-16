"""Claim schema, validation at submission and the shared matching rule (spec §2.3)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from agentcoop.coop.claims import (Claim, Vocabulary, claim_from_dict, claim_to_dict, parse_claims,
                                    same_claim)

VOCAB = Vocabulary(populations=("CD4 T", "CD14 Mono", "B"), groups=("control", "treated", "mimic"),
                   contrasts=(("treated", "control"), ("mimic", "control")))


def make(**kw) -> Claim:
    base = dict(id="S1:1", site="S1", kind="de", population="CD14 Mono", target="treated",
                reference="control", up_genes=("IFIT1", "MX1", "ISG15"), down_genes=(),
                effect=1.25, n_units=8, rank=1, rationale="interferon response")
    return Claim(**{**base, **kw})


def test_vocabulary_normalizes_json_lists_to_tuples():
    v = Vocabulary(populations=["B"], groups=["control", "treated"],
                   contrasts=[["treated", "control"]])
    assert v.populations == ("B",) and v.groups == ("control", "treated")
    assert v.contrasts == (("treated", "control"),) and ("treated", "control") in v.contrasts
    hash(v)


def test_vocabulary_rejects_contrast_groups_outside_its_groups():
    with pytest.raises(ValueError, match="mimic"):
        Vocabulary(populations=("B",), groups=("control", "treated"),
                   contrasts=(("mimic", "control"),))


def test_contrast_and_signed_genes():
    c = make(up_genes=("IFIT1", "MX1"), down_genes=("CD52",))
    assert c.contrast == ("treated", "control")
    assert c.signed_genes() == frozenset({("IFIT1", 1), ("MX1", 1), ("CD52", -1)})
    assert make(kind="composition", up_genes=(), effect=-0.4).signed_genes() == frozenset()


def test_claim_dict_round_trip_is_exact_through_json():
    """Step-1 claims are memoized as JSON; every arm must read back identical claims (RF4)."""
    c = make(down_genes=("CD52",), effect=0.1 + 0.2, rationale="ISGs ↑")
    d = claim_to_dict(c)
    assert d["up_genes"] == ["IFIT1", "MX1", "ISG15"] and d["down_genes"] == ["CD52"]
    assert (d["target"], d["reference"]) == ("treated", "control")
    back = claim_from_dict(json.loads(json.dumps(d)))
    assert back == c and hash(back) == hash(c)
    assert isinstance(back.up_genes, tuple) and isinstance(back.down_genes, tuple)
    comp = make(kind="composition", up_genes=(), effect=-0.4, n_units=None)
    assert claim_from_dict(json.loads(json.dumps(claim_to_dict(comp)))) == comp


def test_claim_from_dict_ignores_report_extras_and_defaults_rationale():
    d = {**claim_to_dict(make()), "label": "corroborated", "scope": None, "cause": None}
    assert claim_from_dict(d) == make()
    del d["rationale"]
    assert claim_from_dict(d).rationale == ""


def test_claim_from_dict_requires_the_schema_fields():
    d = claim_to_dict(make())
    del d["population"]
    with pytest.raises(TypeError):
        claim_from_dict(d)


def test_same_claim_de_uses_signed_gene_jaccard_with_an_inclusive_threshold():
    a = make(up_genes=("A", "B", "C"))
    assert same_claim(a, make(up_genes=("A", "D", "E")))                       # 1/5 = 0.2
    assert not same_claim(a, make(up_genes=("A", "D", "E", "F")))              # 1/6
    assert not same_claim(a, make(up_genes=(), down_genes=("A", "B", "C")))    # opposite signs
    assert same_claim(a, make(up_genes=("A", "B", "X")), min_jaccard=0.5)      # 2/4
    assert not same_claim(a, make(up_genes=("A", "D", "E")), min_jaccard=0.5)


def test_same_claim_needs_the_same_kind_population_and_contrast():
    a = make()
    assert same_claim(a, make(id="S2:4", site="S2", rank=7, effect=None, n_units=None))
    assert not same_claim(a, make(population="B"))
    assert not same_claim(a, make(target="mimic"))
    assert not same_claim(a, make(kind="composition", up_genes=(), effect=1.0))


def test_same_claim_composition_matches_by_direction():
    up = make(kind="composition", up_genes=(), effect=0.3)
    assert same_claim(up, make(kind="composition", up_genes=(), effect=2.0, site="S2"))
    assert not same_claim(up, make(kind="composition", up_genes=(), effect=-0.3))
    assert not same_claim(up, make(kind="composition", up_genes=(), effect=None))


def test_same_claim_is_symmetric_and_reflexive():
    pool = [make(), make(up_genes=("IFIT1", "MX1", "OAS1", "IFI6")),
            make(up_genes=("IFIT1",), down_genes=("MX1", "ISG15")),
            make(up_genes=("A", "B", "C")), make(population="B"), make(target="mimic"),
            make(kind="composition", up_genes=(), effect=0.5),
            make(kind="composition", up_genes=(), effect=-0.5),
            make(kind="composition", up_genes=(), effect=0.1, site="S3")]
    for x in pool:
        assert same_claim(x, x)
        for y in pool:
            for t in (0.0, 0.2, 0.5, 1.0):
                assert same_claim(x, y, t) == same_claim(y, x, t)




# ---------------------------------------------------------------------------
# parse_claims: validation at submission
# ---------------------------------------------------------------------------

def de(**kw) -> dict:
    item = {"kind": "de", "population": "CD14 Mono", "contrast": ["treated", "control"],
            "up_genes": ["IFIT1", "MX1", "ISG15"], "down_genes": [], "effect": 1.25,
            "n_units": 8, "rationale": "interferon response"}
    return {**item, **kw}


def comp(**kw) -> dict:
    item = {"kind": "composition", "population": "B", "contrast": ["treated", "control"],
            "effect": -0.7, "n_units": 8}
    return {**item, **kw}


def without(item: dict, *keys: str) -> dict:
    return {k: v for k, v in item.items() if k not in keys}


def parse(raw: object, max_claims: int = 10) -> tuple[list[Claim], list[str]]:
    return parse_claims(raw, site="S1", vocab=VOCAB, max_claims=max_claims)


def test_parse_valid_claims_assigns_site_ids_and_default_ranks():
    claims, errors = parse([de(), comp()])
    assert errors == []
    assert claims == [make(), make(id="S1:2", kind="composition", population="B", up_genes=(),
                                   effect=-0.7, rank=2, rationale="")]


def test_parse_accepts_a_claims_object_and_is_deterministic():
    raw = [de(), comp()]
    assert parse({"claims": raw}) == parse(raw) == parse(json.loads(json.dumps(raw)))


@pytest.mark.parametrize("raw", [None, '[{"kind": "de"}]', 42, {"claims": None},
                                 {"findings": [de()]}, {"claims": "IFIT1"}])
def test_parse_malformed_container_returns_an_error_not_an_exception(raw):
    claims, errors = parse(raw)
    assert claims == [] and len(errors) == 1 and isinstance(errors[0], str)


def test_parse_skips_non_object_items_and_keeps_the_rest():
    claims, errors = parse([None, "IFIT1 up", ["IFIT1"], de()])
    assert [c.id for c in claims] == ["S1:1"] and claims[0].rank == 4    # default rank = position
    assert [e.split(":")[0] for e in errors] == ["item[0]", "item[1]", "item[2]"]
    assert all("not an object" in e for e in errors)


@pytest.mark.parametrize("item, word", [
    (without(de(), "kind"), "kind"),
    (de(kind="gsea"), "kind"),
    (de(kind=1), "kind"),
    (without(de(), "population"), "population"),
    (de(population="Monocytes"), "population"),
    (de(population=["CD14 Mono"]), "population"),
    (without(de(), "contrast"), "contrast"),
    (de(contrast=["control", "treated"]), "contrast"),       # reversed: not a vocabulary contrast
    (de(contrast=["treated", "mimic"]), "contrast"),
    (de(contrast=["treated"]), "contrast"),
    (de(contrast="treated vs control"), "contrast"),
    (without(de(), "up_genes", "down_genes"), "genes"),
    (de(up_genes="IFIT1, MX1, ISG15"), "up_genes"),
    (de(up_genes=["IFIT1", 7, "ISG15"]), "up_genes"),
    (de(up_genes=["IFIT1", "  ", "MX1", "ISG15"]), "up_genes"),
    (de(down_genes={"CD52": -1}), "down_genes"),
    (de(effect="large"), "effect"),
    (de(effect=float("nan")), "effect"),
    (de(n_units=-1), "n_units"),
    (de(n_units="8"), "n_units"),
    (de(rank=1.5), "rank"),
    (de(rank=True), "rank"),
])
def test_parse_skips_an_invalid_item_with_a_readable_reason(item, word):
    claims, errors = parse([item, comp()])
    assert [c.population for c in claims] == ["B"]
    assert len(errors) == 1 and errors[0].startswith("item[0]: ") and word in errors[0]


def test_parse_normalizes_case_and_whitespace():
    claims, errors = parse([de(kind=" DE ", population=" CD14 Mono ",
                               contrast=[" treated", "control "],
                               up_genes=[" ifit1 ", "Mx1", "IFIT1", "isg15"], down_genes=["cd52"],
                               rationale="  ISGs  ")])
    assert errors == []
    c = claims[0]
    assert (c.kind, c.population, c.contrast) == ("de", "CD14 Mono", ("treated", "control"))
    assert c.up_genes == ("IFIT1", "MX1", "ISG15") and c.down_genes == ("CD52",)
    assert c.rationale == "ISGs"


def test_parse_de_gene_rules():
    _, errors = parse([de(up_genes=["IFIT1", "ifit1", "MX1"])])          # 2 distinct genes
    assert "3 genes" in errors[0]
    _, errors = parse([de(down_genes=["mx1"])])
    assert "both" in errors[0] and "MX1" in errors[0]
    claims, errors = parse([de(up_genes=["IFIT1", "MX1"], down_genes=["CD52"], effect=None),
                            without(de(), "down_genes", "effect", "n_units", "rationale")])
    assert errors == [] and [c.effect for c in claims] == [None, None]
    assert claims[1].down_genes == () and claims[1].n_units is None and claims[1].rationale == ""


@pytest.mark.parametrize("item", [
    comp(up_genes=["CD79A"]), comp(down_genes=["MS4A1"]), without(comp(), "effect"),
    comp(effect=None), comp(effect=0), comp(effect=0.0), comp(effect="-0.7"),
    comp(effect=float("inf")), comp(effect=True),
])
def test_parse_composition_needs_no_genes_and_a_nonzero_effect(item):
    claims, errors = parse([item])
    assert claims == [] and len(errors) == 1


def test_parse_composition_accepts_empty_gene_lists_and_integer_effects():
    claims, errors = parse([comp(up_genes=[], down_genes=None, effect=-2)])
    assert errors == [] and claims[0].effect == -2.0 and isinstance(claims[0].effect, float)
    assert claims[0].up_genes == claims[0].down_genes == ()


def test_parse_accepts_integral_floats_and_numpy_numbers():
    claims, errors = parse([de(n_units=8.0, rank=2.0),
                            comp(effect=np.float64(-0.7), n_units=np.int64(8), rank=np.int64(1))])
    assert errors == []
    assert [(c.kind, c.rank, c.n_units) for c in claims] == [("composition", 1, 8), ("de", 2, 8)]
    assert all(type(c.rank) is int and type(c.n_units) is int for c in claims)
    assert type(claims[0].effect) is float


def test_parse_orders_by_rank_then_position_and_ids_follow_that_order():
    raw = [de(up_genes=["A1", "A2", "A3"], rank=3),
           de(up_genes=["B1", "B2", "B3"]),                  # default rank = position 2
           de(up_genes=["C1", "C2", "C3"], rank=1),
           de(up_genes=["D1", "D2", "D3"], rank=2)]          # ties with B; B came first
    claims, errors = parse(raw)
    assert errors == []
    assert [x.up_genes[0] for x in claims] == ["C1", "B1", "D1", "A1"]
    assert [x.id for x in claims] == ["S1:1", "S1:2", "S1:3", "S1:4"]
    assert [x.rank for x in claims] == [1, 2, 2, 3]


def test_parse_ignores_agent_supplied_id_and_site():
    claims, _ = parse([de(id="S2:1", site="S2")])
    assert (claims[0].id, claims[0].site) == ("S1:1", "S1")


def test_parse_drops_duplicates_keeping_the_better_ranked():
    first = de(up_genes=["IFIT1", "MX1", "ISG15"], rank=2, rationale="first")
    again = de(up_genes=["isg15", "IFIT1", "mx1"], rank=1, rationale="again")   # same signed set
    claims, errors = parse([first, again])
    assert [c.rationale for c in claims] == ["again"]
    assert errors == ["item[0]: duplicate of item[1]"]


def test_parse_duplicates_need_identical_kind_population_contrast_and_signed_content():
    raw = [de(), de(up_genes=["IFIT1", "MX1"], down_genes=["ISG15"]), de(population="CD4 T"),
           de(contrast=["mimic", "control"]), de(up_genes=["IFIT1", "MX1", "ISG15", "OAS1"]),
           comp(), comp(effect=0.7), comp(effect=-0.1)]
    claims, errors = parse(raw)
    assert len(claims) == 7 and errors == ["item[7]: duplicate of item[5]"]


def test_parse_caps_claims_in_rank_order():
    raw = [de(up_genes=[f"G{i}A", f"G{i}B", f"G{i}C"], rank=12 - i) for i in range(12)]
    claims, errors = parse(raw)
    assert [c.rank for c in claims] == list(range(1, 11))
    assert errors == ["item[1]: beyond the cap of 10 claims",
                      "item[0]: beyond the cap of 10 claims"]


def test_parse_duplicates_do_not_use_up_the_cap():
    claims, errors = parse([de(), de(), comp()], max_claims=2)
    assert [c.kind for c in claims] == ["de", "composition"]
    assert errors == ["item[1]: duplicate of item[0]"]


def test_parse_reads_back_claim_to_dict_output():
    """claim_to_dict writes target/reference instead of contrast; the parser takes either form."""
    claims, _ = parse([de(down_genes=["CD52"]), comp()])
    again, errors = parse([claim_to_dict(c) for c in claims])
    assert errors == [] and again == claims
