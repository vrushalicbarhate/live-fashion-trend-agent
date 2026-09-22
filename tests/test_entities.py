import pandas as pd
import pytest

from src.config import load_catalog
from src.nlp.entities import (
    VocabularyError,
    build_item_lookup,
    build_nlp_pipeline,
    build_patterns_and_lookup,
    extract_mentions,
)


def test_real_catalog_has_no_vocabulary_conflicts():
    """
    Regression guard: as config/catalog.csv grows, a new row's aliases
    could accidentally collide with another row's canonical term (this
    caught three real conflicts during development). This test fails
    loudly instead of letting entity extraction silently mislabel data.
    """
    catalog = load_catalog("config/catalog.csv")
    build_patterns_and_lookup(catalog)  # raises VocabularyError on conflict

CATALOG = pd.DataFrame(
    [
        {
            "item_id": "C001",
            "product_type": "t-shirt",
            "material": "cotton",
            "color": "white",
            "google_keyword": "white cotton t-shirt",
            "product_aliases": "tee|t shirt",
            "material_aliases": "cotton",
            "color_aliases": "white|off-white",
        },
        {
            "item_id": "C003",
            "product_type": "cargo pants",
            "material": "denim",
            "color": "washed blue",
            "google_keyword": "denim cargo pants",
            "product_aliases": "cargos|cargo trousers",
            "material_aliases": "denim|jean",
            "color_aliases": "washed blue|light blue",
        },
    ]
)


def test_patterns_include_canonical_and_alias_terms():
    patterns, term_lookup = build_patterns_and_lookup(CATALOG)
    pattern_texts = {p["pattern"] for p in patterns}
    assert "t-shirt" in pattern_texts
    assert "tee" in pattern_texts
    assert term_lookup["PRODUCT:tee"] == ("PRODUCT", "t-shirt")


def test_conflicting_alias_raises_vocabulary_error():
    conflicting = pd.concat(
        [
            CATALOG,
            pd.DataFrame(
                [
                    {
                        "item_id": "C099",
                        "product_type": "jacket",
                        "material": "leather",
                        "color": "black",
                        "google_keyword": "black leather jacket",
                        "product_aliases": "tee",  # clashes with C001's "tee" -> t-shirt
                        "material_aliases": "leather",
                        "color_aliases": "black",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(VocabularyError, match="Ambiguous PRODUCT term"):
        build_patterns_and_lookup(conflicting)


def test_extracts_mention_when_full_triple_in_one_sentence():
    nlp, term_lookup = build_nlp_pipeline(CATALOG)
    item_lookup = build_item_lookup(CATALOG)

    text = "I just bought this white cotton tee and it's amazing."
    mentions = extract_mentions(text, nlp, term_lookup, item_lookup)

    assert len(mentions) == 1
    assert mentions[0].item_id == "C001"
    assert mentions[0].product == "t-shirt"


def test_no_mention_when_entities_span_different_sentences():
    nlp, term_lookup = build_nlp_pipeline(CATALOG)
    item_lookup = build_item_lookup(CATALOG)

    text = "I bought a tee yesterday. Separately, cotton prices are rising this year."
    mentions = extract_mentions(text, nlp, term_lookup, item_lookup)

    # "tee" alone has no material/color in its sentence -> no mention;
    # the second sentence has material but no product/color -> no mention.
    assert mentions == []


def test_no_mention_when_triple_does_not_match_any_catalog_item():
    nlp, term_lookup = build_nlp_pipeline(CATALOG)
    item_lookup = build_item_lookup(CATALOG)

    # "tee" (t-shirt) + denim + washed blue is a real combo of entities,
    # but no catalog item is a denim washed-blue t-shirt.
    text = "This denim washed blue tee looks so odd."
    mentions = extract_mentions(text, nlp, term_lookup, item_lookup)

    assert mentions == []


def test_extracts_mention_for_cargo_pants_using_aliases():
    nlp, term_lookup = build_nlp_pipeline(CATALOG)
    item_lookup = build_item_lookup(CATALOG)

    text = "These denim cargos in washed blue are my favorite right now."
    mentions = extract_mentions(text, nlp, term_lookup, item_lookup)

    assert len(mentions) == 1
    assert mentions[0].item_id == "C003"
