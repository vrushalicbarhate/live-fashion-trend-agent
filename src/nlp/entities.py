"""
Product/Material/Color Entity Extraction (synopsis §5.4, §9.3).

Builds a spaCy EntityRuler purely from the catalog's own vocabulary
(canonical values + aliases) — no trained NER model, no network
dependency beyond installing spaCy itself. Every match traces back to
an explicit pattern, which is what makes this explainable and
testable rather than a black box.

A "mention" (a candidate combination worth scoring) is only created
when a PRODUCT, a MATERIAL, and a COLOR entity all appear in the same
sentence AND that exact (product, material, color) triple matches a
real catalog item. Partial or mismatched sentences are dropped rather
than guessed at — precision over recall, same philosophy as the
EntityRuler itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import spacy
from spacy.pipeline import EntityRuler

LABELS = ("PRODUCT", "MATERIAL", "COLOR")
CATALOG_COLUMN_FOR_LABEL = {
    "PRODUCT": ("product_type", "product_aliases"),
    "MATERIAL": ("material", "material_aliases"),
    "COLOR": ("color", "color_aliases"),
}


class VocabularyError(ValueError):
    """Raised when the catalog's alias vocabulary has an internal conflict."""


def _terms_for_row(row: pd.Series, canonical_col: str, alias_col: str) -> list[str]:
    canonical = row[canonical_col].strip()
    aliases = [a.strip() for a in row[alias_col].split("|") if a.strip()]
    return [canonical] + aliases


def build_patterns_and_lookup(
    catalog_df: pd.DataFrame,
) -> tuple[list[dict], dict[str, tuple[str, str]]]:
    """
    Returns:
      patterns: EntityRuler pattern dicts, e.g. {"label": "PRODUCT", "pattern": "cargo pants"}
      term_lookup: "LABEL:lowercased matched text" -> (label, canonical_value),
                   so a matched span like "cargos" resolves back to
                   ("PRODUCT", "cargo pants").

    Raises VocabularyError if the same term maps to two different
    canonical values under the same label — that's a genuine ambiguity
    in the catalog, not something the ruler can resolve on its own.
    """
    patterns: list[dict] = []
    term_lookup: dict[str, tuple[str, str]] = {}

    for label in LABELS:
        canonical_col, alias_col = CATALOG_COLUMN_FOR_LABEL[label]
        for _, row in catalog_df.iterrows():
            canonical_value = row[canonical_col].strip()
            for term in _terms_for_row(row, canonical_col, alias_col):
                key = f"{label}:{term.lower()}"
                if key in term_lookup and term_lookup[key][1] != canonical_value:
                    raise VocabularyError(
                        f"Ambiguous {label} term '{term}': maps to both "
                        f"'{term_lookup[key][1]}' and '{canonical_value}'. "
                        f"Fix the catalog's alias columns to remove the overlap."
                    )
                term_lookup[key] = (label, canonical_value)
                patterns.append({"label": label, "pattern": term})

    return patterns, term_lookup


def build_nlp_pipeline(catalog_df: pd.DataFrame) -> tuple[spacy.Language, dict]:
    """
    A blank English pipeline — tokenizer + sentence splitter + our
    catalog-driven EntityRuler. No statistical model, so no model
    download is required beyond `pip install spacy`.

    Returns (nlp, term_lookup) — term_lookup is needed later to resolve
    matched spans back to canonical catalog values.
    """
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    ruler: EntityRuler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
    patterns, term_lookup = build_patterns_and_lookup(catalog_df)
    ruler.add_patterns(patterns)

    return nlp, term_lookup


def build_item_lookup(catalog_df: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    """(product_type, material, color) -> item_id, for resolving a
    recognized triple back to a real tracked catalog combination."""
    lookup = {}
    for _, row in catalog_df.iterrows():
        triple = (row["product_type"].strip(), row["material"].strip(), row["color"].strip())
        lookup[triple] = row["item_id"]
    return lookup


@dataclass
class Mention:
    item_id: str
    sentence: str
    product: str
    material: str
    color: str


def _canonical_by_label(ents, term_lookup: dict[str, tuple[str, str]]) -> dict[str, str]:
    """
    A sentence can contain multiple entities of the same label (e.g. two
    colors mentioned). We take the first occurrence of each label —
    good enough for a weekend prototype, and simpler to explain and
    test than a disambiguation heuristic.
    """
    found: dict[str, str] = {}
    for ent in ents:
        key = f"{ent.label_}:{ent.text.lower()}"
        if ent.label_ not in found and key in term_lookup:
            found[ent.label_] = term_lookup[key][1]
    return found


def extract_mentions(
    text: str,
    nlp: spacy.Language,
    term_lookup: dict[str, tuple[str, str]],
    item_lookup: dict[tuple[str, str, str], str],
) -> list[Mention]:
    """
    Runs the pipeline over `text`, walks each sentence, and emits a
    Mention only when that sentence has all three entity types AND the
    resulting triple matches a real catalog item.
    """
    doc = nlp(text)
    mentions: list[Mention] = []

    for sent in doc.sents:
        sent_ents = [ent for ent in doc.ents if ent.start >= sent.start and ent.end <= sent.end]
        by_label = _canonical_by_label(sent_ents, term_lookup)

        if not all(label in by_label for label in LABELS):
            continue  # incomplete triple in this sentence — excluded, not guessed at

        triple = (by_label["PRODUCT"], by_label["MATERIAL"], by_label["COLOR"])
        item_id = item_lookup.get(triple)
        if item_id is None:
            continue  # recognized entities, but not a real catalog combination

        mentions.append(
            Mention(
                item_id=item_id,
                sentence=sent.text.strip(),
                product=by_label["PRODUCT"],
                material=by_label["MATERIAL"],
                color=by_label["COLOR"],
            )
        )

    return mentions
