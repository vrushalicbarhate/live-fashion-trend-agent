"""
Live runner for entity extraction + sentiment analysis — run this on
your own machine (needs to download the model from huggingface.co,
which this sandbox can't reach).

Usage:
    python scripts/run_nlp_live.py
"""

from transformers import pipeline

from src.config import load_catalog, load_settings
from src.nlp.entities import build_item_lookup, build_nlp_pipeline, extract_mentions
from src.nlp.sentiment import score_mentions
from src.nlp.text_cleaning import clean_text

# A few sample Reddit-style sentences to sanity-check the pipeline end to end.
SAMPLE_TEXTS = [
    "Just copped this white cotton tee and honestly it's the softest thing I own.",
    "These denim cargos in washed blue looked great in photos but fit terrible IRL, kind of regret it.",
    "Anyone else obsessed with linen wide leg trousers in sage green this summer? So comfy.",
]


def main():
    settings = load_settings("config/settings.yaml")
    catalog = load_catalog("config/catalog.csv")

    nlp, term_lookup = build_nlp_pipeline(catalog)
    item_lookup = build_item_lookup(catalog)

    model_name = settings["sentiment_model"]["name"]
    classifier = pipeline("text-classification", model=model_name, top_k=None)

    all_mentions = []
    for raw_text in SAMPLE_TEXTS:
        cleaned = clean_text(raw_text)
        mentions = extract_mentions(cleaned, nlp, term_lookup, item_lookup)
        all_mentions.extend(mentions)

    print(f"Extracted {len(all_mentions)} mentions from {len(SAMPLE_TEXTS)} sample texts.")

    sentiment_df = score_mentions(all_mentions, classifier)
    print(sentiment_df.to_string(index=False))


if __name__ == "__main__":
    main()
