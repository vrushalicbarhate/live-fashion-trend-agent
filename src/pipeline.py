"""
Full pipeline orchestration (ties together every layer built so far).

Deliberately NOT inside app.py: Streamlit code is built around widget
callbacks and session state, which makes it awkward to unit-test. By
keeping every fetch -> NLP -> aggregate -> decide step in plain
functions here, the whole pipeline can be exercised end-to-end with
fake clients (see tests/test_pipeline.py) with no browser involved.
app.py's only job is to call run_full_pipeline() and render the result.

All external clients are optional constructor-injected arguments, same
pattern as the individual connectors: pass fakes in tests, and in
production leave them None so this module lazily builds the real
pytrends/PRAW/Hugging Face clients (avoids importing heavy libraries
just to run a test that doesn't need them).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.aggregation import aggregate_google_features, aggregate_reddit_features, merge_features
from src.agent.workflow import run_workflow
from src.ingestion.google_trends import GoogleTrendsConnector
from src.ingestion.reddit import RedditConnector
from src.nlp.entities import build_item_lookup, build_nlp_pipeline, extract_mentions
from src.nlp.sentiment import score_mentions
from src.nlp.text_cleaning import clean_text


def _build_reddit_mentions_with_timestamps(
    reddit_df: pd.DataFrame,
    nlp,
    term_lookup: dict,
    item_lookup: dict,
    sentiment_classifier,
) -> pd.DataFrame:
    """
    Runs entity extraction per Reddit row, re-attaches that row's
    created_utc to each mention it produced (entities.extract_mentions
    doesn't carry timestamps — see its module docstring), then scores
    sentiment on the resulting sentences.

    Returns columns: item_id, created_utc, polarity — the exact shape
    aggregate_reddit_features expects.
    """
    all_mentions = []
    mention_timestamps = []

    for _, row in reddit_df.iterrows():
        cleaned = clean_text(row["text"])
        mentions = extract_mentions(cleaned, nlp, term_lookup, item_lookup)
        for mention in mentions:
            all_mentions.append(mention)
            mention_timestamps.append(row["created_utc"])

    if not all_mentions:
        return pd.DataFrame(columns=["item_id", "created_utc", "polarity"])

    sentiment_df = score_mentions(all_mentions, sentiment_classifier)
    sentiment_df["created_utc"] = mention_timestamps
    return sentiment_df[["item_id", "created_utc", "polarity"]]


def run_full_pipeline(
    settings: dict,
    catalog_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    google_client=None,
    reddit_client=None,
    sentiment_classifier=None,
    cache_dir: str = "data/cache",
) -> dict:
    """
    Runs the entire pipeline: Google Trends + Reddit ingestion -> entity
    extraction + sentiment -> weekly aggregation -> LangChain decision
    workflow. Returns a dict with recommendation_table, trend_table,
    inventory_table, google_status, reddit_status, collected_at.

    A failure in either data source degrades gracefully (falls back to
    cache, per the connectors' own retry/cache logic) rather than
    crashing the whole run — the dashboard surfaces source_status so
    the user knows if they're looking at live or cached data.
    """
    collected_at = datetime.now(timezone.utc)

    # --- Google Trends ---
    if google_client is None:
        from pytrends.request import TrendReq

        google_client = TrendReq(hl="en-US", tz=330)
    google_connector = GoogleTrendsConnector(settings, client=google_client, cache_dir=cache_dir)
    google_df, google_status = google_connector.run_and_cache(catalog_df["google_keyword"].tolist())
    google_features = aggregate_google_features(google_df, catalog_df)

    # --- Reddit ---
    if reddit_client is None:
        import praw

        from src.config import load_reddit_credentials

        creds = load_reddit_credentials(".env")
        reddit_client = praw.Reddit(
            client_id=creds.client_id, client_secret=creds.client_secret, user_agent=creds.user_agent
        )
        reddit_client.read_only = True
    reddit_connector = RedditConnector(settings, client=reddit_client, cache_dir=cache_dir)
    reddit_df, reddit_status = reddit_connector.run_and_cache()

    # --- NLP: entities + sentiment on the Reddit text ---
    nlp, term_lookup = build_nlp_pipeline(catalog_df)
    item_lookup = build_item_lookup(catalog_df)

    if sentiment_classifier is None:
        from transformers import pipeline

        sentiment_classifier = pipeline(
            "text-classification", model=settings["sentiment_model"]["name"], top_k=None
        )

    reddit_mentions = _build_reddit_mentions_with_timestamps(
        reddit_df, nlp, term_lookup, item_lookup, sentiment_classifier
    )
    reddit_features = aggregate_reddit_features(reddit_mentions, reference_time=collected_at.timestamp())

    # --- Merge into the feature table every catalog item has a row in ---
    features_df = merge_features(catalog_df, google_features, reddit_features)

    # --- LangChain decision workflow ---
    result_state = run_workflow(features_df, inventory_df, settings)

    return {
        "recommendation_table": result_state["recommendation_table"],
        "trend_table": result_state["trend_table"],
        "inventory_table": result_state["inventory_table"],
        "google_df": google_df,  # raw weekly interest, for charting actual trend lines in the dashboard
        "google_status": google_status,
        "reddit_status": reddit_status,
        "collected_at": collected_at,
    }
