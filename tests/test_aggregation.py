from datetime import datetime, timedelta, timezone

import pandas as pd

from src.aggregation import aggregate_google_features, aggregate_reddit_features, merge_features

CATALOG = pd.DataFrame(
    [
        {"item_id": "C001", "google_keyword": "white cotton t-shirt"},
        {"item_id": "C002", "google_keyword": "linen wide leg trousers"},
    ]
)

NOW = datetime.now(timezone.utc)


def weeks_ago(n):
    return NOW - timedelta(weeks=n)


def days_ago_ts(n):
    return (NOW - timedelta(days=n)).timestamp()


def test_google_aggregation_splits_latest_and_previous_two_weeks():
    google_df = pd.DataFrame(
        [
            {"google_keyword": "white cotton t-shirt", "date": weeks_ago(0), "google_interest": 80},
            {"google_keyword": "white cotton t-shirt", "date": weeks_ago(1), "google_interest": 70},
            {"google_keyword": "white cotton t-shirt", "date": weeks_ago(2), "google_interest": 40},
            {"google_keyword": "white cotton t-shirt", "date": weeks_ago(3), "google_interest": 30},
        ]
    )
    result = aggregate_google_features(google_df, CATALOG)
    row = result[result["item_id"] == "C001"].iloc[0]

    assert row["google_latest_2wk_mean"] == 75  # mean(80, 70)
    assert row["google_prev_2wk_mean"] == 35    # mean(40, 30)


def test_google_aggregation_falls_back_to_flat_when_insufficient_history():
    google_df = pd.DataFrame(
        [
            {"google_keyword": "white cotton t-shirt", "date": weeks_ago(0), "google_interest": 60},
            {"google_keyword": "white cotton t-shirt", "date": weeks_ago(1), "google_interest": 50},
        ]
    )
    result = aggregate_google_features(google_df, CATALOG)
    row = result[result["item_id"] == "C001"].iloc[0]

    # No previous-window data at all -> prev falls back to equal latest (flat, no momentum)
    assert row["google_prev_2wk_mean"] == row["google_latest_2wk_mean"]


def test_reddit_aggregation_counts_mentions_in_correct_windows():
    mentions = pd.DataFrame(
        [
            {"item_id": "C001", "created_utc": days_ago_ts(2), "polarity": 0.8},   # latest window
            {"item_id": "C001", "created_utc": days_ago_ts(10), "polarity": 0.5},  # latest window
            {"item_id": "C001", "created_utc": days_ago_ts(20), "polarity": -0.2}, # previous window
            {"item_id": "C001", "created_utc": days_ago_ts(40), "polarity": 0.9},  # older, excluded
        ]
    )
    result = aggregate_reddit_features(mentions, reference_time=NOW.timestamp())
    row = result[result["item_id"] == "C001"].iloc[0]

    assert row["reddit_latest_14d"] == 2
    assert row["reddit_prev_14d"] == 1
    assert abs(row["sentiment_polarity_mean"] - 0.65) < 1e-9  # mean(0.8, 0.5)


def test_reddit_aggregation_neutral_polarity_when_no_latest_mentions():
    mentions = pd.DataFrame(
        [{"item_id": "C001", "created_utc": days_ago_ts(20), "polarity": -0.9}]  # only in prev window
    )
    result = aggregate_reddit_features(mentions, reference_time=NOW.timestamp())
    row = result[result["item_id"] == "C001"].iloc[0]

    assert row["reddit_latest_14d"] == 0
    assert row["sentiment_polarity_mean"] == 0.0


def test_merge_features_fills_zero_mentions_item_with_neutral_defaults():
    google_features = pd.DataFrame(
        [
            {"item_id": "C001", "google_latest_2wk_mean": 75, "google_prev_2wk_mean": 35},
            {"item_id": "C002", "google_latest_2wk_mean": 20, "google_prev_2wk_mean": 20},
        ]
    )
    # C002 has no Reddit mentions at all this run.
    reddit_features = pd.DataFrame(
        [{"item_id": "C001", "reddit_latest_14d": 2, "reddit_prev_14d": 1, "sentiment_polarity_mean": 0.65}]
    )

    merged = merge_features(CATALOG, google_features, reddit_features)
    c002 = merged[merged["item_id"] == "C002"].iloc[0]

    assert c002["reddit_latest_14d"] == 0
    assert c002["reddit_prev_14d"] == 0
    assert c002["sentiment_polarity_mean"] == 0.0
    assert len(merged) == 2  # every catalog item present, none dropped
