"""
Weekly Aggregation and Feature Creation (synopsis §4.3 step 4, §9.4).

Bridges raw connector output to the feature table scoring.compute_trend_table
expects. Two independent aggregations, since Google and Reddit arrive in
completely different shapes:

- Google: one row per (keyword, week) -> latest-2-weeks vs previous-2-weeks
  mean per item.
- Reddit: one row per mention (already timestamped by the caller, who joins
  entity-extraction output back to its source row's created_utc) -> mention
  counts in the latest vs previous 14-day windows, plus mean sentiment
  polarity in the latest window.

Both aggregations independently cover only the items they have data for;
merge_features fills in the gaps so every catalog item gets a complete row
(an item nobody mentioned on Reddit this week still needs a score).
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd


def aggregate_google_features(google_df: pd.DataFrame, catalog_df: pd.DataFrame) -> pd.DataFrame:
    """
    google_df: columns [google_keyword, date, google_interest] — the long-format
    output of GoogleTrendsConnector.fetch_all / run_and_cache.

    Returns one row per item_id: google_latest_2wk_mean, google_prev_2wk_mean.
    """
    keyword_to_item = dict(zip(catalog_df["google_keyword"], catalog_df["item_id"]))

    df = google_df.copy()
    df["item_id"] = df["google_keyword"].map(keyword_to_item)
    df = df.dropna(subset=["item_id"])
    df["date"] = pd.to_datetime(df["date"])

    rows = []
    for item_id, group in df.groupby("item_id"):
        sorted_group = group.sort_values("date", ascending=False)
        latest_values = sorted_group["google_interest"].iloc[0:2]
        prev_values = sorted_group["google_interest"].iloc[2:4]

        latest_mean = latest_values.mean() if len(latest_values) > 0 else 0.0
        # Not enough history for a previous window: treat as flat (no
        # momentum signal) rather than crashing or fabricating a trend.
        prev_mean = prev_values.mean() if len(prev_values) > 0 else latest_mean

        rows.append(
            {"item_id": item_id, "google_latest_2wk_mean": latest_mean, "google_prev_2wk_mean": prev_mean}
        )

    return pd.DataFrame(rows)


def aggregate_reddit_features(
    reddit_mentions_df: pd.DataFrame, reference_time: float | None = None
) -> pd.DataFrame:
    """
    reddit_mentions_df: columns [item_id, created_utc, polarity] — one row
    per mention, already joined to its source post/comment's timestamp
    (created_utc as a Unix timestamp, matching PRAW's convention).

    reference_time: Unix timestamp to treat as "now". Defaults to the max
    created_utc in the data, which keeps aggregation reproducible in tests
    and in cached/offline runs; a live run should pass the real current time.

    Returns one row per item_id: reddit_latest_14d (count),
    reddit_prev_14d (count), sentiment_polarity_mean (mean polarity in the
    latest window; 0.0 — neutral — when the latest window has no mentions).
    """
    if reddit_mentions_df.empty:
        return pd.DataFrame(
            columns=["item_id", "reddit_latest_14d", "reddit_prev_14d", "sentiment_polarity_mean"]
        )

    if reference_time is None:
        reference_time = reddit_mentions_df["created_utc"].max()

    latest_start = reference_time - timedelta(days=14).total_seconds()
    prev_start = reference_time - timedelta(days=28).total_seconds()

    df = reddit_mentions_df.copy()
    df["window"] = pd.cut(
        df["created_utc"],
        bins=[float("-inf"), prev_start, latest_start, float("inf")],
        labels=["older", "previous", "latest"],
    )

    rows = []
    for item_id, group in df.groupby("item_id"):
        latest_group = group[group["window"] == "latest"]
        prev_group = group[group["window"] == "previous"]
        rows.append(
            {
                "item_id": item_id,
                "reddit_latest_14d": len(latest_group),
                "reddit_prev_14d": len(prev_group),
                "sentiment_polarity_mean": latest_group["polarity"].mean() if len(latest_group) > 0 else 0.0,
            }
        )

    return pd.DataFrame(rows)


def merge_features(
    catalog_df: pd.DataFrame,
    google_features: pd.DataFrame,
    reddit_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-joins both feature sets onto the full catalog so every tracked
    item gets a row, even one with zero Reddit mentions this run (a real,
    meaningful state — not missing data — so it's filled with 0 counts
    and neutral 0.0 polarity rather than dropped).
    """
    df = catalog_df[["item_id"]].merge(google_features, on="item_id", how="left")
    df = df.merge(reddit_features, on="item_id", how="left")

    df["google_latest_2wk_mean"] = df["google_latest_2wk_mean"].fillna(0.0)
    df["google_prev_2wk_mean"] = df["google_prev_2wk_mean"].fillna(0.0)
    df["reddit_latest_14d"] = df["reddit_latest_14d"].fillna(0).astype(int)
    df["reddit_prev_14d"] = df["reddit_prev_14d"].fillna(0).astype(int)
    df["sentiment_polarity_mean"] = df["sentiment_polarity_mean"].fillna(0.0)

    return df
