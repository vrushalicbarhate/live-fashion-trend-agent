"""
Trend Analysis Tool (synopsis §5.6, §9.6, workflow step 6).

Consumes an already-aggregated weekly feature table (one row per
catalog item) and produces the composite Trend Score and Rising/
Stable/Declining direction. Pure, deterministic functions — no I/O,
no external services — which is what makes this the most exhaustively
testable part of the whole pipeline.

Expected input columns (produced by the upstream aggregation stage,
not yet built — see repo notes):
    item_id
    google_latest_2wk_mean, google_prev_2wk_mean   (0-100 Trends values)
    reddit_latest_14d, reddit_prev_14d              (raw mention counts)
    sentiment_polarity_mean                         (mean positive-negative, [-1, 1])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_FEATURE_COLUMNS = [
    "item_id",
    "google_latest_2wk_mean",
    "google_prev_2wk_mean",
    "reddit_latest_14d",
    "reddit_prev_14d",
    "sentiment_polarity_mean",
]


def _google_component(latest_2wk_mean: pd.Series) -> pd.Series:
    """G is already a 0-100 index; clip defensively in case of bad input."""
    return latest_2wk_mean.clip(0, 100)


def _reddit_component(latest_counts: pd.Series) -> pd.Series:
    """
    R = log(1 + count), then min-max scaled to 0-100 ACROSS the monitored
    items in this run. log1p compresses the long tail (one viral post
    shouldn't dominate the score the way a raw count would); min-max
    scaling makes items comparable to each other within this run, the
    same way Google's own 0-100 index is relative rather than absolute.

    If every item has the same count (no variation to scale), we can't
    divide by a zero range — everyone gets the neutral midpoint (50)
    rather than an arbitrary 0 or 100.
    """
    log_counts = np.log1p(latest_counts)
    value_range = log_counts.max() - log_counts.min()
    if value_range == 0:
        return pd.Series(50.0, index=latest_counts.index)
    return (log_counts - log_counts.min()) / value_range * 100


def _sentiment_component(mean_polarity: pd.Series) -> pd.Series:
    """S = 50 * (1 + mean_polarity); polarity in [-1,1] maps onto [0,100]."""
    return (50 * (1 + mean_polarity)).clip(0, 100)


def _growth(latest: pd.Series, previous: pd.Series) -> pd.Series:
    """
    Relative growth, clipped to [-1, 1] per the synopsis's momentum
    formula. When `previous` is 0, relative growth is mathematically
    undefined (division by zero) — we treat "0 -> 0" as flat (0.0
    growth) and "0 -> something" as maximal growth (+1.0), since any
    increase from nothing is a strong rising signal worth flagging,
    even though we can't quantify it as a ratio.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = np.where(previous == 0, np.where(latest == 0, 0.0, 1.0), (latest - previous) / previous)
    return pd.Series(growth, index=latest.index).clip(-1, 1)


def compute_trend_table(features_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    missing = [c for c in REQUIRED_FEATURE_COLUMNS if c not in features_df.columns]
    if missing:
        raise ValueError(f"features_df is missing required columns: {missing}")

    ts_cfg = settings["trend_score"]
    weights = ts_cfg["weights"]
    momentum_cfg = ts_cfg["momentum"]

    df = features_df.copy()

    df["google_interest"] = _google_component(df["google_latest_2wk_mean"])
    df["reddit_component"] = _reddit_component(df["reddit_latest_14d"])
    df["sentiment_component"] = _sentiment_component(df["sentiment_polarity_mean"])

    df["trend_score"] = (
        weights["google"] * df["google_interest"]
        + weights["reddit_frequency"] * df["reddit_component"]
        + weights["sentiment"] * df["sentiment_component"]
    ).round().astype(int)

    df["google_momentum"] = _growth(df["google_latest_2wk_mean"], df["google_prev_2wk_mean"])
    df["reddit_momentum"] = _growth(df["reddit_latest_14d"], df["reddit_prev_14d"])

    df["weighted_momentum"] = (
        momentum_cfg["google_weight"] * df["google_momentum"]
        + momentum_cfg["reddit_weight"] * df["reddit_momentum"]
    )

    rising_t = momentum_cfg["rising_threshold"]
    declining_t = momentum_cfg["declining_threshold"]
    df["direction"] = np.select(
        [df["weighted_momentum"] >= rising_t, df["weighted_momentum"] <= declining_t],
        ["Rising", "Declining"],
        default="Stable",
    )

    return df[
        [
            "item_id",
            "trend_score",
            "direction",
            "google_interest",
            "reddit_component",
            "sentiment_component",
            "reddit_latest_14d",
            "google_momentum",
            "reddit_momentum",
            "weighted_momentum",
        ]
    ]
