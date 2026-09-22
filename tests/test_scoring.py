import pandas as pd
import pytest

from src.scoring import compute_trend_table

SETTINGS = {
    "trend_score": {
        "weights": {"google": 0.50, "reddit_frequency": 0.30, "sentiment": 0.20},
        "momentum": {
            "google_weight": 0.60,
            "reddit_weight": 0.40,
            "rising_threshold": 0.10,
            "declining_threshold": -0.10,
        },
    }
}


def make_features(**overrides):
    base = {
        "item_id": ["C001"],
        "google_latest_2wk_mean": [50.0],
        "google_prev_2wk_mean": [50.0],
        "reddit_latest_14d": [10],
        "reddit_prev_14d": [10],
        "sentiment_polarity_mean": [0.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_trend_score_is_weighted_combination():
    # Force known component values: G=100 (google flat at 100),
    # single item so reddit min-max scaling collapses to 50 (no variation),
    # sentiment polarity 1.0 -> S=100.
    df = make_features(
        google_latest_2wk_mean=[100.0], google_prev_2wk_mean=[100.0], sentiment_polarity_mean=[1.0]
    )
    result = compute_trend_table(df, SETTINGS)
    row = result.iloc[0]

    assert row["google_interest"] == 100
    assert row["sentiment_component"] == 100
    assert row["reddit_component"] == 50  # single item -> no variation -> midpoint
    # trend_score = 0.5*100 + 0.3*50 + 0.2*100 = 50 + 15 + 20 = 85
    assert row["trend_score"] == 85


def test_reddit_component_scales_relative_to_other_items():
    df = make_features(
        item_id=["C001", "C002"],
        google_latest_2wk_mean=[50.0, 50.0],
        google_prev_2wk_mean=[50.0, 50.0],
        reddit_latest_14d=[1, 100],  # C002 has far more mentions
        reddit_prev_14d=[1, 100],
        sentiment_polarity_mean=[0.0, 0.0],
    )
    result = compute_trend_table(df, SETTINGS)
    r_low = result[result["item_id"] == "C001"].iloc[0]["reddit_component"]
    r_high = result[result["item_id"] == "C002"].iloc[0]["reddit_component"]

    assert r_low == 0    # min gets scaled to 0
    assert r_high == 100  # max gets scaled to 100


def test_growth_from_zero_to_positive_is_treated_as_max_rise():
    df = make_features(reddit_latest_14d=[5], reddit_prev_14d=[0])
    result = compute_trend_table(df, SETTINGS)
    assert result.iloc[0]["reddit_momentum"] == 1.0


def test_growth_from_zero_to_zero_is_flat():
    df = make_features(reddit_latest_14d=[0], reddit_prev_14d=[0])
    result = compute_trend_table(df, SETTINGS)
    assert result.iloc[0]["reddit_momentum"] == 0.0


def test_direction_rising_when_momentum_above_threshold():
    # google doubles (50 -> 100): growth = (100-50)/50 = 1.0, clipped to 1.0
    df = make_features(google_latest_2wk_mean=[100.0], google_prev_2wk_mean=[50.0])
    result = compute_trend_table(df, SETTINGS)
    row = result.iloc[0]
    # weighted_momentum = 0.6*1.0 + 0.4*0.0 = 0.6 >= 0.10
    assert row["direction"] == "Rising"


def test_direction_declining_when_momentum_below_negative_threshold():
    df = make_features(google_latest_2wk_mean=[25.0], google_prev_2wk_mean=[100.0])
    result = compute_trend_table(df, SETTINGS)
    row = result.iloc[0]
    # growth = (25-100)/100 = -0.75; weighted = 0.6*-0.75 = -0.45 <= -0.10
    assert row["direction"] == "Declining"


def test_direction_stable_when_momentum_within_band():
    # tiny google growth, no reddit growth -> weighted momentum near 0
    df = make_features(google_latest_2wk_mean=[51.0], google_prev_2wk_mean=[50.0])
    result = compute_trend_table(df, SETTINGS)
    assert result.iloc[0]["direction"] == "Stable"


def test_raises_on_missing_required_column():
    df = make_features().drop(columns=["reddit_prev_14d"])
    with pytest.raises(ValueError, match="missing required columns"):
        compute_trend_table(df, SETTINGS)
