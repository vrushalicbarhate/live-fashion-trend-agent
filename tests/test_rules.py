import pandas as pd

from src.rules import compute_recommendations

SETTINGS = {
    "production_rules": {
        "increase_production": {"min_trend_score": 70, "direction": "Rising"},
        "plan_for_next_season": {"min_trend_score": 70, "direction": "Rising"},
        "discontinue": {
            "max_trend_score": 40,
            "direction": "Declining",
            "min_days_of_stock_multiplier": 2,
            "min_days_of_stock_floor": 60,
        },
    }
}


def make_row(**overrides):
    base = {
        "item_id": "C001",
        "trend_score": 50,
        "direction": "Stable",
        "current_stock": 100,
        "days_of_stock": 30.0,
        "lead_time_days": 14,
        "is_low_stock": False,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_increase_production_when_rising_high_score_and_low_stock():
    df = make_row(
        trend_score=75, direction="Rising", days_of_stock=10.0, is_low_stock=True, lead_time_days=14
    )
    result = compute_recommendations(df, SETTINGS)
    assert result.iloc[0]["action"] == "Increase Production"
    assert "10 days" in result.iloc[0]["reason"]


def test_plan_for_next_season_when_rising_high_score_and_adequate_stock():
    df = make_row(
        trend_score=75, direction="Rising", days_of_stock=200.0, is_low_stock=False, lead_time_days=14
    )
    result = compute_recommendations(df, SETTINGS)
    assert result.iloc[0]["action"] == "Plan for Next Season"


def test_discontinue_when_declining_low_score_and_excess_days_of_stock():
    df = make_row(
        trend_score=30, direction="Declining", days_of_stock=100.0, lead_time_days=14
    )
    result = compute_recommendations(df, SETTINGS)
    assert result.iloc[0]["action"] == "Discontinue"  # threshold = max(60, 28) = 60; 100 >= 60


def test_discontinue_not_triggered_when_stock_not_excessive():
    df = make_row(
        trend_score=30, direction="Declining", days_of_stock=40.0, lead_time_days=14
    )
    result = compute_recommendations(df, SETTINGS)
    assert result.iloc[0]["action"] == "Maintain"  # 40 < 60 threshold


def test_discontinue_uses_unit_fallback_when_days_of_stock_unavailable():
    df = make_row(
        trend_score=30,
        direction="Declining",
        days_of_stock=pd.NA,
        current_stock=500,
        lead_time_days=14,
        is_low_stock=pd.NA,
    )
    result = compute_recommendations(df, SETTINGS)
    assert result.iloc[0]["action"] == "Discontinue"  # 500 units >= 60 threshold
    assert "no recent sales" in result.iloc[0]["reason"]


def test_maintain_when_declining_but_stock_not_excessive_and_unavailable():
    df = make_row(
        trend_score=30,
        direction="Declining",
        days_of_stock=pd.NA,
        current_stock=20,  # below the 60-unit fallback threshold
        lead_time_days=14,
        is_low_stock=pd.NA,
    )
    result = compute_recommendations(df, SETTINGS)
    assert result.iloc[0]["action"] == "Maintain"


def test_maintain_is_the_default_for_stable_items():
    df = make_row(trend_score=55, direction="Stable")
    result = compute_recommendations(df, SETTINGS)
    assert result.iloc[0]["action"] == "Maintain"


def test_rising_with_zero_sales_falls_back_to_maintain_not_increase():
    # Rising + high score, but days_of_stock unmeasurable -> conservative Maintain,
    # per the synopsis's "falls back to a conservative Maintain action" rule.
    df = make_row(
        trend_score=80,
        direction="Rising",
        days_of_stock=pd.NA,
        current_stock=500,
        is_low_stock=pd.NA,
        lead_time_days=14,
    )
    result = compute_recommendations(df, SETTINGS)
    assert result.iloc[0]["action"] == "Maintain"


def test_priority_rank_orders_by_trend_score_descending():
    df = pd.concat(
        [
            make_row(item_id="C001", trend_score=40),
            make_row(item_id="C002", trend_score=90),
            make_row(item_id="C003", trend_score=60),
        ],
        ignore_index=True,
    )
    result = compute_recommendations(df, SETTINGS)
    assert list(result["item_id"]) == ["C002", "C003", "C001"]
    assert list(result["priority_rank"]) == [1, 2, 3]
