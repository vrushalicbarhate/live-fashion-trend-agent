import pandas as pd
import pytest

from src.agent.workflow import run_workflow
from src.config import load_settings

SETTINGS = load_settings("config/settings.yaml")


def make_features_df():
    return pd.DataFrame(
        [
            {
                "item_id": "C001",
                "google_latest_2wk_mean": 85,
                "google_prev_2wk_mean": 40,
                "reddit_latest_14d": 20,
                "reddit_prev_14d": 5,
                "sentiment_polarity_mean": 0.6,
            },
            {
                "item_id": "C002",
                "google_latest_2wk_mean": 10,
                "google_prev_2wk_mean": 30,
                "reddit_latest_14d": 1,
                "reddit_prev_14d": 10,
                "sentiment_polarity_mean": -0.5,
            },
        ]
    )


def make_inventory_df():
    return pd.DataFrame(
        [
            {
                "item_id": "C001",
                "current_stock": 20,
                "units_sold_last_30_days": 300,
                "production_lead_time_days": 14,
                "last_updated": "2026-09-15",
            },
            {
                "item_id": "C002",
                "current_stock": 900,
                "units_sold_last_30_days": 10,
                "production_lead_time_days": 20,
                "last_updated": "2026-09-15",
            },
        ]
    )


def test_workflow_produces_all_three_tables_in_order():
    result_state = run_workflow(make_features_df(), make_inventory_df(), SETTINGS)

    assert "trend_table" in result_state
    assert "inventory_table" in result_state
    assert "recommendation_table" in result_state

    rec_table = result_state["recommendation_table"]
    assert set(rec_table["item_id"]) == {"C001", "C002"}
    assert "action" in rec_table.columns
    assert "reason" in rec_table.columns
    assert "priority_rank" in rec_table.columns


def test_workflow_produces_expected_actions_for_clear_cut_items():
    result_state = run_workflow(make_features_df(), make_inventory_df(), SETTINGS)
    rec_table = result_state["recommendation_table"]

    c001 = rec_table[rec_table["item_id"] == "C001"].iloc[0]
    c002 = rec_table[rec_table["item_id"] == "C002"].iloc[0]

    # C001: strongly rising, high trend score, very low stock -> Increase Production
    assert c001["action"] == "Increase Production"
    # C002: strongly declining, low trend score, huge excess stock -> Discontinue
    assert c002["action"] == "Discontinue"


def test_workflow_fails_at_trend_analysis_tool_on_missing_feature_column():
    broken_features = make_features_df().drop(columns=["sentiment_polarity_mean"])

    with pytest.raises(ValueError, match="missing required columns"):
        run_workflow(broken_features, make_inventory_df(), SETTINGS)
