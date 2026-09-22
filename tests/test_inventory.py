import pandas as pd

from src.inventory import compute_inventory_table

SETTINGS = {"inventory": {"safety_buffer_days": 7}}


def make_trend_df():
    return pd.DataFrame({"item_id": ["C001", "C002"], "trend_score": [80, 30]})


def make_inventory_df(**overrides):
    base = {
        "item_id": ["C001", "C002"],
        "current_stock": [100, 500],
        "units_sold_last_30_days": [300, 0],
        "production_lead_time_days": [14, 14],
        "last_updated": ["2026-09-15", "2026-09-15"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_days_of_stock_computed_correctly_for_normal_sales():
    result = compute_inventory_table(make_trend_df(), make_inventory_df(), SETTINGS)
    c001 = result[result["item_id"] == "C001"].iloc[0]

    assert c001["avg_daily_sales"] == 10  # 300 / 30
    assert c001["days_of_stock"] == 10   # 100 / 10


def test_days_of_stock_is_na_when_zero_sales():
    result = compute_inventory_table(make_trend_df(), make_inventory_df(), SETTINGS)
    c002 = result[result["item_id"] == "C002"].iloc[0]

    assert c002["avg_daily_sales"] == 0
    assert pd.isna(c002["days_of_stock"])
    assert pd.isna(c002["is_low_stock"])  # unknown, not False


def test_low_stock_threshold_uses_lead_time_plus_safety_buffer():
    result = compute_inventory_table(make_trend_df(), make_inventory_df(), SETTINGS)
    c001 = result[result["item_id"] == "C001"].iloc[0]

    assert c001["low_stock_threshold"] == 21  # 14 + 7
    assert c001["is_low_stock"] == True  # days_of_stock=10 <= 21


def test_adequate_stock_is_not_flagged_low():
    inventory = make_inventory_df(current_stock=[1000, 500])
    result = compute_inventory_table(make_trend_df(), inventory, SETTINGS)
    c001 = result[result["item_id"] == "C001"].iloc[0]

    assert c001["days_of_stock"] == 100  # 1000 / 10
    assert c001["is_low_stock"] == False
