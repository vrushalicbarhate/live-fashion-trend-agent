"""
Inventory Check Tool (synopsis §5.7, §9.5, workflow step 7).

Joins the trend table with inventory.csv and computes stock coverage.
Zero recent sales makes days_of_stock mathematically undefined
(division by zero) — the synopsis calls this out explicitly ("days of
stock is reported as unavailable"), so we surface it as pd.NA rather
than 0 or infinity, and let the Production Recommendation Tool decide
what to do about it.
"""

from __future__ import annotations

import pandas as pd


def compute_inventory_table(
    trend_df: pd.DataFrame, inventory_df: pd.DataFrame, settings: dict
) -> pd.DataFrame:
    safety_buffer = settings["inventory"]["safety_buffer_days"]

    df = trend_df.merge(inventory_df, on="item_id", how="left", validate="one_to_one")

    df["avg_daily_sales"] = df["units_sold_last_30_days"] / 30

    # days_of_stock is NA when avg_daily_sales is 0 (division by zero) —
    # deliberately not 0 or inf, so downstream logic can distinguish
    # "no stock coverage info" from "stock runs out today".
    df["days_of_stock"] = df["current_stock"] / df["avg_daily_sales"].replace(0, pd.NA)

    df["low_stock_threshold"] = df["production_lead_time_days"] + safety_buffer

    # Plain NaN comparisons resolve to False, not "unknown" — that would
    # silently treat an unmeasurable item as adequately stocked, which is
    # the wrong default. Use pandas' nullable boolean dtype so NA stays NA.
    is_low = df["days_of_stock"] <= df["low_stock_threshold"]
    df["is_low_stock"] = is_low.astype("boolean")
    df.loc[df["days_of_stock"].isna(), "is_low_stock"] = pd.NA

    return df.rename(columns={"production_lead_time_days": "lead_time_days"})
