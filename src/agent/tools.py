"""
LangChain Agent Layer — typed tools (synopsis §5.9, §9.6).

Each tool is a thin RunnableLambda wrapper around a pure function we
already built and tested (scoring.compute_trend_table,
inventory.compute_inventory_table, rules.compute_recommendations).
LangChain's job here is orchestration and naming, not decision-making —
there is no LLM call anywhere in this layer, deliberately (see
workflow.py docstring for why).

Each tool reads from and writes to a shared `state` dict, since later
tools need earlier tables (Inventory Check needs trend_table;
Production Recommendation needs both trend_table and inventory_table).
Validation isn't duplicated here — compute_trend_table and friends
already raise clear errors on missing columns, so a broken input fails
at the exact tool that received it, with that tool's name in the
traceback.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableLambda

from src.inventory import compute_inventory_table
from src.rules import compute_recommendations
from src.scoring import compute_trend_table


def build_trend_analysis_tool(settings: dict) -> RunnableLambda:
    """Input: state['features_df']. Output: adds state['trend_table']."""

    def _run(state: dict) -> dict:
        state["trend_table"] = compute_trend_table(state["features_df"], settings)
        return state

    return RunnableLambda(_run, name="trend_analysis_tool")


def build_inventory_check_tool(settings: dict) -> RunnableLambda:
    """Input: state['trend_table'], state['inventory_df']. Output: adds state['inventory_table']."""

    def _run(state: dict) -> dict:
        state["inventory_table"] = compute_inventory_table(
            state["trend_table"], state["inventory_df"], settings
        )
        return state

    return RunnableLambda(_run, name="inventory_check_tool")


def build_production_recommendation_tool(settings: dict) -> RunnableLambda:
    """Input: state['inventory_table']. Output: adds state['recommendation_table']."""

    def _run(state: dict) -> dict:
        state["recommendation_table"] = compute_recommendations(state["inventory_table"], settings)
        return state

    return RunnableLambda(_run, name="production_recommendation_tool")
