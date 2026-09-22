"""
LangChain Agent Layer — fixed workflow (synopsis §5.9, §8.2).

Chains the three tools with LCEL's `|` operator into one
RunnableSequence: Trend Analysis -> Inventory Check -> Production
Recommendation, always in that order.

Why not an LLM-driven agent (AgentExecutor + @tool)? That pattern
exists for when a model must decide which tool to call and in what
order based on the situation. Here the order is fixed by the business
logic itself — there is no decision to make, so an LLM in the loop
would only add cost, latency, and a chance of mis-sequencing, with no
upside. LCEL gives the same "typed tools composed into a pipeline"
vocabulary without introducing an LLM call at all.
"""

from __future__ import annotations

from langchain_core.runnables import Runnable

from src.agent.tools import (
    build_inventory_check_tool,
    build_production_recommendation_tool,
    build_trend_analysis_tool,
)


def build_workflow(settings: dict) -> Runnable:
    return (
        build_trend_analysis_tool(settings)
        | build_inventory_check_tool(settings)
        | build_production_recommendation_tool(settings)
    )


def run_workflow(features_df, inventory_df, settings: dict) -> dict:
    """
    Convenience entry point: builds the workflow and runs it on a fresh
    state dict. Returns the final state, containing trend_table,
    inventory_table, and recommendation_table.
    """
    workflow = build_workflow(settings)
    initial_state = {"features_df": features_df, "inventory_df": inventory_df}
    return workflow.invoke(initial_state)
