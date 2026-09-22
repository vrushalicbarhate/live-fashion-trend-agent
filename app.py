"""
Decision Dashboard (synopsis §5.10, §9.7).

Deliberately thin: every fetch/NLP/scoring step lives in src/pipeline.py
(see that module's docstring for why). This file only triggers a run,
stores the result in st.session_state, and renders it — filtering and
chart selection never re-trigger the live pipeline, only a re-render.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import load_catalog, load_inventory, load_settings
from src.pipeline import run_full_pipeline

st.set_page_config(page_title="Live Fashion Trend & Production Agent", layout="wide")

# --- Clean, minimal styling: white background, white cards, indigo accent, no gradients ---
INDIGO = "#4F46E5"
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #FFFFFF; }}
    .metric-card {{
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: left;
    }}
    .metric-card .label {{ color: #6B7280; font-size: 0.85rem; margin-bottom: 4px; }}
    .metric-card .value {{ color: {INDIGO}; font-size: 1.8rem; font-weight: 600; }}
    .status-pill {{
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 500;
    }}
    .status-live {{ background-color: #EEF2FF; color: {INDIGO}; }}
    .status-cached {{ background-color: #FEF3C7; color: #92400E; }}
    .status-failed {{ background-color: #FEE2E2; color: #991B1B; }}
    div.stButton > button {{
        background-color: {INDIGO}; color: white; border: none; border-radius: 8px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_config():
    settings = load_settings("config/settings.yaml")
    catalog = load_catalog("config/catalog.csv")
    inventory = load_inventory("data/inventory.csv", catalog_df=catalog)
    return settings, catalog, inventory


settings, catalog_df, inventory_df = load_config()

# --- Sidebar: the only place a live pipeline run can be triggered ---
with st.sidebar:
    st.header("Live Fashion Trend Agent")
    st.caption("Google Trends + Reddit \u2192 NLP \u2192 production recommendation")

    if st.button("Run Analysis", use_container_width=True):
        with st.spinner("Fetching live data and scoring trends\u2026 this can take a minute or two."):
            try:
                st.session_state["result"] = run_full_pipeline(settings, catalog_df, inventory_df)
            except Exception as exc:
                st.session_state["run_error"] = str(exc)

    if "run_error" in st.session_state:
        st.error(f"Analysis failed: {st.session_state['run_error']}")

    if "result" in st.session_state:
        result = st.session_state["result"]
        st.divider()
        st.caption(f"Last run: {result['collected_at'].strftime('%Y-%m-%d %H:%M UTC')}")

        def status_pill(label, status):
            css_class = {"live": "status-live", "live_partial": "status-live", "cached": "status-cached"}.get(
                status, "status-failed"
            )
            st.markdown(
                f'<span class="status-pill {css_class}">{label}: {status}</span>', unsafe_allow_html=True
            )

        status_pill("Google Trends", result["google_status"])
        status_pill("Reddit", result["reddit_status"])

# --- Main area ---
if "result" not in st.session_state:
    st.info("Click **Run Analysis** in the sidebar to fetch live data and generate recommendations.")
    st.stop()

result = st.session_state["result"]
rec_df = result["recommendation_table"].merge(
    catalog_df[["item_id", "product_type", "material", "color"]], on="item_id", how="left"
)

# --- Summary cards: one per action ---
ACTIONS = ["Increase Production", "Plan for Next Season", "Maintain", "Discontinue"]
cols = st.columns(4)
for col, action in zip(cols, ACTIONS):
    count = (rec_df["action"] == action).sum()
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="label">{action}</div>'
            f'<div class="value">{count}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")

# --- Filters ---
filter_cols = st.columns(4)
with filter_cols[0]:
    product_filter = st.multiselect("Product", sorted(rec_df["product_type"].dropna().unique()))
with filter_cols[1]:
    material_filter = st.multiselect("Material", sorted(rec_df["material"].dropna().unique()))
with filter_cols[2]:
    color_filter = st.multiselect("Color", sorted(rec_df["color"].dropna().unique()))
with filter_cols[3]:
    action_filter = st.multiselect("Action", ACTIONS)

filtered_df = rec_df.copy()
if product_filter:
    filtered_df = filtered_df[filtered_df["product_type"].isin(product_filter)]
if material_filter:
    filtered_df = filtered_df[filtered_df["material"].isin(material_filter)]
if color_filter:
    filtered_df = filtered_df[filtered_df["color"].isin(color_filter)]
if action_filter:
    filtered_df = filtered_df[filtered_df["action"].isin(action_filter)]

# --- Ranked recommendation table ---
st.subheader("Production Recommendations")
display_cols = [
    "priority_rank", "item_id", "product_type", "material", "color",
    "trend_score", "direction", "days_of_stock", "action", "reason",
]
st.dataframe(
    filtered_df[display_cols].sort_values("priority_rank"),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download recommendations.csv",
    data=filtered_df[display_cols].to_csv(index=False),
    file_name="recommendations.csv",
    mime="text/csv",
)

st.write("")

# --- Charts ---
chart_cols = st.columns(2)

with chart_cols[0]:
    st.subheader("Google Interest Over Time")
    item_options = dict(zip(catalog_df["item_id"], catalog_df["google_keyword"]))
    selected_item = st.selectbox(
        "Item", options=list(item_options.keys()), format_func=lambda i: f"{i} — {item_options[i]}"
    )
    keyword = item_options[selected_item]
    item_google_df = result["google_df"][result["google_df"]["google_keyword"] == keyword].sort_values("date")
    if item_google_df.empty:
        st.caption("No Google Trends data available for this item in the current run.")
    else:
        fig = px.line(item_google_df, x="date", y="google_interest")
        fig.update_traces(line_color=INDIGO)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

with chart_cols[1]:
    st.subheader("Reddit Mentions (last 14 days)")
    mention_chart_df = rec_df[["item_id", "reddit_latest_14d", "sentiment_component"]].sort_values(
        "reddit_latest_14d", ascending=False
    ).head(10)
    if mention_chart_df["reddit_latest_14d"].sum() == 0:
        st.caption("No Reddit mentions recorded in the current run.")
    else:
        fig = px.bar(mention_chart_df, x="item_id", y="reddit_latest_14d", color="sentiment_component",
                      color_continuous_scale=["#EF4444", "#E5E7EB", INDIGO])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
