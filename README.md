# Live Fashion Trend and Production Recommendation Agent

A LangChain-based decision-support system that combines live Google Trends
search interest, Reddit discussion sentiment, and current inventory position
to recommend what a clothing manufacturer should produce next.

Built as a final-year MCA project (MIT World Peace University, Pune) under
the guidance of Dr. Meenal Jabde, Dept. of Computer Science and Applications.

## What it does

For a curated catalog of ~15–20 product–material–color combinations
(e.g. "white cotton t-shirt", "denim cargo pants"), the pipeline:

1. Pulls weekly search-interest data from **Google Trends** (via `pytrends`)
2. Pulls recent posts/comments from four fashion subreddits (via `PRAW`)
3. Extracts product/material/color mentions from Reddit text using a
   **spaCy EntityRuler** built entirely from the catalog's own vocabulary
4. Scores sentiment on each mention's sentence with a local Hugging Face
   transformer (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
5. Combines both signals into a transparent **0–100 Trend Score** and a
   Rising / Stable / Declining momentum label
6. Joins that against the manufacturer's inventory CSV to calculate days
   of stock remaining
7. Applies deterministic business rules to output one of four actions:
   **Increase Production, Plan for Next Season, Maintain, or Discontinue**
   — each with a one-line, template-generated reason
8. Displays everything in a **Streamlit dashboard** with filters, charts,
   and CSV export

No forecasting model, no synthetic data, no autonomous LLM agent — every
number in the final recommendation traces back to a real, inspectable
source.

## Architecture
The three decision-layer tools are chained with LangChain's LCEL (`|`
operator) into a fixed `RunnableSequence` — **not** an LLM-driven agent.
The tool order is fixed by the business logic itself, so there's no
decision for an LLM to make; adding one would only add cost, latency, and
a chance of mis-sequencing.

## Tech stack

Python 3.11+ · pandas / NumPy · pytrends · PRAW · spaCy (EntityRuler) ·
Hugging Face Transformers + PyTorch · scikit-learn · LangChain (LCEL) ·
Streamlit + Plotly · pytest

## Project structure

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Reddit access requires a registered script app:

1. Go to reddit.com/prefs/apps → **create app** → type **script**
2. Copy `.env.example` to `.env` and fill in the client ID, secret, and a
   descriptive user agent

## Running

```bash
# Run the test suite (no network needed — uses fake clients)
python -m pytest tests/ -v

# Sanity-check config loading
python src/config.py

# Test individual live connectors
python scripts/run_google_trends_live.py
python scripts/run_reddit_live.py
python scripts/run_nlp_live.py

# Launch the full dashboard
streamlit run app.py
```

In the dashboard, click **Run Analysis** in the sidebar to fetch live
data and generate recommendations. The first run downloads the sentiment
model (a few hundred MB) and takes a couple of minutes due to Google
Trends request spacing; subsequent runs reuse the cached model.

## Testing approach

Every external dependency (pytrends, PRAW, the Hugging Face model) is
injected behind a small `Protocol` interface, so the full pipeline —
ingestion through the LangChain workflow — is covered by unit and
integration tests using fake clients, with no network access required.
`tests/test_pipeline.py` runs the entire pipeline end-to-end this way.

## Known limitations

- `pytrends` is an unofficial, previously-archived connector; Google
  Trends values are a relative 0–100 index, not absolute search volume
- Reddit access, rate limits, and retention rules are controlled by
  Reddit and may change
- Entity extraction favors precision over recall — a mention is only
  scored when product, material, and color all appear in the same
  sentence and match a real catalog combination; ambiguous text is
  dropped rather than guessed at
- The four recommended actions are prototype decision support, not a
  validated production-planning system — real use requires historical
  sales validation and retailer sign-off

## Academic scope note

This synopsis intentionally excludes demand forecasting, clustering,
vector databases, multi-agent orchestration, a separate API backend,
and any paid inference service, in favor of a small, fully-explainable,
weekend-buildable pipeline.
