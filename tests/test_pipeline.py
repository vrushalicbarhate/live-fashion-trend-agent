from datetime import datetime, timedelta, timezone

import pandas as pd

from src.config import load_catalog, load_settings
from src.pipeline import run_full_pipeline
from tests.test_google_trends import FakeTrendsClient
from tests.test_reddit import FakeComment, FakeRedditClient, FakeSubmission, FakeSubredditHandle
from tests.test_sentiment import FakeClassifier

SETTINGS = load_settings("config/settings.yaml")
CATALOG = load_catalog("config/catalog.csv")

NOW = datetime.now(timezone.utc)


def days_ago_ts(n):
    return (NOW - timedelta(days=n)).timestamp()


def weeks_ago(n):
    return NOW - timedelta(weeks=n)


def build_fake_google_client():
    """
    Builds canned responses for every batch the real catalog will
    produce, all with a rising pattern for the first item's keyword
    and flat/declining for the rest, so the end-to-end test has a
    genuinely non-trivial recommendation to check.
    """
    dates = [weeks_ago(3), weeks_ago(2), weeks_ago(1), weeks_ago(0)]
    anchor = SETTINGS["google_trends"]["anchor_keyword"]
    batch_size = SETTINGS["google_trends"]["batch_size"] - 1

    keywords = CATALOG["google_keyword"].tolist()
    responses = {}
    for i in range(0, len(keywords), batch_size):
        chunk = keywords[i : i + batch_size]
        batch_keywords = chunk + [anchor]
        data = {}
        for kw in chunk:
            if kw == keywords[0]:
                data[kw] = [20, 20, 80, 90]  # clearly rising
            else:
                data[kw] = [30, 30, 30, 30]  # flat
        data[anchor] = [40, 40, 40, 40]
        df = pd.DataFrame(data, index=pd.DatetimeIndex(dates, name="date"))
        df["isPartial"] = False
        responses[frozenset(batch_keywords)] = df

    return FakeTrendsClient(responses)


def build_fake_reddit_client():
    keywords_item = CATALOG.iloc[0]  # the "rising" item from the Google fake data above
    product_term = keywords_item["product_type"]
    material_term = keywords_item["material"]
    color_term = keywords_item["color"]

    post = FakeSubmission(
        id="p1",
        title=f"Loving my new {material_term} {product_term} in {color_term}",
        selftext="Best purchase this year, so comfortable.",
        created_utc=days_ago_ts(2),
        comments=[FakeComment(f"Same, that {color_term} {material_term} {product_term} is great", days_ago_ts(3))],
    )

    handles = {name: FakeSubredditHandle([]) for name in SETTINGS["reddit"]["subreddits"]}
    handles[SETTINGS["reddit"]["subreddits"][0]] = FakeSubredditHandle([post])
    return FakeRedditClient(handles)


def build_fake_sentiment_classifier():
    # Every sentence in this test gets a strongly positive canned score.
    class AlwaysPositiveClassifier:
        def __call__(self, texts):
            return [
                [
                    {"label": "positive", "score": 0.9},
                    {"label": "neutral", "score": 0.08},
                    {"label": "negative", "score": 0.02},
                ]
                for _ in texts
            ]

    return AlwaysPositiveClassifier()


def make_inventory_df():
    rows = []
    for _, row in CATALOG.iterrows():
        is_rising_item = row["item_id"] == CATALOG.iloc[0]["item_id"]
        rows.append(
            {
                "item_id": row["item_id"],
                "current_stock": 20 if is_rising_item else 500,
                "units_sold_last_30_days": 300 if is_rising_item else 20,
                "production_lead_time_days": 14,
                "last_updated": "2026-09-15",
            }
        )
    return pd.DataFrame(rows)


def test_full_pipeline_end_to_end_with_fakes(tmp_path):
    result = run_full_pipeline(
        settings=SETTINGS,
        catalog_df=CATALOG,
        inventory_df=make_inventory_df(),
        google_client=build_fake_google_client(),
        reddit_client=build_fake_reddit_client(),
        sentiment_classifier=build_fake_sentiment_classifier(),
        cache_dir=tmp_path,
    )

    assert result["google_status"] == "live"
    assert result["reddit_status"] == "live"
    assert len(result["recommendation_table"]) == len(CATALOG)

    rising_item_id = CATALOG.iloc[0]["item_id"]
    rec = result["recommendation_table"]
    rising_row = rec[rec["item_id"] == rising_item_id].iloc[0]

    # Rising Google interest + positive Reddit mentions + very low stock
    # relative to sales -> should clear the Increase Production bar.
    assert rising_row["action"] == "Increase Production"
