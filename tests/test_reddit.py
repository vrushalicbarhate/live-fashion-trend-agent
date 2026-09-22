from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.ingestion.reddit import RedditConnector

SETTINGS = {
    "reddit": {
        "subreddits": ["fashion", "streetwear"],
        "lookback_days": 30,
        "max_posts_per_subreddit": 50,
        "max_comments_per_post": 2,
    }
}

NOW = datetime.now(timezone.utc)


def days_ago(n):
    return (NOW - timedelta(days=n)).timestamp()


class FakeComment:
    def __init__(self, body, created_utc):
        self.body = body
        self.created_utc = created_utc
        self.author = "should_never_be_read"  # present to prove we never touch it


class FakeCommentForest(list):
    def replace_more(self, limit=0):
        pass  # no-op stand-in for PRAW's "load more comments" stubs


class FakeSubmission:
    def __init__(self, id, title, selftext, created_utc, comments):
        self.id = id
        self.title = title
        self.selftext = selftext
        self.created_utc = created_utc
        self.comments = FakeCommentForest(comments)
        self.author = "should_never_be_read"


class FakeSubredditHandle:
    def __init__(self, submissions, should_fail=False):
        self._submissions = submissions
        self.should_fail = should_fail

    def new(self, limit):
        if self.should_fail:
            raise ConnectionError("simulated network failure")
        return iter(self._submissions[:limit])


class FakeRedditClient:
    def __init__(self, subreddit_handles: dict[str, FakeSubredditHandle]):
        self._handles = subreddit_handles

    def subreddit(self, name):
        return self._handles[name]


def test_filters_out_posts_older_than_lookback_window():
    recent_post = FakeSubmission("p1", "Recent post", "body text", days_ago(5), [])
    old_post = FakeSubmission("p2", "Old post", "body text", days_ago(60), [])

    client = FakeRedditClient(
        {
            "fashion": FakeSubredditHandle([recent_post, old_post]),
            "streetwear": FakeSubredditHandle([]),
        }
    )
    connector = RedditConnector(SETTINGS, client=client)
    df, status = connector.fetch_all()

    assert status == "live"
    assert "p1" in df["post_id"].values
    assert "p2" not in df["post_id"].values


def test_caps_comments_per_post_and_includes_post_text():
    comments = [
        FakeComment("first comment", days_ago(2)),
        FakeComment("second comment", days_ago(2)),
        FakeComment("third comment should be dropped", days_ago(2)),
    ]
    post = FakeSubmission("p1", "Loving this jacket", "great fit", days_ago(2), comments)

    client = FakeRedditClient(
        {
            "fashion": FakeSubredditHandle([post]),
            "streetwear": FakeSubredditHandle([]),
        }
    )
    connector = RedditConnector(SETTINGS, client=client)
    df, status = connector.fetch_all()

    post_rows = df[(df["post_id"] == "p1") & (df["source_type"] == "post")]
    comment_rows = df[(df["post_id"] == "p1") & (df["source_type"] == "comment")]

    assert len(post_rows) == 1
    assert "Loving this jacket" in post_rows.iloc[0]["text"]
    assert len(comment_rows) == 2  # capped at max_comments_per_post=2
    assert "third comment" not in comment_rows["text"].str.cat()


def test_output_never_contains_author_fields():
    post = FakeSubmission("p1", "title", "body", days_ago(1), [FakeComment("c", days_ago(1))])
    client = FakeRedditClient(
        {
            "fashion": FakeSubredditHandle([post]),
            "streetwear": FakeSubredditHandle([]),
        }
    )
    connector = RedditConnector(SETTINGS, client=client)
    df, _ = connector.fetch_all()

    assert "author" not in df.columns
    assert set(df.columns) == {"subreddit", "post_id", "source_type", "text", "created_utc"}


def test_partial_failure_status_when_one_subreddit_fails():
    post = FakeSubmission("p1", "title", "body", days_ago(1), [])
    client = FakeRedditClient(
        {
            "fashion": FakeSubredditHandle([post]),
            "streetwear": FakeSubredditHandle([], should_fail=True),
        }
    )
    connector = RedditConnector(SETTINGS, client=client)
    df, status = connector.fetch_all()

    assert status == "live_partial"
    assert len(df) >= 1


def test_run_and_cache_falls_back_on_total_failure(tmp_path):
    post = FakeSubmission("p1", "title", "body", days_ago(1), [])
    good_client = FakeRedditClient(
        {
            "fashion": FakeSubredditHandle([post]),
            "streetwear": FakeSubredditHandle([]),
        }
    )
    connector = RedditConnector(SETTINGS, client=good_client, cache_dir=tmp_path)
    df1, status1 = connector.run_and_cache()
    assert status1 == "live"
    assert (tmp_path / "reddit_latest.csv").exists()

    failing_client = FakeRedditClient(
        {
            "fashion": FakeSubredditHandle([], should_fail=True),
            "streetwear": FakeSubredditHandle([], should_fail=True),
        }
    )
    connector2 = RedditConnector(SETTINGS, client=failing_client, cache_dir=tmp_path)
    df2, status2 = connector2.run_and_cache()

    assert status2 == "cached"
    assert not df2.empty
