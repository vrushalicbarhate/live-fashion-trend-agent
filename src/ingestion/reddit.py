"""
Reddit Ingestion Module (synopsis §5.3, §9.2).

Uses PRAW in read-only mode to pull recent posts and a capped number
of comments from the configured fashion subreddits. Deliberately does
NOT retain any author-identifying fields (username, author ID, profile
URL) — only text, subreddit, post_id, and created_utc are kept, per
the privacy scope in the synopsis.

PRAW's `.new()` listing has no built-in date filter — it's newest-first
— so we walk it ourselves and stop once posts fall outside the
lookback window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Protocol

import pandas as pd


class RedditComment(Protocol):
    body: str
    created_utc: float


class RedditSubmission(Protocol):
    id: str
    title: str
    selftext: str
    created_utc: float
    comments: object  # PRAW CommentForest; has .replace_more() and is iterable


class RedditClient(Protocol):
    """Minimal interface we depend on from praw.Reddit, for testability."""

    def subreddit(self, name: str): ...  # returns an object with .new(limit=...)


@dataclass
class SubredditFetchResult:
    subreddit: str
    rows: list[dict]
    success: bool
    error: str | None = None


class RedditConnector:
    def __init__(
        self,
        settings: dict,
        client: RedditClient,
        cache_dir: str | Path = "data/cache",
    ):
        rd = settings["reddit"]
        self.subreddits: list[str] = rd["subreddits"]
        self.lookback_days: int = rd["lookback_days"]
        self.max_posts_per_subreddit: int = rd["max_posts_per_subreddit"]
        self.max_comments_per_post: int = rd["max_comments_per_post"]
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cutoff_timestamp(self) -> float:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        return cutoff.timestamp()

    def _extract_rows_from_submission(self, submission, subreddit_name: str) -> list[dict]:
        """
        Flatten one submission + its top comments into text rows.
        No author fields are read or stored anywhere in this method.
        """
        rows = []

        post_text = f"{submission.title}\n{submission.selftext or ''}".strip()
        if post_text:
            rows.append(
                {
                    "subreddit": subreddit_name,
                    "post_id": submission.id,
                    "source_type": "post",
                    "text": post_text,
                    "created_utc": submission.created_utc,
                }
            )

        # replace_more(limit=0) discards "load more comments" stubs rather
        # than following them — keeps this bounded and fast.
        submission.comments.replace_more(limit=0)
        for comment in list(submission.comments)[: self.max_comments_per_post]:
            body = getattr(comment, "body", None)
            if not body:
                continue
            rows.append(
                {
                    "subreddit": subreddit_name,
                    "post_id": submission.id,
                    "source_type": "comment",
                    "text": body,
                    "created_utc": comment.created_utc,
                }
            )

        return rows

    def _fetch_subreddit(self, name: str, max_retries: int = 3) -> SubredditFetchResult:
        cutoff = self._cutoff_timestamp()
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                rows: list[dict] = []
                listing = self.client.subreddit(name).new(limit=self.max_posts_per_subreddit)
                for submission in listing:
                    if submission.created_utc < cutoff:
                        continue  # outside the lookback window
                    rows.extend(self._extract_rows_from_submission(submission, name))
                return SubredditFetchResult(subreddit=name, rows=rows, success=True)
            except Exception as exc:
                last_error = str(exc)
                if attempt < max_retries:
                    time.sleep(2 * attempt)

        return SubredditFetchResult(subreddit=name, rows=[], success=False, error=last_error)

    def fetch_all(self) -> tuple[pd.DataFrame, str]:
        """
        Fetch posts + comments from every configured subreddit.
        Returns (df, source_status) where source_status is
        'live', 'live_partial' (some subreddits failed), or 'failed'.
        """
        all_rows = []
        any_failure = False

        for name in self.subreddits:
            result = self._fetch_subreddit(name)
            if not result.success:
                any_failure = True
                continue
            all_rows.extend(result.rows)

        if not all_rows:
            columns = ["subreddit", "post_id", "source_type", "text", "created_utc"]
            return pd.DataFrame(columns=columns), "failed"

        df = pd.DataFrame(all_rows)
        status = "live_partial" if any_failure else "live"
        return df, status

    def run_and_cache(self) -> tuple[pd.DataFrame, str]:
        df, status = self.fetch_all()

        if status != "failed":
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            df.to_csv(self.cache_dir / f"reddit_{timestamp}.csv", index=False)
            df.to_csv(self.cache_dir / "reddit_latest.csv", index=False)
            return df, status

        cached = self._load_latest_cache()
        if cached is not None:
            return cached, "cached"
        return df, "failed"

    def _load_latest_cache(self) -> pd.DataFrame | None:
        latest_path = self.cache_dir / "reddit_latest.csv"
        if not latest_path.exists():
            return None
        return pd.read_csv(latest_path)
