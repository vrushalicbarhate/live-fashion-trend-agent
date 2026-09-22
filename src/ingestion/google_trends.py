"""
Google Trends Ingestion Module (synopsis §5.2, §9.2).

pytrends only compares up to 5 keywords per request, and each request
is rescaled independently (values aren't comparable across requests).
To monitor an arbitrarily large catalog, we split keywords into
batches of 4 "candidates" + 1 shared "anchor" keyword, then rescale
every batch's candidate values against the anchor's value in a fixed
reference batch.

pytrends is unofficial and known to break when Google changes its
site, so every live call is wrapped in retries, and every successful
run is cached with a timestamp so the dashboard can fall back to the
last good snapshot instead of failing outright.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd


class TrendsClient(Protocol):
    """
    Minimal interface we depend on from pytrends' TrendReq, so tests
    can inject a fake client instead of hitting Google over the network.
    """

    def build_payload(self, kw_list: list[str], timeframe: str, geo: str) -> None: ...
    def interest_over_time(self) -> pd.DataFrame: ...


@dataclass
class BatchResult:
    keywords: list[str]
    raw: pd.DataFrame          # columns: date index, one column per keyword, 'isPartial'
    success: bool
    error: str | None = None


class GoogleTrendsConnector:
    def __init__(
        self,
        settings: dict,
        client: TrendsClient,
        cache_dir: str | Path = "data/cache",
    ):
        gt = settings["google_trends"]
        self.region = gt["region"]
        self.window_months = gt["window_months"]
        self.anchor_keyword = gt["anchor_keyword"]
        self.batch_size = gt.get("batch_size", 5) - 1  # minus 1 slot reserved for the anchor
        self.request_delay = gt.get("request_delay_seconds", 5)
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _timeframe(self) -> str:
        return f"today {self.window_months}-m"

    def _make_batches(self, keywords: list[str]) -> list[list[str]]:
        """Split candidate keywords into chunks, appending the anchor to each."""
        # De-duplicate but preserve order (a keyword shared by two items
        # would otherwise be fetched and rescaled twice).
        seen = set()
        unique_keywords = [k for k in keywords if not (k in seen or seen.add(k))]

        batches = []
        for i in range(0, len(unique_keywords), self.batch_size):
            chunk = unique_keywords[i : i + self.batch_size]
            batches.append(chunk + [self.anchor_keyword])
        return batches

    def _fetch_with_retries(
        self, keywords: list[str], max_retries: int = 3
    ) -> BatchResult:
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                self.client.build_payload(
                    kw_list=keywords, timeframe=self._timeframe(), geo=self.region
                )
                raw = self.client.interest_over_time()
                if raw is None or raw.empty:
                    raise ValueError("pytrends returned an empty result")
                return BatchResult(keywords=keywords, raw=raw, success=True)
            except Exception as exc:  # pytrends can raise several exception types
                last_error = str(exc)
                if attempt < max_retries:
                    time.sleep(self.request_delay * attempt)  # back off a bit more each retry
        return BatchResult(keywords=keywords, raw=pd.DataFrame(), success=False, error=last_error)

    @staticmethod
    def _drop_partial_week(raw: pd.DataFrame) -> pd.DataFrame:
        """The most recent week from pytrends is usually incomplete and noisy."""
        if "isPartial" not in raw.columns:
            return raw
        complete = raw[raw["isPartial"] == False].copy()  # noqa: E712
        return complete.drop(columns=["isPartial"])

    def _rescale_batch(
        self, raw: pd.DataFrame, reference_anchor_mean: float | None
    ) -> tuple[pd.DataFrame, float]:
        """
        Rescale this batch's candidate columns so they're comparable to the
        reference batch. Returns (rescaled_df, this_batch's_anchor_mean) —
        the caller uses the first batch's anchor mean as the reference for
        every later batch.
        """
        anchor_mean = raw[self.anchor_keyword].mean()

        if reference_anchor_mean is None:
            # This IS the reference batch — nothing to rescale against yet.
            return raw, anchor_mean

        if anchor_mean == 0:
            # Anchor had zero interest in this batch; ratio is undefined,
            # so leave values as-is rather than dividing by zero.
            return raw, anchor_mean

        ratio = reference_anchor_mean / anchor_mean
        candidate_cols = [c for c in raw.columns if c != self.anchor_keyword]
        rescaled = raw.copy()
        rescaled[candidate_cols] = (raw[candidate_cols] * ratio).clip(0, 100)
        return rescaled, anchor_mean

    def fetch_all(self, keywords: list[str]) -> tuple[pd.DataFrame, str]:
        """
        Fetch weekly interest for every keyword. Returns (long_df, source_status)
        where long_df has columns: google_keyword, date, google_interest, and
        source_status is 'live', 'live_partial' (some batches failed), or 'failed'.
        """
        batches = self._make_batches(keywords)
        reference_anchor_mean: float | None = None
        all_rows = []
        any_failure = False

        for i, batch_keywords in enumerate(batches):
            result = self._fetch_with_retries(batch_keywords)
            if not result.success:
                any_failure = True
                continue

            clean = self._drop_partial_week(result.raw)
            rescaled, batch_anchor_mean = self._rescale_batch(clean, reference_anchor_mean)
            if reference_anchor_mean is None:
                reference_anchor_mean = batch_anchor_mean

            candidate_cols = [c for c in rescaled.columns if c != self.anchor_keyword]
            long = rescaled[candidate_cols].reset_index().melt(
                id_vars=rescaled.index.name or "date",
                var_name="google_keyword",
                value_name="google_interest",
            )
            long = long.rename(columns={rescaled.index.name or "date": "date"})
            all_rows.append(long)

            if i < len(batches) - 1:
                time.sleep(self.request_delay)

        if not all_rows:
            return pd.DataFrame(columns=["google_keyword", "date", "google_interest"]), "failed"

        combined = pd.concat(all_rows, ignore_index=True)
        status = "live_partial" if any_failure else "live"
        return combined, status

    def run_and_cache(self, keywords: list[str]) -> tuple[pd.DataFrame, str]:
        """
        Fetch live data; on total failure, fall back to the most recent
        cached snapshot instead of returning nothing.
        """
        df, status = self.fetch_all(keywords)

        if status != "failed":
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            snapshot_path = self.cache_dir / f"google_trends_{timestamp}.csv"
            df.to_csv(snapshot_path, index=False)
            latest_path = self.cache_dir / "google_trends_latest.csv"
            df.to_csv(latest_path, index=False)
            return df, status

        cached = self._load_latest_cache()
        if cached is not None:
            return cached, "cached"

        return df, "failed"

    def _load_latest_cache(self) -> pd.DataFrame | None:
        latest_path = self.cache_dir / "google_trends_latest.csv"
        if not latest_path.exists():
            return None
        return pd.read_csv(latest_path)
