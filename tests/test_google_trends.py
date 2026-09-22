import pandas as pd
import pytest

from src.ingestion.google_trends import GoogleTrendsConnector

SETTINGS = {
    "google_trends": {
        "region": "IN",
        "window_months": 12,
        "anchor_keyword": "denim jacket",
        "batch_size": 5,
        "request_delay_seconds": 0,  # no sleeping in tests
    }
}


def make_dates(n=3):
    return pd.date_range("2026-08-01", periods=n, freq="W")


class FakeTrendsClient:
    """
    Stands in for pytrends.TrendReq. `responses` maps a frozenset of the
    requested keywords to the DataFrame that should be returned for that
    exact batch, so each test controls exactly what each batch "fetches".
    """

    def __init__(self, responses: dict[frozenset, pd.DataFrame]):
        self.responses = responses
        self._last_kw_list = None
        self.call_count = 0

    def build_payload(self, kw_list, timeframe, geo):
        self._last_kw_list = kw_list

    def interest_over_time(self):
        self.call_count += 1
        key = frozenset(self._last_kw_list)
        if key not in self.responses:
            raise ValueError(f"FakeTrendsClient has no canned response for {key}")
        return self.responses[key]


def test_batches_include_anchor_and_respect_batch_size():
    connector = GoogleTrendsConnector(SETTINGS, client=FakeTrendsClient({}))
    keywords = [f"kw{i}" for i in range(9)]  # 9 candidates, batch_size=4 after anchor slot
    batches = connector._make_batches(keywords)

    assert len(batches) == 3  # ceil(9/4)
    for batch in batches:
        assert connector.anchor_keyword in batch
        assert len(batch) <= 5


def test_rescale_batch_uses_reference_anchor_mean():
    connector = GoogleTrendsConnector(SETTINGS, client=FakeTrendsClient({}))
    dates = make_dates(2)
    raw = pd.DataFrame(
        {"candidate_a": [40, 60], "denim jacket": [20, 20]}, index=dates
    )
    # reference anchor mean is 40 (double this batch's anchor mean of 20)
    rescaled, batch_anchor_mean = connector._rescale_batch(raw, reference_anchor_mean=40)

    assert batch_anchor_mean == 20
    # ratio = 40/20 = 2x, so candidate_a should double, clipped at 100
    assert list(rescaled["candidate_a"]) == [80, 100]


def test_fetch_all_combines_two_batches_with_consistent_scale():
    dates = make_dates(2)
    batch1_raw = pd.DataFrame(
        {"kw1": [50, 60], "kw2": [10, 20], "kw3": [5, 5], "kw4": [1, 1], "denim jacket": [40, 40]},
        index=dates,
    )
    batch1_raw.index.name = "date"
    batch1_raw["isPartial"] = False

    batch2_raw = pd.DataFrame(
        {"kw5": [30, 30], "denim jacket": [20, 20]}, index=dates
    )
    batch2_raw.index.name = "date"
    batch2_raw["isPartial"] = False

    client = FakeTrendsClient(
        {
            frozenset(["kw1", "kw2", "kw3", "kw4", "denim jacket"]): batch1_raw,
            frozenset(["kw5", "denim jacket"]): batch2_raw,
        }
    )
    connector = GoogleTrendsConnector(SETTINGS, client=client)
    keywords = ["kw1", "kw2", "kw3", "kw4", "kw5"]

    combined, status = connector.fetch_all(keywords)

    assert status == "live"
    assert set(combined["google_keyword"]) == {"kw1", "kw2", "kw3", "kw4", "kw5"}
    # batch2's anchor (20) is half of batch1's reference anchor (40),
    # so kw5's values should be rescaled 2x: 30 -> 60
    kw5_values = combined[combined["google_keyword"] == "kw5"]["google_interest"].tolist()
    assert kw5_values == [60, 60]


def test_run_and_cache_falls_back_to_cache_on_total_failure(tmp_path):
    dates = make_dates(2)
    good_raw = pd.DataFrame({"kw1": [10, 10], "denim jacket": [5, 5]}, index=dates)
    good_raw.index.name = "date"
    good_raw["isPartial"] = False

    client = FakeTrendsClient({frozenset(["kw1", "denim jacket"]): good_raw})
    connector = GoogleTrendsConnector(SETTINGS, client=client, cache_dir=tmp_path)

    # First run succeeds and writes the cache.
    df1, status1 = connector.run_and_cache(["kw1"])
    assert status1 == "live"
    assert (tmp_path / "google_trends_latest.csv").exists()

    # Second run: client now has no matching response for any batch -> fails.
    failing_client = FakeTrendsClient({})
    connector2 = GoogleTrendsConnector(SETTINGS, client=failing_client, cache_dir=tmp_path)
    df2, status2 = connector2.run_and_cache(["kw1"])

    assert status2 == "cached"
    assert not df2.empty
