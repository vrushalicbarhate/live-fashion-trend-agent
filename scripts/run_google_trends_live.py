"""
Live runner for the Google Trends connector — run this on your own
machine (not in this sandbox, which can't reach trends.google.com).

Usage:
    python scripts/run_google_trends_live.py
"""

from pytrends.request import TrendReq

from src.config import load_catalog, load_settings
from src.ingestion.google_trends import GoogleTrendsConnector


def main():
    settings = load_settings("config/settings.yaml")
    catalog = load_catalog("config/catalog.csv")

    pytrends_client = TrendReq(hl="en-US", tz=330)  # tz=330 = IST offset in minutes

    connector = GoogleTrendsConnector(settings, client=pytrends_client)
    df, status = connector.run_and_cache(catalog["google_keyword"].tolist())

    print(f"Source status: {status}")
    print(f"Rows fetched: {len(df)}")
    print(df.head(10))


if __name__ == "__main__":
    main()
