"""
Live runner for the Reddit connector — run this on your own machine
with a .env file containing REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET,
REDDIT_USER_AGENT (see .env.example).

Usage:
    python scripts/run_reddit_live.py
"""

import praw

from src.config import load_reddit_credentials, load_settings
from src.ingestion.reddit import RedditConnector


def main():
    settings = load_settings("config/settings.yaml")
    creds = load_reddit_credentials(".env")

    reddit_client = praw.Reddit(
        client_id=creds.client_id,
        client_secret=creds.client_secret,
        user_agent=creds.user_agent,
    )
    reddit_client.read_only = True  # we never need to post/vote/comment

    connector = RedditConnector(settings, client=reddit_client)
    df, status = connector.run_and_cache()

    print(f"Source status: {status}")
    print(f"Rows fetched: {len(df)}")
    print(df.head(10))


if __name__ == "__main__":
    main()
