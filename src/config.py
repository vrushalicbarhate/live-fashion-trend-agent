"""
Configuration and Catalog Module (synopsis §9.1).

Loads settings.yaml, .env credentials, and the two core input CSVs
(fashion_catalog.csv, inventory.csv). Validates schema and rejects
duplicate item IDs or incomplete rows so every downstream module can
trust the data it receives without re-checking it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

# Columns every catalog row must have, and must not be empty.
CATALOG_REQUIRED_COLUMNS = [
    "item_id",
    "product_type",
    "material",
    "color",
    "google_keyword",
    "product_aliases",
    "material_aliases",
    "color_aliases",
]

# Inventory rows: item_id must match a catalog item_id; other fields
# must be present, but sales/stock can legitimately be 0.
INVENTORY_REQUIRED_COLUMNS = [
    "item_id",
    "current_stock",
    "units_sold_last_30_days",
    "production_lead_time_days",
    "last_updated",
]


class ConfigError(ValueError):
    """Raised when settings, catalog, or inventory fail validation."""


@dataclass
class RedditCredentials:
    client_id: str
    client_secret: str
    user_agent: str


def load_settings(settings_path: str | Path = "config/settings.yaml") -> dict:
    """Load settings.yaml into a plain dict."""
    path = Path(settings_path)
    if not path.exists():
        raise ConfigError(f"settings file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    if not settings:
        raise ConfigError(f"settings file is empty: {path}")
    return settings


def load_reddit_credentials(env_path: str | Path = ".env") -> RedditCredentials:
    """
    Load Reddit OAuth credentials from environment variables (.env).
    Kept separate from settings.yaml so secrets never end up in a
    file that gets committed to the portfolio repo.
    """
    load_dotenv(dotenv_path=env_path)
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    missing = [
        name
        for name, value in [
            ("REDDIT_CLIENT_ID", client_id),
            ("REDDIT_CLIENT_SECRET", client_secret),
            ("REDDIT_USER_AGENT", user_agent),
        ]
        if not value
    ]
    if missing:
        raise ConfigError(
            f"missing Reddit credentials in environment: {', '.join(missing)}. "
            f"Copy .env.example to .env and fill in your values."
        )
    return RedditCredentials(client_id, client_secret, user_agent)


def load_catalog(catalog_path: str | Path = "config/catalog.csv") -> pd.DataFrame:
    """
    Load and validate the fashion catalog.

    Raises ConfigError if:
      - required columns are missing
      - any required field is blank
      - item_id values are duplicated
    """
    path = Path(catalog_path)
    if not path.exists():
        raise ConfigError(f"catalog file not found: {path}")

    df = pd.read_csv(path, dtype=str)

    missing_cols = [c for c in CATALOG_REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ConfigError(f"catalog is missing required columns: {missing_cols}")

    df = df[CATALOG_REQUIRED_COLUMNS].copy()

    # Every required field must be non-blank — an incomplete combination
    # can't be scored later, so we reject it now rather than letting it
    # silently produce NaNs downstream.
    blank_mask = df.isna() | (df.astype(str).apply(lambda col: col.str.strip()) == "")
    incomplete_rows = df[blank_mask.any(axis=1)]
    if not incomplete_rows.empty:
        raise ConfigError(
            f"catalog has incomplete rows at item_id(s): "
            f"{incomplete_rows['item_id'].tolist()}"
        )

    duplicates = df["item_id"][df["item_id"].duplicated()].tolist()
    if duplicates:
        raise ConfigError(f"catalog has duplicate item_id values: {duplicates}")

    return df.reset_index(drop=True)


def load_inventory(
    inventory_path: str | Path = "data/inventory.csv",
    catalog_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Load and validate the inventory CSV. If catalog_df is provided,
    also checks that every inventory item_id exists in the catalog
    (and warns — via ConfigError — about catalog items missing inventory).
    """
    path = Path(inventory_path)
    if not path.exists():
        raise ConfigError(f"inventory file not found: {path}")

    df = pd.read_csv(path)

    missing_cols = [c for c in INVENTORY_REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ConfigError(f"inventory is missing required columns: {missing_cols}")

    numeric_cols = ["current_stock", "units_sold_last_30_days", "production_lead_time_days"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ConfigError(f"inventory column '{col}' must be numeric")
        if (df[col] < 0).any():
            raise ConfigError(f"inventory column '{col}' has negative values")

    duplicates = df["item_id"][df["item_id"].duplicated()].tolist()
    if duplicates:
        raise ConfigError(f"inventory has duplicate item_id values: {duplicates}")

    if catalog_df is not None:
        catalog_ids = set(catalog_df["item_id"])
        inventory_ids = set(df["item_id"])
        orphaned = inventory_ids - catalog_ids
        if orphaned:
            raise ConfigError(
                f"inventory references item_id(s) not in catalog: {sorted(orphaned)}"
            )
        uncovered = catalog_ids - inventory_ids
        if uncovered:
            raise ConfigError(
                f"catalog item_id(s) have no inventory row: {sorted(uncovered)}"
            )

    return df.reset_index(drop=True)


if __name__ == "__main__":
    # Quick manual check: run `python src/config.py` from the project root.
    settings = load_settings()
    catalog = load_catalog()
    inventory = load_inventory(catalog_df=catalog)
    print(f"Settings loaded: {len(settings)} top-level keys")
    print(f"Catalog loaded: {len(catalog)} combinations")
    print(f"Inventory loaded: {len(inventory)} rows, all matched to catalog")
