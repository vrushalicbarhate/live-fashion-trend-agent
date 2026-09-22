import pandas as pd
import pytest

from src.config import ConfigError, load_catalog, load_inventory, load_settings


def test_settings_loads_expected_keys():
    settings = load_settings("config/settings.yaml")
    assert "google_trends" in settings
    assert "trend_score" in settings
    assert settings["trend_score"]["weights"]["google"] == 0.50


def test_catalog_loads_and_has_no_duplicate_ids():
    catalog = load_catalog("config/catalog.csv")
    assert len(catalog) > 0
    assert catalog["item_id"].is_unique


def test_catalog_rejects_duplicate_item_id(tmp_path):
    bad_csv = tmp_path / "bad_catalog.csv"
    bad_csv.write_text(
        "item_id,product_type,material,color,google_keyword,"
        "product_aliases,material_aliases,color_aliases\n"
        "C001,t-shirt,cotton,white,white cotton t-shirt,tee,cotton,white\n"
        "C001,jeans,denim,blue,blue denim jeans,denims,denim,blue\n"
    )
    with pytest.raises(ConfigError, match="duplicate item_id"):
        load_catalog(bad_csv)


def test_catalog_rejects_incomplete_row(tmp_path):
    bad_csv = tmp_path / "bad_catalog.csv"
    bad_csv.write_text(
        "item_id,product_type,material,color,google_keyword,"
        "product_aliases,material_aliases,color_aliases\n"
        "C001,t-shirt,cotton,,white cotton t-shirt,tee,cotton,white\n"
    )
    with pytest.raises(ConfigError, match="incomplete rows"):
        load_catalog(bad_csv)


def test_inventory_matches_catalog_exactly():
    catalog = load_catalog("config/catalog.csv")
    inventory = load_inventory("data/inventory.csv", catalog_df=catalog)
    assert set(inventory["item_id"]) == set(catalog["item_id"])


def test_inventory_rejects_orphaned_item_id(tmp_path):
    catalog = pd.DataFrame(
        {
            "item_id": ["C001"],
            "product_type": ["t-shirt"],
            "material": ["cotton"],
            "color": ["white"],
            "google_keyword": ["white cotton t-shirt"],
            "product_aliases": ["tee"],
            "material_aliases": ["cotton"],
            "color_aliases": ["white"],
        }
    )
    bad_csv = tmp_path / "bad_inventory.csv"
    bad_csv.write_text(
        "item_id,current_stock,units_sold_last_30_days,"
        "production_lead_time_days,last_updated\n"
        "C999,10,5,14,2026-09-15\n"
    )
    with pytest.raises(ConfigError, match="not in catalog"):
        load_inventory(bad_csv, catalog_df=catalog)
