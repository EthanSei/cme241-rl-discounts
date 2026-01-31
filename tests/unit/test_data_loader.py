"""Tests for data loading utilities."""

from pathlib import Path

import pandas as pd
import pytest

from discount_engine.utils.data_loader import (
    filter_tables_by_category,
    ingest_raw_to_processed,
    list_raw_tables,
    load_processed_tables,
)


def _write_csv(path: Path, data: dict[str, list[object]]) -> None:
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def test_list_raw_tables_missing_dir(tmp_path: Path) -> None:
    missing_dir = tmp_path / "raw"
    with pytest.raises(FileNotFoundError):
        list_raw_tables(missing_dir)


def test_ingest_and_load_csv_round_trip(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()

    _write_csv(raw_dir / "transactions.csv", {"Household ID": [1], "Sales": [9.99]})
    _write_csv(raw_dir / "products.csv", {"Product ID": [10], "Category": ["A"]})

    output_paths = ingest_raw_to_processed(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        file_format="csv",
    )

    assert output_paths["transactions"].exists()
    assert output_paths["products"].exists()

    tables = load_processed_tables(processed_dir=processed_dir, file_format="csv")

    assert set(tables.keys()) == {"transactions", "products"}
    assert "household_id" in tables["transactions"].columns
    assert "product_id" in tables["products"].columns


def test_filter_tables_by_category() -> None:
    products = pd.DataFrame(
        {
            "product_id": [1, 2, 3],
            "commodity_desc": ["SOFT DRINKS", "SOFT DRINKS", "CANDY"],
        }
    )
    transactions = pd.DataFrame(
        {"product_id": [1, 2, 3, 3], "sales_value": [1.0, 2.0, 3.0, 4.0]}
    )
    coupons = pd.DataFrame({"coupon_upc": [10, 20], "product_id": [1, 3]})
    redemptions = pd.DataFrame({"coupon_upc": [10, 20]})
    tables = {
        "product": products,
        "transaction_data": transactions,
        "coupon": coupons,
        "coupon_redempt": redemptions,
    }

    filtered = filter_tables_by_category(tables, category_value="soft drinks")

    assert set(filtered["product"]["product_id"]) == {1, 2}
    assert set(filtered["transaction_data"]["product_id"]) == {1, 2}
    assert set(filtered["coupon"]["product_id"]) == {1}
    assert set(filtered["coupon_redempt"]["coupon_upc"]) == {10}

