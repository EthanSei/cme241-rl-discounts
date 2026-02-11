"""Tests for dataset ingestion and loading utilities."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from discount_engine.utils.io import (
    ingest_raw_to_processed,
    list_raw_tables,
    load_processed_tables,
)


def _write_raw_fixture(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    transactions = pd.DataFrame(
        {
            "Household Key": [1, 1, 2, 2],
            "Day": [1, 2, 1, 2],
            "Product ID": [10, 11, 10, 12],
            "Quantity": [1, 1, 1, 1],
            "Sales Value": [10.0, 9.0, 10.0, 8.0],
            "Retail Disc": [0.0, -1.0, 0.0, -2.0],
            "Coupon Disc": [0.0, 0.0, 0.0, 0.0],
            "Coupon Match Disc": [0.0, 0.0, 0.0, 0.0],
        }
    )
    products = pd.DataFrame(
        {
            "Product ID": [10, 11, 12],
            "Commodity Desc": ["A", "A", "B"],
        }
    )
    transactions.to_csv(raw_dir / "transaction_data.csv", index=False)
    products.to_csv(raw_dir / "product.csv", index=False)


def test_list_raw_tables_finds_csv_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw_fixture(raw_dir)
    table_map = list_raw_tables(raw_dir)
    assert set(table_map) == {"transaction_data", "product"}


def test_list_raw_tables_raises_for_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_raw_tables(tmp_path / "missing")


def test_ingest_and_load_processed_csv(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_raw_fixture(raw_dir)

    output_paths = ingest_raw_to_processed(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        file_format="csv",
    )
    assert set(output_paths) == {"transaction_data", "product"}
    assert output_paths["transaction_data"].exists()

    tables = load_processed_tables(
        processed_dir=processed_dir,
        file_format="csv",
        table_names={"transaction_data", "product"},
    )
    assert "transaction_data" in tables
    assert "product" in tables
    assert "household_key" in tables["transaction_data"].columns
    assert "commodity_desc" in tables["product"].columns


def test_preprocess_script_cli_runs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_raw_fixture(raw_dir)

    project_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")

    cmd = [
        sys.executable,
        str(project_root / "scripts" / "data" / "preprocess_data.py"),
        "--raw-dir",
        str(raw_dir),
        "--processed-dir",
        str(processed_dir),
        "--format",
        "csv",
    ]
    result = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert (processed_dir / "transaction_data.csv").exists()
    assert (processed_dir / "product.csv").exists()
