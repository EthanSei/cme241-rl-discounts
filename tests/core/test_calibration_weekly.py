"""Tests for weekly aggregation and category override in calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from discount_engine.core.calibration import (
    _coarsen_to_weekly,
    _detect_time_column,
    calibrate_mdp_params,
)
from discount_engine.utils.io import ingest_raw_to_processed


def _write_raw_fixture(raw_dir: Path, n_days: int = 21) -> None:
    """Create a small raw dataset spanning multiple weeks."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    products = pd.DataFrame(
        {
            "Product ID": [100, 200, 300],
            "Commodity Desc": ["ALPHA", "BETA", "GAMMA"],
        }
    )
    rows: list[dict[str, float | int]] = []
    prices = {100: 10.0, 200: 7.0, 300: 5.0}
    for hh in (1, 2, 3):
        for day in range(1, n_days + 1):
            for pid in (100, 200, 300):
                if (hh + day + pid) % 3 == 0:
                    full_price = prices[pid]
                    disc = -1.0 if day % 5 == 0 else 0.0
                    rows.append(
                        {
                            "Household Key": hh,
                            "Day": day,
                            "Product ID": pid,
                            "Quantity": 1,
                            "Sales Value": full_price + disc,
                            "Retail Disc": disc,
                            "Coupon Disc": 0.0,
                            "Coupon Match Disc": 0.0,
                        }
                    )
    transactions = pd.DataFrame(rows)
    transactions.to_csv(raw_dir / "transaction_data.csv", index=False)
    products.to_csv(raw_dir / "product.csv", index=False)


# -- _coarsen_to_weekly unit tests --


def test_coarsen_produces_week_column_and_drops_day() -> None:
    df = pd.DataFrame(
        {
            "household_key": [1, 1, 1, 2, 2],
            "day": [0, 1, 7, 0, 8],
            "cat": ["A", "A", "A", "A", "A"],
            "sales_value": [10.0, 5.0, 8.0, 3.0, 4.0],
            "quantity": [1, 1, 1, 1, 1],
            "discount_rate": [0.1, 0.2, 0.05, 0.0, 0.3],
            "unit_price": [10.0, 5.0, 8.0, 3.0, 4.0],
        }
    )
    result = _coarsen_to_weekly(df, day_column="day", category_column="cat")

    assert "week" in result.columns
    assert "day" not in result.columns

    # Household 1: days 0,1 -> week 0; day 7 -> week 1
    hh1_w0 = result[(result["household_key"] == 1) & (result["week"] == 0)]
    assert len(hh1_w0) == 1
    assert float(hh1_w0["sales_value"].iloc[0]) == pytest.approx(15.0)
    assert int(hh1_w0["quantity"].iloc[0]) == 2
    assert float(hh1_w0["discount_rate"].iloc[0]) == pytest.approx(0.2)

    hh1_w1 = result[(result["household_key"] == 1) & (result["week"] == 1)]
    assert len(hh1_w1) == 1
    assert float(hh1_w1["sales_value"].iloc[0]) == pytest.approx(8.0)


def test_coarsen_aggregates_discount_as_max() -> None:
    df = pd.DataFrame(
        {
            "household_key": [1, 1, 1],
            "day": [0, 3, 6],
            "cat": ["X", "X", "X"],
            "sales_value": [5.0, 5.0, 5.0],
            "quantity": [1, 1, 1],
            "discount_rate": [0.05, 0.40, 0.10],
            "unit_price": [5.0, 5.0, 5.0],
        }
    )
    result = _coarsen_to_weekly(df, day_column="day", category_column="cat")

    assert len(result) == 1
    assert float(result["discount_rate"].iloc[0]) == pytest.approx(0.40)


# -- _detect_time_column tests --


def test_detect_time_column_prefers_day() -> None:
    df = pd.DataFrame({"day": [1], "week": [0]})
    assert _detect_time_column(df) == "day"


def test_detect_time_column_finds_week_when_day_absent() -> None:
    df = pd.DataFrame({"week": [0], "other": [1]})
    assert _detect_time_column(df) == "week"


# -- selected_categories override --


def test_selected_categories_override(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    output_path = processed_dir / "mdp_params.yaml"
    _write_raw_fixture(raw_dir)
    ingest_raw_to_processed(raw_dir=raw_dir, processed_dir=processed_dir, file_format="csv")

    payload = calibrate_mdp_params(
        processed_dir=processed_dir,
        output_path=output_path,
        n_categories=3,
        selected_categories=["ALPHA", "BETA"],
        alpha_grid=(0.0, 0.5),
    )
    assert len(payload["categories"]) == 2
    cat_names = {c["name"] for c in payload["categories"]}
    assert cat_names == {"ALPHA", "BETA"}


def test_selected_categories_missing_raises(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    output_path = processed_dir / "mdp_params.yaml"
    _write_raw_fixture(raw_dir)
    ingest_raw_to_processed(raw_dir=raw_dir, processed_dir=processed_dir, file_format="csv")

    with pytest.raises(ValueError, match="not found in data"):
        calibrate_mdp_params(
            processed_dir=processed_dir,
            output_path=output_path,
            n_categories=3,
            selected_categories=["NONEXISTENT"],
            alpha_grid=(0.0,),
        )


# -- Weekly end-to-end --


def test_weekly_calibration_produces_valid_params(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    output_path = processed_dir / "mdp_params.yaml"
    _write_raw_fixture(raw_dir, n_days=21)
    ingest_raw_to_processed(raw_dir=raw_dir, processed_dir=processed_dir, file_format="csv")

    payload = calibrate_mdp_params(
        processed_dir=processed_dir,
        output_path=output_path,
        n_categories=2,
        alpha_grid=(0.0, 0.5),
        time_resolution="weekly",
        inactivity_horizon=2,
    )
    assert output_path.exists()
    parsed = yaml.safe_load(output_path.read_text())

    assert parsed["metadata"]["time_resolution"] == "weekly"
    assert parsed["metadata"]["time_column"] == "week"
    assert parsed["beta_p"] > 0.0
    assert parsed["eta"] > 0.0
    assert 0.0 < parsed["c0"] < 1.0


def test_invalid_time_resolution_raises(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    output_path = processed_dir / "mdp_params.yaml"
    _write_raw_fixture(raw_dir)
    ingest_raw_to_processed(raw_dir=raw_dir, processed_dir=processed_dir, file_format="csv")

    with pytest.raises(ValueError, match="time_resolution"):
        calibrate_mdp_params(
            processed_dir=processed_dir,
            output_path=output_path,
            n_categories=2,
            alpha_grid=(0.0,),
            time_resolution="hourly",
        )


def test_beta_m_floor_overrides_fitted_value(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    output_path = processed_dir / "mdp_params.yaml"
    _write_raw_fixture(raw_dir)
    ingest_raw_to_processed(raw_dir=raw_dir, processed_dir=processed_dir, file_format="csv")

    payload = calibrate_mdp_params(
        processed_dir=processed_dir,
        output_path=output_path,
        n_categories=2,
        alpha_grid=(0.0,),
        beta_m_floor=0.15,
    )
    assert payload["beta_m"] >= 0.15
    parsed = yaml.safe_load(output_path.read_text())
    assert parsed["beta_m"] >= 0.15
    assert parsed["metadata"]["beta_m_floor"] == 0.15
    assert parsed["metadata"]["beta_m_floor_active"] is True


def test_beta_m_floor_none_uses_fitted(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    output_path = processed_dir / "mdp_params.yaml"
    _write_raw_fixture(raw_dir)
    ingest_raw_to_processed(raw_dir=raw_dir, processed_dir=processed_dir, file_format="csv")

    payload = calibrate_mdp_params(
        processed_dir=processed_dir,
        output_path=output_path,
        n_categories=2,
        alpha_grid=(0.0,),
        beta_m_floor=None,
    )
    parsed = yaml.safe_load(output_path.read_text())
    assert parsed["metadata"]["beta_m_floor"] is None
    assert parsed["metadata"]["beta_m_floor_active"] is False
