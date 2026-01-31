"""Tests for sequence builder utilities."""

import pandas as pd
import pytest

from pathlib import Path

from discount_engine.utils.sequence_builder import (
    DEFAULT_SEQUENCE_ORDER_COLUMNS,
    build_and_save_training_sequences,
    build_training_sequences,
    validate_training_sequences,
)


def test_build_training_sequences_coupon_flags() -> None:
    tables = {
        "transaction_data": pd.DataFrame(
            {
                "household_key": [1, 1],
                "day": [5, 6],
                "basket_id": [10, 11],
                "trans_time": [100, 200],
                "product_id": [1000, 2000],
                "coupon_disc": [0.0, 0.0],
                "coupon_match_disc": [0.0, 0.0],
            }
        ),
        "coupon": pd.DataFrame(
            {
                "coupon_upc": [555],
                "product_id": [1000],
                "campaign": [1],
            }
        ),
        "coupon_redempt": pd.DataFrame(
            {
                "household_key": [1],
                "day": [5],
                "coupon_upc": [555],
                "campaign": [1],
            }
        ),
        "campaign_table": pd.DataFrame(
            {
                "household_key": [1],
                "campaign": [1],
            }
        ),
        "campaign_desc": pd.DataFrame(
            {
                "campaign": [1],
                "start_day": [4],
                "end_day": [7],
            }
        ),
    }

    sequences = build_training_sequences(tables)

    first = sequences.loc[sequences["product_id"] == 1000].iloc[0]
    second = sequences.loc[sequences["product_id"] == 2000].iloc[0]

    assert bool(first["coupon_available"]) is True
    assert bool(first["coupon_used"]) is True
    assert bool(second["coupon_available"]) is False
    assert bool(second["coupon_used"]) is False
    assert list(sequences["sequence_index"]) == [0, 1]


def test_build_and_save_training_sequences(tmp_path: Path) -> None:
    tables = {
        "transaction_data": pd.DataFrame(
            {
                "household_key": [1],
                "day": [5],
                "basket_id": [10],
                "trans_time": [100],
                "product_id": [1000],
                "coupon_disc": [0.0],
                "coupon_match_disc": [0.0],
            }
        ),
        "coupon": pd.DataFrame(
            {
                "coupon_upc": [555],
                "product_id": [1000],
                "campaign": [1],
            }
        ),
        "coupon_redempt": pd.DataFrame(
            {
                "household_key": [1],
                "day": [5],
                "coupon_upc": [555],
                "campaign": [1],
            }
        ),
        "campaign_table": pd.DataFrame(
            {
                "household_key": [1],
                "campaign": [1],
            }
        ),
        "campaign_desc": pd.DataFrame(
            {
                "campaign": [1],
                "start_day": [4],
                "end_day": [7],
            }
        ),
    }

    output_path = tmp_path / "sequences.csv"
    saved_path = build_and_save_training_sequences(tables, output_path)

    assert saved_path.exists()


def test_validate_training_sequences_ordering() -> None:
    tables = {
        "transaction_data": pd.DataFrame(
            {
                "household_key": [1, 1],
                "day": [5, 6],
                "basket_id": [10, 11],
                "trans_time": [100, 200],
                "product_id": [1000, 2000],
                "coupon_disc": [0.0, 0.0],
                "coupon_match_disc": [0.0, 0.0],
            }
        ),
        "coupon": pd.DataFrame(
            {
                "coupon_upc": [555],
                "product_id": [1000],
                "campaign": [1],
            }
        ),
        "coupon_redempt": pd.DataFrame(
            {
                "household_key": [1],
                "day": [5],
                "coupon_upc": [555],
                "campaign": [1],
            }
        ),
        "campaign_table": pd.DataFrame(
            {
                "household_key": [1],
                "campaign": [1],
            }
        ),
        "campaign_desc": pd.DataFrame(
            {
                "campaign": [1],
                "start_day": [4],
                "end_day": [7],
            }
        ),
    }

    sequences = build_training_sequences(tables)
    validate_training_sequences(sequences, order_columns=DEFAULT_SEQUENCE_ORDER_COLUMNS)

    broken = sequences.copy()
    broken.loc[broken.index[0], "sequence_index"] = 3
    with pytest.raises(ValueError):
        validate_training_sequences(broken, order_columns=DEFAULT_SEQUENCE_ORDER_COLUMNS)

