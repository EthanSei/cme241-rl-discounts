"""Sequence and coupon feature utilities for training data."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

DEFAULT_SEQUENCE_ORDER_COLUMNS = ("day", "basket_id", "trans_time", "transaction_id")


def build_training_sequences(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build time-ordered household sequences with coupon flags.

    Args:
        tables: Mapping of table name to DataFrame. Requires
            `transaction_data`, `coupon`, `coupon_redempt`,
            `campaign_table`, and `campaign_desc`.

    Returns:
        Transaction-level DataFrame with sequence index and coupon flags.

    Raises:
        KeyError: If required tables or columns are missing.
    """
    _require_tables(
        tables,
        {
            "transaction_data",
            "coupon",
            "coupon_redempt",
            "campaign_table",
            "campaign_desc",
        },
    )

    transactions = tables["transaction_data"].copy()
    coupons = tables["coupon"].copy()
    redemptions = tables["coupon_redempt"].copy()
    campaigns = tables["campaign_table"].copy()
    campaign_desc = tables["campaign_desc"].copy()

    _require_columns(
        transactions,
        {
            "household_key",
            "day",
            "basket_id",
            "trans_time",
            "product_id",
            "coupon_disc",
            "coupon_match_disc",
        },
        table_name="transaction_data",
    )
    _require_columns(coupons, {"coupon_upc", "product_id", "campaign"}, "coupon")
    _require_columns(
        redemptions,
        {"household_key", "day", "coupon_upc", "campaign"},
        "coupon_redempt",
    )
    _require_columns(
        campaigns, {"household_key", "campaign"}, table_name="campaign_table"
    )
    _require_columns(
        campaign_desc, {"campaign", "start_day", "end_day"}, "campaign_desc"
    )

    transactions = transactions.reset_index().rename(columns={"index": "transaction_id"})

    availability = _build_coupon_availability(
        transactions=transactions,
        coupons=coupons,
        campaigns=campaigns,
        campaign_desc=campaign_desc,
    )
    usage = _build_coupon_usage(
        transactions=transactions,
        coupons=coupons,
        redemptions=redemptions,
    )

    transactions = transactions.merge(availability, on="transaction_id", how="left")
    transactions = transactions.merge(usage, on="transaction_id", how="left")

    transactions["coupon_available"] = transactions["coupon_available"].astype(
        "boolean"
    )
    transactions["coupon_available"] = transactions["coupon_available"].fillna(False)
    transactions["coupon_used"] = transactions["coupon_used"].astype("boolean")
    transactions["coupon_used"] = transactions["coupon_used"].fillna(False)
    transactions["coupon_used"] = transactions["coupon_used"] | (
        (transactions["coupon_disc"] > 0) | (transactions["coupon_match_disc"] > 0)
    )

    transactions = transactions.sort_values(
        ["household_key", *DEFAULT_SEQUENCE_ORDER_COLUMNS]
    )
    transactions["sequence_index"] = (
        transactions.groupby("household_key").cumcount()
    )

    return transactions


def build_and_save_training_sequences(
    tables: dict[str, pd.DataFrame],
    output_path: Path,
) -> Path:
    """Build sequences and persist them to disk.

    Args:
        tables: Mapping of table name to DataFrame.
        output_path: Destination file path (.csv or .parquet).

    Returns:
        Path to the saved file.

    Raises:
        ValueError: If the output file extension is unsupported.
        ImportError: If parquet support is requested but unavailable.
    """
    sequences = build_training_sequences(tables)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        try:
            sequences.to_parquet(output_path, index=False)
        except ImportError as exc:
            raise ImportError(
                "Parquet support is missing. Install 'pyarrow' or use a CSV path."
            ) from exc
    elif suffix == ".csv":
        sequences.to_csv(output_path, index=False)
    else:
        raise ValueError("output_path must end with .csv or .parquet")

    return output_path


def validate_training_sequences(
    sequences: pd.DataFrame,
    order_columns: Iterable[str] | None = None,
) -> None:
    """Validate sequence ordering and continuity for training data.

    Args:
        sequences: DataFrame containing training sequences.
        order_columns: Optional columns to validate ordering before checking
            sequence continuity.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If sequence indices are invalid or non-consecutive.
    """
    _require_columns(sequences, {"household_key", "sequence_index"}, "sequences")
    if sequences["sequence_index"].isna().any():
        raise ValueError("sequence_index contains missing values.")
    if (sequences["sequence_index"] < 0).any():
        raise ValueError("sequence_index contains negative values.")

    sort_columns = ["household_key", "sequence_index"]
    if order_columns:
        order_columns_list = list(order_columns)
        _require_columns(
            sequences, set(order_columns_list), table_name="sequences"
        )
        sort_columns = ["household_key", *order_columns_list]

    sorted_sequences = sequences.sort_values(sort_columns).copy()
    expected_sequence = sorted_sequences.groupby("household_key").cumcount()
    mismatched = sorted_sequences["sequence_index"] != expected_sequence
    if mismatched.any():
        raise ValueError("sequence_index is not consecutive within household_key.")


def _build_coupon_availability(
    transactions: pd.DataFrame,
    coupons: pd.DataFrame,
    campaigns: pd.DataFrame,
    campaign_desc: pd.DataFrame,
) -> pd.DataFrame:
    coupons = coupons[["coupon_upc", "product_id", "campaign"]]
    campaigns = campaigns[["household_key", "campaign"]]
    campaign_desc = campaign_desc[["campaign", "start_day", "end_day"]]

    available = campaigns.merge(coupons, on="campaign", how="inner").merge(
        campaign_desc, on="campaign", how="inner"
    )

    merged = transactions[["transaction_id", "household_key", "product_id", "day"]].merge(
        available, on=["household_key", "product_id"], how="left"
    )
    in_window = merged["day"].between(merged["start_day"], merged["end_day"])
    available_rows = merged[in_window].copy()

    if available_rows.empty:
        return pd.DataFrame(
            columns=["transaction_id", "coupon_available", "coupon_upc", "campaign"]
        )

    first_available = (
        available_rows.sort_values(["transaction_id", "campaign", "coupon_upc"])
        .groupby("transaction_id")
        .first()
        .reset_index()
    )
    first_available["coupon_available"] = True

    return first_available[
        ["transaction_id", "coupon_available", "coupon_upc", "campaign"]
    ]


def _build_coupon_usage(
    transactions: pd.DataFrame,
    coupons: pd.DataFrame,
    redemptions: pd.DataFrame,
) -> pd.DataFrame:
    coupons = coupons[["coupon_upc", "product_id", "campaign"]]
    redemptions = redemptions[["household_key", "day", "coupon_upc", "campaign"]]
    redemptions = redemptions.merge(coupons, on=["coupon_upc", "campaign"], how="left")

    merged = transactions[["transaction_id", "household_key", "day", "product_id"]].merge(
        redemptions, on=["household_key", "day", "product_id"], how="left"
    )
    used = merged[merged["coupon_upc"].notna()].copy()

    if used.empty:
        return pd.DataFrame(columns=["transaction_id", "coupon_used", "coupon_upc_used"])

    first_used = (
        used.sort_values(["transaction_id", "coupon_upc"])
        .groupby("transaction_id")
        .first()
        .reset_index()
    )
    first_used["coupon_used"] = True
    first_used = first_used.rename(columns={"coupon_upc": "coupon_upc_used"})

    return first_used[["transaction_id", "coupon_used", "coupon_upc_used"]]


def _require_tables(tables: dict[str, pd.DataFrame], required: set[str]) -> None:
    missing = required - set(tables)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Missing required tables: {missing_list}")


def _require_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    table_name: str,
) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Missing columns in {table_name}: {missing_list}")

