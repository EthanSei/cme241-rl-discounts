"""Feature engineering utilities for pricing models."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd


FeatureBuilder = Callable[[dict[str, pd.DataFrame]], pd.DataFrame]

_DEFAULT_ID_COLUMNS = ("household_key", "basket_id", "week_no", "product_id")


def build_basic_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build basic transaction-level features for modeling.

    Args:
        tables: Mapping of table name to DataFrame. Requires
            `transaction_data`, `product`, and `hh_demographic`.

    Returns:
        DataFrame with transaction-level features and selected joins.

    Raises:
        KeyError: If required tables or columns are missing.
    """
    _require_tables(tables, {"transaction_data", "product", "hh_demographic"})
    transactions = tables["transaction_data"].copy()
    products = tables["product"].copy()
    households = tables["hh_demographic"].copy()

    _require_columns(
        transactions,
        {
            "household_key",
            "basket_id",
            "week_no",
            "product_id",
            "quantity",
            "sales_value",
            "store_id",
            "retail_disc",
            "coupon_disc",
            "coupon_match_disc",
        },
        table_name="transaction_data",
    )
    _require_columns(
        products,
        {"product_id", "department", "commodity_desc", "brand"},
        table_name="product",
    )
    _require_columns(
        households,
        {"household_key", "age_desc", "income_desc", "homeowner_desc"},
        table_name="hh_demographic",
    )

    transactions = _coerce_numeric(
        transactions,
        ["quantity", "sales_value", "retail_disc", "coupon_disc", "coupon_match_disc"],
    )
    transactions["total_discount"] = (
        transactions["retail_disc"]
        + transactions["coupon_disc"]
        + transactions["coupon_match_disc"]
    )
    transactions["price_per_unit"] = transactions["sales_value"] / transactions[
        "quantity"
    ].replace(0, pd.NA)
    transactions["discount_rate"] = transactions["total_discount"] / transactions[
        "sales_value"
    ].replace(0, pd.NA)
    transactions["promo_flag"] = transactions["total_discount"] > 0

    features = transactions.merge(
        products[["product_id", "department", "commodity_desc", "brand"]],
        on="product_id",
        how="left",
    ).merge(
        households[["household_key", "age_desc", "income_desc", "homeowner_desc"]],
        on="household_key",
        how="left",
    )

    return features


FEATURE_BUILDERS: dict[str, FeatureBuilder] = {"basic": build_basic_features}


def available_feature_sets() -> tuple[str, ...]:
    """Return available named feature sets.

    Returns:
        Tuple of available feature set names.
    """
    return tuple(FEATURE_BUILDERS.keys())


def build_features(
    tables: dict[str, pd.DataFrame],
    feature_sets: Iterable[str] | None = None,
    include_columns: Iterable[str] | None = None,
    exclude_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build features for training with optional selection controls.

    Args:
        tables: Mapping of table name to DataFrame.
        feature_sets: Optional iterable of feature set names to include.
        include_columns: Optional iterable of columns to keep.
        exclude_columns: Optional iterable of columns to drop.

    Returns:
        DataFrame containing the selected features.

    Raises:
        KeyError: If requested feature sets or columns are missing.
        ValueError: If feature set outputs are incompatible.
    """
    if isinstance(feature_sets, str):
        selected_sets = (feature_sets,)
    else:
        selected_sets = tuple(feature_sets) if feature_sets is not None else ("basic",)
    if not selected_sets:
        raise ValueError("feature_sets must include at least one feature set.")

    missing_sets = set(selected_sets) - set(FEATURE_BUILDERS)
    if missing_sets:
        missing_list = ", ".join(sorted(missing_sets))
        raise KeyError(f"Unknown feature sets: {missing_list}")

    frames = [FEATURE_BUILDERS[name](tables) for name in selected_sets]
    features = _merge_feature_frames(frames)
    return _select_feature_columns(
        features,
        include_columns=include_columns,
        exclude_columns=exclude_columns,
        always_keep=_DEFAULT_ID_COLUMNS,
    )


def _merge_feature_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames_list = list(frames)
    if not frames_list:
        raise ValueError("No feature frames provided for merging.")

    merged = frames_list[0].reset_index(drop=True)
    for frame in frames_list[1:]:
        frame = frame.reset_index(drop=True)
        if len(frame) != len(merged):
            raise ValueError("Feature frames must have the same number of rows.")

        overlapping = set(merged.columns) & set(frame.columns)
        if overlapping:
            overlap_list = ", ".join(sorted(overlapping))
            raise ValueError(f"Duplicate feature columns detected: {overlap_list}")

        merged = pd.concat([merged, frame], axis=1)

    return merged


def _select_feature_columns(
    df: pd.DataFrame,
    include_columns: Iterable[str] | None,
    exclude_columns: Iterable[str] | None,
    always_keep: Iterable[str],
) -> pd.DataFrame:
    include_set = set(include_columns) if include_columns else None
    exclude_set = set(exclude_columns) if exclude_columns else set()
    always_keep_set = set(always_keep)

    if include_set:
        missing = include_set - set(df.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise KeyError(f"Missing requested feature columns: {missing_list}")

    selected_columns: list[str] = []
    for column in df.columns:
        keep = column in always_keep_set
        if include_set is not None:
            keep = keep or column in include_set
        else:
            keep = keep or column not in exclude_set
        if (keep and column not in exclude_set) or column in always_keep_set:
            selected_columns.append(column)

    return df[selected_columns]


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


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df

