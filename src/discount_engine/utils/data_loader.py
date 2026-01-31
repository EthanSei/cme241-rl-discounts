"""Data loading and ingestion utilities for raw/processed datasets."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def list_raw_tables(raw_dir: Path) -> dict[str, Path]:
    """List raw CSV tables in the raw data directory.

    Args:
        raw_dir: Path to the raw data directory.

    Returns:
        Mapping of table names (file stem) to CSV paths.

    Raises:
        FileNotFoundError: If the directory does not exist or has no CSV files.
    """
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in raw data dir: {raw_dir}")

    return {path.stem: path for path in csv_files}


def ingest_raw_to_processed(
    raw_dir: Path,
    processed_dir: Path,
    file_format: str = "parquet",
) -> dict[str, Path]:
    """Ingest raw CSVs into processed artifacts with basic standardization.

    Args:
        raw_dir: Path to the raw data directory.
        processed_dir: Path to the processed data directory.
        file_format: Output format ("parquet" or "csv").

    Returns:
        Mapping of table names to processed file paths.

    Raises:
        FileNotFoundError: If raw CSV files are missing.
        ValueError: If the file_format is unsupported.
        ImportError: If parquet support is requested but unavailable.
    """
    normalized_format = file_format.lower()
    if normalized_format not in {"parquet", "csv"}:
        raise ValueError("file_format must be 'parquet' or 'csv'")

    tables = list_raw_tables(raw_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}
    for name, path in tables.items():
        df = pd.read_csv(path)
        df = _standardize_columns(df)
        output_path = processed_dir / f"{name}.{normalized_format}"
        if normalized_format == "parquet":
            try:
                df.to_parquet(output_path, index=False)
            except ImportError as exc:
                raise ImportError(
                    "Parquet support is missing. Install 'pyarrow' or use "
                    "file_format='csv'."
                ) from exc
        else:
            df.to_csv(output_path, index=False)
        output_paths[name] = output_path

    return output_paths


def load_processed_tables(
    processed_dir: Path,
    file_format: str = "parquet",
    table_names: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load processed tables into pandas DataFrames.

    Args:
        processed_dir: Path to the processed data directory.
        file_format: File format to load ("parquet" or "csv").
        table_names: Optional set of table names to load.

    Returns:
        Mapping of table names to DataFrames.

    Raises:
        FileNotFoundError: If processed files are missing.
        ValueError: If the file_format is unsupported.
        ImportError: If parquet support is requested but unavailable.
    """
    normalized_format = file_format.lower()
    if normalized_format not in {"parquet", "csv"}:
        raise ValueError("file_format must be 'parquet' or 'csv'")

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")

    file_paths = sorted(processed_dir.glob(f"*.{normalized_format}"))
    if table_names:
        file_paths = [path for path in file_paths if path.stem in table_names]
    if not file_paths:
        raise FileNotFoundError(
            f"No {normalized_format} files found in processed dir: {processed_dir}"
        )

    tables: dict[str, pd.DataFrame] = {}
    for path in file_paths:
        if normalized_format == "parquet":
            try:
                df = pd.read_parquet(path)
            except ImportError as exc:
                raise ImportError(
                    "Parquet support is missing. Install 'pyarrow' or use "
                    "file_format='csv'."
                ) from exc
        else:
            df = pd.read_csv(path)
        tables[path.stem] = df

    return tables


def filter_tables_by_category(
    tables: dict[str, pd.DataFrame],
    category_value: str,
    category_column: str = "commodity_desc",
) -> dict[str, pd.DataFrame]:
    """Filter tables to a single product category.

    Args:
        tables: Mapping of table name to DataFrame. Requires `product` and
            `transaction_data`.
        category_value: Category value to filter on (case-insensitive).
        category_column: Column on the product table containing category labels.

    Returns:
        New mapping with filtered tables for the specified category.

    Raises:
        KeyError: If required tables or columns are missing.
        ValueError: If the category has no matching products.
    """
    if not category_value:
        return tables

    _require_tables(tables, {"product", "transaction_data"})
    products = tables["product"].copy()
    _require_columns(products, {"product_id", category_column}, "product")

    normalized_category = str(category_value).strip().upper()
    product_category = products[category_column].astype(str).str.upper()
    filtered_products = products[product_category == normalized_category].copy()
    if filtered_products.empty:
        raise ValueError(
            f"No products found for {category_column}={category_value!r}."
        )

    product_ids = set(filtered_products["product_id"].unique())
    transactions = tables["transaction_data"].copy()
    _require_columns(transactions, {"product_id"}, "transaction_data")
    filtered_transactions = transactions[transactions["product_id"].isin(product_ids)]

    updated_tables = dict(tables)
    updated_tables["product"] = filtered_products
    updated_tables["transaction_data"] = filtered_transactions.copy()

    if "coupon" in updated_tables:
        coupons = updated_tables["coupon"].copy()
        _require_columns(coupons, {"product_id", "coupon_upc"}, "coupon")
        filtered_coupons = coupons[coupons["product_id"].isin(product_ids)]
        updated_tables["coupon"] = filtered_coupons.copy()

        if "coupon_redempt" in updated_tables:
            redemptions = updated_tables["coupon_redempt"].copy()
            _require_columns(redemptions, {"coupon_upc"}, "coupon_redempt")
            filtered_redemptions = redemptions[
                redemptions["coupon_upc"].isin(filtered_coupons["coupon_upc"])
            ]
            updated_tables["coupon_redempt"] = filtered_redemptions.copy()

    return updated_tables


def _require_tables(tables: dict[str, pd.DataFrame], required: set[str]) -> None:
    missing = required - set(tables)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Missing required tables: {missing_list}")


def _require_columns(
    df: pd.DataFrame,
    required: set[str],
    table_name: str,
) -> None:
    missing = required - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Missing columns in {table_name}: {missing_list}")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase snake case."""
    standardized = [
        column.strip().lower().replace(" ", "_") for column in df.columns
    ]
    if len(set(standardized)) != len(standardized):
        raise ValueError("Standardized column names are not unique.")
    df = df.copy()
    df.columns = standardized
    return df


