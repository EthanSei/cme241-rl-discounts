"""I/O helpers for dataset ingestion and processed-table loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_raw_tables(raw_dir: Path) -> dict[str, Path]:
    """List raw CSV tables available under ``raw_dir``."""
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
    """Convert raw CSV tables into normalized processed artifacts."""
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
    """Load processed tables into memory."""
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


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase snake-case-like strings."""
    standardized = [column.strip().lower().replace(" ", "_") for column in df.columns]
    if len(set(standardized)) != len(standardized):
        raise ValueError("Standardized column names are not unique.")
    out = df.copy()
    out.columns = standardized
    return out
