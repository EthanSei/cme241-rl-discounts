"""Build processed data, sequences, and feature artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from discount_engine.utils.data_loader import (
    filter_tables_by_category,
    ingest_raw_to_processed,
    load_processed_tables,
)
from discount_engine.utils.elasticity import estimate_elasticity_parameters
from discount_engine.utils.feature_engineering import available_feature_sets, build_features
from discount_engine.utils.sequence_builder import (
    DEFAULT_SEQUENCE_ORDER_COLUMNS,
    build_training_sequences,
    validate_training_sequences,
)


def _save_dataframe(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(output_path, index=False)
        except ImportError as exc:
            raise ImportError(
                "Parquet support is missing. Install 'pyarrow' or use a CSV path."
            ) from exc
    elif suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError("output_path must end with .csv or .parquet")
    return output_path


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _run_steps_with_progress(steps: list[tuple[str, Callable[[], None]]]) -> None:
    with tqdm(
        total=len(steps),
        desc="Preprocess",
        unit="step",
        disable=not sys.stdout.isatty(),
    ) as progress:
        for label, step in steps:
            progress.set_postfix_str(label)
            step()
            progress.update(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build processed data, sequences, and features."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw data directory for dataset files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Processed data directory for output files.",
    )
    parser.add_argument(
        "--format",
        default="parquet",
        help="Processed file format: parquet or csv.",
    )
    parser.add_argument(
        "--sequences-path",
        type=Path,
        default=None,
        help="Output path for sequences (default based on format).",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=None,
        help="Output path for features (default based on format).",
    )
    parser.add_argument(
        "--elasticity-path",
        type=Path,
        default=None,
        help="Output path for elasticity parameters (default: processed dir).",
    )
    parser.add_argument(
        "--category",
        default="SOFT DRINKS",
        help="Product category value to filter on (default: SOFT DRINKS).",
    )
    parser.add_argument(
        "--category-column",
        default="commodity_desc",
        help="Product table column containing category values.",
    )
    parser.add_argument(
        "--feature-sets",
        default="basic",
        help=(
            "Comma-separated feature sets to build. Available: "
            f"{', '.join(available_feature_sets())}."
        ),
    )
    parser.add_argument(
        "--feature-columns",
        default=None,
        help="Comma-separated list of feature columns to include.",
    )
    parser.add_argument(
        "--drop-feature-columns",
        default=None,
        help="Comma-separated list of feature columns to drop.",
    )
    parser.add_argument(
        "--sequence-check",
        choices=["none", "basic"],
        default="basic",
        help="Validation level for sequences (default: basic).",
    )
    args = parser.parse_args()
    file_format = args.format.lower()
    feature_sets = _parse_csv_arg(args.feature_sets)
    include_columns = _parse_csv_arg(args.feature_columns)
    exclude_columns = _parse_csv_arg(args.drop_feature_columns)

    required_tables = {
        "transaction_data",
        "coupon",
        "coupon_redempt",
        "campaign_table",
        "campaign_desc",
        "product",
        "hh_demographic",
    }

    tables: dict[str, pd.DataFrame] = {}
    sequences_path = args.sequences_path or (
        args.processed_dir / f"training_sequences.{file_format}"
    )
    features_path = args.features_path or (
        args.processed_dir / f"training_features.{file_format}"
    )
    sequences: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    elasticity_params: dict[str, float] | None = None
    elasticity_path = args.elasticity_path or (
        args.processed_dir / "elasticity_params.json"
    )

    def _ingest() -> None:
        nonlocal file_format
        try:
            ingest_raw_to_processed(
                raw_dir=args.raw_dir,
                processed_dir=args.processed_dir,
                file_format=file_format,
            )
        except ImportError:
            if file_format != "parquet":
                raise
            file_format = "csv"
            print("Parquet support missing; falling back to CSV for processed data.")
            ingest_raw_to_processed(
                raw_dir=args.raw_dir,
                processed_dir=args.processed_dir,
                file_format=file_format,
            )

    def _load_tables() -> None:
        nonlocal tables
        tables = load_processed_tables(
            processed_dir=args.processed_dir,
            file_format=file_format,
            table_names=required_tables,
        )

    def _filter_tables() -> None:
        nonlocal tables
        tables = filter_tables_by_category(
            tables,
            category_value=args.category,
            category_column=args.category_column,
        )

    def _estimate_elasticity() -> None:
        nonlocal elasticity_params
        elasticity_params = estimate_elasticity_parameters(tables["transaction_data"])
        elasticity_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": elasticity_params["alpha"],
            "beta": elasticity_params["beta"],
            "intercept": elasticity_params["intercept"],
            "memory_coef": elasticity_params["memory_coef"],
            "mse": elasticity_params["mse"],
            "n_samples": elasticity_params["n_samples"],
            "category": args.category,
            "category_column": args.category_column,
        }
        elasticity_path.write_text(json.dumps(payload, indent=2))

    def _build_sequences() -> None:
        nonlocal sequences
        sequences = build_training_sequences(tables)
        if args.sequence_check == "basic":
            validate_training_sequences(
                sequences, order_columns=DEFAULT_SEQUENCE_ORDER_COLUMNS
            )

    def _save_sequences() -> None:
        if sequences is None:
            raise ValueError("Sequences are not available to save.")
        _save_dataframe(sequences, sequences_path)

    def _build_features() -> None:
        nonlocal features
        features = build_features(
            tables,
            feature_sets=feature_sets,
            include_columns=include_columns,
            exclude_columns=exclude_columns,
        )

    def _save_features() -> None:
        if features is None:
            raise ValueError("Features are not available to save.")
        _save_dataframe(features, features_path)

    _run_steps_with_progress(
        [
            ("ingest raw", _ingest),
            ("load tables", _load_tables),
            ("filter category", _filter_tables),
            ("estimate elasticity", _estimate_elasticity),
            ("build sequences", _build_sequences),
            ("save sequences", _save_sequences),
            ("build features", _build_features),
            ("save features", _save_features),
        ]
    )

    print(f"Processed data saved to {args.processed_dir} ({file_format}).")
    print(f"Saved sequences to {sequences_path}.")
    print(f"Saved features to {features_path}.")
    print(f"Saved elasticity params to {elasticity_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

