"""Preprocess raw dataset into cleaned tables (v2 scaffold)."""

from __future__ import annotations

import argparse
from pathlib import Path

from discount_engine.utils.io import ingest_raw_to_processed


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess raw dataset into data/processed.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    args = parser.parse_args()

    ingest_raw_to_processed(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        file_format=args.format,
    )
    print(f"Preprocessed tables saved to {args.processed_dir} ({args.format}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
