"""Download a Kaggle dataset into data/raw/ if missing."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

DEFAULT_DATASET_ID = "frtgnn/dunnhumby-the-complete-journey"


def _load_token_payload(token_value: str) -> dict[str, object] | None:
    token_path = Path(token_value)
    try:
        if token_path.exists():
            return json.loads(token_path.read_text())
        return json.loads(token_value)
    except json.JSONDecodeError:
        return None


def _apply_kaggle_token_env() -> None:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return

    token_value = os.environ.get("KAGGLE_TOKEN_JSON")
    if not token_value:
        return

    payload = _load_token_payload(token_value)
    if not payload:
        raise ValueError(
            "KAGGLE_TOKEN_JSON must be a path to kaggle.json or a JSON payload."
        )

    username = payload.get("username")
    key = payload.get("key")
    if not username or not key:
        raise ValueError("Kaggle token payload must contain 'username' and 'key'.")

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key


def _raw_dir_has_files(raw_dir: Path) -> bool:
    return raw_dir.exists() and any(raw_dir.iterdir())


def _copy_dataset_to_raw(source_dir: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        destination = raw_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def download_dataset(raw_dir: Path, dataset_id: str) -> Path:
    """Download dataset via kagglehub and sync to raw_dir.

    Args:
        raw_dir: Target raw data directory.
        dataset_id: Kaggle dataset identifier (owner/dataset).
    """
    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required. Install with `pip install kagglehub`."
        ) from exc

    _apply_kaggle_token_env()
    dataset_path = Path(kagglehub.dataset_download(dataset_id))
    if not dataset_path.exists():
        raise RuntimeError("kagglehub did not return a valid dataset path.")
    _copy_dataset_to_raw(dataset_path, raw_dir)
    return raw_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a Kaggle dataset to data/raw/ if missing."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw data directory for dataset files.",
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=(
            "Kaggle dataset identifier (owner/dataset). Defaults to "
            "frtgnn/dunnhumby-the-complete-journey."
        ),
    )
    args = parser.parse_args()

    if _raw_dir_has_files(args.raw_dir):
        print(f"Raw data already present at {args.raw_dir}. Skipping download.")
        return 0

    download_dataset(args.raw_dir, dataset_id=args.dataset_id)
    print(f"Downloaded dataset to {args.raw_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

