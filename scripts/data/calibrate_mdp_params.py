"""Build calibrated MDP parameter artifact from processed dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from discount_engine.core.calibration import calibrate_mdp_params


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in config file: {config_path}")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate MDP parameters from processed tables."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file (e.g., configs/data/calibration.yaml).",
    )
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--n-categories", type=int, default=None)
    parser.add_argument("--category-column", default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--inactivity-horizon", type=int, default=None)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help="Fraction of latest time periods reserved for alpha validation.",
    )
    args = parser.parse_args()

    config: dict[str, Any] = {}
    if args.config is not None:
        config = _load_yaml_config(args.config)

    processed_dir = Path(
        args.processed_dir or config.get("processed_dir", "data/processed")
    )
    output_path = Path(
        args.output_path or config.get("params_output", "data/processed/mdp_params.yaml")
    )
    n_categories = int(args.n_categories or config.get("n_categories", 3))
    category_column = str(args.category_column or config.get("category_column", "commodity_desc"))
    delta = float(args.delta if args.delta is not None else config.get("delta", 0.30))
    gamma = float(args.gamma if args.gamma is not None else config.get("gamma", 0.99))
    inactivity_horizon = int(
        args.inactivity_horizon
        if args.inactivity_horizon is not None
        else config.get("inactivity_horizon", 28)
    )
    validation_fraction = float(
        args.validation_fraction
        if args.validation_fraction is not None
        else config.get("validation_fraction", 0.20)
    )

    calibrate_mdp_params(
        processed_dir=processed_dir,
        output_path=output_path,
        n_categories=n_categories,
        category_column=category_column,
        delta=delta,
        gamma=gamma,
        inactivity_horizon=inactivity_horizon,
        validation_fraction=validation_fraction,
    )
    print(f"Wrote calibrated parameters to {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
