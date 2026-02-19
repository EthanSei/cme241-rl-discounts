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
    parser.add_argument(
        "--deal-signal-mode",
        default=None,
        help=(
            "Deal-signal contract for calibration and downstream DP. "
            "Supported: positive_centered_anomaly, binary_delta_indicator, "
            "price_delta_dollars."
        ),
    )
    parser.add_argument("--inactivity-horizon", type=int, default=None)
    parser.add_argument(
        "--selected-categories",
        nargs="+",
        default=None,
        help="Override automatic category selection with explicit names.",
    )
    parser.add_argument(
        "--time-resolution",
        default=None,
        choices=["daily", "weekly"],
        help="Time resolution for panel construction (daily or weekly).",
    )
    parser.add_argument(
        "--beta-m-floor",
        type=float,
        default=None,
        help="Impose a minimum beta_m (memory penalty). Set to null/omit to use fitted value.",
    )
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
    deal_signal_mode = str(
        args.deal_signal_mode
        or config.get("deal_signal_mode", "price_delta_dollars")
    )
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
    selected_categories: list[str] | None = (
        args.selected_categories or config.get("selected_categories")
    )
    time_resolution = str(
        args.time_resolution or config.get("time_resolution", "daily")
    )
    beta_m_floor_raw = (
        args.beta_m_floor if args.beta_m_floor is not None
        else config.get("beta_m_floor")
    )
    beta_m_floor: float | None = float(beta_m_floor_raw) if beta_m_floor_raw is not None else None

    calibrate_mdp_params(
        processed_dir=processed_dir,
        output_path=output_path,
        n_categories=n_categories,
        category_column=category_column,
        delta=delta,
        gamma=gamma,
        deal_signal_mode=deal_signal_mode,
        inactivity_horizon=inactivity_horizon,
        validation_fraction=validation_fraction,
        selected_categories=selected_categories,
        time_resolution=time_resolution,
        beta_m_floor=beta_m_floor,
    )
    print(f"Wrote calibrated parameters to {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
