"""Build calibrated MDP parameter artifact from processed dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from discount_engine.core.calibration import calibrate_mdp_params


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate MDP parameters from processed tables."
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/mdp_params.yaml"),
    )
    parser.add_argument("--n-categories", type=int, default=5)
    parser.add_argument("--category-column", default="commodity_desc")
    parser.add_argument("--delta", type=float, default=0.30)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--inactivity-horizon", type=int, default=28)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Fraction of latest time periods reserved for alpha validation.",
    )
    args = parser.parse_args()

    calibrate_mdp_params(
        processed_dir=args.processed_dir,
        output_path=args.output_path,
        n_categories=args.n_categories,
        category_column=args.category_column,
        delta=args.delta,
        gamma=args.gamma,
        inactivity_horizon=args.inactivity_horizon,
        validation_fraction=args.validation_fraction,
    )
    print(f"Wrote calibrated parameters to {args.output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
