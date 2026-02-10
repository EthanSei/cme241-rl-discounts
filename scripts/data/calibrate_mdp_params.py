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
    args = parser.parse_args()

    calibrate_mdp_params(
        processed_dir=args.processed_dir,
        output_path=args.output_path,
    )
    print(f"Wrote calibrated parameters to {args.output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
