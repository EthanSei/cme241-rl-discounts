"""Dataset-to-parameter calibration interfaces for v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def calibrate_mdp_params(
    processed_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Calibrate and persist MDP parameters from processed data.

    This is intentionally scaffold-only until the first implementation step.
    """
    raise NotImplementedError(
        "MDP parameter calibration is pending implementation."
    )
