"""Tests for elasticity parameter estimation."""

from __future__ import annotations

import pandas as pd

from discount_engine.utils.elasticity import estimate_elasticity_parameters


def test_estimate_elasticity_parameters_basic() -> None:
    data = pd.DataFrame(
        {
            "household_key": [1, 1, 1, 2, 2, 2],
            "day": [1, 2, 3, 1, 2, 3],
            "product_id": [10, 10, 10, 10, 10, 10],
            "quantity": [8.0, 7.0, 6.5, 8.5, 7.5, 6.0],
            "sales_value": [8.0, 7.5, 7.0, 8.5, 7.5, 7.0],
            "retail_disc": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "coupon_disc": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "coupon_match_disc": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    params = estimate_elasticity_parameters(
        data, alpha_grid=[0.0, 0.5], min_observations=5
    )

    assert params["alpha"] in {0.0, 0.5}
    assert params["beta"] >= 0.0
    assert params["n_samples"] == 6.0

