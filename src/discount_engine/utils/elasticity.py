"""Elasticity and memory decay estimation utilities."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def estimate_elasticity_parameters(
    transactions: pd.DataFrame,
    alpha_grid: Iterable[float] | None = None,
    min_observations: int = 50,
) -> dict[str, float]:
    """Estimate price elasticity (beta) and memory decay (alpha).

    This fits a simple linear demand curve with a discount-memory feature:
        quantity = intercept - beta * price_per_unit + memory_coef * memory

    The memory term is an exponentially decayed average of past discount rates,
    computed per household-product sequence.

    Args:
        transactions: Transaction-level data containing price and discount info.
        alpha_grid: Optional iterable of alpha values to evaluate. Defaults to
            a coarse grid from 0.0 to 0.95.
        min_observations: Minimum number of rows required to fit the model.

    Returns:
        Dictionary containing alpha, beta, intercept, memory_coef, mse, and
        n_samples.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If there is insufficient data to fit the model.
    """
    required = {
        "household_key",
        "day",
        "product_id",
        "quantity",
        "sales_value",
        "retail_disc",
        "coupon_disc",
        "coupon_match_disc",
    }
    missing = required - set(transactions.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Missing columns in transactions: {missing_list}")

    data = transactions.copy()
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce")
    data["sales_value"] = pd.to_numeric(data["sales_value"], errors="coerce")
    data["price_per_unit"] = data["sales_value"] / data["quantity"].replace(0, np.nan)
    data["discount_rate"] = (
        data["retail_disc"] + data["coupon_disc"] + data["coupon_match_disc"]
    ) / data["sales_value"].replace(0, np.nan)

    data = data.dropna(
        subset=["quantity", "sales_value", "price_per_unit", "discount_rate"]
    )
    if len(data) < min_observations:
        raise ValueError("Insufficient data to estimate elasticity parameters.")

    data = data.sort_values(["household_key", "product_id", "day"]).reset_index(drop=True)
    alpha_candidates = list(alpha_grid) if alpha_grid is not None else [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
    ]
    if not alpha_candidates:
        raise ValueError("alpha_grid must contain at least one value.")

    best_alpha = alpha_candidates[0]
    best_mse = float("inf")
    best_params: dict[str, float] = {}

    grouped = data.groupby(["household_key", "product_id"], sort=False)

    for alpha in alpha_candidates:
        memory = np.zeros(len(data), dtype=float)
        for _, group in grouped:
            idx = group.index.to_numpy()
            discount_values = data.loc[idx, "discount_rate"].to_numpy(dtype=float)
            mem = np.zeros(len(discount_values), dtype=float)
            for i in range(1, len(discount_values)):
                mem[i] = alpha * mem[i - 1] + discount_values[i - 1]
            memory[idx] = mem

        design = np.column_stack(
            [
                np.ones(len(data)),
                data["price_per_unit"].to_numpy(dtype=float),
                memory,
            ]
        )
        target = data["quantity"].to_numpy(dtype=float)
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
        predictions = design @ coeffs
        mse = float(np.mean((target - predictions) ** 2))

        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
            best_params = {
                "intercept": float(coeffs[0]),
                "beta": float(-coeffs[1]),
                "memory_coef": float(coeffs[2]),
            }

    return {
        "alpha": best_alpha,
        "beta": best_params["beta"],
        "intercept": best_params["intercept"],
        "memory_coef": best_params["memory_coef"],
        "mse": best_mse,
        "n_samples": float(len(data)),
    }


