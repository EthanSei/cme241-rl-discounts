"""Dataset-to-parameter calibration interfaces for v2.

This module estimates MDP parameters from processed transaction/product tables.
The calibration flow intentionally keeps the final parameter schema simple, but
adds safeguards against common estimation pitfalls:

1. Avoid target leakage in deal features.
2. Use chronological feature construction for recency and memory.
3. Select memory decay alpha using a time-based validation split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tqdm
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from discount_engine.core.params import CategoryParams, MDPParams, save_mdp_params
from discount_engine.utils.io import load_processed_tables


_TIME_COLUMNS = ("day", "week", "week_no")
_DP_CHURN_CENTER_RANGE = (0.05, 0.50)
_DEAL_SIGNAL_MODE_POSITIVE_CENTERED_ANOMALY = "positive_centered_anomaly"
_DEAL_SIGNAL_MODE_BINARY_DELTA_INDICATOR = "binary_delta_indicator"
_DEAL_SIGNAL_MODE_PRICE_DELTA_DOLLARS = "price_delta_dollars"
_SUPPORTED_DEAL_SIGNAL_MODES = (
    _DEAL_SIGNAL_MODE_POSITIVE_CENTERED_ANOMALY,
    _DEAL_SIGNAL_MODE_BINARY_DELTA_INDICATOR,
    _DEAL_SIGNAL_MODE_PRICE_DELTA_DOLLARS,
)


@dataclass(frozen=True)
class _FittedLogit:
    """Container for one fitted logistic model at a given alpha."""

    alpha: float
    intercepts: np.ndarray
    deal_coef: float
    recency_coef: float
    memory_coef: float
    neg_log_likelihood: float
    train_neg_log_likelihood: float
    val_neg_log_likelihood: float


@dataclass(frozen=True)
class _SequenceCache:
    """Precomputed contiguous sequence metadata for fast memory updates."""

    starts: np.ndarray
    ends: np.ndarray
    times: np.ndarray
    deal_signal: np.ndarray


def calibrate_mdp_params(
    processed_dir: Path,
    output_path: Path,
    n_categories: int = 10,
    category_column: str = "commodity_desc",
    delta: float = 0.30,
    gamma: float = 0.99,
    inactivity_horizon: int = 90,
    deal_signal_mode: str = _DEAL_SIGNAL_MODE_PRICE_DELTA_DOLLARS,
    alpha_grid: tuple[float, ...] | None = None,
    validation_fraction: float = 0.20,
    selected_categories: list[str] | None = None,
    time_resolution: str = "daily",
    beta_m_floor: float | None = None,
) -> dict[str, Any]:
    """Calibrate and persist MDP parameters from processed data.
    """
    if n_categories <= 0:
        raise ValueError("n_categories must be positive.")
    if not (0.0 <= validation_fraction < 1.0):
        raise ValueError("validation_fraction must satisfy 0.0 <= value < 1.0.")
    if time_resolution not in ("daily", "weekly"):
        raise ValueError(
            f"Unsupported time_resolution={time_resolution!r}. "
            "Expected 'daily' or 'weekly'."
        )
    _validate_deal_signal_mode(deal_signal_mode)

    alpha_candidates = alpha_grid or (
        0.00,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
    )

    # 1) Load and clean source tables.
    tables, file_format = _load_processed_tables_with_fallback(processed_dir)
    transactions = _prepare_transactions(tables["transaction_data"])
    products = _prepare_products(
        tables["product"],
        product_ids=set(transactions["product_id"]),
        category_column=category_column,
    )

    merged = transactions.merge(
        products[["product_id", category_column]],
        on="product_id",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No transactions matched product categories.")

    # 2) Build transaction-level features and restrict to tractable top categories.
    merged = _compute_price_and_discount_signals(merged)
    if selected_categories is not None:
        available = set(merged[category_column].unique())
        missing = [c for c in selected_categories if c not in available]
        if missing:
            raise ValueError(
                "Selected categories not found in data: " + ", ".join(missing)
            )
    else:
        selected_categories = _select_top_categories(
            merged=merged,
            category_column=category_column,
            n_categories=n_categories,
        )
    merged = merged[merged[category_column].isin(selected_categories)].copy()
    if merged.empty:
        raise ValueError("No rows remain after category filtering.")

    # 2b) Optionally coarsen to weekly resolution before panel construction.
    if time_resolution == "weekly":
        day_column = _detect_time_column(merged)
        category_prices = _estimate_category_prices(
            merged=merged,
            category_column=category_column,
            selected_categories=selected_categories,
        )
        merged = _coarsen_to_weekly(
            merged, day_column=day_column, category_column=category_column,
        )

    # 3) Build representative category prices and a dense panel, then apply the
    # configured deal-signal contract before fitting.
    if time_resolution != "weekly":
        category_prices = _estimate_category_prices(
            merged=merged,
            category_column=category_column,
            selected_categories=selected_categories,
        )
    time_column = _detect_time_column(merged)
    panel, category_to_idx = _build_purchase_panel(
        merged=merged,
        category_column=category_column,
        selected_categories=selected_categories,
        time_column=time_column,
    )
    panel = _apply_deal_signal_contract(
        panel=panel,
        category_column=category_column,
        selected_categories=selected_categories,
        category_prices=category_prices,
        deal_signal_mode=deal_signal_mode,
        delta=float(delta),
    )
    promotion_deal_signals = _estimate_category_promotion_deal_signals(
        panel=panel,
        category_column=category_column,
        selected_categories=selected_categories,
        category_prices=category_prices,
        deal_signal_mode=deal_signal_mode,
        delta=float(delta),
    )

    fitted = _fit_logistic_grid(
        panel=panel,
        category_to_idx=category_to_idx,
        alpha_candidates=alpha_candidates,
        validation_fraction=validation_fraction,
    )

    # 4) Calibrate churn dynamics.
    c0, eta, churn_stats = _calibrate_churn_dynamics(
        merged=merged,
        time_column=time_column,
        inactivity_horizon=inactivity_horizon,
    )

    categories = tuple(
        CategoryParams(
            name=category,
            price=float(category_prices[category]),
            beta_0=float(fitted.intercepts[category_to_idx[category]]),
            promotion_deal_signal=float(promotion_deal_signals[category]),
        )
        for category in selected_categories
    )

    # 5) Assemble final MDP parameter bundle and persist to YAML.
    params = MDPParams(
        delta=float(delta),
        gamma=float(gamma),
        alpha=float(fitted.alpha),
        beta_p=max(float(fitted.deal_coef), 1e-6),
        beta_l=max(float(fitted.recency_coef), 1e-6),
        beta_m=max(float(fitted.memory_coef), beta_m_floor if beta_m_floor is not None else 1e-6),
        eta=float(eta),
        c0=float(c0),
        categories=categories,
        metadata={
            "file_format": file_format,
            "time_column": time_column,
            "category_column": category_column,
            "n_panel_rows": int(len(panel)),
            "selected_categories": list(selected_categories),
            "inactivity_horizon": int(inactivity_horizon),
            "churn_stats": churn_stats,
            "churn_bucketing": churn_stats.get("discretization", {}),
            "fit_neg_log_likelihood": float(fitted.neg_log_likelihood),
            "fit_train_neg_log_likelihood": float(fitted.train_neg_log_likelihood),
            "fit_val_neg_log_likelihood": float(fitted.val_neg_log_likelihood),
            "alpha_selection_metric": "validation_nll",
            "validation_fraction": float(validation_fraction),
            "discount_signal_mode": "category_time_mean",
            "time_resolution": time_resolution,
            "deal_signal_mode": deal_signal_mode,
            "promotion_deal_signals": {
                category: float(promotion_deal_signals[category])
                for category in selected_categories
            },
            "memory_mode": "gap_aware_ewma",
            "beta_m_floor": float(beta_m_floor) if beta_m_floor is not None else None,
            "beta_m_floor_active": (
                beta_m_floor is not None
                and float(fitted.memory_coef) < beta_m_floor
            ),
        },
    )
    save_mdp_params(params=params, output_path=output_path)
    return params.to_dict()


def _load_processed_tables_with_fallback(
    processed_dir: Path,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Load required processed tables, preferring parquet with CSV fallback."""
    required = {"transaction_data", "product"}
    try:
        return (
            load_processed_tables(
                processed_dir=processed_dir,
                file_format="parquet",
                table_names=required,
            ),
            "parquet",
        )
    except (FileNotFoundError, ImportError):
        return (
            load_processed_tables(
                processed_dir=processed_dir,
                file_format="csv",
                table_names=required,
            ),
            "csv",
        )


def _prepare_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate transaction rows used for calibration."""
    required = {"household_key", "product_id", "quantity", "sales_value"}
    _require_columns(transactions, required, "transaction_data")
    out = transactions.copy()
    # Keep household key as-is and coerce numeric columns used downstream.
    out["household_key"] = out["household_key"]
    out["product_id"] = pd.to_numeric(out["product_id"], errors="coerce")
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["sales_value"] = pd.to_numeric(out["sales_value"], errors="coerce")

    for discount_col in ("retail_disc", "coupon_disc", "coupon_match_disc"):
        if discount_col not in out.columns:
            # Missing discount columns are interpreted as no discount.
            out[discount_col] = 0.0
        out[discount_col] = pd.to_numeric(out[discount_col], errors="coerce").fillna(0.0)

    out = out.dropna(subset=["household_key", "product_id", "quantity", "sales_value"])
    out = out[out["quantity"] > 0].copy()
    if out.empty:
        raise ValueError("No valid transaction rows after cleaning.")
    out["product_id"] = out["product_id"].astype(int)
    return out


def _prepare_products(
    products: pd.DataFrame,
    product_ids: set[int],
    category_column: str,
) -> pd.DataFrame:
    """Normalize product table and keep only products observed in transactions."""
    required = {"product_id", category_column}
    _require_columns(products, required, "product")
    out = products.copy()
    out["product_id"] = pd.to_numeric(out["product_id"], errors="coerce")
    out = out.dropna(subset=["product_id", category_column]).copy()
    out["product_id"] = out["product_id"].astype(int)
    out[category_column] = out[category_column].astype(str)
    out = out[out["product_id"].isin(product_ids)].copy()
    if out.empty:
        raise ValueError("No product rows matched transaction product IDs.")
    return out


def _compute_price_and_discount_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute unit price and bounded discount rate signals from raw sales columns."""
    out = df.copy()
    out["total_discount"] = (
        out["retail_disc"] + out["coupon_disc"] + out["coupon_match_disc"]
    )
    out["unit_price"] = out["sales_value"] / out["quantity"].replace(0, np.nan)
    original_price = out["sales_value"] - out["total_discount"]
    # Retail/coupon discounts are typically negative in this dataset; negate so
    # positive values mean "more discounted".
    out["discount_rate"] = (-out["total_discount"]) / original_price.replace(0, np.nan)
    out["discount_rate"] = (
        out["discount_rate"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 0.95)
    )
    out = out.dropna(subset=["unit_price"]).copy()
    out = out[out["unit_price"] > 0].copy()
    if out.empty:
        raise ValueError("No valid rows for price/discount signals.")
    return out


def _coarsen_to_weekly(
    merged: pd.DataFrame,
    day_column: str,
    category_column: str,
) -> pd.DataFrame:
    """Aggregate daily transactions to weekly resolution.

    Replaces the daily time column with a ``week`` column (``day // 7``).
    Each output row represents one household-category-week triple with
    aggregated sales, quantity, and promotion fields.
    """
    out = merged.copy()
    out["week"] = out[day_column].astype(int) // 7

    agg = out.groupby(
        ["household_key", "week", category_column], as_index=False,
    ).agg(
        sales_value=("sales_value", "sum"),
        quantity=("quantity", "sum"),
        discount_rate=("discount_rate", "max"),
        unit_price=("unit_price", "median"),
    )
    return agg


def _select_top_categories(
    merged: pd.DataFrame,
    category_column: str,
    n_categories: int,
) -> list[str]:
    """Pick the highest-volume categories to keep the state space tractable."""
    counts = (
        merged.groupby(category_column, dropna=False)["household_key"]
        .count()
        .sort_values(ascending=False)
    )
    selected = [str(category) for category in counts.head(n_categories).index.tolist()]
    if not selected:
        raise ValueError("Could not select any categories for calibration.")
    return selected


def _detect_time_column(df: pd.DataFrame) -> str:
    """Return the preferred supported time column present in the dataframe."""
    for column in _TIME_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(f"No supported time column found. Expected one of {_TIME_COLUMNS}.")


def _build_purchase_panel(
    merged: pd.DataFrame,
    category_column: str,
    selected_categories: list[str],
    time_column: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build a dense household-time-category panel for binary purchase modeling.

    Key design choices:
    - We build a full grid over observed household-time pairs and selected categories.
    - Purchase target (`purchased`) is household-specific.
    - Deal signal is leakage-safe and not equal to "purchase happened":
      1) leave-one-household-out category-time mean discount,
      2) centered against long-run category baseline,
      3) clipped at zero so larger value means "stronger than usual promotion."
    """
    # Build dense candidate states: each observed household-time pair crossed with
    # selected categories.
    base_pairs = merged[["household_key", time_column]].drop_duplicates().copy()
    categories_df = pd.DataFrame({category_column: selected_categories})
    base_pairs["__key"] = 1
    categories_df["__key"] = 1
    grid = base_pairs.merge(categories_df, on="__key", how="inner").drop(columns="__key")

    # Purchase target: household-level binary indicator.
    purchases = (
        merged.groupby(["household_key", time_column, category_column], as_index=False)
        .size()
        .rename(columns={"size": "purchased"})
    )

    # Category-time discount stats (across households) for deal exposure context.
    category_time_stats = (
        merged.groupby([time_column, category_column], as_index=False)
        .agg(
            discount_sum=("discount_rate", "sum"),
            discount_count=("discount_rate", "size"),
        )
    )

    # Household contribution to category-time stats for leave-one-household-out logic.
    household_time_stats = (
        merged.groupby(["household_key", time_column, category_column], as_index=False)
        .agg(
            household_discount_sum=("discount_rate", "sum"),
            household_discount_count=("discount_rate", "size"),
        )
    )

    # Long-run category baseline used to convert level discount into promotion bump.
    category_baseline = (
        merged.groupby(category_column, as_index=False)["discount_rate"]
        .median()
        .rename(columns={"discount_rate": "category_discount_baseline"})
    )

    panel = grid.merge(
        category_time_stats,
        on=[time_column, category_column],
        how="left",
    )
    panel = panel.merge(
        household_time_stats,
        on=["household_key", time_column, category_column],
        how="left",
    )
    panel = panel.merge(
        category_baseline,
        on=[category_column],
        how="left",
    )
    panel = panel.merge(
        purchases,
        on=["household_key", time_column, category_column],
        how="left",
    )

    # Fill missing aggregates for combinations with no observed transaction rows.
    panel["discount_sum"] = panel["discount_sum"].fillna(0.0)
    panel["discount_count"] = panel["discount_count"].fillna(0.0)
    panel["household_discount_sum"] = panel["household_discount_sum"].fillna(0.0)
    panel["household_discount_count"] = panel["household_discount_count"].fillna(0.0)
    panel["category_discount_baseline"] = panel["category_discount_baseline"].fillna(0.0)

    # Leave-one-household-out category-time discount rate:
    # - for non-purchase rows, this is the plain category-time mean,
    # - for purchase rows, this excludes that household's own transaction(s).
    loo_sum = panel["discount_sum"] - panel["household_discount_sum"]
    loo_count = panel["discount_count"] - panel["household_discount_count"]
    panel["discount_rate"] = np.where(
        loo_count > 0,
        loo_sum / loo_count,
        np.where(panel["discount_count"] > 0, panel["discount_sum"] / panel["discount_count"], 0.0),
    )

    # Convert level discount into a positive "promotion anomaly" signal.
    panel["deal_signal"] = (
        panel["discount_rate"] - panel["category_discount_baseline"]
    ).clip(lower=0.0)

    # If the anomaly collapses to all zeros, fall back to the level signal.
    if float(panel["deal_signal"].max()) <= 1e-12:
        panel["deal_signal"] = panel["discount_rate"]

    # Binary target.
    panel["purchased"] = (panel["purchased"].fillna(0) > 0).astype(int)

    # Keep deterministic chronological order; downstream sequence code assumes this.
    panel = panel.sort_values(["household_key", category_column, time_column]).reset_index(
        drop=True
    )
    panel["recency"] = _compute_recency(panel, time_column, category_column)
    panel["cat_idx"] = pd.Categorical(
        panel[category_column], categories=selected_categories, ordered=True
    ).codes
    category_to_idx = {category: idx for idx, category in enumerate(selected_categories)}
    return panel, category_to_idx


def _validate_deal_signal_mode(deal_signal_mode: str) -> None:
    if deal_signal_mode not in _SUPPORTED_DEAL_SIGNAL_MODES:
        raise ValueError(
            "Unsupported deal_signal_mode="
            f"{deal_signal_mode!r}. Expected one of {_SUPPORTED_DEAL_SIGNAL_MODES}."
        )


def _apply_deal_signal_contract(
    *,
    panel: pd.DataFrame,
    category_column: str,
    selected_categories: list[str],
    category_prices: dict[str, float],
    deal_signal_mode: str,
    delta: float,
) -> pd.DataFrame:
    """Transform panel deal signal to match the configured DP demand contract."""
    _validate_deal_signal_mode(deal_signal_mode)
    out = panel.copy()
    raw_deal = out["deal_signal"].to_numpy(dtype=float)

    missing_prices = [category for category in selected_categories if category not in category_prices]
    if missing_prices:
        raise ValueError(
            "Missing category price(s) for deal-signal contract: "
            + ", ".join(missing_prices)
        )

    if deal_signal_mode == _DEAL_SIGNAL_MODE_POSITIVE_CENTERED_ANOMALY:
        transformed = raw_deal
    elif deal_signal_mode == _DEAL_SIGNAL_MODE_BINARY_DELTA_INDICATOR:
        transformed = np.where(raw_deal > 1e-12, float(delta), 0.0)
    else:
        # Convert rate anomaly to dollar magnitude for each category.
        price_series = out[category_column].map(category_prices)
        if price_series.isna().any():
            missing = sorted(
                str(value)
                for value in out.loc[price_series.isna(), category_column].dropna().unique()
            )
            raise ValueError(
                "Found categories without mapped prices for dollar deal contract: "
                + ", ".join(missing)
            )
        transformed = raw_deal * price_series.to_numpy(dtype=float)

    out["deal_signal"] = np.clip(
        np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        np.inf,
    )
    return out


def _estimate_category_promotion_deal_signals(
    *,
    panel: pd.DataFrame,
    category_column: str,
    selected_categories: list[str],
    category_prices: dict[str, float],
    deal_signal_mode: str,
    delta: float,
) -> dict[str, float]:
    """Estimate one representative promoted deal signal per category."""
    _validate_deal_signal_mode(deal_signal_mode)
    if deal_signal_mode == _DEAL_SIGNAL_MODE_BINARY_DELTA_INDICATOR:
        return {category: float(delta) for category in selected_categories}
    if deal_signal_mode == _DEAL_SIGNAL_MODE_PRICE_DELTA_DOLLARS:
        return {
            category: float(category_prices[category] * delta)
            for category in selected_categories
        }

    signals: dict[str, float] = {}
    for category in selected_categories:
        values = panel.loc[panel[category_column] == category, "deal_signal"].to_numpy(
            dtype=float
        )
        positives = values[values > 1e-12]
        if positives.size == 0:
            signals[category] = 0.0
            continue
        representative = float(np.quantile(positives, 0.90))
        if not np.isfinite(representative):
            representative = 0.0
        signals[category] = max(0.0, representative)
    return signals


def _compute_recency(
    panel: pd.DataFrame,
    time_column: str,
    category_column: str,
) -> np.ndarray:
    """Compute days/weeks since previous purchase within each household-category."""
    recency = np.zeros(len(panel), dtype=float)
    for _, group in panel.groupby(["household_key", category_column], sort=False):
        ordered = group.sort_values(time_column, kind="mergesort")
        idx_list = ordered.index.to_numpy(dtype=int)
        times = pd.to_numeric(ordered[time_column], errors="coerce").to_numpy(dtype=float)
        purchases = ordered["purchased"].to_numpy(dtype=int)
        last_purchase_time: float | None = None
        for local_i, row_idx in enumerate(idx_list):
            current_time = times[local_i]
            if np.isnan(current_time):
                current_time = float(local_i)
            if last_purchase_time is None:
                recency[row_idx] = 0.0
            else:
                recency[row_idx] = max(0.0, float(current_time - last_purchase_time))
            if purchases[local_i] == 1:
                last_purchase_time = float(current_time)
    return recency


def _fit_logistic_grid(
    panel: pd.DataFrame,
    category_to_idx: dict[str, int],
    alpha_candidates: tuple[float, ...],
    validation_fraction: float = 0.20,
) -> _FittedLogit:
    """Fit one logistic model per alpha and select alpha using validation NLL."""
    time_column = _detect_time_column(panel)
    panel_for_memory = panel.sort_values(
        ["household_key", "cat_idx", time_column], kind="mergesort"
    ).reset_index(drop=True)
    sequence_cache = _build_sequence_cache(panel_for_memory, time_column=time_column)

    y = panel_for_memory["purchased"].to_numpy(dtype=float)
    deal = panel_for_memory["deal_signal"].to_numpy(dtype=float)
    recency = panel_for_memory["recency"].to_numpy(dtype=float)
    cat_idx = panel_for_memory["cat_idx"].to_numpy(dtype=int)
    times = pd.to_numeric(panel_for_memory[time_column], errors="coerce").to_numpy(dtype=float)

    # Time-based split ensures alpha is chosen by forward-looking validation,
    # not by in-sample fit.
    train_mask, val_mask, split_cutoff = _build_time_split_masks(
        times=times,
        validation_fraction=validation_fraction,
    )
    # Fall back to full-sample estimation if the split becomes degenerate.
    if (
        not train_mask.any()
        or np.unique(y[train_mask]).size < 2
        or (val_mask.any() and np.unique(y[val_mask]).size < 2)
    ):
        train_mask = np.ones(len(y), dtype=bool)
        val_mask = np.zeros(len(y), dtype=bool)
        split_cutoff = float("nan")

    best_fit: _FittedLogit | None = None
    best_selection_nll: float | None = None
    theta_seed: np.ndarray | None = None

    for alpha in tqdm.tqdm(alpha_candidates, desc="Fitting logistic grid"):
        memory = _compute_memory_feature(
            panel_for_memory,
            alpha=alpha,
            sequence_cache=sequence_cache,
        )
        if len(memory) != len(panel_for_memory):
            raise ValueError("Memory feature length mismatch.")

        # Fit on training period only.
        fit = _fit_logistic_once(
            y=y[train_mask],
            category_idx=cat_idx[train_mask],
            deal=deal[train_mask],
            recency=recency[train_mask],
            memory=memory[train_mask],
            n_categories=len(category_to_idx),
            alpha=alpha,
            initial_theta=theta_seed,
        )

        # Evaluate unregularized likelihood on train/validation windows.
        train_nll = _evaluate_neg_log_likelihood(
            y=y[train_mask],
            category_idx=cat_idx[train_mask],
            deal=deal[train_mask],
            recency=recency[train_mask],
            memory=memory[train_mask],
            intercepts=fit.intercepts,
            deal_coef=fit.deal_coef,
            recency_coef=fit.recency_coef,
            memory_coef=fit.memory_coef,
        )
        if val_mask.any():
            val_nll = _evaluate_neg_log_likelihood(
                y=y[val_mask],
                category_idx=cat_idx[val_mask],
                deal=deal[val_mask],
                recency=recency[val_mask],
                memory=memory[val_mask],
                intercepts=fit.intercepts,
                deal_coef=fit.deal_coef,
                recency_coef=fit.recency_coef,
                memory_coef=fit.memory_coef,
            )
            selection_nll = val_nll
        else:
            val_nll = train_nll
            selection_nll = train_nll

        fit_with_metrics = _FittedLogit(
            alpha=fit.alpha,
            intercepts=fit.intercepts,
            deal_coef=fit.deal_coef,
            recency_coef=fit.recency_coef,
            memory_coef=fit.memory_coef,
            neg_log_likelihood=selection_nll,
            train_neg_log_likelihood=train_nll,
            val_neg_log_likelihood=val_nll,
        )

        # Warm-start next alpha from this optimum.
        theta_seed = np.concatenate(
            [
                fit.intercepts,
                np.array([fit.deal_coef, fit.recency_coef, fit.memory_coef], dtype=float),
            ]
        )

        if best_fit is None or best_selection_nll is None or selection_nll < best_selection_nll:
            best_fit = fit_with_metrics
            best_selection_nll = selection_nll

    if best_fit is None:
        raise RuntimeError("Failed to fit purchase model for any alpha candidate.")

    # If no validation split was available, keep metric semantics consistent.
    if not np.isfinite(split_cutoff):
        best_fit = _FittedLogit(
            alpha=best_fit.alpha,
            intercepts=best_fit.intercepts,
            deal_coef=best_fit.deal_coef,
            recency_coef=best_fit.recency_coef,
            memory_coef=best_fit.memory_coef,
            neg_log_likelihood=best_fit.train_neg_log_likelihood,
            train_neg_log_likelihood=best_fit.train_neg_log_likelihood,
            val_neg_log_likelihood=best_fit.val_neg_log_likelihood,
        )

    return best_fit


def _build_sequence_cache(panel: pd.DataFrame, time_column: str) -> _SequenceCache:
    """Precompute contiguous household-category segments for sequence features."""
    n_rows = len(panel)
    if n_rows == 0:
        return _SequenceCache(
            starts=np.array([], dtype=int),
            ends=np.array([], dtype=int),
            times=np.array([], dtype=float),
            deal_signal=np.array([], dtype=float),
        )

    keys = panel[["household_key", "cat_idx"]].to_numpy()
    is_start = np.ones(n_rows, dtype=bool)
    is_start[1:] = np.any(keys[1:] != keys[:-1], axis=1)
    starts = np.flatnonzero(is_start)
    ends = np.concatenate([starts[1:], np.array([n_rows], dtype=int)])

    times = pd.to_numeric(panel[time_column], errors="coerce").to_numpy(dtype=float)
    deal_signal = panel["deal_signal"].to_numpy(dtype=float)
    return _SequenceCache(starts=starts, ends=ends, times=times, deal_signal=deal_signal)


def _build_time_split_masks(
    times: np.ndarray,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Construct chronological train/validation masks from the time column."""
    n_rows = len(times)
    train_mask = np.ones(n_rows, dtype=bool)
    val_mask = np.zeros(n_rows, dtype=bool)
    if n_rows == 0 or validation_fraction <= 0.0:
        return train_mask, val_mask, float("nan")

    finite_times = times[np.isfinite(times)]
    if len(finite_times) == 0:
        return train_mask, val_mask, float("nan")

    cutoff = float(np.quantile(finite_times, 1.0 - validation_fraction))
    train_mask = ~np.isfinite(times) | (times <= cutoff)
    val_mask = np.isfinite(times) & (times > cutoff)
    return train_mask, val_mask, cutoff


def _evaluate_neg_log_likelihood(
    y: np.ndarray,
    category_idx: np.ndarray,
    deal: np.ndarray,
    recency: np.ndarray,
    memory: np.ndarray,
    intercepts: np.ndarray,
    deal_coef: float,
    recency_coef: float,
    memory_coef: float,
) -> float:
    """Evaluate pure negative log-likelihood (without regularization)."""
    if len(y) == 0:
        return float("nan")
    logits = (
        intercepts[category_idx]
        + deal_coef * deal
        - recency_coef * recency
        - memory_coef * memory
    )
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    eps = 1e-9
    nll = -np.sum(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))
    return float(nll)


def _compute_memory_feature(
    panel: pd.DataFrame,
    alpha: float,
    sequence_cache: _SequenceCache | None = None,
) -> np.ndarray:
    """Compute gap-aware EWMA memory using chronological household-category sequences."""
    time_column = _detect_time_column(panel)
    cache = sequence_cache or _build_sequence_cache(panel, time_column=time_column)

    memory = np.zeros(len(panel), dtype=float)
    for start, end in zip(cache.starts, cache.ends, strict=False):
        if end - start <= 1:
            continue

        prev_memory = 0.0
        prev_time = cache.times[start]
        if not np.isfinite(prev_time):
            prev_time = 0.0

        for idx in range(start + 1, end):
            current_time = cache.times[idx]
            if not np.isfinite(current_time):
                current_time = prev_time + 1.0
            # Gap-aware decay: larger time gaps imply stronger decay.
            delta_t = max(0.0, float(current_time - prev_time))
            prev_memory = (alpha**delta_t) * prev_memory + max(
                0.0, float(cache.deal_signal[idx - 1])
            )
            memory[idx] = prev_memory
            prev_time = current_time

    return memory


def _fit_logistic_once(
    y: np.ndarray,
    category_idx: np.ndarray,
    deal: np.ndarray,
    recency: np.ndarray,
    memory: np.ndarray,
    n_categories: int,
    alpha: float,
    initial_theta: np.ndarray | None = None,
) -> _FittedLogit:
    """Fit one bounded logistic model for a fixed alpha value."""
    n = len(y)
    if not (len(category_idx) == len(deal) == len(recency) == len(memory) == n):
        raise ValueError("Feature length mismatch in logistic fit.")

    def objective(theta: np.ndarray) -> float:
        intercepts = theta[:n_categories]
        deal_coef, recency_coef, memory_coef = theta[n_categories:]
        logits = (
            intercepts[category_idx]
            + deal_coef * deal
            - recency_coef * recency
            - memory_coef * memory
        )
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        eps = 1e-9
        nll = -np.sum(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))
        reg = 1e-4 * float(np.sum(theta * theta))
        return float(nll + reg)

    if initial_theta is not None:
        theta0 = np.asarray(initial_theta, dtype=float).copy()
        if theta0.shape != (n_categories + 3,):
            raise ValueError("initial_theta has incompatible shape for logistic fit.")
    else:
        theta0 = np.zeros(n_categories + 3, dtype=float)
        theta0[n_categories:] = np.array([1.0, 0.01, 0.01], dtype=float)

    # Constrain coefficients to preserve monotonic interpretation in demand:
    # higher deal -> higher purchase; longer recency/memory -> lower purchase.
    bounds = [(None, None)] * n_categories + [(0.0, None), (0.0, None), (0.0, None)]
    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 120},
    )
    if not result.success:
        raise RuntimeError(f"Logistic fit failed for alpha={alpha}: {result.message}")

    theta = result.x
    return _FittedLogit(
        alpha=float(alpha),
        intercepts=theta[:n_categories].copy(),
        deal_coef=float(theta[n_categories]),
        recency_coef=float(theta[n_categories + 1]),
        memory_coef=float(theta[n_categories + 2]),
        neg_log_likelihood=float(result.fun),
        train_neg_log_likelihood=float(result.fun),
        val_neg_log_likelihood=float(result.fun),
    )


def _estimate_category_prices(
    merged: pd.DataFrame,
    category_column: str,
    selected_categories: list[str],
) -> dict[str, float]:
    """Estimate typical non-promo category prices used in reward calculations."""
    prices: dict[str, float] = {}
    for category in selected_categories:
        subset = merged[merged[category_column] == category]
        non_promo = subset[subset["discount_rate"] <= 1e-3]
        source = non_promo if not non_promo.empty else subset
        median_price = float(source["unit_price"].median())
        if not np.isfinite(median_price) or median_price <= 0:
            median_price = 1.0
        prices[category] = median_price
    return prices


def _calibrate_churn_dynamics(
    merged: pd.DataFrame,
    time_column: str,
    inactivity_horizon: int,
) -> tuple[float, float, dict[str, Any]]:
    """Estimate coarse churn dynamics from inactivity run lengths."""
    pairs: list[tuple[int, int]] = []
    churn_events = 0
    total_points = 0

    # Convert sparse activity traces into run-length/no-purchase observations.
    activity = merged[["household_key", time_column]].drop_duplicates()
    for household, sub in activity.groupby("household_key", sort=False):
        times = pd.to_numeric(sub[time_column], errors="coerce").dropna().astype(int)
        if times.empty:
            continue
        min_t = int(times.min())
        max_t = int(times.max())
        active_set = set(times.tolist())
        run = 0
        for t in range(min_t, max_t):
            is_active = t in active_set
            if is_active:
                run = 0
            else:
                run += 1
            next_no_purchase = 0 if (t + 1 in active_set) else 1
            pairs.append((run, next_no_purchase))
            total_points += 1
            if run >= inactivity_horizon:
                churn_events += 1

    if not pairs:
        default_c0 = _DP_CHURN_CENTER_RANGE[0]
        default_eta = 0.05
        discretization = _build_data_driven_churn_discretization(
            runs=np.array([0.0, 1.0, 2.0], dtype=float),
            c0=default_c0,
            eta=default_eta,
            n_buckets=3,
            center_range=_DP_CHURN_CENTER_RANGE,
        )
        return (
            float(discretization["grid"][0]),
            default_eta,
            {
                "churn_rate_at_horizon": 0.0,
                "raw_c0": default_c0,
                "raw_eta": default_eta,
                "discretization": discretization,
            },
        )

    run_df = pd.DataFrame(pairs, columns=["run", "next_no_purchase"])
    run_df["run_bucket"] = run_df["run"].clip(upper=30)
    hazard = (
        run_df.groupby("run_bucket", as_index=False)["next_no_purchase"].mean().sort_values(
            "run_bucket"
        )
    )

    c0 = float(hazard.loc[hazard["run_bucket"] == 0, "next_no_purchase"].mean())
    if not np.isfinite(c0):
        c0 = float(run_df["next_no_purchase"].mean())
    c0 = float(np.clip(c0, 0.01, 0.99))

    if len(hazard) >= 2:
        slope, _ = np.polyfit(
            hazard["run_bucket"].to_numpy(dtype=float),
            hazard["next_no_purchase"].to_numpy(dtype=float),
            deg=1,
        )
        eta = float(np.clip(slope, 0.001, 0.25))
    else:
        eta = 0.05

    churn_rate_at_horizon = churn_events / total_points if total_points > 0 else 0.0
    discretization = _build_data_driven_churn_discretization(
        runs=run_df["run"].to_numpy(dtype=float),
        c0=c0,
        eta=eta,
        n_buckets=3,
        center_range=_DP_CHURN_CENTER_RANGE,
    )
    effective_c0 = float(discretization["grid"][0])
    return effective_c0, eta, {
        "churn_rate_at_horizon": float(churn_rate_at_horizon),
        "raw_c0": float(c0),
        "raw_eta": float(eta),
        "effective_c0": effective_c0,
        "discretization": discretization,
    }


def _build_data_driven_churn_discretization(
    *,
    runs: np.ndarray,
    c0: float,
    eta: float,
    n_buckets: int,
    center_range: tuple[float, float] = _DP_CHURN_CENTER_RANGE,
) -> dict[str, Any]:
    """Build interpretable churn buckets from empirical inactivity runs."""
    if n_buckets <= 0:
        raise ValueError("n_buckets must be positive.")
    if len(center_range) != 2:
        raise ValueError("center_range must be a (low, high) tuple.")
    center_low, center_high = float(center_range[0]), float(center_range[1])
    if not (0.0 <= center_low < center_high <= 1.0):
        raise ValueError("center_range must satisfy 0.0 <= low < high <= 1.0.")

    run_values = np.asarray(runs, dtype=float)
    run_values = run_values[np.isfinite(run_values)]
    if run_values.size == 0:
        run_values = np.array([0.0], dtype=float)
    run_values = np.maximum(run_values, 0.0)

    if n_buckets == 1:
        bucket_arrays = [run_values]
        run_bounds: list[tuple[int, int | None]] = [
            (0, int(np.max(run_values))),
        ]
    else:
        quantile_points = np.linspace(0.0, 1.0, n_buckets + 1)[1:-1]
        cutoffs = [
            int(np.floor(float(np.quantile(run_values, q))))
            for q in quantile_points
        ]
        bucket_ids = np.digitize(run_values, bins=cutoffs, right=True)
        bucket_arrays = [run_values[bucket_ids == idx] for idx in range(n_buckets)]
        if any(arr.size == 0 for arr in bucket_arrays):
            # Fallback for degenerate ties at quantile boundaries.
            bucket_arrays = np.array_split(np.sort(run_values), n_buckets)
            run_bounds = []
            for idx, arr in enumerate(bucket_arrays):
                lower = int(arr.min()) if idx == 0 else int(bucket_arrays[idx - 1].max()) + 1
                upper = int(arr.max()) if idx < n_buckets - 1 else None
                run_bounds.append((lower, upper))
        else:
            run_bounds = []
            for idx in range(n_buckets):
                if idx == 0:
                    lower = 0
                    upper: int | None = cutoffs[0]
                elif idx < n_buckets - 1:
                    lower = cutoffs[idx - 1] + 1
                    upper = cutoffs[idx]
                else:
                    lower = cutoffs[-1] + 1
                    upper = None
                run_bounds.append((lower, upper))

    n_effective = len(bucket_arrays)
    label_pool = _churn_bucket_label_pool(n_effective)

    bucket_rows: list[dict[str, Any]] = []
    centers: list[float] = []
    for idx, arr in enumerate(bucket_arrays):
        run_min, run_max = run_bounds[idx]
        if run_max is None:
            run_max = int(arr.max())
        run_median = float(np.median(arr))
        raw_churn_center = float(np.clip(c0 + eta * run_median, 0.0, 1.0))
        centers.append(raw_churn_center)
        bucket_rows.append(
            {
                "index": idx,
                "label": label_pool[idx],
                "run_min_inclusive": run_min,
                "run_max_inclusive": run_max,
                "run_median": run_median,
                "raw_churn_center": raw_churn_center,
                "share": float(arr.size / run_values.size),
            }
        )

    churn_grid = _damp_centers_to_range(
        centers=centers,
        target_low=center_low,
        target_high=center_high,
    )
    for idx, center in enumerate(churn_grid):
        bucket_rows[idx]["churn_center"] = float(center)

    return {
        "mode": "run_quantile_v2_damped",
        "n_buckets": n_effective,
        "center_range": [center_low, center_high],
        "grid": [float(x) for x in churn_grid],
        "labels": [row["label"] for row in bucket_rows],
        "buckets": bucket_rows,
    }


def _churn_bucket_label_pool(n_buckets: int) -> list[str]:
    if n_buckets == 3:
        return [
            "Engaged (low churn risk)",
            "At-Risk (medium churn risk)",
            "Lapsing (high churn risk)",
        ]
    if n_buckets == 2:
        return [
            "Engaged (lower churn risk)",
            "Lapsing (higher churn risk)",
        ]
    return [f"Churn Segment {idx + 1}" for idx in range(n_buckets)]


def _strictly_increasing_grid(
    values: list[float],
    *,
    lower: float,
    upper: float,
    min_gap: float,
) -> tuple[float, ...]:
    arr = np.clip(np.asarray(values, dtype=float), lower, upper)
    if arr.size == 0:
        return tuple()
    if arr.size == 1:
        return (float(arr[0]),)

    for idx in range(arr.size - 2, -1, -1):
        if arr[idx] >= arr[idx + 1]:
            arr[idx] = arr[idx + 1] - min_gap

    if arr[0] < lower:
        arr = arr + (lower - arr[0])
    if arr[-1] > upper:
        arr = arr - (arr[-1] - upper)

    if np.any(np.diff(arr) <= 0.0):
        arr = np.linspace(
            max(lower, float(np.min(arr))),
            min(upper, float(np.max(arr))),
            arr.size,
        )
    return tuple(float(x) for x in arr)


def _damp_centers_to_range(
    *,
    centers: list[float],
    target_low: float,
    target_high: float,
) -> tuple[float, ...]:
    arr = np.asarray(centers, dtype=float)
    if arr.size == 0:
        return tuple()
    if arr.size == 1:
        midpoint = 0.5 * (target_low + target_high)
        return (float(midpoint),)

    raw_min = float(np.min(arr))
    raw_max = float(np.max(arr))
    if raw_max - raw_min <= 1e-12:
        normalized = np.linspace(0.0, 1.0, arr.size)
    else:
        normalized = (arr - raw_min) / (raw_max - raw_min)

    damped = target_low + normalized * (target_high - target_low)
    return _strictly_increasing_grid(
        damped.tolist(),
        lower=target_low,
        upper=target_high,
        min_gap=1e-4,
    )


def _require_columns(df: pd.DataFrame, required: set[str], table_name: str) -> None:
    """Raise an explicit error when required schema columns are missing."""
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {table_name}: {', '.join(sorted(missing))}")
