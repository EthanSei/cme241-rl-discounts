"""DP transition probability-mass tests for Version C."""

from __future__ import annotations

import numpy as np

from discount_engine.core.params import CategoryParams, MDPParams
from discount_engine.core.types import DiscreteState
from discount_engine.dp.discretization import enumerate_all_states, resolve_churn_grid
from discount_engine.dp.transitions import (
    _compute_buy_probs_and_prices,
    enumerate_transition_distribution,
)


def _small_params() -> MDPParams:
    return MDPParams(
        delta=0.30,
        gamma=0.99,
        alpha=0.90,
        beta_p=1.2,
        beta_l=0.01,
        beta_m=0.4,
        eta=0.01,
        c0=0.25,
        categories=(
            CategoryParams(name="A", price=1.5, beta_0=0.2),
            CategoryParams(name="B", price=2.0, beta_0=-0.1),
        ),
    )


def test_transition_distribution_is_normalized_and_non_negative() -> None:
    params = _small_params()
    state = DiscreteState(churn_bucket=0, memory_buckets=(1, 1), recency_buckets=(0, 1))
    action = 1

    outcomes = enumerate_transition_distribution(
        state=state,
        action=action,
        params=params,
    )

    total_prob = sum(outcome.probability for outcome in outcomes)
    assert abs(total_prob - 1.0) <= 1e-9
    assert all(outcome.probability >= 0.0 for outcome in outcomes)


def test_interpolated_bucketization_fans_out_transition_branches() -> None:
    params = MDPParams(
        delta=0.30,
        gamma=0.99,
        alpha=0.90,
        beta_p=1.1,
        beta_l=0.02,
        beta_m=0.3,
        eta=0.04,
        c0=0.20,
        categories=(
            CategoryParams(name="A", price=1.5, beta_0=0.2),
            CategoryParams(name="B", price=2.0, beta_0=-0.1),
            CategoryParams(name="C", price=2.5, beta_0=0.05),
        ),
    )
    state = DiscreteState(churn_bucket=0, memory_buckets=(0, 0, 0), recency_buckets=(0, 0, 0))
    action = 2
    outcomes = enumerate_transition_distribution(
        state=state,
        action=action,
        params=params,
    )

    subset_bound = (1 << len(params.categories)) + 1
    interpolated_bound = len(enumerate_all_states(n_categories=len(params.categories))) + 1
    assert len(outcomes) > subset_bound
    assert len(outcomes) <= interpolated_bound


def test_binary_delta_indicator_mode_is_price_invariant_for_purchase_uplift() -> None:
    params = MDPParams(
        delta=0.30,
        gamma=0.99,
        alpha=0.90,
        beta_p=1.2,
        beta_l=0.01,
        beta_m=0.4,
        eta=0.01,
        c0=0.25,
        categories=(
            CategoryParams(name="Cheap", price=1.0, beta_0=0.2),
            CategoryParams(name="Expensive", price=3.0, beta_0=0.2),
        ),
        metadata={"deal_signal_mode": "binary_delta_indicator"},
    )
    state = DiscreteState(churn_bucket=0, memory_buckets=(0, 0), recency_buckets=(0, 0))
    churn_grid = resolve_churn_grid(params)

    buy_probs_action1, _ = _compute_buy_probs_and_prices(
        state=state,
        action=1,
        params=params,
        churn_grid=churn_grid,
    )
    buy_probs_action2, _ = _compute_buy_probs_and_prices(
        state=state,
        action=2,
        params=params,
        churn_grid=churn_grid,
    )

    assert np.isclose(buy_probs_action1[0], buy_probs_action2[1], atol=1e-12)


def test_price_delta_dollars_mode_is_price_sensitive_for_purchase_uplift() -> None:
    params = MDPParams(
        delta=0.30,
        gamma=0.99,
        alpha=0.90,
        beta_p=1.2,
        beta_l=0.01,
        beta_m=0.4,
        eta=0.01,
        c0=0.25,
        categories=(
            CategoryParams(name="Cheap", price=1.0, beta_0=0.2),
            CategoryParams(name="Expensive", price=3.0, beta_0=0.2),
        ),
        metadata={"deal_signal_mode": "price_delta_dollars"},
    )
    state = DiscreteState(churn_bucket=0, memory_buckets=(0, 0), recency_buckets=(0, 0))
    churn_grid = resolve_churn_grid(params)

    buy_probs_action1, _ = _compute_buy_probs_and_prices(
        state=state,
        action=1,
        params=params,
        churn_grid=churn_grid,
    )
    buy_probs_action2, _ = _compute_buy_probs_and_prices(
        state=state,
        action=2,
        params=params,
        churn_grid=churn_grid,
    )

    assert buy_probs_action2[1] > buy_probs_action1[0]
