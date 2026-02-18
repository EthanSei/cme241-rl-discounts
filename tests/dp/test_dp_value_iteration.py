"""DP value iteration tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from discount_engine.core.params import CategoryParams, load_mdp_params
from discount_engine.core.types import DiscreteState
from discount_engine.dp.value_iteration import (
    ValueIterationConfig,
    bellman_backup,
    solve_value_iteration,
)


def _fixture_params():
    fixture_path = Path("tests/fixtures/dp_params_small.yaml")
    return load_mdp_params(fixture_path)


def test_value_iteration_converges_and_is_deterministic() -> None:
    params = _fixture_params()
    config = ValueIterationConfig(gamma=params.gamma, epsilon=1e-8, max_iters=500)

    result1 = solve_value_iteration(params=params, config=config)
    result2 = solve_value_iteration(params=params, config=config)

    assert result1.converged is True
    assert result1.iterations <= config.max_iters
    assert result1.policy == result2.policy
    assert result1.max_delta_history[-1] < config.epsilon
    assert result1.final_bellman_residual <= 1e-6

    terminal = DiscreteState(
        churn_bucket=-1,
        memory_buckets=(0,) * len(params.categories),
        recency_buckets=(0,) * len(params.categories),
    )
    assert abs(result1.values[terminal]) <= 1e-12


def test_n5_feasibility_smoke_for_single_bellman_pass() -> None:
    base = _fixture_params()
    categories = tuple(
        CategoryParams(name=f"C{i+1}", price=1.5 + 0.1 * i, beta_0=0.1 - 0.02 * i)
        for i in range(5)
    )
    params = replace(base, categories=categories)

    state = DiscreteState(
        churn_bucket=0,
        memory_buckets=(1, 1, 1, 1, 1),
        recency_buckets=(0, 1, 0, 1, 0),
    )
    best_value, best_action, q_values = bellman_backup(
        state=state,
        values={},
        params=params,
        gamma=params.gamma,
    )

    assert len(q_values) == 6
    assert 0 <= best_action <= 5
    assert best_value == max(q_values.values())


def test_value_iteration_respects_data_driven_three_churn_buckets() -> None:
    base = _fixture_params()
    params = replace(
        base,
        metadata={
            **base.metadata,
            "churn_bucketing": {
                "grid": [0.10, 0.45, 0.85],
                "labels": [
                    "Engaged (low churn risk)",
                    "At-Risk (medium churn risk)",
                    "Lapsing (high churn risk)",
                ],
            },
        },
    )
    result = solve_value_iteration(
        params=params,
        config=ValueIterationConfig(gamma=params.gamma, epsilon=1e-8, max_iters=500),
    )

    live_states = [state for state in result.policy if state.churn_bucket != -1]
    assert live_states
    assert set(state.churn_bucket for state in live_states) == {0, 1, 2}

    # n_states = churn(3) * memory(3^N) * recency(2^N) + terminal
    expected_states = 3 * (3 ** len(params.categories)) * (2 ** len(params.categories)) + 1
    assert len(result.values) == expected_states
