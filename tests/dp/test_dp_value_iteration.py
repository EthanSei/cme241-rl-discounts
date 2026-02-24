"""DP value iteration tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from discount_engine.core.params import CategoryParams, load_mdp_params
from discount_engine.core.types import DiscreteState
from discount_engine.dp.value_iteration import (
    ValueIterationConfig,
    bellman_backup,
    evaluate_policy,
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


# --- evaluate_policy tests ---


def test_evaluate_policy_matches_optimal_values() -> None:
    """V^π with the optimal policy should equal V* from value iteration."""
    params = _fixture_params()
    config = ValueIterationConfig(gamma=params.gamma, epsilon=1e-10, max_iters=2000)
    result = solve_value_iteration(params=params, config=config)

    v_pi = evaluate_policy(policy=result.policy, params=params, config=config)

    for state, v_star in result.values.items():
        assert abs(v_pi[state] - v_star) < 1e-6, (
            f"State {state}: V^π={v_pi[state]:.8f} != V*={v_star:.8f}"
        )


def test_evaluate_policy_never_promote_leq_optimal() -> None:
    """Never-promote policy value should be <= optimal for all live states."""
    params = _fixture_params()
    config = ValueIterationConfig(gamma=params.gamma, epsilon=1e-10, max_iters=2000)
    result = solve_value_iteration(params=params, config=config)

    never_promote = {s: 0 for s in result.policy}
    v_never = evaluate_policy(policy=never_promote, params=params, config=config)

    for state in result.values:
        if state.churn_bucket != -1:
            assert v_never[state] <= result.values[state] + 1e-8, (
                f"State {state}: V^never={v_never[state]:.6f} > V*={result.values[state]:.6f}"
            )


def test_evaluate_policy_terminal_is_zero() -> None:
    """Terminal state should have value 0 under any policy."""
    params = _fixture_params()
    config = ValueIterationConfig(gamma=params.gamma, epsilon=1e-10, max_iters=2000)
    result = solve_value_iteration(params=params, config=config)

    v_pi = evaluate_policy(policy=result.policy, params=params, config=config)

    terminal = DiscreteState(
        churn_bucket=-1,
        memory_buckets=(0,) * len(params.categories),
        recency_buckets=(0,) * len(params.categories),
    )
    assert abs(v_pi[terminal]) < 1e-12


def test_evaluate_policy_raises_on_missing_state() -> None:
    """Policy evaluation should fail fast when policy is incomplete."""
    params = _fixture_params()
    config = ValueIterationConfig(gamma=params.gamma, epsilon=1e-10, max_iters=2000)
    result = solve_value_iteration(params=params, config=config)

    broken_policy = dict(result.policy)
    broken_policy.pop(next(iter(broken_policy)))

    with pytest.raises(ValueError, match="missing"):
        evaluate_policy(policy=broken_policy, params=params, config=config)


def test_evaluate_policy_raises_on_invalid_action() -> None:
    """Policy evaluation should validate action bounds."""
    params = _fixture_params()
    config = ValueIterationConfig(gamma=params.gamma, epsilon=1e-10, max_iters=2000)
    result = solve_value_iteration(params=params, config=config)

    broken_policy = dict(result.policy)
    sample_state = next(iter(broken_policy))
    broken_policy[sample_state] = len(params.categories) + 1

    with pytest.raises(ValueError, match="Invalid action"):
        evaluate_policy(policy=broken_policy, params=params, config=config)


def test_evaluate_policy_raises_when_not_converged() -> None:
    """Policy evaluation should fail fast if convergence tolerance is not reached."""
    params = _fixture_params()
    solve_cfg = ValueIterationConfig(gamma=params.gamma, epsilon=1e-10, max_iters=2000)
    result = solve_value_iteration(params=params, config=solve_cfg)

    non_convergent_cfg = ValueIterationConfig(
        gamma=params.gamma,
        epsilon=1e-14,
        max_iters=1,
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        evaluate_policy(policy=result.policy, params=params, config=non_convergent_cfg)
