"""DP Bellman backup tests."""

from __future__ import annotations

from discount_engine.core.params import CategoryParams, MDPParams
from discount_engine.core.types import DiscreteState
from discount_engine.dp.value_iteration import bellman_action_value, bellman_backup


def _hand_check_params() -> MDPParams:
    # This fixture forces p_buy = 0.5 for the only category under both actions.
    return MDPParams(
        delta=0.30,
        gamma=0.95,
        alpha=0.90,
        beta_p=0.0,
        beta_l=0.0,
        beta_m=0.0,
        eta=0.05,
        c0=0.20,
        categories=(CategoryParams(name="only", price=10.0, beta_0=0.0),),
    )


def test_bellman_backup_matches_hand_checked_one_step_values() -> None:
    params = _hand_check_params()
    state = DiscreteState(churn_bucket=0, memory_buckets=(0,), recency_buckets=(0,))
    values = {
        DiscreteState(churn_bucket=0, memory_buckets=(0,), recency_buckets=(0,)): 0.0,
        DiscreteState(churn_bucket=1, memory_buckets=(0,), recency_buckets=(1,)): 0.0,
        DiscreteState(churn_bucket=-1, memory_buckets=(0,), recency_buckets=(0,)): 0.0,
    }

    # Analytical expected rewards when V=0:
    # action=0 -> 10 * 0.5 = 5
    # action=1 -> (10*(1-0.3)) * 0.5 = 3.5
    q0 = bellman_action_value(
        state=state,
        action=0,
        values=values,
        params=params,
        gamma=params.gamma,
    )
    q1 = bellman_action_value(
        state=state,
        action=1,
        values=values,
        params=params,
        gamma=params.gamma,
    )
    assert abs(q0 - 5.0) <= 1e-9
    assert abs(q1 - 3.5) <= 1e-9

    best_value, best_action, q_values = bellman_backup(
        state=state,
        values=values,
        params=params,
        gamma=params.gamma,
    )
    assert abs(best_value - 5.0) <= 1e-9
    assert best_action == 0
    assert abs(q_values[0] - 5.0) <= 1e-9
    assert abs(q_values[1] - 3.5) <= 1e-9
