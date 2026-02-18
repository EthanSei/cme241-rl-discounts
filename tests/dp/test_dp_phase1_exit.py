"""Phase 1 exit-criteria tests for DP transition/reward behavior."""

from __future__ import annotations

from discount_engine.core.params import CategoryParams, MDPParams
from discount_engine.core.types import DiscreteState
from discount_engine.dp.discretization import enumerate_all_states
from discount_engine.dp.transitions import enumerate_transition_distribution


def _params_for_n(n_categories: int) -> MDPParams:
    categories = tuple(
        CategoryParams(name=f"C{i+1}", price=1.0 + i, beta_0=0.15 - 0.1 * i)
        for i in range(n_categories)
    )
    return MDPParams(
        delta=0.30,
        gamma=0.99,
        alpha=0.90,
        beta_p=1.2,
        beta_l=0.01,
        beta_m=0.4,
        eta=0.02,
        c0=0.20,
        categories=categories,
    )


def _terminal_probability(outcomes: list) -> float:
    return sum(outcome.probability for outcome in outcomes if outcome.next_state.churn_bucket == -1)


def test_transition_kernel_probability_and_rewards_for_n2_and_n3() -> None:
    for n_categories in (2, 3):
        params = _params_for_n(n_categories)
        state = DiscreteState(
            churn_bucket=0,
            memory_buckets=(1,) * n_categories,
            recency_buckets=(0,) * n_categories,
        )

        for action in range(n_categories + 1):
            outcomes = enumerate_transition_distribution(
                state=state,
                action=action,
                params=params,
            )
            total_prob = sum(outcome.probability for outcome in outcomes)
            assert abs(total_prob - 1.0) <= 1e-9
            assert all(outcome.probability >= 0.0 for outcome in outcomes)
            assert all(outcome.reward >= 0.0 for outcome in outcomes)
            # Interpolation-based bucketization can fan out one subset into multiple
            # discretized next states. Bound by full state support plus terminal.
            max_outcomes = len(enumerate_all_states(n_categories=n_categories)) + 1
            assert 1 <= len(outcomes) <= max_outcomes


def test_higher_churn_bucket_increases_terminal_branch_probability() -> None:
    n_categories = 3
    params = _params_for_n(n_categories)
    low_churn = DiscreteState(
        churn_bucket=0,
        memory_buckets=(1,) * n_categories,
        recency_buckets=(0,) * n_categories,
    )
    high_churn = DiscreteState(
        churn_bucket=1,
        memory_buckets=(1,) * n_categories,
        recency_buckets=(0,) * n_categories,
    )

    low_outcomes = enumerate_transition_distribution(state=low_churn, action=0, params=params)
    high_outcomes = enumerate_transition_distribution(state=high_churn, action=0, params=params)
    assert _terminal_probability(high_outcomes) > _terminal_probability(low_outcomes)
