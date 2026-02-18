"""DP terminal-state behavior tests for Version C."""

from __future__ import annotations

from discount_engine.core.params import CategoryParams, MDPParams
from discount_engine.dp.discretization import terminal_state
from discount_engine.dp.transitions import enumerate_transition_distribution


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


def test_terminal_state_is_absorbing_with_zero_reward() -> None:
    params = _small_params()
    state = terminal_state(n_categories=len(params.categories))

    for action in range(len(params.categories) + 1):
        outcomes = enumerate_transition_distribution(
            state=state,
            action=action,
            params=params,
        )

        assert len(outcomes) == 1
        only = outcomes[0]
        assert only.probability == 1.0
        assert only.next_state == state
        assert only.reward == 0.0
