"""DP policy sanity tests."""

from __future__ import annotations

from pathlib import Path

from discount_engine.core.params import load_mdp_params
from discount_engine.dp.discretization import is_terminal_state
from discount_engine.dp.value_iteration import ValueIterationConfig, solve_value_iteration


def _params():
    return load_mdp_params(Path("tests/fixtures/dp_params_small.yaml"))


def test_policy_is_not_degenerate_and_uses_no_promo_in_high_memory_states() -> None:
    params = _params()
    result = solve_value_iteration(
        params=params,
        config=ValueIterationConfig(gamma=params.gamma, epsilon=1e-8, max_iters=500),
    )

    live_actions = [
        action
        for state, action in result.policy.items()
        if not is_terminal_state(state)
    ]
    assert any(action == 0 for action in live_actions)

    high_memory_recent_actions = [
        action
        for state, action in result.policy.items()
        if (
            not is_terminal_state(state)
            and all(bucket == 2 for bucket in state.memory_buckets)
            and all(bucket == 0 for bucket in state.recency_buckets)
        )
    ]
    assert high_memory_recent_actions
    assert any(action == 0 for action in high_memory_recent_actions)


def test_stale_states_are_not_less_promoted_than_recent_states() -> None:
    params = _params()
    result = solve_value_iteration(
        params=params,
        config=ValueIterationConfig(gamma=params.gamma, epsilon=1e-8, max_iters=500),
    )

    recent_actions: list[int] = []
    stale_actions: list[int] = []
    for state, action in result.policy.items():
        if is_terminal_state(state):
            continue
        mean_recency = sum(state.recency_buckets) / len(state.recency_buckets)
        if mean_recency <= 0.25:
            recent_actions.append(action)
        if mean_recency >= 0.75:
            stale_actions.append(action)

    assert recent_actions
    assert stale_actions
    recent_rate = sum(1 for action in recent_actions if action != 0) / len(recent_actions)
    stale_rate = sum(1 for action in stale_actions if action != 0) / len(stale_actions)
    assert stale_rate + 0.05 >= recent_rate
