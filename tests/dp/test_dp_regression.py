"""DP regression snapshot tests."""

from __future__ import annotations

from pathlib import Path

from discount_engine.core.params import load_mdp_params
from discount_engine.core.types import DiscreteState
from discount_engine.dp.value_iteration import ValueIterationConfig, solve_value_iteration


EXPECTED_ACTIONS = {
    "c0_m00_l00": 0,
    "c1_m00_l11": 0,
    "c0_m22_l00": 0,
    "c1_m22_l11": 0,
    "terminal": 0,
}

EXPECTED_VALUES = {
    "c0_m00_l00": 22.137321603434735,
    "c1_m00_l11": 8.554405804439625,
    "c0_m22_l00": 5.503779201570762,
    "c1_m22_l11": 0.534681135569914,
    "terminal": 0.0,
}


def test_regression_selected_states_match_snapshot() -> None:
    params = load_mdp_params(Path("tests/fixtures/dp_params_small.yaml"))
    result = solve_value_iteration(
        params=params,
        config=ValueIterationConfig(gamma=params.gamma, epsilon=1e-8, max_iters=500),
    )

    selected_states = {
        "c0_m00_l00": DiscreteState(
            churn_bucket=0,
            memory_buckets=(0, 0),
            recency_buckets=(0, 0),
        ),
        "c1_m00_l11": DiscreteState(
            churn_bucket=1,
            memory_buckets=(0, 0),
            recency_buckets=(1, 1),
        ),
        "c0_m22_l00": DiscreteState(
            churn_bucket=0,
            memory_buckets=(2, 2),
            recency_buckets=(0, 0),
        ),
        "c1_m22_l11": DiscreteState(
            churn_bucket=1,
            memory_buckets=(2, 2),
            recency_buckets=(1, 1),
        ),
        "terminal": DiscreteState(
            churn_bucket=-1,
            memory_buckets=(0, 0),
            recency_buckets=(0, 0),
        ),
    }

    for key, state in selected_states.items():
        assert result.policy[state] == EXPECTED_ACTIONS[key]
        assert abs(result.values[state] - EXPECTED_VALUES[key]) <= 1e-6
