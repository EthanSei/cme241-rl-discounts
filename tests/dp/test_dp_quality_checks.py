"""DP quality-check tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from discount_engine.core.params import load_mdp_params
from discount_engine.dp.quality_checks import run_quality_checks
from discount_engine.dp.value_iteration import ValueIterationConfig, solve_value_iteration


def _solved_fixture():
    params = load_mdp_params(Path("tests/fixtures/dp_params_small.yaml"))
    result = solve_value_iteration(
        params=params,
        config=ValueIterationConfig(gamma=params.gamma, epsilon=1e-8, max_iters=500),
    )
    return params, result


def test_hard_checks_fail_on_invalid_parameter_domain() -> None:
    params, result = _solved_fixture()
    invalid = replace(params, delta=1.2)

    report = run_quality_checks(
        params=invalid,
        values=result.values,
        policy=result.policy,
        q_values=result.q_values,
        gamma=invalid.gamma,
        strict_conceptual=False,
    )
    assert report.passed is False
    assert any(check.name == "parameter_domains" for check in report.hard_failures)


def test_conceptual_warnings_do_not_fail_by_default() -> None:
    params, result = _solved_fixture()
    collapsed_policy = {state: 0 for state in result.policy}

    report = run_quality_checks(
        params=params,
        values=result.values,
        policy=collapsed_policy,
        q_values=result.q_values,
        gamma=params.gamma,
        strict_conceptual=False,
    )
    assert report.hard_failures == ()
    assert len(report.conceptual_warnings) >= 1
    assert report.passed is True


def test_strict_mode_escalates_conceptual_warnings_to_failure() -> None:
    params, result = _solved_fixture()
    collapsed_policy = {state: 0 for state in result.policy}

    report = run_quality_checks(
        params=params,
        values=result.values,
        policy=collapsed_policy,
        q_values=result.q_values,
        gamma=params.gamma,
        strict_conceptual=True,
    )
    assert len(report.conceptual_warnings) >= 1
    assert report.passed is False
