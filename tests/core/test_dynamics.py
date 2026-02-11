"""Tests for core dynamics interfaces."""

import pytest

from discount_engine.core.dynamics import (
    TransitionOutcome,
    step_continuous_state,
    step_discrete_state,
)


def test_transition_outcome_fields() -> None:
    outcome = TransitionOutcome(
        next_state={"c": 0.2},
        reward=1.5,
        terminated=False,
        info={"p_buy": [0.3, 0.1]},
    )
    assert outcome.reward == 1.5
    assert outcome.terminated is False
    assert outcome.next_state == {"c": 0.2}


def test_transition_interface_placeholders_raise() -> None:
    with pytest.raises(NotImplementedError):
        step_continuous_state()
    with pytest.raises(NotImplementedError):
        step_discrete_state()
