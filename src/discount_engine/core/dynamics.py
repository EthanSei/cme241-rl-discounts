"""Shared state-transition interfaces for DP and RL workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TransitionOutcome:
    """Single transition outcome emitted by environment/simulator steps."""

    next_state: Any
    reward: float
    terminated: bool
    info: dict[str, Any]


def step_continuous_state(*args: Any, **kwargs: Any) -> TransitionOutcome:
    """Apply one transition in Version B continuous state space."""
    raise NotImplementedError("Continuous dynamics are pending implementation.")


def step_discrete_state(*args: Any, **kwargs: Any) -> TransitionOutcome:
    """Apply one transition in Version C discretized state space."""
    raise NotImplementedError("Discrete dynamics are pending implementation.")
