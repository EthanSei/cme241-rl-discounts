"""Training interfaces for Version B RL agents."""

from __future__ import annotations

from typing import Any


def train_agent(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Train an RL policy and return training artifacts metadata."""
    raise NotImplementedError("RL training implementation is pending.")
