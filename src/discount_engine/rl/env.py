"""Gymnasium environment interface for Version B."""

from __future__ import annotations

from typing import Any


class DiscountEnv:
    """Environment scaffold for item-level discount targeting."""

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError("RL env reset is pending implementation.")

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        raise NotImplementedError("RL env step is pending implementation.")

