"""Shared state and action types used across DP and RL modules."""

from __future__ import annotations

from dataclasses import dataclass


Action = int


@dataclass(frozen=True)
class ContinuousState:
    """Continuous state for Version B.

    Attributes:
        churn_propensity: Scalar churn propensity in [0, 1].
        discount_memory: Per-category discount memory values.
        purchase_recency: Per-category recency values.
    """

    churn_propensity: float
    discount_memory: tuple[float, ...]
    purchase_recency: tuple[float, ...]


@dataclass(frozen=True)
class DiscreteState:
    """Discretized state for Version C."""

    churn_bucket: int
    memory_buckets: tuple[int, ...]
    recency_buckets: tuple[int, ...]
