"""Shared demand-model interfaces for MDP dynamics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemandInputs:
    """Inputs required by the per-category purchase model."""

    baseline_logit: float
    deal_signal: float
    recency_value: float
    beta_p: float
    beta_l: float


def logistic_purchase_probability(inputs: DemandInputs) -> float:
    """Return purchase probability for one category.

    This is intentionally left as a scaffold for the initial implementation
    phase. The finalized formula will be wired to calibrated MDP parameters.
    """
    raise NotImplementedError("Demand model wiring is pending implementation.")
