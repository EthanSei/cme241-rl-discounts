"""Shared demand-model interfaces for MDP dynamics."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DemandInputs:
    """Inputs required by the per-category purchase model."""

    baseline_logit: float
    deal_signal: float
    recency_value: float
    memory_value: float
    beta_p: float
    beta_l: float
    beta_m: float


def logistic_purchase_probability(inputs: DemandInputs) -> float:
    """Return purchase probability for one category.
    """
    logit = (
        inputs.baseline_logit
        + inputs.beta_p * inputs.deal_signal
        - inputs.beta_l * inputs.recency_value
        - inputs.beta_m * inputs.memory_value
    )
    # Numerically stable sigmoid.
    if logit >= 0:
        exp_neg = math.exp(-logit)
        return 1.0 / (1.0 + exp_neg)

    exp_pos = math.exp(logit)
    return exp_pos / (1.0 + exp_pos)
