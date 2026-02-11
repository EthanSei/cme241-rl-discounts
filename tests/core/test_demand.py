"""Tests for core demand model."""

from discount_engine.core.demand import DemandInputs, logistic_purchase_probability


def test_purchase_probability_bounds() -> None:
    inputs = DemandInputs(
        baseline_logit=0.0,
        deal_signal=0.0,
        recency_value=0.0,
        memory_value=0.0,
        beta_p=1.0,
        beta_l=1.0,
        beta_m=1.0,
    )
    prob = logistic_purchase_probability(inputs)
    assert 0.0 < prob < 1.0


def test_deal_signal_increases_purchase_probability() -> None:
    base = DemandInputs(
        baseline_logit=0.1,
        deal_signal=0.0,
        recency_value=1.0,
        memory_value=0.5,
        beta_p=1.5,
        beta_l=0.2,
        beta_m=0.3,
    )
    better_deal = DemandInputs(
        baseline_logit=base.baseline_logit,
        deal_signal=0.5,
        recency_value=base.recency_value,
        memory_value=base.memory_value,
        beta_p=base.beta_p,
        beta_l=base.beta_l,
        beta_m=base.beta_m,
    )

    assert logistic_purchase_probability(better_deal) > logistic_purchase_probability(base)


def test_higher_recency_and_memory_reduce_probability() -> None:
    lower_friction = DemandInputs(
        baseline_logit=0.3,
        deal_signal=0.2,
        recency_value=1.0,
        memory_value=0.1,
        beta_p=1.0,
        beta_l=0.4,
        beta_m=0.5,
    )
    higher_friction = DemandInputs(
        baseline_logit=lower_friction.baseline_logit,
        deal_signal=lower_friction.deal_signal,
        recency_value=4.0,
        memory_value=1.0,
        beta_p=lower_friction.beta_p,
        beta_l=lower_friction.beta_l,
        beta_m=lower_friction.beta_m,
    )

    assert logistic_purchase_probability(higher_friction) < logistic_purchase_probability(
        lower_friction
    )
