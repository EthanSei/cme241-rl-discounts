"""Shared fixtures for RL tests."""

from __future__ import annotations

import pytest

from discount_engine.rl.env import DiscountEnv


def _make_small_env(**overrides) -> DiscountEnv:
    """Build a 2-product DiscountEnv with deterministic defaults."""
    product_params = {
        0: {"beta_0": -1.0, "logit_bump": 0.5, "raw_deal_signal": 0.8, "price": 10.0, "category": "A"},
        1: {"beta_0": -2.0, "logit_bump": 0.3, "raw_deal_signal": 0.6, "price": 20.0, "category": "B"},
    }
    defaults = dict(
        product_params=product_params,
        product_order=[0, 1],
        beta_p=0.5,
        beta_l=0.1,
        beta_m=0.2,
        alpha=0.8,
        delta=0.3,
        gamma=0.99,
        c0=0.05,
        eta=0.01,
        max_steps=50,
        churn_cost=0.0,
        recency_sentinel=52.0,
        randomize_init=False,
    )
    defaults.update(overrides)
    return DiscountEnv(**defaults)


@pytest.fixture
def small_env() -> DiscountEnv:
    """Deterministic 2-product env (randomize_init=False)."""
    return _make_small_env()


@pytest.fixture
def small_env_random() -> DiscountEnv:
    """2-product env with randomized initial states."""
    return _make_small_env(randomize_init=True)
