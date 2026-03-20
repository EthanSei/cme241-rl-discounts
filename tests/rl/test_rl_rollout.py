"""Tests for run_rollout."""

from __future__ import annotations

import numpy as np

from discount_engine.rl.env import DiscountEnv
from discount_engine.rl.rollout import run_rollout


class TestRunRollout:
    """run_rollout returns correct shapes and handles different policies."""

    def test_output_keys_and_shapes(self, small_env: DiscountEnv) -> None:
        result = run_rollout(small_env, n_episodes=5, policy="no_promo", seed=0)
        assert result["rewards"].shape == (5,)
        assert result["lengths"].shape == (5,)
        assert result["actions"].ndim == 1
        assert result["purchases_per_step"].ndim == 1
        # Total steps across episodes should match action count
        assert len(result["actions"]) == int(result["lengths"].sum())

    def test_no_promo_only_action_zero(self, small_env: DiscountEnv) -> None:
        result = run_rollout(small_env, n_episodes=3, policy="no_promo", seed=0)
        assert np.all(result["actions"] == 0)

    def test_fixed_action_policy(self, small_env: DiscountEnv) -> None:
        result = run_rollout(small_env, n_episodes=3, policy=1, seed=0)
        assert np.all(result["actions"] == 1)

    def test_callable_policy(self, small_env: DiscountEnv) -> None:
        result = run_rollout(small_env, n_episodes=3, policy=lambda obs: 2, seed=0)
        assert np.all(result["actions"] == 2)

    def test_random_policy_uses_multiple_actions(self, small_env: DiscountEnv) -> None:
        result = run_rollout(small_env, n_episodes=10, policy="random", seed=0)
        unique_actions = set(result["actions"])
        assert len(unique_actions) >= 2

    def test_deterministic_with_seed(self, small_env: DiscountEnv) -> None:
        r1 = run_rollout(small_env, n_episodes=5, policy="no_promo", seed=42)
        r2 = run_rollout(small_env, n_episodes=5, policy="no_promo", seed=42)
        np.testing.assert_array_equal(r1["rewards"], r2["rewards"])
