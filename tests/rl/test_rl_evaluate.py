"""Tests for evaluate helpers."""

from __future__ import annotations

import numpy as np
import pytest

from discount_engine.rl.env import DiscountEnv
from discount_engine.rl.evaluate import action_distribution, q_value_sweep, summarize_seeds


class TestActionDistribution:

    def test_sums_to_one(self) -> None:
        actions = np.array([0, 0, 1, 2, 2, 2])
        dist = action_distribution(actions, n_actions=3)
        assert dist.shape == (3,)
        assert dist.sum() == pytest.approx(1.0)

    def test_correct_frequencies(self) -> None:
        actions = np.array([1, 1, 1, 0])
        dist = action_distribution(actions, n_actions=3)
        np.testing.assert_array_almost_equal(dist, [0.25, 0.75, 0.0])


class TestQValueSweep:

    def test_output_shape(self, small_env: DiscountEnv) -> None:
        """q_value_sweep returns (len(sweep_range), n_actions)."""
        from discount_engine.rl.agent import DQNAgent

        agent = DQNAgent(
            obs_dim=5, n_actions=3, hidden=16,
            obs_scale=small_env.build_obs_scale(),
        )
        sweep = np.linspace(0.0, 0.5, 5)
        q = q_value_sweep(agent, small_env, sweep_dim="churn", sweep_range=sweep)
        assert q.shape == (5, 3)

    def test_recency_sweep_varies_product(self, small_env: DiscountEnv) -> None:
        from discount_engine.rl.agent import DQNAgent

        agent = DQNAgent(
            obs_dim=5, n_actions=3, hidden=16,
            obs_scale=small_env.build_obs_scale(),
        )
        sweep = np.array([1.0, 26.0, 52.0])
        q = q_value_sweep(
            agent, small_env, sweep_dim="recency",
            sweep_range=sweep, product_idx=0,
        )
        # Different recency values should produce different Q-values
        assert not np.allclose(q[0], q[2])


class TestSummarizeSeeds:

    def test_keys_and_types(self) -> None:
        rewards = np.array([10.0, 12.0, 11.0])
        result = summarize_seeds(rewards, baseline_reward=8.0)
        assert result["n_seeds"] == 3
        assert result["mean_reward"] == pytest.approx(11.0)
        assert "lift_mean_pct" in result
        assert "se_reward" in result

    def test_lift_calculation(self) -> None:
        rewards = np.array([12.0, 12.0])  # uniform for easy math
        result = summarize_seeds(rewards, baseline_reward=10.0)
        # lift = (12 - 10) / 10 * 100 = 20%
        assert result["lift_mean_pct"] == pytest.approx(20.0)
        assert result["lift_se_pct"] == pytest.approx(0.0)

    def test_zero_baseline_raises(self) -> None:
        with pytest.raises(ValueError, match="too close to zero"):
            summarize_seeds(np.array([10.0]), baseline_reward=0.0)
