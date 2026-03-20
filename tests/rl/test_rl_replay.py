"""Tests for PrioritizedReplayBuffer."""

from __future__ import annotations

import numpy as np
import pytest

from discount_engine.rl.replay import PrioritizedReplayBuffer


class TestPrioritizedReplayBuffer:
    """Array-backed PER: push, sample, priorities, reproducibility."""

    def test_push_and_len(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=3)
        assert len(buf) == 0
        buf.push(np.zeros(3), 0, 1.0, np.zeros(3), 0.0)
        assert len(buf) == 1

    def test_circular_overwrites(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=3, obs_dim=2)
        for i in range(5):
            obs = np.full(2, float(i), dtype=np.float32)
            buf.push(obs, 0, float(i), obs, 0.0)
        assert len(buf) == 3

    def test_sample_returns_correct_shapes(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=3)
        for i in range(20):
            buf.push(
                np.array([i, 0.0, 0.0]), 0, 1.0,
                np.array([i + 1, 0.0, 0.0]), 0.0,
            )
        states, actions, rewards, next_states, dones, weights, indices = (
            buf.sample(8, beta=0.4)
        )
        assert states.shape == (8, 3)
        assert actions.shape == (8,)
        assert rewards.shape == (8,)
        assert next_states.shape == (8, 3)
        assert dones.shape == (8,)
        assert weights.shape == (8,)
        assert indices.shape == (8,)

    def test_high_priority_sampled_more(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=1)
        for i in range(11):
            buf.push(np.array([float(i)]), 0, 0.0, np.array([0.0]), 0.0)
        buf.update_priorities([10], [100.0])

        counts = np.zeros(11)
        for _ in range(500):
            _, _, _, _, _, _, indices = buf.sample(4, beta=0.4)
            for idx in indices:
                counts[idx] += 1
        assert counts[10] > counts[0] * 3

    def test_uniform_sampling_with_alpha_zero(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=1, alpha=0.0)
        for i in range(10):
            buf.push(np.array([0.0]), 0, 0.0, np.array([0.0]), 0.0)
        buf.update_priorities([0], [100.0])
        counts = np.zeros(10)
        for _ in range(1000):
            _, _, _, _, _, _, indices = buf.sample(1)
            counts[indices[0]] += 1
        # Roughly uniform despite different priorities
        assert counts.std() / counts.mean() < 0.5

    def test_weights_are_normalized(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=1)
        for i in range(20):
            buf.push(np.array([float(i)]), 0, 0.0, np.array([0.0]), 0.0)
        buf.update_priorities([0], [100.0])
        _, _, _, _, _, weights, _ = buf.sample(16, beta=1.0)
        assert abs(float(weights.max()) - 1.0) < 1e-6
        assert float(weights.min()) > 0.0

    def test_update_priorities(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=1)
        for i in range(5):
            buf.push(np.array([0.0]), 0, 0.0, np.array([0.0]), 0.0)
        buf.update_priorities([0, 1], [10.0, 20.0])
        assert buf.max_priority == 20.0

    def test_recompute_max_priority(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=10, obs_dim=1)
        for _ in range(5):
            buf.push(np.array([0.0]), 0, 0.0, np.array([0.0]), 0.0)
        buf.update_priorities([0], [100.0])
        assert buf.max_priority == 100.0
        buf.priorities[0] = 1.0
        buf.recompute_max_priority()
        assert buf.max_priority == pytest.approx(1.0)

    def test_rng_reproducibility(self) -> None:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        buf1 = PrioritizedReplayBuffer(capacity=50, obs_dim=2, rng=rng1)
        buf2 = PrioritizedReplayBuffer(capacity=50, obs_dim=2, rng=rng2)
        obs = np.ones(2, dtype=np.float32)
        for i in range(20):
            buf1.push(obs * i, 0, float(i), obs, 0.0)
            buf2.push(obs * i, 0, float(i), obs, 0.0)
        _, _, r1, _, _, _, _ = buf1.sample(8)
        _, _, r2, _, _, _, _ = buf2.sample(8)
        np.testing.assert_array_equal(r1, r2)