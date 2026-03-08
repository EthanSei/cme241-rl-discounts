"""Tests for PrioritizedReplayBuffer."""

import numpy as np

from discount_engine.rl.replay import PrioritizedReplayBuffer


class TestPrioritizedReplayBuffer:
    """PER: priority-proportional sampling with importance-sampling weights."""

    def test_sample_returns_correct_shapes(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
        for i in range(20):
            buf.push(np.array([i, 0.0, 0.0]), 0, 1.0, np.array([i + 1, 0.0, 0.0]), False)

        states, actions, rewards, next_states, dones, weights, indices = buf.sample(8, beta=0.4)
        assert states.shape == (8, 3)
        assert actions.shape == (8,)
        assert rewards.shape == (8,)
        assert next_states.shape == (8, 3)
        assert dones.shape == (8,)
        assert weights.shape == (8,)
        assert len(indices) == 8

    def test_high_priority_sampled_more(self) -> None:
        """Transitions with higher TD error should be sampled more frequently."""
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
        # Push 10 low-priority, then 1 high-priority
        for i in range(10):
            buf.push(np.array([0.0]), 0, 0.0, np.array([0.0]), False)
        buf.push(np.array([1.0]), 1, 1.0, np.array([1.0]), False)
        # Boost priority of last transition
        buf.update_priorities([10], [100.0])

        # Sample many times and count how often index 10 appears
        counts = np.zeros(11)
        for _ in range(500):
            _, _, _, _, _, _, indices = buf.sample(4, beta=0.4)
            for idx in indices:
                counts[idx] += 1

        # High-priority transition should appear much more than average
        assert counts[10] > counts[0] * 3

    def test_weights_are_normalized(self) -> None:
        """IS weights should be normalized so max weight = 1."""
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
        for i in range(20):
            buf.push(np.array([float(i)]), 0, 0.0, np.array([0.0]), False)
        buf.update_priorities([0], [100.0])

        _, _, _, _, _, weights, _ = buf.sample(16, beta=1.0)
        assert abs(float(weights.max()) - 1.0) < 1e-6
        assert float(weights.min()) > 0.0

    def test_update_priorities(self) -> None:
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
        for i in range(5):
            buf.push(np.array([0.0]), 0, 0.0, np.array([0.0]), False)
        buf.update_priorities([0, 1], [10.0, 20.0])
        # Should not raise; priorities are internal state
        assert len(buf) == 5
