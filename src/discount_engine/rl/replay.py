"""Prioritized Experience Replay buffer."""

from __future__ import annotations

import numpy as np


class PrioritizedReplayBuffer:
    """Proportional PER (Schaul et al., 2016).

    Stores (s, a, r, s', done) with per-transition priorities.
    Sampling probability is proportional to priority^alpha.
    Returns importance-sampling weights normalized so max weight = 1.

    Returns numpy arrays. Callers (e.g. DQN agent) convert to tensors.
    """

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: list[tuple] = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[int],
    ]:
        n = len(self.buffer)
        priors = self.priorities[:n] ** self.alpha
        probs = priors / priors.sum()

        indices = np.random.choice(n, size=batch_size, p=probs)

        # Importance-sampling weights, normalized so max = 1
        weights = (n * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        states, actions, rewards, next_states, dones = zip(
            *(self.buffer[i] for i in indices)
        )
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            weights.astype(np.float32),
            list(indices),
        )

    def update_priorities(self, indices: list[int], priorities: list[float]) -> None:
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = p
            self.max_priority = max(self.max_priority, p)

    def __len__(self) -> int:
        return len(self.buffer)
