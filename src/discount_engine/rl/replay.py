"""Prioritized Experience Replay buffer."""

from __future__ import annotations

import numpy as np


class PrioritizedReplayBuffer:
    """Proportional PER (Schaul et al., 2016).

    Stores (s, a, r, s', done) in pre-allocated numpy arrays for fast sampling.
    Sampling probability is proportional to priority^alpha.
    Returns importance-sampling weights normalized so max weight = 1.

    Returns numpy arrays. Callers (e.g. DQN agent) convert to tensors.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        alpha: float = 0.6,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.rng = rng if rng is not None else np.random.default_rng()

        # Pre-allocated storage
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        # Recompute max_priority on wrap-around to prevent stale inflation
        if self.pos == 0 and self.size == self.capacity:
            self.recompute_max_priority()

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        n = self.size
        priors = self.priorities[:n] ** self.alpha
        total = priors.sum()
        if total == 0:
            probs = np.ones(n) / n
        else:
            probs = priors / total

        indices = self.rng.choice(n, size=batch_size, p=probs)

        # Importance-sampling weights in log-space to avoid overflow
        log_w = -beta * np.log(n * probs[indices] + 1e-10)
        weights = np.exp(log_w - log_w.max())  # normalized so max = 1

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            weights.astype(np.float32),
            indices,
        )

    def update_priorities(
        self, indices: np.ndarray, priorities: np.ndarray | list[float]
    ) -> None:
        p_arr = np.asarray(priorities, dtype=np.float64)
        self.priorities[indices] = p_arr
        self.max_priority = max(self.max_priority, float(p_arr.max()))

    def recompute_max_priority(self) -> None:
        """Recompute max_priority from active entries to prevent stale inflation."""
        if self.size > 0:
            self.max_priority = float(self.priorities[:self.size].max())

    def __len__(self) -> int:
        return self.size