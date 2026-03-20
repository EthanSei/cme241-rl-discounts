"""DQN agent model for promotions"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from discount_engine.rl.replay import PrioritizedReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """2-layer MLP Q-network"""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class DQNAgent:
    """Double DQN with PER, Huber loss, and observation normalization."""

    def __init__(
            self,
            obs_dim: int,
            n_actions: int,
            lr: float = 3e-4,
            gamma: float = 0.99,
            buffer_size: int = 100_000,
            batch_size: int = 128,
            target_update_freq: int = 500,
            eps_start: float = 1.0,
            eps_end: float = 0.05,
            eps_decay_steps: int = 20_000,
            hidden: int = 128,
            per_alpha: float = 0.6,
            per_beta_start: float = 0.4,
            per_beta_end: float = 1.0,
            per_beta_steps: int = 50_000,
            obs_scale: np.ndarray | None = None,
            seed: int = 42,
            reward_scale: float = 1.0,
            grad_clip: float = 10.0,
    ):
        if reward_scale <= 0:
            raise ValueError(f"reward_scale must be positive, got {reward_scale}")
        if grad_clip <= 0:
            raise ValueError(f"grad_clip must be positive, got {grad_clip}")
        self.n_actions = n_actions
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        self.q_net = QNetwork(obs_dim, n_actions, hidden).to(device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.rng = np.random.default_rng(seed)
        buf_rng = self.rng.spawn(1)[0]
        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size, obs_dim=obs_dim, alpha=per_alpha,
            rng=buf_rng,
        )
        self.per_beta_start = per_beta_start
        self.per_beta_end = per_beta_end
        self.per_beta_steps = per_beta_steps

        scale = obs_scale if obs_scale is not None else np.ones(obs_dim)
        self.obs_scale = torch.FloatTensor(
            np.maximum(scale, 1e-8)
        ).to(device)
        self.step_count = 0       # env steps (drives epsilon/beta schedules)
        self.update_count = 0    # gradient updates (drives target net sync)

    def tick(self) -> None:
        """Advance env-step counter. Call once per env step, independent of update()."""
        self.step_count += 1

    def get_epsilon(self) -> float:
        return max(
            self.eps_end,
            self.eps_start - (self.eps_start - self.eps_end) * self.step_count / self.eps_decay_steps,
        )

    def get_beta(self) -> float:
        return min(
            self.per_beta_end,
            self.per_beta_start + (self.per_beta_end - self.per_beta_start) * self.step_count / self.per_beta_steps,
        )

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        """Return Q-values for a single observation."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device) / self.obs_scale
            return self.q_net(obs_t).squeeze(0).cpu().numpy()

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        # Explore
        if not greedy and self.rng.random() < self.get_epsilon():
            return int(self.rng.integers(self.n_actions))
        # Exploit
        return int(self.get_q_values(obs).argmax())

    def store(
        self, obs: np.ndarray, action: int, reward: float,
        next_obs: np.ndarray, done: float,
    ) -> None:
        self.buffer.push(obs, action, reward / self.reward_scale, next_obs, done)

    def update(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        
        beta = self.get_beta()
        states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(
            self.batch_size, beta=beta
        )

        # Convert numpy arrays to tensors, move to device, and normalize
        states = torch.from_numpy(states).to(device) / self.obs_scale
        next_states = torch.from_numpy(next_states).to(device) / self.obs_scale
        actions = torch.from_numpy(actions).to(device)
        rewards = torch.from_numpy(rewards).to(device)
        dones = torch.from_numpy(dones).to(device)
        weights = torch.from_numpy(weights).to(device)

        # Current Q-values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: online net selects, target net evaluates
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss
        td_errors = q_values - targets
        element_wise = nn.functional.smooth_l1_loss(q_values, targets, reduction="none")
        loss = (weights * element_wise).mean()

        # Step backwards
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Update PER priorities
        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.buffer.update_priorities(indices, new_priorities)

        # Target network update (based on gradient steps, not env steps)
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())
