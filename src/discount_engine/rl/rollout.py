"""Rollout and simulation helpers for RL experiments."""

from __future__ import annotations

import numpy as np
from typing import Any


def run_rollout(env, n_episodes=500, policy='random', seed=42) -> dict[str, Any]:
    """Run rollouts and collect statistics."""
    episode_rewards = np.zeros(n_episodes)
    episode_lengths = np.zeros(n_episodes)
    all_actions: list[int] = []
    all_purchases: list[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            if policy == 'random':
                action = env.action_space.sample()
            elif policy == 'no_promo':
                action = 0
            elif isinstance(policy, int):
                action = policy
            else:
                action = policy(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            all_actions.append(action)
            all_purchases.append(info['purchases'].sum())
            done = terminated or truncated
        episode_rewards[ep] = total_reward
        episode_lengths[ep] = steps

    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "actions": np.array(all_actions),
        "purchases_per_step": np.array(all_purchases),
    }