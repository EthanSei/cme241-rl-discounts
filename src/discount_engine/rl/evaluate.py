"""Evaluation helpers for RL agents."""

from __future__ import annotations

import numpy as np
from typing import Any


def action_distribution(actions: np.ndarray, n_actions: int) -> np.ndarray:
    """Return normalized action frequencies from a rollout's action array."""
    counts = np.bincount(actions, minlength=n_actions)
    return counts / counts.sum()


def q_value_sweep(
    agent,
    env,
    sweep_dim: str,
    sweep_range: np.ndarray,
    base_churn: float = 0.10,
    base_memory: np.ndarray | None = None,
    base_recency: np.ndarray | None = None,
    product_idx: int | None = None,
) -> np.ndarray:
    """Sweep one state dimension and return Q-values at each point.

    Args:
        agent: Agent with a get_q_values(obs) method.
        env: DiscountEnv (uses make_obs and action_space).
        sweep_dim: One of "churn", "recency", "memory".
        sweep_range: 1-D array of values to sweep.
        base_churn: Default churn for non-swept dimensions.
        base_memory: Default memory vector (zeros if None).
        base_recency: Default recency vector (sentinel if None).
        product_idx: Which product index to vary for recency/memory sweeps.

    Returns:
        Q-values array of shape (len(sweep_range), n_actions).
    """
    if base_memory is None:
        base_memory = np.zeros(env.N)
    if base_recency is None:
        base_recency = np.full(env.N, env.recency_sentinel)

    q_vals = np.zeros((len(sweep_range), env.action_space.n))

    for i, val in enumerate(sweep_range):
        churn = base_churn
        mem = base_memory.copy()
        rec = base_recency.copy()

        if sweep_dim == "churn":
            churn = val
        elif sweep_dim == "recency":
            if product_idx is None:
                raise ValueError("product_idx required for recency sweep")
            rec[product_idx] = val
        elif sweep_dim == "memory":
            if product_idx is None:
                raise ValueError("product_idx required for memory sweep")
            mem[product_idx] = val
        else:
            raise ValueError(f"Unknown sweep_dim: {sweep_dim!r}")

        obs = env.make_obs(churn, memory=mem, recency=rec)
        q_vals[i] = agent.get_q_values(obs)

    return q_vals


def summarize_seeds(
    rewards: np.ndarray,
    baseline_reward: float,
) -> dict[str, Any]:
    """Aggregate multi-seed reward results with lift vs baseline.

    Args:
        rewards: 1-D array of per-seed mean rewards.
        baseline_reward: No-promo baseline reward for lift calculation.

    Returns:
        Dict with mean, se, lift_mean, lift_se.
    """
    rewards = np.asarray(rewards)
    n = len(rewards)
    if abs(baseline_reward) < 1e-12:
        raise ValueError("baseline_reward too close to zero for lift calculation")
    lifts = (rewards - baseline_reward) / abs(baseline_reward) * 100

    return {
        "mean_reward": float(rewards.mean()),
        "se_reward": float(rewards.std() / np.sqrt(n)),
        "lift_mean_pct": float(lifts.mean()),
        "lift_se_pct": float(lifts.std() / np.sqrt(n)),
        "n_seeds": n,
        "baseline_reward": float(baseline_reward),
    }
