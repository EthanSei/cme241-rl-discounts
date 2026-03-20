"""Training interfaces for Version B RL agents."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from typing import Any


class Trainer:
    """Manages training, evaluation, and checkpointing for an RL agent."""

    def __init__(
        self,
        env,
        agent,
        seed: int = 42,
        snapshot_dir: Path | None = None,
        label: str = "agent",
        update_every: int = 4,
    ):
        self.env = env
        self.agent = agent
        self.seed = seed
        self.label = label
        self.update_every = update_every

        self.snapshot_dir: Path | None = None
        if snapshot_dir is not None:
            self.snapshot_dir = Path(snapshot_dir)
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ── Core training loop ────────���──────────────────────────────────────

    def train(
        self,
        n_episodes: int = 3_000,
        eval_interval: int = 500,
        eval_episodes: int = 50,
        snapshot_interval: int | None = None,
    ) -> dict[str, Any]:
        """Train the agent and collect metrics.

        Args:
            n_episodes: Total training episodes.
            eval_interval: Evaluate greedy policy every this many episodes.
            eval_episodes: Episodes per evaluation.
            snapshot_interval: Save a checkpoint every this many episodes.
                Defaults to eval_interval when snapshot_dir is set.
        """
        if snapshot_interval is None:
            snapshot_interval = eval_interval

        n_evals = n_episodes // eval_interval
        train_rewards = np.zeros(n_episodes)
        eval_rewards = np.zeros(n_evals)
        eval_points = np.zeros(n_evals, dtype=np.int64)
        eval_mean_q = np.zeros(n_evals)
        eval_max_q = np.zeros(n_evals)
        eval_mean_loss = np.zeros(n_evals)
        losses: list[float] = []
        interval_losses: list[float] = []
        action_counts = np.zeros(self.env.action_space.n)
        snapshot_paths: list[str] = []
        eval_idx = 0
        best_eval = -np.inf
        best_weights: dict[str, Any] | None = None
        best_episode = 0

        pbar = tqdm(range(n_episodes), desc=f"Training {self.label}")
        for ep in pbar:
            reward, ep_losses, ep_actions = self._run_episode(ep)
            train_rewards[ep] = reward
            losses.extend(ep_losses)
            interval_losses.extend(ep_losses)
            for a in ep_actions:
                action_counts[a] += 1

            # Periodic eval
            if (ep + 1) % eval_interval == 0:
                mean_r, mean_q, max_q = self._evaluate(eval_episodes)
                eval_rewards[eval_idx] = mean_r
                eval_mean_q[eval_idx] = mean_q
                eval_max_q[eval_idx] = max_q
                eval_mean_loss[eval_idx] = (
                    float(np.mean(interval_losses)) if interval_losses else 0.0
                )
                interval_losses.clear()
                eval_points[eval_idx] = ep + 1
                eval_idx += 1

                # Track best model
                if mean_r > best_eval:
                    best_eval = mean_r
                    best_episode = ep + 1
                    best_weights = {
                        k: v.clone() for k, v in self.agent.q_net.state_dict().items()
                    }

                pbar.set_postfix({
                    "eval": f"{mean_r:.1f}",
                    "mean_q": f"{mean_q:.1f}",
                    "eps": f"{self.agent.get_epsilon():.3f}",
                    "steps": f"{self.agent.step_count:,}",
                })
                tqdm.write(
                    f"  [ep {ep+1:>5d}] eval={mean_r:.2f}  "
                    f"mean_q={mean_q:.2f}  max_q={max_q:.2f}  "
                    f"loss={eval_mean_loss[eval_idx-1]:.4f}  "
                    f"eps={self.agent.get_epsilon():.3f}"
                )

            # Periodic checkpoint
            if self.snapshot_dir is not None and (ep + 1) % snapshot_interval == 0:
                path = self._save_snapshot(
                    ep + 1,
                    eval_reward=float(eval_rewards[eval_idx - 1]) if eval_idx > 0 else None,
                )
                snapshot_paths.append(path)

        pbar.close()

        # Restore best model
        if best_weights is not None:
            self.agent.q_net.load_state_dict(best_weights)
            self.agent.target_net.load_state_dict(best_weights)
            tqdm.write(f"  Restored best model from ep {best_episode} (eval={best_eval:.2f})")

        return {
            "train_rewards": train_rewards,
            "eval_rewards": eval_rewards,
            "eval_points": eval_points,
            "eval_mean_q": eval_mean_q,
            "eval_max_q": eval_max_q,
            "eval_mean_loss": eval_mean_loss,
            "losses": np.array(losses),
            "action_counts": action_counts,
            "snapshot_paths": snapshot_paths,
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    def _run_episode(self, ep: int) -> tuple[float, list[float], list[int]]:
        """Run one training episode. Returns (total_reward, losses, actions)."""
        obs, _ = self.env.reset(seed=self.seed + ep)
        total_reward = 0.0
        done = False
        losses: list[float] = []
        actions: list[int] = []

        while not done:
            action = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.agent.store(obs, action, reward, next_obs, float(done))
            self.agent.tick()
            if self.agent.step_count % self.update_every == 0:
                loss = self.agent.update()
                if loss is not None:
                    losses.append(loss)
            actions.append(action)
            total_reward += reward
            obs = next_obs

        return total_reward, losses, actions

    def _evaluate(self, n_episodes: int) -> tuple[float, float, float]:
        """Run greedy evaluation episodes. Returns (mean_reward, mean_q, max_q)."""
        eval_rs = np.zeros(n_episodes)
        q_values_collected: list[float] = []
        for i in range(n_episodes):
            obs, _ = self.env.reset(seed=1_000_000 + i)
            r = 0.0
            done = False
            # Collect Q-values from first observation of each episode
            q = self.agent.get_q_values(obs)
            q_values_collected.append(float(q.max()))
            while not done:
                a = self.agent.select_action(obs, greedy=True)
                obs, rew, term, trunc, _ = self.env.step(a)
                r += rew
                done = term or trunc
            eval_rs[i] = r
        q_arr = np.array(q_values_collected)
        return float(eval_rs.mean()), float(q_arr.mean()), float(q_arr.max())

    def _save_snapshot(self, episode: int, eval_reward: float | None = None) -> str:
        """Save a model checkpoint and return the path."""
        path = self.snapshot_dir / f"{self.label}_seed{self.seed}_ep{episode}.pt"
        torch.save({
            "q_net": self.agent.q_net.state_dict(),
            "target_net": self.agent.target_net.state_dict(),
            "step_count": self.agent.step_count,
            "episode": episode,
            "eval_reward": eval_reward,
        }, path)
        return str(path)
