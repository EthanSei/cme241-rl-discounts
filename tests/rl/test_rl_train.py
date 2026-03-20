"""Tests for Trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from discount_engine.rl.agent import DQNAgent
from discount_engine.rl.env import DiscountEnv
from discount_engine.rl.train import Trainer


@pytest.fixture
def trainer(small_env: DiscountEnv) -> Trainer:
    agent = DQNAgent(
        obs_dim=5,
        n_actions=3,
        hidden=32,
        buffer_size=500,
        batch_size=8,
        target_update_freq=50,
        eps_decay_steps=50,
        obs_scale=small_env.build_obs_scale(),
    )
    return Trainer(small_env, agent, seed=0, update_every=1)


class TestTrainer:
    """Trainer runs training loop and returns correct metrics."""

    def test_train_returns_expected_keys(self, trainer: Trainer) -> None:
        result = trainer.train(n_episodes=10, eval_interval=5, eval_episodes=3)
        assert result["train_rewards"].shape == (10,)
        assert result["eval_rewards"].shape == (2,)  # 10 // 5
        assert result["eval_points"].shape == (2,)
        assert result["eval_mean_q"].shape == (2,)
        assert result["eval_max_q"].shape == (2,)
        assert result["eval_mean_loss"].shape == (2,)
        assert isinstance(result["losses"], np.ndarray)
        assert result["action_counts"].shape == (3,)

    def test_eval_points_are_correct(self, trainer: Trainer) -> None:
        result = trainer.train(n_episodes=20, eval_interval=10, eval_episodes=2)
        np.testing.assert_array_equal(result["eval_points"], [10, 20])

    def test_snapshot_saves_files(self, trainer: Trainer, tmp_path: Path) -> None:
        trainer.snapshot_dir = tmp_path
        result = trainer.train(
            n_episodes=10, eval_interval=5, eval_episodes=2, snapshot_interval=5
        )
        assert len(result["snapshot_paths"]) == 2
        for p in result["snapshot_paths"]:
            assert Path(p).exists()


class TestConvergenceTracking:
    """Q-value and loss tracking at eval checkpoints."""

    def test_eval_q_values_are_finite(self, trainer: Trainer) -> None:
        result = trainer.train(n_episodes=10, eval_interval=5, eval_episodes=3)
        assert np.all(np.isfinite(result["eval_mean_q"]))
        assert np.all(np.isfinite(result["eval_max_q"]))
        assert np.all(result["eval_max_q"] >= result["eval_mean_q"])

    def test_eval_mean_loss_populated(self, trainer: Trainer) -> None:
        result = trainer.train(n_episodes=10, eval_interval=5, eval_episodes=3)
        # After some training, at least the second interval should have losses
        assert result["eval_mean_loss"].shape == (2,)
        assert np.all(np.isfinite(result["eval_mean_loss"]))
        # Second interval should have positive loss (buffer has enough samples)
        assert result["eval_mean_loss"][1] > 0


class TestUpdateEvery:
    """update_every gates gradient updates; step_count tracks env steps."""

    def test_update_every_reduces_updates(self, small_env: DiscountEnv) -> None:
        """With update_every=4, agent.update_count should be ~1/4 of step_count."""
        agent = DQNAgent(
            obs_dim=5, n_actions=3, hidden=32,
            buffer_size=500, batch_size=8,
            target_update_freq=50, eps_decay_steps=50,
            obs_scale=small_env.build_obs_scale(),
        )
        trainer = Trainer(small_env, agent, seed=0, update_every=4)
        trainer.train(n_episodes=20, eval_interval=20, eval_episodes=1)
        # step_count = total env steps, update_count = gradient steps
        assert agent.step_count > 0
        assert agent.update_count > 0
        assert agent.update_count <= agent.step_count // 4 + 1

    def test_step_count_decoupled_from_updates(
        self, small_env: DiscountEnv
    ) -> None:
        """step_count should advance every env step regardless of update_every."""
        agent = DQNAgent(
            obs_dim=5, n_actions=3, hidden=32,
            buffer_size=500, batch_size=8,
            target_update_freq=50, eps_decay_steps=50,
            obs_scale=small_env.build_obs_scale(),
        )
        # With update_every=100 (very high), updates rarely happen
        trainer = Trainer(small_env, agent, seed=0, update_every=100)
        trainer.train(n_episodes=5, eval_interval=5, eval_episodes=1)
        # step_count should still reflect total env steps
        assert agent.step_count > 10
        # epsilon should have decayed (driven by step_count, not update_count)
        assert agent.get_epsilon() < 1.0
