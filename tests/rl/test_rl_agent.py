"""Tests for DQNAgent."""

from __future__ import annotations

import numpy as np
import pytest

from discount_engine.rl.agent import DQNAgent
from discount_engine.rl.env import DiscountEnv


@pytest.fixture
def agent(small_env: DiscountEnv) -> DQNAgent:
    """Small DQN agent matching the 2-product env."""
    obs_scale = small_env.build_obs_scale()
    return DQNAgent(
        obs_dim=5,
        n_actions=3,  # no_promo + 2 products
        hidden=32,
        buffer_size=500,
        batch_size=8,
        target_update_freq=50,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=100,
        obs_scale=obs_scale,
    )


class TestDQNAgent:
    """Core agent functionality: Q-values, action selection, learning."""

    def test_get_q_values_shape(self, agent: DQNAgent, small_env: DiscountEnv) -> None:
        obs, _ = small_env.reset(seed=0)
        q = agent.get_q_values(obs)
        assert q.shape == (3,)
        assert q.dtype == np.float32

    def test_greedy_action_is_argmax(self, agent: DQNAgent, small_env: DiscountEnv) -> None:
        obs, _ = small_env.reset(seed=0)
        q = agent.get_q_values(obs)
        action = agent.select_action(obs, greedy=True)
        assert action == int(q.argmax())

    def test_epsilon_decays(self, agent: DQNAgent) -> None:
        assert agent.get_epsilon() == 1.0
        agent.step_count = 50
        eps_mid = agent.get_epsilon()
        assert 0.05 < eps_mid < 1.0
        agent.step_count = 200  # past decay steps
        assert agent.get_epsilon() == pytest.approx(0.05)

    def test_store_and_update_returns_loss(self, agent: DQNAgent, small_env: DiscountEnv) -> None:
        """After enough transitions, update() should return a finite loss."""
        obs, _ = small_env.reset(seed=0)
        for _ in range(20):
            action = small_env.action_space.sample()
            next_obs, reward, term, trunc, _ = small_env.step(action)
            agent.store(obs, action, reward, next_obs, float(term or trunc))
            if term or trunc:
                obs, _ = small_env.reset(seed=0)
            else:
                obs = next_obs

        loss = agent.update()
        assert loss is not None
        assert np.isfinite(loss)

    def test_update_returns_none_when_buffer_small(self, agent: DQNAgent) -> None:
        """update() returns None when buffer has fewer samples than batch_size."""
        assert agent.update() is None

    def test_exploration_at_full_epsilon(self, agent: DQNAgent, small_env: DiscountEnv) -> None:
        """With epsilon=1.0, select_action(greedy=False) should explore randomly."""
        obs, _ = small_env.reset(seed=0)
        assert agent.get_epsilon() == 1.0  # step_count=0
        actions = [agent.select_action(obs, greedy=False) for _ in range(100)]
        unique = set(actions)
        # With 3 actions and 100 samples at epsilon=1.0, all actions should appear
        assert len(unique) >= 2

    def test_tick_advances_step_count(self, agent: DQNAgent) -> None:
        """tick() should increment step_count and decay epsilon."""
        assert agent.step_count == 0
        for _ in range(50):
            agent.tick()
        assert agent.step_count == 50
        assert agent.get_epsilon() < 1.0

    def test_obs_scale_zero_guard(self, small_env: DiscountEnv) -> None:
        """obs_scale with zeros should not produce NaN/Inf Q-values."""
        zero_scale = np.array([0.0, 1.0, 0.0, 5.0, 0.0])
        agent = DQNAgent(
            obs_dim=5, n_actions=3, hidden=32,
            buffer_size=500, batch_size=8,
            obs_scale=zero_scale,
        )
        obs, _ = small_env.reset(seed=0)
        q = agent.get_q_values(obs)
        assert np.all(np.isfinite(q))

    def test_reward_scaling_stores_scaled(self, small_env: DiscountEnv) -> None:
        """With reward_scale=10, buffer should store reward/10."""
        agent = DQNAgent(
            obs_dim=5, n_actions=3, hidden=32,
            buffer_size=500, batch_size=8,
            obs_scale=small_env.build_obs_scale(),
            reward_scale=10.0,
        )
        obs, _ = small_env.reset(seed=0)
        next_obs, reward, _, _, _ = small_env.step(1)
        agent.store(obs, 1, reward, next_obs, 0.0)
        # Buffer should contain reward / 10
        stored = agent.buffer.rewards[0]
        assert stored == pytest.approx(reward / 10.0, rel=1e-5)

    def test_reward_scaling_produces_finite_loss(self, small_env: DiscountEnv) -> None:
        """Agent with reward_scale=10 should produce finite loss after training."""
        agent = DQNAgent(
            obs_dim=5, n_actions=3, hidden=32,
            buffer_size=500, batch_size=8,
            target_update_freq=50,
            obs_scale=small_env.build_obs_scale(),
            reward_scale=10.0,
            grad_clip=1.0,
        )
        obs, _ = small_env.reset(seed=0)
        for _ in range(20):
            action = small_env.action_space.sample()
            next_obs, reward, term, trunc, _ = small_env.step(action)
            agent.store(obs, action, reward, next_obs, float(term or trunc))
            if term or trunc:
                obs, _ = small_env.reset(seed=0)
            else:
                obs = next_obs
        loss = agent.update()
        assert loss is not None
        assert np.isfinite(loss)

    def test_grad_clip_limits_gradients(self, small_env: DiscountEnv) -> None:
        """grad_clip should limit gradient norms during update()."""
        agent = DQNAgent(
            obs_dim=5, n_actions=3, hidden=32,
            buffer_size=500, batch_size=8,
            obs_scale=small_env.build_obs_scale(),
            grad_clip=0.01,  # very tight clip
        )
        obs, _ = small_env.reset(seed=0)
        for _ in range(20):
            action = small_env.action_space.sample()
            next_obs, reward, term, trunc, _ = small_env.step(action)
            agent.store(obs, action, reward, next_obs, float(term or trunc))
            if term or trunc:
                obs, _ = small_env.reset(seed=0)
            else:
                obs = next_obs
        agent.update()
        # After update with tight clip, all param grads should be clipped
        for p in agent.q_net.parameters():
            if p.grad is not None:
                assert p.grad.norm().item() <= 0.01 + 1e-6

    def test_reward_scale_zero_raises(self) -> None:
        """reward_scale=0 should raise ValueError."""
        with pytest.raises(ValueError, match="reward_scale"):
            DQNAgent(obs_dim=5, n_actions=3, reward_scale=0.0)

    def test_grad_clip_zero_raises(self) -> None:
        """grad_clip=0 should raise ValueError."""
        with pytest.raises(ValueError, match="grad_clip"):
            DQNAgent(obs_dim=5, n_actions=3, grad_clip=0.0)
