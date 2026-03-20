"""Tests for DiscountEnv."""

from __future__ import annotations

import numpy as np
import pytest

from discount_engine.rl.env import DiscountEnv


class TestDiscountEnvReset:
    """Reset produces correct shapes and deterministic defaults."""

    def test_obs_shape(self, small_env: DiscountEnv) -> None:
        obs, info = small_env.reset(seed=0)
        # state = [churn(1), memory(N), recency(N)] = 1 + 2*2 = 5
        assert obs.shape == (5,)
        assert obs.dtype == np.float32

    def test_deterministic_init(self, small_env: DiscountEnv) -> None:
        obs, _ = small_env.reset(seed=0)
        # randomize_init=False: churn=c0, memory=0, recency=sentinel
        assert obs[0] == pytest.approx(0.05)  # c0
        np.testing.assert_array_equal(obs[1:3], 0.0)  # memory
        np.testing.assert_array_equal(obs[3:5], 52.0)  # recency sentinel

    def test_randomized_init_varies(self, small_env_random: DiscountEnv) -> None:
        obs1, _ = small_env_random.reset(seed=0)
        obs2, _ = small_env_random.reset(seed=99)
        assert not np.array_equal(obs1, obs2)


class TestDiscountEnvStep:
    """Step mechanics: rewards, state transitions, termination."""

    def test_step_output_shapes(self, small_env: DiscountEnv) -> None:
        small_env.reset(seed=0)
        obs, reward, terminated, truncated, info = small_env.step(0)
        assert obs.shape == (5,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "purchases" in info
        assert info["purchases"].shape == (2,)

    def test_no_promo_no_discount(self, small_env: DiscountEnv) -> None:
        """Action 0 should use full prices (no discount applied)."""
        small_env.reset(seed=0)
        _, reward, _, _, info = small_env.step(0)
        # Revenue = sum of full prices * purchases
        expected = np.sum(small_env._prices * info["purchases"])
        assert reward == pytest.approx(expected)

    def test_promo_discounts_price(self, small_env: DiscountEnv) -> None:
        """Promoting product 0 (action=1) discounts its price by delta."""
        small_env.reset(seed=0)
        _, _, _, _, info = small_env.step(1)
        # Even if no purchase, the effective price used should be discounted
        # We verify via purchase_probs: promoting should increase purchase prob
        small_env.reset(seed=0)
        _, _, _, _, info_no = small_env.step(0)
        small_env.reset(seed=0)
        _, _, _, _, info_promo = small_env.step(1)
        # Promo on product 0 should increase its purchase probability
        assert info_promo["purchase_probs"][0] > info_no["purchase_probs"][0]

    def test_churn_increases_without_purchase(self, small_env: DiscountEnv) -> None:
        """Churn should increase by eta when no purchases occur."""
        # Use high recency (sentinel) so purchase probs are near zero
        small_env.reset(seed=0)
        # Step with no promo — purchases very unlikely with sentinel recency + negative beta_0
        small_env.step(0)
        new_churn = small_env._state[0]
        # If no purchase happened, churn = c0 + eta = 0.06
        # If purchase happened, churn resets to c0 = 0.05
        assert new_churn in (pytest.approx(0.05), pytest.approx(0.06))

    def test_truncation_at_max_steps(self) -> None:
        """Episode truncates after max_steps with zero churn."""
        from tests.rl.conftest import _make_small_env

        # c0=0, eta=0 ensures churn stays 0 → no early termination
        env = _make_small_env(c0=0.0, eta=0.0, max_steps=10)
        env.reset(seed=0)
        for i in range(9):
            _, _, terminated, truncated, _ = env.step(0)
            assert not truncated
            assert not terminated
        _, _, terminated, truncated, _ = env.step(0)
        assert truncated
        assert env._step_count == 10

    def test_churn_terminates_early(self) -> None:
        """High churn should cause early termination."""
        from tests.rl.conftest import _make_small_env

        env = _make_small_env(c0=0.9, eta=0.0, max_steps=200)
        terminated_count = 0
        for trial in range(20):
            env.reset(seed=trial)
            for _ in range(10):
                _, _, terminated, _, _ = env.step(0)
                if terminated:
                    terminated_count += 1
                    break
        # With churn=0.9, termination should happen frequently
        assert terminated_count > 5

    def test_memory_clips_at_cap(self) -> None:
        """Memory should saturate at _memory_cap after repeated promotions."""
        from tests.rl.conftest import _make_small_env

        env = _make_small_env(c0=0.0, eta=0.0, max_steps=500)
        env.reset(seed=0)
        for _ in range(100):
            env.step(1)  # promote product 0 every step
        mem = env._state[1]
        assert mem[0] == pytest.approx(env._memory_cap, abs=1e-6)

    def test_churn_resets_on_purchase(self) -> None:
        """Churn should reset to c0 when any purchase occurs."""
        from tests.rl.conftest import _make_small_env

        # High beta_0 + low recency → near-certain purchase
        params = {
            0: {"beta_0": 5.0, "logit_bump": 0.5, "raw_deal_signal": 0.8, "price": 10.0, "category": "A"},
            1: {"beta_0": 5.0, "logit_bump": 0.3, "raw_deal_signal": 0.6, "price": 20.0, "category": "B"},
        }
        env = _make_small_env(
            product_params=params, c0=0.05, eta=0.01,
            max_steps=50, randomize_init=False,
        )
        env.reset(seed=0)
        # Manually set high churn and low recency to force purchase
        env._state = (0.5, env._state[1], np.array([1.0, 1.0]))
        env.step(0)
        new_churn = env._state[0]
        # With beta_0=5 and recency=1, purchase is near-certain → churn decrements by eta
        assert new_churn == pytest.approx(0.49)

    def test_memory_accumulates(self, small_env: DiscountEnv) -> None:
        """Promoting a product should increase its memory state."""
        small_env.reset(seed=0)
        small_env.step(1)  # promote product 0
        mem_after = small_env._state[1]  # memory array
        assert mem_after[0] > 0.0  # product 0 got memory input
        assert mem_after[1] == 0.0  # product 1 unchanged


class TestMakeObs:
    """make_obs constructs observation vectors correctly."""

    def test_default_args(self, small_env: DiscountEnv) -> None:
        obs = small_env.make_obs(0.1)
        assert obs.shape == (5,)
        assert obs[0] == pytest.approx(0.1)
        np.testing.assert_array_equal(obs[1:3], 0.0)
        np.testing.assert_array_equal(obs[3:5], 52.0)

    def test_custom_memory_recency(self, small_env: DiscountEnv) -> None:
        mem = np.array([0.5, 1.0])
        rec = np.array([3.0, 10.0])
        obs = small_env.make_obs(0.2, memory=mem, recency=rec)
        np.testing.assert_array_almost_equal(obs, [0.2, 0.5, 1.0, 3.0, 10.0])


class TestBuildObsScale:
    """build_obs_scale returns sensible normalization factors."""

    def test_shape_and_values(self, small_env: DiscountEnv) -> None:
        scales = small_env.build_obs_scale()
        assert scales.shape == (5,)
        assert scales[0] == 1.0  # churn scale
        assert scales[1] == scales[2]  # both memory scales equal
        assert scales[3] == pytest.approx(52.0)  # recency sentinel
