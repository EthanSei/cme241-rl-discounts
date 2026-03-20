"""Gymnasium environment interface for Version B."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any

eps = 1e-9


class DiscountEnv(gym.Env):
    """Product-level discount pricing environment.

    State: [churn, memory[N], recency[N]]  (1 + 2N dimensions)
    Action: 0=no_promo, 1..N=promote product i

    Deal signals use raw_deal_signal (matching calibration scale).
    The logistic model applies beta_p internally, so deal_signals must be
    raw signals — NOT pre-multiplied logit_bumps (that would double-count beta_p).

    V2: beta_p, beta_l, beta_m can be per-product arrays (length N) or scalars.
    Per-product values enable category-specific fatigue coefficients.
    """
    def __init__(
            self,
            product_params: dict, # prod_id -> {beta_0, raw_deal_signal, price, category}
            product_order: list[int], # ordered list of prod ids
            beta_p: float | np.ndarray, # deal coefficient (scalar or per-product)
            beta_l: float | np.ndarray, # recency coefficient (scalar or per-product)
            beta_m: float | np.ndarray, # memory/fatigue coefficient (scalar or per-product)
            alpha: float, # memory decay
            delta: float = 0.3, # discount depth (e.g. 30% off)
            gamma: float = 0.99, # time-reward discount rate
            c0: float = 0.05, # base churn propensity
            eta: float = 0.01, # base churn propensity
            max_steps: int = 200,
            churn_cost: float = 0.0,
            recency_sentinel: float = 52.0,
            randomize_init: bool = True, # for exploration
            init_memory_frac: float = 0.5, # max initial memory as fraction of cap
            decouple_memory_init: bool = False, # randomize memory even for sentinel-recency products
    ):
        super().__init__()
        self.products = product_order
        self.N = len(product_order)
        self.params = product_params
        self.beta_p = beta_p
        self.beta_l = beta_l
        self.beta_m = beta_m
        self.alpha = alpha
        self.delta = delta
        self.gamma_discount = gamma
        self.c0 = c0
        self.eta = eta
        self.max_steps = max_steps
        self.churn_cost = churn_cost
        self.recency_sentinel = recency_sentinel
        self.randomize_init = randomize_init
        self.init_memory_frac = init_memory_frac
        self.decouple_memory_init = decouple_memory_init

        # Spaces
        state_dim = 1 + 2 * self.N
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(state_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.N + 1)

        # Pre-extract product arrays for vectorized computation
        self._beta_0 = np.array([product_params[pid]["beta_0"] for pid in product_order])
        self._prices = np.array([product_params[pid]["price"] for pid in product_order])
        # Raw deal signals for BOTH purchase probability and memory update
        self._raw_deal_signal = np.array([
            product_params[pid]["raw_deal_signal"] for pid in product_order
        ])

        # Memory cap: max_signal / (1 - alpha) if alpha < 1, else fixed
        max_signal = float(self._raw_deal_signal.max() if len(self._raw_deal_signal) > 0 else 1.0)
        self._memory_cap = 6.0 if alpha >= 1.0 else min(6.0, max_signal / (1.0 - alpha + eps))

        self._state = None
        self._step_count = 0

    def _get_state(self) -> np.ndarray:
        churn, mem, rec = self._state
        return np.concatenate([[churn], mem, rec]).astype(np.float32)

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if self.randomize_init:
            # Randomize initial state: customers arrive at diverse engagement levels
            churn = self.np_random.uniform(self.c0, 0.15)
            # Each product: 30% chance of being known (low recency) vs cold (sentinel)
            recency = np.where(
                self.np_random.random(self.N) < 0.3,
                self.np_random.uniform(1, 10, size=self.N), # recently active (1-10 weeks)
                self.recency_sentinel, # never purchased
            )
            # Random memory for products with past deal exposure (or all, if decoupled)
            max_init_mem = self.init_memory_frac * self._memory_cap
            if self.decouple_memory_init:
                # Broader state coverage for training: all products get random memory
                memory = self.np_random.uniform(0, max_init_mem, size=self.N)
            else:
                memory = np.where(
                    recency < self.recency_sentinel,
                    self.np_random.uniform(0, max_init_mem, size=self.N),
                    0.0
                )
        else:
            # low c0 churn, no memory, never purchased
            churn = self.c0
            memory = np.zeros(self.N, dtype=np.float64)
            recency = np.full(self.N, self.recency_sentinel, dtype=np.float64)
        self._state = (float(churn), memory.astype(np.float64), recency.astype(np.float64))
        self._step_count = 0
        return self._get_state(), {}
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        churn, memory, recency = self._state
        self._step_count += 1

        eff_prices = np.copy(self._prices)
        deal_signals = np.zeros(self.N)
        if action > 0:
            promoted_product_id = action - 1 # (action 0 = no promo)
            eff_prices[promoted_product_id] *= (1.0 - self.delta)
            deal_signals[promoted_product_id] = self._raw_deal_signal[promoted_product_id]
        
        # Purchase probabilities
        # logit = beta_0 + beta_p * deal - beta_m * memory - beta_l * recency
        logits = (
            self._beta_0
            + self.beta_p * deal_signals
            - self.beta_m * memory
            - self.beta_l * recency
        )
        purchase_probs = self._sigmoid(logits)

        # Sample purchases
        purchases = self.np_random.random(self.N) < purchase_probs

        # Update churn
        any_purchase = purchases.any()
        if any_purchase:
            new_churn = max(0.0, churn - self.eta)
        else:
            new_churn = min(1.0, churn + self.eta)
        
        # Memory update
        memory_input = np.zeros(self.N)
        if action > 0:
            memory_input[action - 1] = self._raw_deal_signal[action - 1]
        new_memory = self.alpha * memory + memory_input
        new_memory = np.clip(new_memory, 0.0, self._memory_cap)

        # Recency update (0.0 on purchase, consistent with DP)
        new_recency = np.where(purchases, 0.0, recency + 1.0)

        # Reward: revenue from purchases (discounted if promoted)
        revenue = np.sum(eff_prices * purchases)
        reward = revenue - self.churn_cost * churn

        # Churn termination only on no-purchase steps (consistent with DP)
        if any_purchase:
            terminated = False
        else:
            terminated = bool(self.np_random.random() < churn)
        truncated = self._step_count >= self.max_steps

        self._state = (new_churn, new_memory, new_recency)
        return self._get_state(), float(reward), terminated, truncated, {
            "purchases": purchases.copy(),
            "purchase_probs": purchase_probs.copy(),
            "revenue": revenue,
            "churn": new_churn,
        }
    
    def make_obs(
        self,
        churn: float,
        memory: np.ndarray | None = None,
        recency: np.ndarray | None = None,
    ) -> np.ndarray:
        """Construct an observation vector from components."""
        mem = np.zeros(self.N) if memory is None else np.asarray(memory)
        rec = np.full(self.N, self.recency_sentinel) if recency is None else np.asarray(recency)
        return np.concatenate([[churn], mem, rec]).astype(np.float32)

    def build_obs_scale(self) -> np.ndarray:
        """Builds the normalization factor for obs: [churn(1), memory(N), recency(N)]"""

        scales = np.ones(1 + 2 * self.N, dtype=np.float32)
        scales[1:1+self.N] = self._memory_cap
        scales[1+self.N:] = self.recency_sentinel
        return scales

