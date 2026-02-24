"""Value-iteration solver for Version C tabular DP."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from discount_engine.core.params import MDPParams
from discount_engine.core.types import DiscreteState
from discount_engine.dp.discretization import (
    enumerate_all_states,
    is_terminal_state,
    resolve_churn_grid,
    validate_n_categories,
)
from discount_engine.dp.transitions import enumerate_transition_distribution

_TIE_TOL = 1e-12


@dataclass(frozen=True)
class ValueIterationConfig:
    """Configuration for tabular value iteration."""

    gamma: float
    epsilon: float
    max_iters: int
    show_progress: bool = False
    progress_desc: str = "Value Iteration"

    def validate(self) -> None:
        if not (0.0 < self.gamma < 1.0):
            raise ValueError("gamma must be in (0, 1).")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if self.max_iters <= 0:
            raise ValueError("max_iters must be positive.")


@dataclass(frozen=True)
class ValueIterationResult:
    """Outputs from value-iteration solve."""

    values: dict[DiscreteState, float]
    policy: dict[DiscreteState, int]
    q_values: dict[DiscreteState, dict[int, float]]
    iterations: int
    converged: bool
    max_delta_history: tuple[float, ...]
    final_bellman_residual: float


@dataclass(frozen=True)
class _ActionKernel:
    """Precomputed transition kernel for one state-action pair."""

    next_indices: np.ndarray
    rewards: np.ndarray
    probabilities: np.ndarray


def bellman_action_value(
    state: DiscreteState,
    action: int,
    values: dict[DiscreteState, float],
    params: MDPParams,
    gamma: float,
) -> float:
    """Compute Q(s, a) from a fixed value function."""
    branches = enumerate_transition_distribution(state=state, action=action, params=params)
    total = 0.0
    for branch in branches:
        total += branch.probability * (
            branch.reward + gamma * values.get(branch.next_state, 0.0)
        )
    return total


def bellman_backup(
    state: DiscreteState,
    values: dict[DiscreteState, float],
    params: MDPParams,
    gamma: float,
) -> tuple[float, int, dict[int, float]]:
    """Compute Bellman-optimal value and argmax action at one state."""
    n_categories = len(params.categories)
    if is_terminal_state(state):
        q_values = {action: 0.0 for action in range(n_categories + 1)}
        return 0.0, 0, q_values

    best_action = 0
    best_value = float("-inf")
    q_values: dict[int, float] = {}
    for action in range(n_categories + 1):
        q_val = bellman_action_value(
            state=state,
            action=action,
            values=values,
            params=params,
            gamma=gamma,
        )
        q_values[action] = q_val
        if q_val > best_value + _TIE_TOL:
            best_value = q_val
            best_action = action
        elif abs(q_val - best_value) <= _TIE_TOL and action < best_action:
            best_action = action

    return best_value, best_action, q_values


def solve_value_iteration(
    params: MDPParams,
    config: ValueIterationConfig,
) -> ValueIterationResult:
    """Compute optimal values/policy for Version C using exact tabular DP."""
    config.validate()
    n_categories = len(params.categories)
    validate_n_categories(n_categories)
    churn_grid = resolve_churn_grid(params)

    states = enumerate_all_states(n_categories=n_categories, churn_grid=churn_grid)
    n_states = len(states)
    n_actions = n_categories + 1
    state_to_index = {state: idx for idx, state in enumerate(states)}
    transition_cache = _build_transition_cache(
        states=states,
        state_to_index=state_to_index,
        params=params,
    )
    terminal_mask = np.array([is_terminal_state(state) for state in states], dtype=bool)

    values_vec = np.zeros(n_states, dtype=np.float64)
    policy_vec = np.zeros(n_states, dtype=np.int64)
    max_delta_history: list[float] = []
    converged = False
    iterations = 0

    iterator = range(1, config.max_iters + 1)
    show_tqdm = config.show_progress
    progress = iterator
    if show_tqdm:
        # Import tqdm lazily to avoid notebook-side effects when progress is disabled.
        from tqdm.auto import tqdm

        progress = tqdm(
            iterator,
            desc=config.progress_desc,
            dynamic_ncols=True,
            leave=False,
        )

    for iteration in progress:
        next_values = np.zeros_like(values_vec)
        next_policy = np.zeros_like(policy_vec)
        max_delta = 0.0

        for state_idx in range(n_states):
            if terminal_mask[state_idx]:
                continue

            best_action = 0
            best_value = -math.inf
            state_kernels = transition_cache[state_idx]
            for action in range(n_actions):
                q_val = _q_from_kernel(
                    kernel=state_kernels[action],
                    values=values_vec,
                    gamma=config.gamma,
                )
                if q_val > best_value + _TIE_TOL:
                    best_value = q_val
                    best_action = action
                elif abs(q_val - best_value) <= _TIE_TOL and action < best_action:
                    best_action = action

            next_values[state_idx] = best_value
            next_policy[state_idx] = best_action
            max_delta = max(max_delta, abs(best_value - values_vec[state_idx]))

        values_vec = next_values
        policy_vec = next_policy
        max_delta_history.append(max_delta)
        iterations = iteration

        if show_tqdm:
            progress.set_postfix({"delta": f"{max_delta:.3e}"}, refresh=False)

        if max_delta < config.epsilon:
            converged = True
            break

    if show_tqdm:
        progress.close()

    values = {state: float(values_vec[idx]) for idx, state in enumerate(states)}
    policy = {state: int(policy_vec[idx]) for idx, state in enumerate(states)}

    q_values: dict[DiscreteState, dict[int, float]] = {}
    final_bellman_residual = 0.0
    for state_idx, state in enumerate(states):
        q_map: dict[int, float] = {}
        state_best = -math.inf
        state_kernels = transition_cache[state_idx]
        for action in range(n_actions):
            q_val = _q_from_kernel(
                kernel=state_kernels[action],
                values=values_vec,
                gamma=config.gamma,
            )
            q_map[action] = q_val
            state_best = max(state_best, q_val)
        q_values[state] = q_map
        final_bellman_residual = max(
            final_bellman_residual,
            abs(values[state] - state_best),
        )

    return ValueIterationResult(
        values=values,
        policy=policy,
        q_values=q_values,
        iterations=iterations,
        converged=converged,
        max_delta_history=tuple(max_delta_history),
        final_bellman_residual=final_bellman_residual,
    )


def evaluate_policy(
    policy: dict[DiscreteState, int],
    params: MDPParams,
    config: ValueIterationConfig,
) -> dict[DiscreteState, float]:
    """Compute V^π(s) for a fixed policy via iterative policy evaluation.

    Instead of max_a Q(s,a), this computes Q(s, π(s)) at each step,
    giving the true long-run value of always following the given policy.

    Args:
        policy: Mapping from state to fixed action.
        params: Calibrated MDP parameters.
        config: Solver configuration (gamma, epsilon, max_iters).

    Returns:
        Mapping from each state to its converged V^π value.
    """
    config.validate()
    n_categories = len(params.categories)
    validate_n_categories(n_categories)
    churn_grid = resolve_churn_grid(params)

    states = enumerate_all_states(n_categories=n_categories, churn_grid=churn_grid)
    n_states = len(states)
    _validate_policy_map(
        policy=policy,
        states=states,
        n_categories=n_categories,
    )
    state_to_index = {state: idx for idx, state in enumerate(states)}
    transition_cache = _build_transition_cache(
        states=states,
        state_to_index=state_to_index,
        params=params,
    )
    terminal_mask = np.array([is_terminal_state(state) for state in states], dtype=bool)

    policy_actions = np.array([policy[state] for state in states], dtype=np.int64)

    values_vec = np.zeros(n_states, dtype=np.float64)

    converged = False
    final_delta = math.inf

    for _ in range(1, config.max_iters + 1):
        next_values = np.zeros_like(values_vec)
        max_delta = 0.0

        for state_idx in range(n_states):
            if terminal_mask[state_idx]:
                continue
            action = int(policy_actions[state_idx])
            q_val = _q_from_kernel(
                kernel=transition_cache[state_idx][action],
                values=values_vec,
                gamma=config.gamma,
            )
            next_values[state_idx] = q_val
            max_delta = max(max_delta, abs(q_val - values_vec[state_idx]))

        values_vec = next_values
        final_delta = max_delta
        if max_delta < config.epsilon:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            "Policy evaluation did not converge within max_iters "
            f"(max_iters={config.max_iters}, epsilon={config.epsilon:.3e}, "
            f"final_delta={final_delta:.3e})."
        )

    return {state: float(values_vec[idx]) for idx, state in enumerate(states)}


def _build_transition_cache(
    *,
    states: tuple[DiscreteState, ...],
    state_to_index: dict[DiscreteState, int],
    params: MDPParams,
) -> list[list[_ActionKernel]]:
    n_categories = len(params.categories)
    n_actions = n_categories + 1
    cache: list[list[_ActionKernel]] = []

    for state in states:
        row: list[_ActionKernel] = []
        for action in range(n_actions):
            branches = enumerate_transition_distribution(
                state=state,
                action=action,
                params=params,
            )
            next_indices = np.fromiter(
                (state_to_index[branch.next_state] for branch in branches),
                dtype=np.int64,
                count=len(branches),
            )
            rewards = np.fromiter(
                (branch.reward for branch in branches),
                dtype=np.float64,
                count=len(branches),
            )
            probabilities = np.fromiter(
                (branch.probability for branch in branches),
                dtype=np.float64,
                count=len(branches),
            )
            row.append(
                _ActionKernel(
                    next_indices=next_indices,
                    rewards=rewards,
                    probabilities=probabilities,
                )
            )
        cache.append(row)
    return cache


def _validate_policy_map(
    *,
    policy: dict[DiscreteState, int],
    states: tuple[DiscreteState, ...],
    n_categories: int,
) -> None:
    state_set = set(states)
    missing = [state for state in states if state not in policy]
    if missing:
        sample = missing[0]
        raise ValueError(
            f"Policy is missing {len(missing)} required state(s); "
            f"example missing state: {sample}"
        )

    extra = [state for state in policy if state not in state_set]
    if extra:
        sample = extra[0]
        raise ValueError(
            f"Policy contains {len(extra)} unknown state(s); "
            f"example unknown state: {sample}"
        )

    invalid_actions = [
        (state, action)
        for state, action in policy.items()
        if action < 0 or action > n_categories
    ]
    if invalid_actions:
        state, action = invalid_actions[0]
        raise ValueError(
            f"Invalid action {action} for state {state}; "
            f"expected action in [0, {n_categories}]"
        )


def _q_from_kernel(
    *,
    kernel: _ActionKernel,
    values: np.ndarray,
    gamma: float,
) -> float:
    discounted_next = gamma * values[kernel.next_indices]
    return float(np.dot(kernel.probabilities, kernel.rewards + discounted_next))
