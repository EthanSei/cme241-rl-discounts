"""Tabular transition enumeration for Version C."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from discount_engine.core.demand import DemandInputs, logistic_purchase_probability
from discount_engine.core.params import MDPParams
from discount_engine.core.types import DiscreteState
from discount_engine.dp.discretization import (
    bucketize_churn_distribution,
    bucketize_memory_distribution,
    bucketize_recency_distribution,
    decode_state,
    is_terminal_state,
    resolve_churn_grid,
    terminal_state,
    validate_n_categories,
    validate_state_shape,
)


@dataclass(frozen=True)
class TransitionBranch:
    """One probabilistic branch in the DP transition distribution."""

    next_state: DiscreteState
    reward: float
    probability: float


def enumerate_transition_distribution(
    state: DiscreteState,
    action: int,
    params: MDPParams,
) -> list[TransitionBranch]:
    """Enumerate all transition branches for a state-action pair.

    For non-terminal states, this enumerates every purchase subset over categories.
    Churn branching is applied only on the zero-purchase subset.
    """
    n_categories = len(params.categories)
    churn_grid = resolve_churn_grid(params)
    validate_n_categories(n_categories)
    if not (0 <= action <= n_categories):
        raise ValueError(f"Invalid action {action}. Expected in [0, {n_categories}].")

    validate_state_shape(state=state, n_categories=n_categories, churn_grid=churn_grid)

    if is_terminal_state(state):
        return [TransitionBranch(next_state=state, reward=0.0, probability=1.0)]

    decoded = decode_state(state, churn_grid=churn_grid)
    buy_probs, effective_prices = _compute_buy_probs_and_prices(
        state=state,
        action=action,
        params=params,
        churn_grid=churn_grid,
    )

    outcomes: dict[tuple[DiscreteState, float], float] = {}

    # Enumerate purchase subsets with integer bitmasks.
    for mask in range(1 << n_categories):
        purchases = tuple(bool(mask & (1 << idx)) for idx in range(n_categories))
        prob = _purchase_subset_probability(purchases=purchases, buy_probs=buy_probs)
        if prob <= 0.0:
            continue

        reward = sum(
            effective_prices[idx] for idx, bought in enumerate(purchases) if bought
        )
        any_purchase = any(purchases)
        next_distribution = _next_non_terminal_distribution(
            state=state,
            action=action,
            purchases=purchases,
            params=params,
            churn_grid=churn_grid,
        )

        if any_purchase:
            for next_state, next_prob in next_distribution:
                key = (next_state, reward)
                outcomes[key] = outcomes.get(key, 0.0) + (prob * next_prob)
            continue

        churn_prob = decoded.churn_propensity
        stay_prob = prob * (1.0 - churn_prob)
        term_prob = prob * churn_prob

        if stay_prob > 0.0:
            for next_state, next_prob in next_distribution:
                key = (next_state, reward)
                outcomes[key] = outcomes.get(key, 0.0) + (stay_prob * next_prob)

        if term_prob > 0.0:
            terminal = terminal_state(n_categories=n_categories)
            key = (terminal, reward)
            outcomes[key] = outcomes.get(key, 0.0) + term_prob

    branches = [
        TransitionBranch(next_state=next_state, reward=reward, probability=prob)
        for (next_state, reward), prob in outcomes.items()
    ]
    _normalize_probabilities_in_place(branches)
    return branches


def _compute_buy_probs_and_prices(
    state: DiscreteState,
    action: int,
    params: MDPParams,
    churn_grid: tuple[float, ...],
) -> tuple[list[float], list[float]]:
    decoded = decode_state(state, churn_grid=churn_grid)
    buy_probs: list[float] = []
    effective_prices: list[float] = []

    for idx, category in enumerate(params.categories):
        promoted = action == (idx + 1)
        eff_price = category.price * (1.0 - params.delta) if promoted else category.price
        effective_prices.append(eff_price)

        deal_signal = category.price - eff_price
        p_buy = logistic_purchase_probability(
            DemandInputs(
                baseline_logit=category.beta_0,
                deal_signal=deal_signal,
                recency_value=decoded.purchase_recency[idx],
                memory_value=decoded.discount_memory[idx],
                beta_p=params.beta_p,
                beta_l=params.beta_l,
                beta_m=params.beta_m,
            )
        )
        buy_probs.append(p_buy)

    return buy_probs, effective_prices


def _purchase_subset_probability(
    purchases: tuple[bool, ...], buy_probs: list[float]
) -> float:
    prob = 1.0
    for idx, bought in enumerate(purchases):
        p_buy = buy_probs[idx]
        prob *= p_buy if bought else (1.0 - p_buy)
    return prob


def _next_non_terminal_distribution(
    state: DiscreteState,
    action: int,
    purchases: tuple[bool, ...],
    params: MDPParams,
    churn_grid: tuple[float, ...],
) -> tuple[tuple[DiscreteState, float], ...]:
    decoded = decode_state(state, churn_grid=churn_grid)
    current_churn = decoded.churn_propensity

    next_memory_dists: list[tuple[tuple[int, float], ...]] = []
    next_recency_dists: list[tuple[tuple[int, float], ...]] = []
    for idx in range(len(params.categories)):
        promoted = action == (idx + 1)
        promo_bump = params.delta if promoted else 0.0
        mem_val = (params.alpha * decoded.discount_memory[idx]) + promo_bump
        next_memory_dists.append(bucketize_memory_distribution(mem_val))

        if purchases[idx]:
            rec_val = 0.0
        else:
            rec_val = decoded.purchase_recency[idx] + 1.0
        next_recency_dists.append(bucketize_recency_distribution(rec_val))

    if any(purchases):
        churn_val = max(0.0, current_churn - params.eta)
    else:
        churn_val = min(1.0, current_churn + params.eta)
    churn_dist = bucketize_churn_distribution(churn_val, churn_grid=churn_grid)

    outcomes: dict[DiscreteState, float] = {}
    for churn_bucket, churn_prob in churn_dist:
        for memory_choices in product(*next_memory_dists):
            memory_buckets = tuple(choice[0] for choice in memory_choices)
            memory_prob = 1.0
            for _, p in memory_choices:
                memory_prob *= p

            for recency_choices in product(*next_recency_dists):
                recency_buckets = tuple(choice[0] for choice in recency_choices)
                recency_prob = 1.0
                for _, p in recency_choices:
                    recency_prob *= p

                next_state = DiscreteState(
                    churn_bucket=churn_bucket,
                    memory_buckets=memory_buckets,
                    recency_buckets=recency_buckets,
                )
                joint_prob = churn_prob * memory_prob * recency_prob
                outcomes[next_state] = outcomes.get(next_state, 0.0) + joint_prob

    return tuple(outcomes.items())


def _normalize_probabilities_in_place(branches: list[TransitionBranch]) -> None:
    total = sum(branch.probability for branch in branches)
    if total <= 0.0:
        raise ValueError("Transition distribution has non-positive total probability.")
    if abs(total - 1.0) <= 1e-12:
        return

    scale = 1.0 / total
    for idx, branch in enumerate(branches):
        branches[idx] = TransitionBranch(
            next_state=branch.next_state,
            reward=branch.reward,
            probability=branch.probability * scale,
        )
