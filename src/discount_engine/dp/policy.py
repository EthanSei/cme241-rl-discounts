"""Policy extraction and reporting helpers for Version C DP."""

from __future__ import annotations

from collections import Counter
from statistics import mean

from discount_engine.core.params import MDPParams
from discount_engine.core.types import DiscreteState
from discount_engine.dp.discretization import (
    enumerate_all_states,
    is_terminal_state,
    resolve_churn_grid,
    resolve_churn_labels,
    validate_n_categories,
)
from discount_engine.dp.value_iteration import bellman_backup


def extract_greedy_policy(
    values: dict[DiscreteState, float],
    params: MDPParams,
    gamma: float,
) -> tuple[dict[DiscreteState, int], dict[DiscreteState, dict[int, float]]]:
    """Extract deterministic greedy policy against a value function."""
    n_categories = len(params.categories)
    validate_n_categories(n_categories)
    states = enumerate_all_states(
        n_categories=n_categories,
        churn_grid=resolve_churn_grid(params),
    )

    policy: dict[DiscreteState, int] = {}
    q_values: dict[DiscreteState, dict[int, float]] = {}
    for state in states:
        _, action, q_map = bellman_backup(
            state=state,
            values=values,
            params=params,
            gamma=gamma,
        )
        policy[state] = action
        q_values[state] = q_map
    return policy, q_values


def state_to_id(state: DiscreteState) -> str:
    """Encode a discrete state as a stable string identifier."""
    if is_terminal_state(state):
        return "terminal"
    memory = ",".join(str(v) for v in state.memory_buckets)
    recency = ",".join(str(v) for v in state.recency_buckets)
    return f"c:{state.churn_bucket}|m:{memory}|l:{recency}"


def state_from_id(state_id: str) -> DiscreteState:
    """Decode a state identifier emitted by :func:`state_to_id`."""
    if state_id == "terminal":
        return DiscreteState(churn_bucket=-1, memory_buckets=(), recency_buckets=())

    parts = state_id.split("|")
    if len(parts) != 3:
        raise ValueError(f"Invalid state id: {state_id}")
    churn_bucket = int(parts[0].split(":")[1])
    memory_raw = parts[1].split(":")[1]
    recency_raw = parts[2].split(":")[1]
    memory = tuple(int(x) for x in memory_raw.split(",") if x != "")
    recency = tuple(int(x) for x in recency_raw.split(",") if x != "")
    return DiscreteState(
        churn_bucket=churn_bucket,
        memory_buckets=memory,
        recency_buckets=recency,
    )


def canonicalize_loaded_state(
    loaded_state: DiscreteState, n_categories: int
) -> DiscreteState:
    """Normalize parsed state ids to canonical shapes for map lookups."""
    if loaded_state.churn_bucket == -1:
        return DiscreteState(
            churn_bucket=-1,
            memory_buckets=(0,) * n_categories,
            recency_buckets=(0,) * n_categories,
        )
    return loaded_state


def policy_rows(
    policy: dict[DiscreteState, int],
    values: dict[DiscreteState, float],
    *,
    params: MDPParams | None = None,
) -> list[dict[str, object]]:
    """Build flat tabular rows for policy/value inspection."""
    churn_labels = resolve_churn_labels(params)
    rows: list[dict[str, object]] = []
    for state in sorted(policy, key=_state_sort_key):
        action = policy[state]
        value = values.get(state, 0.0)
        terminal = is_terminal_state(state)
        memory = tuple(state.memory_buckets)
        recency = tuple(state.recency_buckets)
        rows.append(
            {
                "state_id": state_to_id(state),
                "is_terminal": terminal,
                "churn_bucket": state.churn_bucket,
                "churn_label": _churn_label_for_bucket(state.churn_bucket, churn_labels),
                "memory_buckets": ",".join(str(v) for v in memory),
                "recency_buckets": ",".join(str(v) for v in recency),
                "memory_mean_bucket": _safe_mean(memory),
                "recency_mean_bucket": _safe_mean(recency),
                "action": action,
                "value": value,
            }
        )
    return rows


def summarize_policy_by_cluster(rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarize policy behavior across intuitive state clusters."""
    live_rows = [row for row in rows if not bool(row["is_terminal"])]
    if not live_rows:
        return {
            "n_live_states": 0,
            "action_histogram": {},
            "no_promotion_rate": 0.0,
            "promotion_rate": 0.0,
            "by_churn_bucket": {},
        }

    action_counter = Counter(int(row["action"]) for row in live_rows)
    n_live = len(live_rows)
    no_promo = action_counter.get(0, 0)

    by_churn: dict[str, dict[str, object]] = {}
    churn_values = sorted(set(int(row["churn_bucket"]) for row in live_rows))
    for churn_bucket in churn_values:
        subset = [row for row in live_rows if int(row["churn_bucket"]) == churn_bucket]
        sub_counter = Counter(int(row["action"]) for row in subset)
        by_churn[str(churn_bucket)] = {
            "count": len(subset),
            "action_histogram": dict(sorted(sub_counter.items())),
            "promotion_rate": _promotion_rate(subset),
            "mean_value": float(mean(float(r["value"]) for r in subset)),
        }

    return {
        "n_live_states": n_live,
        "action_histogram": dict(sorted(action_counter.items())),
        "no_promotion_rate": no_promo / n_live,
        "promotion_rate": 1.0 - (no_promo / n_live),
        "by_churn_bucket": by_churn,
    }


def build_evaluation_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """Build report-oriented summary payload from policy rows."""
    live_rows = [row for row in rows if not bool(row["is_terminal"])]
    live_values = [float(row["value"]) for row in live_rows]

    if live_rows:
        value_summary = {
            "min": min(live_values),
            "max": max(live_values),
            "mean": float(sum(live_values) / len(live_values)),
        }
    else:
        value_summary = {"min": 0.0, "max": 0.0, "mean": 0.0}

    top_states = sorted(live_rows, key=lambda row: float(row["value"]), reverse=True)[:10]
    bottom_states = sorted(live_rows, key=lambda row: float(row["value"]))[:10]

    return {
        "policy_cluster_summary": summarize_policy_by_cluster(rows),
        "value_summary": value_summary,
        "top_states_by_value": top_states,
        "bottom_states_by_value": bottom_states,
    }


def _state_sort_key(state: DiscreteState) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    if is_terminal_state(state):
        return (99, (), ())
    return (state.churn_bucket, state.memory_buckets, state.recency_buckets)


def _safe_mean(values: tuple[int, ...]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _promotion_rate(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    promoted = sum(1 for row in rows if int(row["action"]) != 0)
    return promoted / len(rows)


def _churn_label_for_bucket(churn_bucket: int, labels: tuple[str, ...]) -> str:
    if churn_bucket < 0:
        return "Terminal"
    if churn_bucket >= len(labels):
        return f"Churn Segment {churn_bucket}"
    return labels[churn_bucket]
