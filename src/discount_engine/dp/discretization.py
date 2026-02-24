"""Continuous-to-discrete utilities for Version C DP state handling."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
import math
from typing import Iterable, Iterator

from discount_engine.core.params import MDPParams
from discount_engine.core.types import DiscreteState

# Version C default bucket grids (canonical weekly-resolution values).
# These are aligned with configs/dp/solver.yaml and the final Phase 2 run.
CHURN_GRID: tuple[float, ...] = (0.05, 0.50)
MEMORY_GRID: tuple[float, ...] = (0.0, 0.9, 2.0)
RECENCY_GRID: tuple[float, ...] = (1.0, 4.0)

# Global Version C feasibility bound.
MAX_DP_CATEGORIES = 5

# A dedicated churn bucket index marks the absorbing churn state.
TERMINAL_CHURN_BUCKET = -1


@dataclass(frozen=True)
class DecodedDiscreteState:
    """Continuous-valued interpretation of a discretized DP state."""

    churn_propensity: float
    discount_memory: tuple[float, ...]
    purchase_recency: tuple[float, ...]


def configure_bucket_grids(
    *,
    memory_grid: Iterable[float] | None = None,
    recency_grid: Iterable[float] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Configure module-level memory/recency grids used by DP discretization."""
    global MEMORY_GRID, RECENCY_GRID

    if memory_grid is not None:
        MEMORY_GRID = _normalize_bucket_grid(
            values=memory_grid,
            name="memory_grid",
            expected_len=3,
        )
    if recency_grid is not None:
        RECENCY_GRID = _normalize_bucket_grid(
            values=recency_grid,
            name="recency_grid",
            expected_len=2,
        )
    return MEMORY_GRID, RECENCY_GRID


@contextmanager
def temporary_bucket_grids(
    *,
    memory_grid: Iterable[float] | None = None,
    recency_grid: Iterable[float] | None = None,
) -> Iterator[tuple[tuple[float, ...], tuple[float, ...]]]:
    """Temporarily apply bucket grids and restore previous values on exit."""
    global MEMORY_GRID, RECENCY_GRID

    previous_memory = MEMORY_GRID
    previous_recency = RECENCY_GRID
    try:
        resolved = configure_bucket_grids(
            memory_grid=memory_grid,
            recency_grid=recency_grid,
        )
        yield resolved
    finally:
        MEMORY_GRID = previous_memory
        RECENCY_GRID = previous_recency


def resolve_churn_grid(params: MDPParams | None = None) -> tuple[float, ...]:
    """Resolve churn grid from params metadata, falling back to legacy defaults."""
    if params is None:
        return CHURN_GRID

    metadata = params.metadata if isinstance(params.metadata, dict) else {}
    raw_grid: object | None = None

    bucketing = metadata.get("churn_bucketing")
    if isinstance(bucketing, dict):
        raw_grid = bucketing.get("grid")
    if raw_grid is None:
        raw_grid = metadata.get("churn_grid")
    if raw_grid is None:
        return CHURN_GRID
    if not isinstance(raw_grid, (list, tuple)):
        return CHURN_GRID

    try:
        candidate = tuple(float(value) for value in raw_grid)
    except (TypeError, ValueError):
        return CHURN_GRID

    if not _is_valid_churn_grid(candidate):
        return CHURN_GRID
    return candidate


def resolve_churn_labels(params: MDPParams | None = None) -> tuple[str, ...]:
    """Resolve human-readable churn labels aligned with the active churn grid."""
    grid = resolve_churn_grid(params)
    default_labels = _default_churn_labels(len(grid))
    if params is None:
        return default_labels

    metadata = params.metadata if isinstance(params.metadata, dict) else {}
    bucketing = metadata.get("churn_bucketing")
    if not isinstance(bucketing, dict):
        return default_labels
    raw_labels = bucketing.get("labels")
    if not isinstance(raw_labels, (list, tuple)):
        return default_labels
    labels = tuple(str(label) for label in raw_labels)
    if len(labels) != len(grid):
        return default_labels
    return labels


def validate_n_categories(n_categories: int) -> None:
    """Validate Version C category dimension constraints."""
    if n_categories <= 0:
        raise ValueError("n_categories must be positive.")
    if n_categories > MAX_DP_CATEGORIES:
        raise ValueError(
            f"n_categories={n_categories} exceeds Version C cap "
            f"MAX_DP_CATEGORIES={MAX_DP_CATEGORIES}."
        )


def terminal_state(n_categories: int) -> DiscreteState:
    """Return the absorbing terminal state for a problem dimension."""
    validate_n_categories(n_categories)
    return DiscreteState(
        churn_bucket=TERMINAL_CHURN_BUCKET,
        memory_buckets=(0,) * n_categories,
        recency_buckets=(0,) * n_categories,
    )


def is_terminal_state(state: DiscreteState) -> bool:
    """Return whether the state is the absorbing terminal state."""
    return state.churn_bucket == TERMINAL_CHURN_BUCKET


def decode_state(
    state: DiscreteState,
    *,
    churn_grid: tuple[float, ...] | None = None,
) -> DecodedDiscreteState:
    """Decode bucket indices into representative continuous values."""
    if is_terminal_state(state):
        raise ValueError("Terminal state has no continuous decoding.")

    grid = _resolve_churn_grid_arg(churn_grid)
    validate_state_shape(
        state=state,
        n_categories=len(state.memory_buckets),
        churn_grid=grid,
    )

    if not (0 <= state.churn_bucket < len(grid)):
        raise ValueError(f"Invalid churn bucket: {state.churn_bucket}.")

    memory = tuple(_value_from_bucket(idx, MEMORY_GRID) for idx in state.memory_buckets)
    recency = tuple(_value_from_bucket(idx, RECENCY_GRID) for idx in state.recency_buckets)
    return DecodedDiscreteState(
        churn_propensity=grid[state.churn_bucket],
        discount_memory=memory,
        purchase_recency=recency,
    )


def bucketize_churn(
    value: float,
    *,
    churn_grid: tuple[float, ...] | None = None,
) -> int:
    """Map a churn propensity value to the nearest churn bucket index."""
    return _nearest_bucket(value=value, grid=_resolve_churn_grid_arg(churn_grid))


def bucketize_churn_distribution(
    value: float,
    *,
    churn_grid: tuple[float, ...] | None = None,
) -> tuple[tuple[int, float], ...]:
    """Map churn value to an interpolated bucket distribution."""
    return _bucket_distribution(value=value, grid=_resolve_churn_grid_arg(churn_grid))


def bucketize_memory(value: float) -> int:
    """Map a memory value to the nearest memory bucket index."""
    return _nearest_bucket(value=value, grid=MEMORY_GRID)


def bucketize_memory_distribution(value: float) -> tuple[tuple[int, float], ...]:
    """Map memory value to an interpolated bucket distribution."""
    return _bucket_distribution(value=value, grid=MEMORY_GRID)


def bucketize_recency(value: float) -> int:
    """Map a recency value to the nearest recency bucket index."""
    return _nearest_bucket(value=value, grid=RECENCY_GRID)


def bucketize_recency_distribution(value: float) -> tuple[tuple[int, float], ...]:
    """Map recency value to an interpolated bucket distribution."""
    return _bucket_distribution(value=value, grid=RECENCY_GRID)


def validate_state_shape(
    state: DiscreteState,
    n_categories: int,
    *,
    churn_grid: tuple[float, ...] | None = None,
) -> None:
    """Validate that a discrete state matches expected dimensions and bins."""
    validate_n_categories(n_categories)
    grid = _resolve_churn_grid_arg(churn_grid)
    if len(state.memory_buckets) != n_categories:
        raise ValueError(
            f"Expected {n_categories} memory buckets, got {len(state.memory_buckets)}."
        )
    if len(state.recency_buckets) != n_categories:
        raise ValueError(
            f"Expected {n_categories} recency buckets, got {len(state.recency_buckets)}."
        )

    if is_terminal_state(state):
        return

    if not (0 <= state.churn_bucket < len(grid)):
        raise ValueError(f"Invalid churn bucket: {state.churn_bucket}.")
    for idx in state.memory_buckets:
        if not (0 <= idx < len(MEMORY_GRID)):
            raise ValueError(f"Invalid memory bucket: {idx}.")
    for idx in state.recency_buckets:
        if not (0 <= idx < len(RECENCY_GRID)):
            raise ValueError(f"Invalid recency bucket: {idx}.")


def enumerate_live_states(
    n_categories: int,
    *,
    churn_grid: tuple[float, ...] | None = None,
) -> tuple[DiscreteState, ...]:
    """Enumerate all non-terminal states in deterministic lexicographic order."""
    validate_n_categories(n_categories)
    grid = _resolve_churn_grid_arg(churn_grid)
    states: list[DiscreteState] = []
    for churn_bucket in range(len(grid)):
        for memory_buckets in product(range(len(MEMORY_GRID)), repeat=n_categories):
            for recency_buckets in product(range(len(RECENCY_GRID)), repeat=n_categories):
                states.append(
                    DiscreteState(
                        churn_bucket=churn_bucket,
                        memory_buckets=tuple(memory_buckets),
                        recency_buckets=tuple(recency_buckets),
                    )
                )
    return tuple(states)


def enumerate_all_states(
    n_categories: int,
    *,
    churn_grid: tuple[float, ...] | None = None,
) -> tuple[DiscreteState, ...]:
    """Enumerate live states followed by terminal state."""
    live = enumerate_live_states(n_categories=n_categories, churn_grid=churn_grid)
    return live + (terminal_state(n_categories=n_categories),)


def state_index_maps(
    n_categories: int,
    *,
    churn_grid: tuple[float, ...] | None = None,
) -> tuple[dict[DiscreteState, int], dict[int, DiscreteState]]:
    """Build forward/reverse maps for stable state indices."""
    states = enumerate_all_states(n_categories=n_categories, churn_grid=churn_grid)
    state_to_index = {state: idx for idx, state in enumerate(states)}
    index_to_state = {idx: state for idx, state in enumerate(states)}
    return state_to_index, index_to_state


def _value_from_bucket(bucket_index: int, grid: tuple[float, ...]) -> float:
    if not (0 <= bucket_index < len(grid)):
        raise ValueError(f"Bucket index {bucket_index} out of bounds for grid.")
    return grid[bucket_index]


def _nearest_bucket(value: float, grid: tuple[float, ...]) -> int:
    return min(range(len(grid)), key=lambda idx: abs(grid[idx] - value))


def _bucket_distribution(value: float, grid: tuple[float, ...]) -> tuple[tuple[int, float], ...]:
    """Return a piecewise-linear interpolation over neighboring bucket indices."""
    if len(grid) == 1:
        return ((0, 1.0),)

    if value <= grid[0]:
        return ((0, 1.0),)
    if value >= grid[-1]:
        return ((len(grid) - 1, 1.0),)

    for idx in range(len(grid) - 1):
        left = grid[idx]
        right = grid[idx + 1]
        if not (left <= value <= right):
            continue
        if right == left:
            return ((idx, 1.0),)
        right_weight = (value - left) / (right - left)
        left_weight = 1.0 - right_weight
        return (
            (idx, left_weight),
            (idx + 1, right_weight),
        )

    # Defensive fallback in case of floating-point boundary issues.
    nearest = _nearest_bucket(value=value, grid=grid)
    return ((nearest, 1.0),)


def _resolve_churn_grid_arg(churn_grid: tuple[float, ...] | None) -> tuple[float, ...]:
    grid = CHURN_GRID if churn_grid is None else tuple(float(value) for value in churn_grid)
    if not _is_valid_churn_grid(grid):
        raise ValueError(
            "Invalid churn_grid: expected at least 2 strictly increasing finite values."
        )
    return grid


def _is_valid_churn_grid(grid: tuple[float, ...]) -> bool:
    if len(grid) < 2:
        return False
    for value in grid:
        if not isinstance(value, float):
            return False
        if not math.isfinite(value):
            return False
        if value < 0.0 or value > 1.0:
            return False
    return all(grid[idx] < grid[idx + 1] for idx in range(len(grid) - 1))


def _default_churn_labels(n_buckets: int) -> tuple[str, ...]:
    if n_buckets == 3:
        return (
            "Engaged (low churn risk)",
            "At-Risk (medium churn risk)",
            "Lapsing (high churn risk)",
        )
    if n_buckets == 2:
        return (
            "Engaged (lower churn risk)",
            "Lapsing (higher churn risk)",
        )
    return tuple(f"Churn Segment {idx + 1}" for idx in range(n_buckets))


def _normalize_bucket_grid(
    *,
    values: Iterable[float],
    name: str,
    expected_len: int,
) -> tuple[float, ...]:
    try:
        grid = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an iterable of finite numbers.") from exc

    if len(grid) != expected_len:
        raise ValueError(f"{name} must contain exactly {expected_len} values.")
    if not all(math.isfinite(value) for value in grid):
        raise ValueError(f"{name} must contain only finite values.")
    if not all(grid[idx] < grid[idx + 1] for idx in range(len(grid) - 1)):
        raise ValueError(f"{name} must be strictly increasing.")
    return grid
