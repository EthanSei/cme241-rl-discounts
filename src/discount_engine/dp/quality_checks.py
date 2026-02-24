"""Quality checks for Version C DP outputs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from discount_engine.core.params import MDPParams
from discount_engine.core.types import DiscreteState
from discount_engine.dp.discretization import (
    MEMORY_GRID,
    RECENCY_GRID,
    enumerate_all_states,
    is_terminal_state,
    resolve_churn_grid,
)
from discount_engine.dp.transitions import enumerate_transition_distribution
from discount_engine.dp.value_iteration import bellman_backup


@dataclass(frozen=True)
class CheckResult:
    """One quality-check result."""

    name: str
    passed: bool
    details: str
    metric: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
        }
        if self.metric is not None:
            payload["metric"] = self.metric
        return payload


@dataclass(frozen=True)
class QualityReport:
    """Aggregated hard/conceptual checks."""

    hard_checks: tuple[CheckResult, ...]
    conceptual_checks: tuple[CheckResult, ...]
    strict_conceptual: bool

    @property
    def hard_failures(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.hard_checks if not check.passed)

    @property
    def conceptual_warnings(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.conceptual_checks if not check.passed)

    @property
    def passed(self) -> bool:
        if self.hard_failures:
            return False
        if self.strict_conceptual and self.conceptual_warnings:
            return False
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "hard_checks": [check.to_dict() for check in self.hard_checks],
            "conceptual_checks": [check.to_dict() for check in self.conceptual_checks],
            "hard_failures": [check.to_dict() for check in self.hard_failures],
            "conceptual_warnings": [
                check.to_dict() for check in self.conceptual_warnings
            ],
            "strict_conceptual": self.strict_conceptual,
            "passed": self.passed,
        }


def run_quality_checks(
    *,
    params: MDPParams,
    values: dict[DiscreteState, float],
    policy: dict[DiscreteState, int],
    q_values: dict[DiscreteState, dict[int, float]],
    gamma: float,
    bellman_atol: float = 1e-6,
    strict_conceptual: bool = False,
) -> QualityReport:
    """Run hard + conceptual checks against DP artifacts."""
    hard_checks = (
        _check_parameter_domains(params=params, gamma=gamma),
        _check_policy_action_range(policy=policy, n_categories=len(params.categories)),
        _check_terminal_invariants(params=params),
        _check_transition_probabilities(params=params),
        _check_q_values_coverage(
            q_values=q_values,
            n_categories=len(params.categories),
            states=tuple(values.keys()),
        ),
        _check_bellman_residual(
            params=params,
            values=values,
            gamma=gamma,
            bellman_atol=bellman_atol,
        ),
    )

    conceptual_checks = (
        _check_value_ordering_by_churn(
            values=values,
            n_categories=len(params.categories),
            n_churn_buckets=len(resolve_churn_grid(params)),
        ),
        _check_policy_collapse(policy=policy),
        _check_memory_addiction_intuition(policy=policy),
        _check_recency_intuition(policy=policy),
    )

    return QualityReport(
        hard_checks=hard_checks,
        conceptual_checks=conceptual_checks,
        strict_conceptual=strict_conceptual,
    )


def _check_parameter_domains(params: MDPParams, gamma: float) -> CheckResult:
    checks: list[str] = []
    if not (0.0 < gamma < 1.0):
        checks.append("gamma must be in (0, 1)")
    if not (0.0 < params.delta < 1.0):
        checks.append("delta must be in (0, 1)")
    if not (0.0 <= params.alpha < 1.0):
        checks.append("alpha must be in [0, 1)")
    if params.eta <= 0.0:
        checks.append("eta must be positive")
    if len(params.categories) == 0:
        checks.append("at least one category required")
    for category in params.categories:
        if category.price <= 0.0:
            checks.append(f"category {category.name} has non-positive price")
    passed = len(checks) == 0
    details = "parameter domains valid" if passed else "; ".join(checks)
    return CheckResult(name="parameter_domains", passed=passed, details=details)


def _check_policy_action_range(
    policy: dict[DiscreteState, int], n_categories: int
) -> CheckResult:
    invalid = [
        (state, action)
        for state, action in policy.items()
        if action < 0 or action > n_categories
    ]
    if invalid:
        sample = invalid[0]
        return CheckResult(
            name="policy_action_range",
            passed=False,
            details=(
                f"invalid action {sample[1]} for state {sample[0]} "
                f"(expected 0..{n_categories})"
            ),
        )
    return CheckResult(
        name="policy_action_range",
        passed=True,
        details=f"all policy actions in [0, {n_categories}]",
    )


def _check_terminal_invariants(params: MDPParams) -> CheckResult:
    n_categories = len(params.categories)
    terminal = DiscreteState(
        churn_bucket=-1,
        memory_buckets=(0,) * n_categories,
        recency_buckets=(0,) * n_categories,
    )
    for action in range(n_categories + 1):
        branches = enumerate_transition_distribution(
            state=terminal,
            action=action,
            params=params,
        )
        if len(branches) != 1:
            return CheckResult(
                name="terminal_invariants",
                passed=False,
                details=f"terminal has {len(branches)} branches for action={action}",
            )
        only = branches[0]
        if only.next_state != terminal or only.reward != 0.0 or only.probability != 1.0:
            return CheckResult(
                name="terminal_invariants",
                passed=False,
                details="terminal branch does not match absorbing zero-reward contract",
            )
    return CheckResult(
        name="terminal_invariants",
        passed=True,
        details="terminal state is absorbing with zero reward",
    )


def _check_transition_probabilities(params: MDPParams) -> CheckResult:
    n_categories = len(params.categories)
    states = enumerate_all_states(
        n_categories=n_categories,
        churn_grid=resolve_churn_grid(params),
    )
    checked = 0
    for state in states:
        for action in range(n_categories + 1):
            branches = enumerate_transition_distribution(
                state=state,
                action=action,
                params=params,
            )
            total = sum(branch.probability for branch in branches)
            if any(branch.probability < -1e-15 for branch in branches):
                return CheckResult(
                    name="transition_probabilities",
                    passed=False,
                    details=f"negative probability at state={state}, action={action}",
                )
            if abs(total - 1.0) > 1e-9:
                return CheckResult(
                    name="transition_probabilities",
                    passed=False,
                    details=(
                        f"probability mass={total:.12f} at state={state}, action={action}"
                    ),
                    metric=total,
                )
            checked += 1
    return CheckResult(
        name="transition_probabilities",
        passed=True,
        details=f"validated {checked} state-action distributions",
    )


def _check_q_values_coverage(
    *,
    q_values: dict[DiscreteState, dict[int, float]],
    n_categories: int,
    states: tuple[DiscreteState, ...],
) -> CheckResult:
    expected_actions = set(range(n_categories + 1))
    for state in states:
        q_map = q_values.get(state)
        if q_map is None:
            return CheckResult(
                name="q_values_coverage",
                passed=False,
                details=f"missing q-values for state {state}",
            )
        if set(q_map.keys()) != expected_actions:
            return CheckResult(
                name="q_values_coverage",
                passed=False,
                details=f"action keys mismatch for state {state}",
            )
    return CheckResult(
        name="q_values_coverage",
        passed=True,
        details="q-values cover all states and actions",
    )


def _check_bellman_residual(
    *,
    params: MDPParams,
    values: dict[DiscreteState, float],
    gamma: float,
    bellman_atol: float,
) -> CheckResult:
    n_categories = len(params.categories)
    states = enumerate_all_states(
        n_categories=n_categories,
        churn_grid=resolve_churn_grid(params),
    )
    missing_states = [state for state in states if state not in values]
    if missing_states:
        sample = missing_states[0]
        return CheckResult(
            name="bellman_residual",
            passed=False,
            details=(
                f"value function missing {len(missing_states)} required state(s); "
                f"example missing state: {sample}"
            ),
        )

    residual = 0.0
    for state in states:
        bellman_value, _, _ = bellman_backup(
            state=state,
            values=values,
            params=params,
            gamma=gamma,
        )
        residual = max(residual, abs(values[state] - bellman_value))

    passed = residual <= bellman_atol
    details = (
        f"bellman residual {residual:.3e} <= atol {bellman_atol:.3e}"
        if passed
        else f"bellman residual {residual:.3e} exceeds atol {bellman_atol:.3e}"
    )
    return CheckResult(
        name="bellman_residual",
        passed=passed,
        details=details,
        metric=residual,
    )


def _check_value_ordering_by_churn(
    *,
    values: dict[DiscreteState, float],
    n_categories: int,
    n_churn_buckets: int,
) -> CheckResult:
    if n_churn_buckets < 2:
        return CheckResult(
            name="value_ordering_by_churn",
            passed=True,
            details="single churn bucket configured; ordering check skipped",
        )

    comparisons = 0
    violations = 0
    for memory in product(range(len(MEMORY_GRID)), repeat=n_categories):
        for recency in product(range(len(RECENCY_GRID)), repeat=n_categories):
            for churn_bucket in range(n_churn_buckets - 1):
                lower_state = DiscreteState(
                    churn_bucket=churn_bucket,
                    memory_buckets=tuple(memory),
                    recency_buckets=tuple(recency),
                )
                higher_state = DiscreteState(
                    churn_bucket=churn_bucket + 1,
                    memory_buckets=tuple(memory),
                    recency_buckets=tuple(recency),
                )
                if lower_state not in values or higher_state not in values:
                    continue
                comparisons += 1
                if values[lower_state] + 1e-9 < values[higher_state]:
                    violations += 1

    if comparisons == 0:
        return CheckResult(
            name="value_ordering_by_churn",
            passed=True,
            details="insufficient states for churn ordering comparison",
        )

    violation_rate = violations / comparisons
    passed = violation_rate <= 0.05
    details = (
        f"{violations}/{comparisons} churn-order violations (rate={violation_rate:.3f})"
    )
    return CheckResult(
        name="value_ordering_by_churn",
        passed=passed,
        details=details,
        metric=violation_rate,
    )


def _check_policy_collapse(policy: dict[DiscreteState, int]) -> CheckResult:
    live_actions = [
        action for state, action in policy.items() if not is_terminal_state(state)
    ]
    if not live_actions:
        return CheckResult(
            name="policy_collapse",
            passed=True,
            details="no live-state actions found",
        )

    counts: dict[int, int] = {}
    for action in live_actions:
        counts[action] = counts.get(action, 0) + 1
    dominant_share = max(counts.values()) / len(live_actions)
    passed = dominant_share < 0.95
    details = f"dominant action share={dominant_share:.3f}"
    return CheckResult(
        name="policy_collapse",
        passed=passed,
        details=details,
        metric=dominant_share,
    )


def _check_memory_addiction_intuition(policy: dict[DiscreteState, int]) -> CheckResult:
    low_group: list[int] = []
    high_group: list[int] = []

    for state, action in policy.items():
        if is_terminal_state(state):
            continue
        mean_memory = sum(state.memory_buckets) / len(state.memory_buckets)
        if mean_memory <= 0.5:
            low_group.append(action)
        if mean_memory >= 1.5:
            high_group.append(action)

    if not low_group or not high_group:
        return CheckResult(
            name="memory_addiction_intuition",
            passed=True,
            details="insufficient low/high memory states for comparison",
        )

    low_rate = _promotion_rate(low_group)
    high_rate = _promotion_rate(high_group)
    passed = high_rate <= low_rate + 0.05
    details = (
        f"promotion_rate(high_memory)={high_rate:.3f}, "
        f"promotion_rate(low_memory)={low_rate:.3f}"
    )
    return CheckResult(
        name="memory_addiction_intuition",
        passed=passed,
        details=details,
        metric=high_rate - low_rate,
    )


def _check_recency_intuition(policy: dict[DiscreteState, int]) -> CheckResult:
    recent_group: list[int] = []
    stale_group: list[int] = []

    for state, action in policy.items():
        if is_terminal_state(state):
            continue
        mean_recency = sum(state.recency_buckets) / len(state.recency_buckets)
        if mean_recency <= 0.25:
            recent_group.append(action)
        if mean_recency >= 0.75:
            stale_group.append(action)

    if not recent_group or not stale_group:
        return CheckResult(
            name="recency_intuition",
            passed=True,
            details="insufficient recent/stale states for comparison",
        )

    recent_rate = _promotion_rate(recent_group)
    stale_rate = _promotion_rate(stale_group)
    passed = stale_rate + 0.05 >= recent_rate
    details = (
        f"promotion_rate(stale)={stale_rate:.3f}, "
        f"promotion_rate(recent)={recent_rate:.3f}"
    )
    return CheckResult(
        name="recency_intuition",
        passed=passed,
        details=details,
        metric=stale_rate - recent_rate,
    )


def _promotion_rate(actions: list[int]) -> float:
    if not actions:
        return 0.0
    promoted = sum(1 for action in actions if action != 0)
    return promoted / len(actions)
