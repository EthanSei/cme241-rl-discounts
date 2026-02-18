"""Serialization helpers for DP run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml

from discount_engine.core.params import MDPParams
from discount_engine.core.types import DiscreteState
from discount_engine.dp.policy import canonicalize_loaded_state, state_from_id, state_to_id


REQUIRED_RUN_FILES: tuple[str, ...] = (
    "config_resolved.yaml",
    "values.json",
    "policy.json",
    "q_values.json",
    "solver_metrics.json",
    "policy_table.csv",
    "quality_report.json",
    "quality_warnings.json",
    "evaluation_summary.json",
)


@dataclass(frozen=True)
class LoadedRunArtifacts:
    """Structured artifacts loaded from a DP run directory."""

    run_dir: Path
    params: MDPParams
    config_resolved: dict[str, Any]
    values: dict[DiscreteState, float]
    policy: dict[DiscreteState, int]
    q_values: dict[DiscreteState, dict[int, float]]
    solver_metrics: dict[str, Any]
    quality_report: dict[str, Any]
    evaluation_summary: dict[str, Any]


def ensure_run_dir(run_dir: Path) -> None:
    """Create run directory and parent paths."""
    run_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with stable formatting."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: Any) -> None:
    """Write YAML with stable formatting."""
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def encode_values(values: dict[DiscreteState, float]) -> dict[str, float]:
    """Encode value table keyed by state id."""
    return {state_to_id(state): float(value) for state, value in values.items()}


def decode_values(payload: dict[str, float], n_categories: int) -> dict[DiscreteState, float]:
    """Decode value table from serialized state ids."""
    decoded: dict[DiscreteState, float] = {}
    for state_id, value in payload.items():
        loaded = state_from_id(state_id)
        state = canonicalize_loaded_state(loaded, n_categories=n_categories)
        decoded[state] = float(value)
    return decoded


def encode_policy(policy: dict[DiscreteState, int]) -> dict[str, int]:
    """Encode policy table keyed by state id."""
    return {state_to_id(state): int(action) for state, action in policy.items()}


def decode_policy(payload: dict[str, int], n_categories: int) -> dict[DiscreteState, int]:
    """Decode policy table from serialized state ids."""
    decoded: dict[DiscreteState, int] = {}
    for state_id, action in payload.items():
        loaded = state_from_id(state_id)
        state = canonicalize_loaded_state(loaded, n_categories=n_categories)
        decoded[state] = int(action)
    return decoded


def encode_q_values(
    q_values: dict[DiscreteState, dict[int, float]]
) -> dict[str, dict[str, float]]:
    """Encode q-values table with state/action string keys."""
    payload: dict[str, dict[str, float]] = {}
    for state, action_map in q_values.items():
        payload[state_to_id(state)] = {
            str(action): float(value) for action, value in action_map.items()
        }
    return payload


def decode_q_values(
    payload: dict[str, dict[str, float]], n_categories: int
) -> dict[DiscreteState, dict[int, float]]:
    """Decode q-values table from serialized state/action keys."""
    decoded: dict[DiscreteState, dict[int, float]] = {}
    for state_id, action_map in payload.items():
        loaded = state_from_id(state_id)
        state = canonicalize_loaded_state(loaded, n_categories=n_categories)
        decoded[state] = {int(action): float(value) for action, value in action_map.items()}
    return decoded


def load_run_artifacts(run_dir: Path) -> LoadedRunArtifacts:
    """Load all required artifacts from a run directory."""
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required run artifacts in {run_dir}: {', '.join(missing)}"
        )

    config_resolved = yaml.safe_load((run_dir / "config_resolved.yaml").read_text())
    if not isinstance(config_resolved, dict):
        raise ValueError("config_resolved.yaml must contain a YAML mapping.")

    params_payload = config_resolved.get("params")
    if not isinstance(params_payload, dict):
        raise ValueError("config_resolved.yaml missing 'params' payload.")
    params = MDPParams.from_dict(params_payload)
    n_categories = len(params.categories)

    values_payload = json.loads((run_dir / "values.json").read_text())
    policy_payload = json.loads((run_dir / "policy.json").read_text())
    q_payload = json.loads((run_dir / "q_values.json").read_text())
    solver_metrics = json.loads((run_dir / "solver_metrics.json").read_text())
    quality_report = json.loads((run_dir / "quality_report.json").read_text())
    evaluation_summary = json.loads((run_dir / "evaluation_summary.json").read_text())

    return LoadedRunArtifacts(
        run_dir=run_dir,
        params=params,
        config_resolved=config_resolved,
        values=decode_values(values_payload, n_categories=n_categories),
        policy=decode_policy(policy_payload, n_categories=n_categories),
        q_values=decode_q_values(q_payload, n_categories=n_categories),
        solver_metrics=solver_metrics,
        quality_report=quality_report,
        evaluation_summary=evaluation_summary,
    )

