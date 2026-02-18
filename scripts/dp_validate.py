"""Validate DP run artifacts and quality checks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from discount_engine.dp.artifacts import REQUIRED_RUN_FILES, load_run_artifacts, write_json
from discount_engine.dp.discretization import temporary_bucket_grids
from discount_engine.dp.quality_checks import CheckResult, run_quality_checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DP run artifacts.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory. If omitted, latest runs/dp/* is used.",
    )
    parser.add_argument(
        "--strict-conceptual",
        action="store_true",
        help="Fail on conceptual warnings in addition to hard checks.",
    )
    parser.add_argument(
        "--bellman-atol",
        type=float,
        default=None,
        help=(
            "Tolerance for Bellman residual hard check. "
            "Defaults to run config value when available."
        ),
    )
    args = parser.parse_args()

    run_dir = args.run_dir or _latest_run_dir(root=Path("runs") / "dp")
    loaded = load_run_artifacts(run_dir)
    solver_cfg = _solver_config(loaded.config_resolved)
    memory_grid = _resolve_solver_grid(solver_cfg.get("memory_grid"), name="memory_grid")
    recency_grid = _resolve_solver_grid(solver_cfg.get("recency_grid"), name="recency_grid")
    bellman_atol = float(
        args.bellman_atol
        if args.bellman_atol is not None
        else solver_cfg.get("bellman_atol", 1e-6)
    )

    artifact_check = _check_artifact_completeness(run_dir)
    with temporary_bucket_grids(memory_grid=memory_grid, recency_grid=recency_grid):
        quality = run_quality_checks(
            params=loaded.params,
            values=loaded.values,
            policy=loaded.policy,
            q_values=loaded.q_values,
            gamma=float(solver_cfg["gamma"]),
            bellman_atol=bellman_atol,
            strict_conceptual=args.strict_conceptual,
        )
    report_payload = quality.to_dict()
    report_payload["artifact_completeness"] = artifact_check.to_dict()

    warnings_payload = [warning.to_dict() for warning in quality.conceptual_warnings]
    write_json(run_dir / "quality_report.json", report_payload)
    write_json(run_dir / "quality_warnings.json", warnings_payload)

    print(f"Run directory: {run_dir}")
    print(f"Hard failures: {len(quality.hard_failures)}")
    print(f"Conceptual warnings: {len(quality.conceptual_warnings)}")

    if not artifact_check.passed:
        print(artifact_check.details)
        return 1
    if not quality.passed:
        print("Validation failed; see quality_report.json.")
        return 1
    return 0


def _solver_config(config_resolved: dict[str, Any]) -> dict[str, Any]:
    solver_cfg = config_resolved.get("solver")
    if not isinstance(solver_cfg, dict):
        raise ValueError("config_resolved.yaml missing 'solver' mapping.")
    if "gamma" not in solver_cfg:
        raise ValueError("config_resolved.yaml missing solver.gamma.")
    return solver_cfg


def _resolve_solver_grid(
    raw_grid: object,
    *,
    name: str,
) -> tuple[float, ...] | None:
    if raw_grid is None:
        return None
    if not isinstance(raw_grid, (list, tuple)):
        raise ValueError(f"solver.{name} must be a list/tuple in config_resolved.yaml.")
    try:
        return tuple(float(value) for value in raw_grid)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"solver.{name} must contain numeric values in config_resolved.yaml."
        ) from exc


def _check_artifact_completeness(run_dir: Path) -> CheckResult:
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).exists()]
    if missing:
        return CheckResult(
            name="artifact_completeness",
            passed=False,
            details=f"missing artifact files: {', '.join(missing)}",
        )

    empty = [name for name in REQUIRED_RUN_FILES if (run_dir / name).stat().st_size == 0]
    if empty:
        return CheckResult(
            name="artifact_completeness",
            passed=False,
            details=f"empty artifact files: {', '.join(empty)}",
        )

    return CheckResult(
        name="artifact_completeness",
        passed=True,
        details="all required artifacts present and non-empty",
    )


def _latest_run_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Run root does not exist: {root}")
    candidates = sorted([path for path in root.iterdir() if path.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {root}")
    return candidates[-1]


if __name__ == "__main__":
    raise SystemExit(main())
