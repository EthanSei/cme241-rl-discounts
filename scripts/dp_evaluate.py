"""Evaluate DP run artifacts and write report summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from discount_engine.dp.artifacts import load_run_artifacts, write_json
from discount_engine.dp.discretization import temporary_bucket_grids
from discount_engine.dp.policy import build_evaluation_summary, policy_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DP run artifacts.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory. If omitted, the latest runs/dp/* directory is used.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or _latest_run_dir(root=Path("runs") / "dp")
    loaded = load_run_artifacts(run_dir)
    solver_cfg = _solver_config(loaded.config_resolved)
    memory_grid = _resolve_solver_grid(solver_cfg.get("memory_grid"), name="memory_grid")
    recency_grid = _resolve_solver_grid(solver_cfg.get("recency_grid"), name="recency_grid")

    with temporary_bucket_grids(memory_grid=memory_grid, recency_grid=recency_grid):
        rows = policy_rows(policy=loaded.policy, values=loaded.values, params=loaded.params)
    summary = build_evaluation_summary(rows)
    summary["solver_config"] = {
        "gamma": float(solver_cfg["gamma"]),
        "bellman_atol": float(solver_cfg.get("bellman_atol", 1e-6)),
        "memory_grid": [float(value) for value in memory_grid]
        if memory_grid is not None
        else None,
        "recency_grid": [float(value) for value in recency_grid]
        if recency_grid is not None
        else None,
    }
    write_json(run_dir / "evaluation_summary.json", summary)

    cluster = summary["policy_cluster_summary"]
    print(f"Run directory: {run_dir}")
    print(f"Live states: {cluster['n_live_states']}")
    print(f"No-promotion rate: {cluster['no_promotion_rate']:.3f}")
    print(f"Promotion rate: {cluster['promotion_rate']:.3f}")
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


def _latest_run_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Run root does not exist: {root}")
    candidates = sorted([path for path in root.iterdir() if path.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {root}")
    return candidates[-1]


if __name__ == "__main__":
    raise SystemExit(main())
