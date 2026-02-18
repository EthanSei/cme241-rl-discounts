"""Run Version C tabular DP solve and persist run artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from discount_engine.core.params import MDPParams, load_mdp_params
from discount_engine.dp.artifacts import (
    ensure_run_dir,
    encode_policy,
    encode_q_values,
    encode_values,
    write_json,
    write_yaml,
)
from discount_engine.dp.discretization import MAX_DP_CATEGORIES, temporary_bucket_grids
from discount_engine.dp.policy import build_evaluation_summary, policy_rows
from discount_engine.dp.quality_checks import run_quality_checks
from discount_engine.dp.value_iteration import ValueIterationConfig, solve_value_iteration


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve Version C via value iteration.")
    parser.add_argument(
        "--params-path",
        type=Path,
        default=Path("data/processed/mdp_params.yaml"),
        help="Path to calibrated MDP params YAML.",
    )
    parser.add_argument(
        "--solver-config",
        type=Path,
        default=Path("configs/dp/solver.yaml"),
        help="Path to DP solver config YAML.",
    )
    parser.add_argument(
        "--n-categories",
        type=int,
        default=3,
        help="Number of categories to use (default=3, max=5).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional explicit output run directory.",
    )
    parser.add_argument("--tag", default="manual", help="Tag used in default run directory.")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--bellman-atol", type=float, default=None)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar with current Bellman delta.",
    )
    args = parser.parse_args()

    params = load_mdp_params(args.params_path)
    subset_params = _subset_params(params=params, n_categories=args.n_categories)
    solver_cfg = _load_solver_config(args.solver_config)
    memory_grid = _resolve_solver_grid(
        raw_grid=solver_cfg.get("memory_grid"),
        name="memory_grid",
    )
    recency_grid = _resolve_solver_grid(
        raw_grid=solver_cfg.get("recency_grid"),
        name="recency_grid",
    )
    gamma = float(
        args.gamma if args.gamma is not None else solver_cfg.get("gamma", subset_params.gamma)
    )
    epsilon = float(
        args.epsilon if args.epsilon is not None else solver_cfg.get("epsilon", 1.0e-8)
    )
    max_iters = int(
        args.max_iters if args.max_iters is not None else solver_cfg.get("max_iters", 10000)
    )
    bellman_atol = float(
        args.bellman_atol
        if args.bellman_atol is not None
        else solver_cfg.get("bellman_atol", 1.0e-6)
    )
    config = ValueIterationConfig(
        gamma=gamma,
        epsilon=epsilon,
        max_iters=max_iters,
        show_progress=not args.no_progress,
        progress_desc=f"Value Iteration (N={args.n_categories})",
    )
    with temporary_bucket_grids(
        memory_grid=memory_grid,
        recency_grid=recency_grid,
    ) as (resolved_memory_grid, resolved_recency_grid):
        result = solve_value_iteration(params=subset_params, config=config)
        rows = policy_rows(policy=result.policy, values=result.values, params=subset_params)
        evaluation_summary = build_evaluation_summary(rows)
        quality = run_quality_checks(
            params=subset_params,
            values=result.values,
            policy=result.policy,
            q_values=result.q_values,
            gamma=config.gamma,
            bellman_atol=bellman_atol,
            strict_conceptual=False,
        )

    run_dir = args.run_dir or _default_run_dir(tag=args.tag)
    ensure_run_dir(run_dir)
    _write_policy_table(run_dir=run_dir, rows=rows)

    config_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "params_path": str(args.params_path),
        "solver_config_path": str(args.solver_config),
        "n_categories": args.n_categories,
        "params": subset_params.to_dict(),
        "solver": {
            "gamma": config.gamma,
            "epsilon": config.epsilon,
            "max_iters": config.max_iters,
            "bellman_atol": bellman_atol,
            "memory_grid": [float(value) for value in resolved_memory_grid],
            "recency_grid": [float(value) for value in resolved_recency_grid],
        },
        "notes": {
            "recommended_default_n_categories": 3,
            "n_4_or_5_is_exploratory": True,
        },
    }

    solver_metrics = {
        "iterations": result.iterations,
        "converged": result.converged,
        "max_delta_history": list(result.max_delta_history),
        "final_bellman_residual": result.final_bellman_residual,
        "n_states": len(result.values),
        "n_actions": len(subset_params.categories) + 1,
    }

    write_yaml(run_dir / "config_resolved.yaml", config_payload)
    write_json(run_dir / "values.json", encode_values(result.values))
    write_json(run_dir / "policy.json", encode_policy(result.policy))
    write_json(run_dir / "q_values.json", encode_q_values(result.q_values))
    write_json(run_dir / "solver_metrics.json", solver_metrics)
    write_json(run_dir / "quality_report.json", quality.to_dict())
    write_json(
        run_dir / "quality_warnings.json",
        [warning.to_dict() for warning in quality.conceptual_warnings],
    )
    write_json(run_dir / "evaluation_summary.json", evaluation_summary)

    print(f"Run directory: {run_dir}")
    print(
        f"Value Iteration: converged={result.converged}, "
        f"iterations={result.iterations}, residual={result.final_bellman_residual:.3e}"
    )
    if quality.conceptual_warnings:
        print(
            "Conceptual warnings: "
            + ", ".join(warning.name for warning in quality.conceptual_warnings)
        )

    if quality.hard_failures:
        print("Hard quality checks failed; see quality_report.json.")
        return 1
    return 0


def _subset_params(params: MDPParams, n_categories: int) -> MDPParams:
    if n_categories < 1:
        raise ValueError("n_categories must be at least 1.")
    if n_categories > MAX_DP_CATEGORIES:
        raise ValueError(
            f"n_categories={n_categories} exceeds Version C max={MAX_DP_CATEGORIES}."
        )
    if n_categories > len(params.categories):
        raise ValueError(
            f"n_categories={n_categories} exceeds available categories "
            f"{len(params.categories)} in params artifact."
        )
    return replace(params, categories=params.categories[:n_categories])


def _load_solver_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in solver config: {path}")
    return raw


def _resolve_solver_grid(*, raw_grid: object, name: str) -> tuple[float, ...] | None:
    if raw_grid is None:
        return None
    if not isinstance(raw_grid, (list, tuple)):
        raise ValueError(f"{name} must be a list/tuple in solver config.")
    try:
        return tuple(float(value) for value in raw_grid)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values in solver config.") from exc


def _default_run_dir(tag: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_tag = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in tag)
    return Path("runs") / "dp" / f"{timestamp}_{safe_tag}"


def _write_policy_table(run_dir: Path, rows: list[dict[str, object]]) -> None:
    output_path = run_dir / "policy_table.csv"
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
