"""DP script end-to-end tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from discount_engine.dp.artifacts import REQUIRED_RUN_FILES


def _run_script(
    *,
    project_root: Path,
    env: dict[str, str],
    script_name: str,
    args: list[str],
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(project_root / "scripts" / script_name), *args]
    return subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)


def test_dp_solve_script_writes_expected_run_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    run_dir = tmp_path / "dp_run"
    params_path = project_root / "tests" / "fixtures" / "dp_params_small.yaml"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")
    result = _run_script(
        project_root=project_root,
        env=env,
        script_name="dp_solve.py",
        args=[
            "--params-path",
            str(params_path),
            "--n-categories",
            "2",
            "--run-dir",
            str(run_dir),
            "--tag",
            "pytest",
        ],
    )

    assert result.returncode == 0, result.stderr
    for required in REQUIRED_RUN_FILES:
        assert (run_dir / required).exists(), required
        assert (run_dir / required).stat().st_size > 0, required

    config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text())
    assert config["n_categories"] == 2
    assert float(config["solver"]["gamma"]) > 0.0
    assert config["solver"]["memory_grid"] == [0.0, 0.9, 2.0]
    assert config["solver"]["recency_grid"] == [1.0, 4.0]

    policy_header = (run_dir / "policy_table.csv").read_text().splitlines()[0]
    assert "churn_label" in policy_header

    validate = _run_script(
        project_root=project_root,
        env=env,
        script_name="dp_validate.py",
        args=["--run-dir", str(run_dir)],
    )
    assert validate.returncode == 0, validate.stderr

    evaluate = _run_script(
        project_root=project_root,
        env=env,
        script_name="dp_evaluate.py",
        args=["--run-dir", str(run_dir)],
    )
    assert evaluate.returncode == 0, evaluate.stderr


def test_dp_solve_allows_solver_yaml_grid_override(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    run_dir = tmp_path / "dp_run_override"
    params_path = project_root / "tests" / "fixtures" / "dp_params_small.yaml"
    solver_path = tmp_path / "solver_override.yaml"
    solver_payload = {
        "gamma": 0.95,
        "epsilon": 1.0e-8,
        "max_iters": 5000,
        "bellman_atol": 1.0e-6,
        "memory_grid": [0.0, 0.4, 3.0],
        "recency_grid": [0.0, 20.0],
    }
    solver_path.write_text(yaml.safe_dump(solver_payload, sort_keys=False), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")
    solve = _run_script(
        project_root=project_root,
        env=env,
        script_name="dp_solve.py",
        args=[
            "--params-path",
            str(params_path),
            "--solver-config",
            str(solver_path),
            "--n-categories",
            "2",
            "--run-dir",
            str(run_dir),
            "--tag",
            "pytest",
            "--no-progress",
        ],
    )
    assert solve.returncode == 0, solve.stderr

    config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text())
    assert config["solver"]["memory_grid"] == [0.0, 0.4, 3.0]
    assert config["solver"]["recency_grid"] == [0.0, 20.0]
    assert abs(float(config["solver"]["gamma"]) - 0.95) <= 1e-12

    validate = _run_script(
        project_root=project_root,
        env=env,
        script_name="dp_validate.py",
        args=["--run-dir", str(run_dir)],
    )
    assert validate.returncode == 0, validate.stderr

    evaluate = _run_script(
        project_root=project_root,
        env=env,
        script_name="dp_evaluate.py",
        args=["--run-dir", str(run_dir)],
    )
    assert evaluate.returncode == 0, evaluate.stderr

    summary = json.loads((run_dir / "evaluation_summary.json").read_text())
    assert summary["solver_config"]["memory_grid"] == [0.0, 0.4, 3.0]
    assert summary["solver_config"]["recency_grid"] == [0.0, 20.0]


def test_dp_evaluate_uses_metadata_churn_labels(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    run_dir = tmp_path / "dp_run_labeled"
    fixture_path = project_root / "tests" / "fixtures" / "dp_params_small.yaml"
    params_payload = yaml.safe_load(fixture_path.read_text())
    custom_labels = ["Segment A", "Segment B", "Segment C"]
    params_payload["metadata"] = {
        **dict(params_payload.get("metadata", {})),
        "churn_bucketing": {
            "grid": [0.10, 0.45, 0.85],
            "labels": custom_labels,
        },
    }
    params_path = tmp_path / "dp_params_custom_labels.yaml"
    params_path.write_text(yaml.safe_dump(params_payload, sort_keys=False), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")
    solve = _run_script(
        project_root=project_root,
        env=env,
        script_name="dp_solve.py",
        args=[
            "--params-path",
            str(params_path),
            "--n-categories",
            "2",
            "--run-dir",
            str(run_dir),
            "--tag",
            "pytest",
            "--no-progress",
        ],
    )
    assert solve.returncode == 0, solve.stderr

    evaluate = _run_script(
        project_root=project_root,
        env=env,
        script_name="dp_evaluate.py",
        args=["--run-dir", str(run_dir)],
    )
    assert evaluate.returncode == 0, evaluate.stderr

    summary = json.loads((run_dir / "evaluation_summary.json").read_text())
    states = summary["top_states_by_value"] + summary["bottom_states_by_value"]
    observed_labels = {
        str(row["churn_label"])
        for row in states
        if int(row["churn_bucket"]) >= 0
    }
    assert observed_labels
    assert observed_labels.issubset(set(custom_labels))
