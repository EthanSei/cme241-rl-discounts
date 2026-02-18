# Repo Map
## File Tree
```text
cme241-rl-discounts
├── .github/
│   └── workflows/
│       └── python-app.yml
├── configs/
│   ├── data/
│   │   └── calibration.yaml
│   ├── dp/
│   │   └── solver.yaml
│   └── rl/
│       ├── agent.yaml
│       └── env.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_dp_solution.ipynb
│   ├── 03_rl_results.ipynb
│   ├── 04_calibration_walkthrough.ipynb
│   └── dp_walkthrough.ipynb
├── runs/
│   └── dp/
├── scripts/
│   ├── data/
│   │   ├── calibrate_mdp_params.py
│   │   ├── download_dataset.py
│   │   └── preprocess_data.py
│   ├── dp_evaluate.py
│   ├── dp_solve.py
│   ├── dp_validate.py
│   ├── rl_evaluate.py
│   ├── rl_run_simulation.py
│   └── rl_train.py
├── specs/
│   ├── 00_repo_map.md
│   ├── 01_project_core.md
│   ├── 02_roadmap.md
│   ├── 03_memory.md
│   ├── 04_alpha_experiments.md
│   ├── active_brief.md
│   ├── project_guidelines.md
│   └── repo_map_v2.md
├── src/
│   └── discount_engine/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── calibration.py
│       │   ├── demand.py
│       │   ├── dynamics.py
│       │   ├── params.py
│       │   └── types.py
│       ├── dp/
│       │   ├── __init__.py
│       │   ├── artifacts.py
│       │   ├── discretization.py
│       │   ├── policy.py
│       │   ├── quality_checks.py
│       │   ├── transitions.py
│       │   └── value_iteration.py
│       ├── rl/
│       │   ├── __init__.py
│       │   ├── baselines.py
│       │   ├── env.py
│       │   ├── evaluate.py
│       │   ├── rollout.py
│       │   └── train.py
│       └── utils/
│           ├── __init__.py
│           ├── io.py
│           ├── plotting.py
│           └── seeding.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── core/
│   │   ├── test_calibration.py
│   │   ├── test_demand.py
│   │   ├── test_dynamics.py
│   │   ├── test_io.py
│   │   └── test_params.py
│   ├── dp/
│   │   ├── test_dp_bellman_backup.py
│   │   ├── test_dp_phase1_exit.py
│   │   ├── test_dp_policy_sanity.py
│   │   ├── test_dp_quality_checks.py
│   │   ├── test_dp_regression.py
│   │   ├── test_dp_script_e2e.py
│   │   ├── test_dp_terminal.py
│   │   ├── test_dp_transitions.py
│   │   └── test_dp_value_iteration.py
│   ├── fixtures/
│   │   └── dp_params_small.yaml
│   └── rl/
│       ├── test_rl_env.py
│       └── test_rl_rollout.py
├── Makefile
├── pyproject.toml
└── README.md
```

## Python Source Files
- `src/discount_engine/__init__.py`
- `src/discount_engine/core/__init__.py`
- `src/discount_engine/core/calibration.py`
- `src/discount_engine/core/demand.py`
- `src/discount_engine/core/dynamics.py`
- `src/discount_engine/core/params.py`
- `src/discount_engine/core/types.py`
- `src/discount_engine/dp/__init__.py`
- `src/discount_engine/dp/artifacts.py`
- `src/discount_engine/dp/discretization.py`
- `src/discount_engine/dp/policy.py`
- `src/discount_engine/dp/quality_checks.py`
- `src/discount_engine/dp/transitions.py`
- `src/discount_engine/dp/value_iteration.py`
- `src/discount_engine/rl/__init__.py`
- `src/discount_engine/rl/baselines.py`
- `src/discount_engine/rl/env.py`
- `src/discount_engine/rl/evaluate.py`
- `src/discount_engine/rl/rollout.py`
- `src/discount_engine/rl/train.py`
- `src/discount_engine/utils/__init__.py`
- `src/discount_engine/utils/io.py`
- `src/discount_engine/utils/plotting.py`
- `src/discount_engine/utils/seeding.py`

## Component Interaction
- `core` must not import `dp` or `rl`.
- `dp` and `rl` may import from `core`.
- `scripts` should call package APIs in `src/`; avoid business logic in script files.

## Coding Standards
- **Type Hinting:** Strict typing for public APIs.
- **Docstrings:** Use concise Google-style docstrings on public classes/functions.
- **Path Handling:** Use `pathlib.Path` over `os.path`.
- **Imports:** Absolute imports, grouped as stdlib, third-party, local.
- **Testing:** Use deterministic fixtures for numerical tests and explicit tolerances.
