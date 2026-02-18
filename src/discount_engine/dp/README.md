# Phase 2 DP Code Reading Guide

This guide is for understanding the Version C tabular DP implementation end-to-end.
Paths are repo-root relative.

## Scope of Phase 2
- Objective: solve the discretized DP exactly with Value Iteration.
- State: churn bucket + per-category memory bucket + per-category recency bucket + terminal.
- Action: `0..N` where `0` is no promotion and `i` promotes category `i`.
- Output: optimal value function, policy, quality checks, and report-friendly artifacts.

## Recommended Reading Order
Read in this exact order.

1. `src/discount_engine/core/types.py`
- Defines `DiscreteState` and shared state/action types.
- This is the base object passed through all DP modules.

2. `src/discount_engine/core/params.py`
- Defines calibrated parameter schema (`MDPParams`, `CategoryParams`).
- Includes load/save helpers used by scripts and tests.

3. `src/discount_engine/core/demand.py`
- Defines `DemandInputs` and `logistic_purchase_probability`.
- This is the per-category purchase model used by transition enumeration.

4. `src/discount_engine/dp/discretization.py`
- Defines bucket grids and state-space construction.
- Key items:
- `CHURN_GRID`, `MEMORY_GRID`, `RECENCY_GRID`, `MAX_DP_CATEGORIES`.
- `temporary_bucket_grids(...)` and `configure_bucket_grids(...)` for run-scoped grid overrides.
- `decode_state(...)` and `bucketize_*_distribution(...)` for continuous/discrete conversion.
- `enumerate_all_states(...)` for deterministic state ordering.

5. `src/discount_engine/dp/transitions.py`
- Implements full transition kernel for one `(state, action)`.
- Key flow inside `enumerate_transition_distribution(...)`:
- compute purchase probabilities,
- enumerate all purchase subsets (`2^N`),
- compute reward and next-state distribution,
- apply churn only on no-purchase branch,
- normalize probability mass.

6. `src/discount_engine/dp/value_iteration.py`
- Solves the DP exactly.
- Key classes/functions:
- `ValueIterationConfig`, `ValueIterationResult`,
- `bellman_action_value(...)`, `bellman_backup(...)`,
- `solve_value_iteration(...)`.
- Uses cached transition kernels for speed and deterministic tie-breaking.

7. `src/discount_engine/dp/policy.py`
- Converts solved outputs into analysis tables.
- Key helpers:
- `policy_rows(...)` for row-level export,
- `build_evaluation_summary(...)` for cluster and value summaries,
- `state_to_id(...)` and `state_from_id(...)` for artifact serialization.

8. `src/discount_engine/dp/quality_checks.py`
- Hard checks (must pass) and conceptual checks (warnings by default).
- Entry point: `run_quality_checks(...)`.
- Hard checks include probability mass, Bellman residual, terminal behavior, action range.

9. `src/discount_engine/dp/artifacts.py`
- Defines artifact contract and serialization helpers.
- `REQUIRED_RUN_FILES` is the run-directory completeness contract.
- `load_run_artifacts(...)` is used by validate/evaluate scripts.

10. `src/discount_engine/dp/__init__.py`
- Public DP package surface (`solve_value_iteration`, config/result types, cap constant).

## Script-Level Flow (How Modules Are Used)
Read scripts after module internals.

1. `configs/dp/solver.yaml`
- Source of default solver settings and default data-driven grids.

2. `scripts/dp_solve.py`
- Main orchestration:
- load params + solver config,
- apply run-scoped grid override via `temporary_bucket_grids(...)`,
- run solver, build summaries, run quality checks,
- write artifacts under `runs/dp/<timestamp>_<tag>/`.

3. `scripts/dp_validate.py`
- Loads an existing run and re-runs quality checks.
- Uses run-resolved grids and run-config `bellman_atol` by default.

4. `scripts/dp_evaluate.py`
- Loads run artifacts and regenerates `evaluation_summary.json`.
- Uses run-resolved grids so summaries match solved artifacts.

## Test Reading Order (Behavior Contracts)
Read tests in this order to understand intended guarantees.

1. `tests/dp/test_dp_terminal.py`
2. `tests/dp/test_dp_transitions.py`
3. `tests/dp/test_dp_bellman_backup.py`
4. `tests/dp/test_dp_value_iteration.py`
5. `tests/dp/test_dp_policy_sanity.py`
6. `tests/dp/test_dp_quality_checks.py`
7. `tests/dp/test_dp_regression.py`
8. `tests/dp/test_dp_script_e2e.py`
9. `tests/dp/test_dp_phase1_exit.py`

## Practical Walkthrough Loop
If you want to study code and behavior together:

1. Run solve:
```bash
python scripts/dp_solve.py --tag readthrough --no-progress
```

2. Validate that run:
```bash
python scripts/dp_validate.py --run-dir runs/dp/<your_run_dir>
```

3. Rebuild summaries:
```bash
python scripts/dp_evaluate.py --run-dir runs/dp/<your_run_dir>
```

4. Inspect outputs:
- `config_resolved.yaml`
- `solver_metrics.json`
- `quality_report.json`
- `quality_warnings.json`
- `evaluation_summary.json`
- `policy_table.csv`

## Notebook Companion
- `notebooks/dp_walkthrough.ipynb` is the narrative analysis layer on top of these artifacts.
- Use it after understanding modules/scripts above.
