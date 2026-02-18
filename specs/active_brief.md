# Active Brief

## Status
- Architecture migration is complete at the file-tree level (`core/dp/rl`, script namespaces, test layout).
- Data calibration artifact is available at `data/processed/mdp_params.yaml` (default Phase 2 runs use `N=3`).
- Phase 2 implementation is complete and audited: solver, scripts, quality checks, walkthrough analysis, and DP test coverage are in place.
- DP defaults now use data-driven low-risk discretization centers from constrained sweep artifacts:
- memory buckets `(0.0, 0.9, 2.0)` and recency buckets `(2.0, 12.0)` (`runs/dp/grid_search/20260218_034643_constrained_40/`).
- Latest default Phase 2 run is validated at `runs/dp/20260218_054956_phase2_polish_final/` (hard checks pass; one conceptual warning remains in `quality_warnings.json`).

## Active Objective
Keep Phase 2 (Version C tabular DP) stable and documented while preparing the Phase 3 handoff.

## Version C Contract (Authoritative for Phase 2)
- Categories supported up to `N <= 5` for Version C, with default `N=3` for routine Phase 2 runs.
- Discrete DP state:
- `c_t` in calibrated finite churn buckets (legacy 2-bucket support plus metadata-driven 3-bucket runs),
- `m_t^(i) in {Low, Med, High}`,
- `l_t^(i) in {Recent, Stale}`,
- absorbing churn terminal state.
- Action set is fixed-depth promotion choice: `A = {0, 1, ..., N}`, `0` means no promotion.
- Transition model is tabular and analytically computed from independent logistic per-category purchases.
- Exact Value Iteration is the Phase 2 solution method.

## Current Implementation Snapshot
- Implemented:
- repository structure matches `specs/repo_map_v2.md`,
- `specs/00_repo_map.md` matches current tree,
- DP script/module/test filenames exist.
- `src/discount_engine/dp/transitions.py` and `src/discount_engine/dp/discretization.py` contain production Phase 2 logic, including interpolation-based bucket distributions for smoother transitions.
- DP discretization defaults are centralized in `configs/dp/solver.yaml` and applied by `scripts/dp_solve.py`:
- `MEMORY_GRID=(0.0, 0.9, 2.0)`, `RECENCY_GRID=(2.0, 12.0)`.
- `tests/dp/test_dp_transitions.py` and `tests/dp/test_dp_terminal.py` are now real behavior tests and passing.
- Implemented (new in this revision):
- `src/discount_engine/dp/value_iteration.py` now contains exact tabular Value Iteration.
- `src/discount_engine/dp/policy.py` now provides greedy extraction and reporting helpers.
- `src/discount_engine/dp/quality_checks.py` now provides hard checks plus conceptual diagnostics.
- `scripts/dp_{solve,validate,evaluate}.py` now execute end-to-end using `runs/dp/<timestamp>_<tag>/`.
- `scripts/dp_evaluate.py` now preserves metadata-driven churn labels when regenerating summaries from run artifacts.
- `scripts/dp_{validate,evaluate}.py` now apply run-resolved solver bucket grids from `config_resolved.yaml` to keep checks/summaries consistent with solved artifacts.

## Architecture Decision (Approved Direction)
- Use one package with three subpackages:
- `src/discount_engine/core/` for shared equations, state/param schemas, and calibration IO.
- `src/discount_engine/dp/` for discretization and DP solvers.
- `src/discount_engine/rl/` for environment and RL workflows.
- Do not split into two isolated top-level modules; shared logic must stay centralized in `core/`.

## Data Pipeline Locations
- Data download lives under `scripts/data/download_dataset.py`.
- Preprocessing/panel construction lives under `scripts/data/preprocess_data.py`.
- Parameter calibration lives under `scripts/data/calibrate_mdp_params.py`.
- Data artifacts:
- raw files -> `data/raw/`
- processed panels/params -> `data/processed/`
- During migration, legacy script names may remain as wrappers only.

## Locked Problem Contract (from `submissions/phase1.ipynb`)
- State (live): `s_t = (c_t, m_t, l_t)` where:
- `c_t` is churn propensity in `[0, 1]`,
- `m_t` is per-category discount memory,
- `l_t` is per-category purchase recency.
- Terminal state: absorbing churn state with zero future reward.
- Action: `a in {0, 1, ..., N}` with fixed discount depth `delta`; `0` means no promotion.
- Transition drivers:
- independent logistic purchase per category,
- memory decay and promotion bump,
- recency reset/increment,
- churn only when no purchase occurs.
- Reward: expected immediate revenue from effective prices.

## Parameter Estimation Plan (from Dunnhumby)
- Build a household-time-category panel with purchase indicator, unit price, promo flag, and recency.
- Choose tractable categories (default `N=3`, supported up to `N <= 5` for Version C) and fix crosswalk used by all configs.
- Estimate `p_j` from non-promo unit-price distributions per category.
- Fit purchase probability model:
- logistic target: purchased vs not purchased per category-time row,
- features: category intercepts, effective deal term, recency term, memory term,
- tune `alpha` via grid search using validation log-likelihood.
- Map regression outputs to proposal parameters:
- `beta_0^{(j)}`, `beta_p`, `beta_l`, `beta_m`.
- Calibrate churn parameters:
- define operational churn event from inactivity window `H`,
- fit hazard curve vs inactivity and calibrate `eta` and initial `c`.
- Persist final parameter artifact for both DP and RL paths (`data/processed/mdp_params.yaml`).

Current calibration status:
- Completed and aligned to Phase 2 scope (`N=3` categories present in `selected_categories`).
- Default config and Make target now support config-driven calibration runs.

## Environment Action Simulation Contract
For each step with action `a_t`:
1. Convert action to effective prices with fixed `delta`.
2. Compute per-category buy probabilities from current state and calibrated params.
3. Sample purchase outcomes by category.
4. Compute reward as realized revenue of purchased basket at effective prices.
5. Update memory vector `m` and recency vector `l`.
6. Update churn propensity `c` based on any purchase vs no purchase.
7. Apply churn transition on no-purchase branch; if churn, move to terminal state.
8. Return standard env outputs plus diagnostics in `info` (buy probs, purchases, churn prob).

## Phase 2 Execution Record (Completed)
1. Replaced placeholder DP tests (`bellman`, `value_iteration`, `policy_sanity`, `regression`, `script_e2e`) with behavior tests.
2. Implemented Value Iteration and policy extraction with deterministic convergence controls.
3. Replaced DP script scaffolds with functional CLI flows that write versioned artifacts.
4. Ran DP validation suites and generated report-ready diagnostics (policy tables + value summaries/heatmap).
5. Completed DP policy analysis and documented findings before RL handoff.

## DP-to-RL Gate (Professor Feedback)
- DP implementation supports `N <= 5`; default analysis mode remains `N=3`.
- RL work starts only after DP policy analysis is complete and reviewed.
- Version B RL algorithm direction is discrete-action; use `DQN` as the primary method.

## Version C (DP) Validation Tests
- `test_dp_transitions.py`: probability mass conservation and non-negativity for all `(s, a)`.
- `test_dp_terminal.py`: absorbing terminal dynamics and zero-reward guarantee.
- `test_dp_bellman_backup.py`: Bellman operator correctness on hand-computable fixture.
- `test_dp_value_iteration.py`: convergence and stopping-rule correctness.
- `test_dp_policy_sanity.py`: expected directional policy behavior in canonical high-memory/high-churn states.
- `test_dp_regression.py`: stable value/policy snapshots under fixed params and seed.
- `test_dp_script_e2e.py`: end-to-end CLI solve and artifact generation.

Acceptance gate for DP phase:
- All DP validation tests pass locally and in CI.
- DP scripts run without scaffold placeholder output and produce expected artifacts.

## Script Naming Convention Decision
- Use `dp_` prefix for DP scripts in `scripts/` to separate solver tooling from RL/eval tooling.
- Canonical DP entrypoint: `scripts/dp_solve.py`.
- If renaming from existing script names, keep short-lived compatibility shims until CI/docs references are migrated.
- Use `rl_` prefix for RL scripts and `scripts/data/*` namespace for data workflows.
- Canonical RL script names: `scripts/rl_train.py`, `scripts/rl_run_simulation.py`, `scripts/rl_evaluate.py`.
- Canonical data script names: `scripts/data/download_dataset.py`, `scripts/data/preprocess_data.py`, `scripts/data/calibrate_mdp_params.py`.

## Documentation Hygiene Rule
- After each completed implementation step, update `specs/00_repo_map.md` to reflect actual repository state.
- A step is not marked complete until code changes and repo-map updates are both committed.

## Immediate Queue (Now)
1. Keep regression snapshots and walkthrough outputs synced whenever solver defaults or calibration metadata changes.
2. Review conceptual warnings (`quality_warnings.json`) as part of DP sign-off before RL experiments.
3. Keep docs (`active_brief`, roadmap, repo map) synced with implementation status.
4. Start Phase 3 only after explicit DP policy sign-off.

## Phase 2 Start Checklist
- [x] Repository layout and script namespaces are in place.
- [x] Calibrated MDP parameter artifact exists and is aligned to `N=3`.
- [x] DP scope support (`N <= 5`, default `N=3`) is documented across roadmap/brief.
- [x] Begin test-first implementation in `tests/dp/` and `src/discount_engine/dp/`.

## Blockers and Decisions Already Resolved
- Discount depth is fixed; action does not include discount magnitude.
- DP target uses small `N` with full transition enumeration.
- RL target uses continuous state and unknown transitions to the agent.
- DP implementation supports up to `N <= 5`, with `N=3` as the standard Phase 2 default.
- RL method choice for Version B is discrete-action (`DQN` primary).
