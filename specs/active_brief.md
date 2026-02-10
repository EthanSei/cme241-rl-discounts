# Active Brief

## Status
- Planning mode after Phase 1 proposal finalization.
- No implementation work starts until this plan is approved.

## Active Objective
Align near-term implementation to the approved MDP and deliver Version C (DP) first.

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
- Choose tractable categories (`N <= 5` for Version C) and fix crosswalk used by all configs.
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

## Initial Implementation Phases to Approve
1. Phase 0: Spec and Interface Freeze
- Freeze parameter schema and action/state conventions.
- Lock module ownership for demand, transitions, and env interface.
- Add acceptance checks for probability normalization and terminal behavior.

2. Phase 1: Version C Dynamics Core
- Implement demand and transition equations for discretized state space.
- Implement purchase-subset enumeration (`2^N`, `N <= 5`) and churn branching.
- Add unit tests for monotonicity and transition correctness.

3. Phase 2: Version C DP Solver
- Implement Value Iteration policy solve path.
- Generate policy/value artifacts for report-ready analysis.
- Add reproducible script and config flow.

4. Phase 3: Version B Env Scaffold
- Implement continuous simulator + Gymnasium environment.
- Add baselines and rollout evaluation harness for later RL training.

## Version C (DP) Validation Tests
- `test_dp_transitions.py`: probability mass conservation and non-negativity for all `(s, a)`.
- `test_dp_terminal.py`: absorbing terminal dynamics and zero-reward guarantee.
- `test_dp_bellman_backup.py`: Bellman operator correctness on hand-computable fixture.
- `test_dp_value_iteration.py`: convergence and stopping-rule correctness.
- `test_dp_policy_sanity.py`: expected directional policy behavior in canonical high-memory/high-churn states.
- `test_dp_regression.py`: stable value/policy snapshots under fixed params and seed.
- `test_dp_script_e2e.py`: end-to-end CLI solve and artifact generation.

Acceptance gate for DP phase:
- All DP validation tests must pass locally and in CI before Phase 2 is considered complete.

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

## Ordered First Work Queue (after approval)
1. Create `core/dp/rl` package scaffolding and update `specs/00_repo_map.md`.
2. Create `scripts/data/*` scaffolding and keep legacy wrappers if needed.
3. Implement calibration script to produce first `mdp_params` artifact from dataset.
4. Define shared dataclasses/interfaces for continuous and discrete states in `core/`.
5. Implement demand model and deterministic state update functions in `core/`.
6. Integrate DP transitions/solver in `dp/`, add DP tests/snapshots, then run Value Iteration end-to-end.

## Blockers and Decisions Already Resolved
- Discount depth is fixed; action does not include discount magnitude.
- DP target uses small `N` with full transition enumeration.
- RL target uses continuous state and unknown transitions to the agent.
