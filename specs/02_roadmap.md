# Project Roadmap: Item-Level Discount Targeting

This roadmap aligns implementation with the finalized Phase 1 proposal and CME 241 project guidelines.

## Scope Lock
- Objective: maximize discounted long-run expected revenue while balancing churn risk, cannibalization, and discount addiction.
- Discount depth is fixed at `delta`; actions choose one product category to promote (or no promotion).
- Core live state is `(c_t, m_t, l_t)` with terminal churn state `s_empty`.
- Phase 2 target is Version C (tabular DP). Phase 3 target is Version B (model-free RL).
- Course dates: Phase 2 due `February 23, 2026`; Phase 3 presentation `March 13, 2026`; final submission `March 16, 2026`.

## Architecture Decision (Shared Core + DP + RL)
Decision:
- Keep one package `src/discount_engine/` with three subpackages:
- `core/`: shared state/parameter models, demand equations, transition primitives, calibration loaders.
- `dp/`: discretization, transition enumeration, value-iteration solvers, DP policy utilities.
- `rl/`: Gymnasium environment, rollouts, RL baselines/training/evaluation helpers.

Rationale:
- Prevent duplicated business logic between DP and RL.
- Keep one source of truth for calibrated parameters and transition equations.
- Make ownership boundaries explicit for implementation and testing.

Migration plan:
- [ ] Introduce `src/discount_engine/core`, `src/discount_engine/dp`, `src/discount_engine/rl`.
- [ ] Move files from legacy folders (`agents`, `simulators`, `envs`) into new ownership boundaries.
- [ ] Maintain temporary compatibility imports while scripts/tests migrate.
- [ ] Remove compatibility shims once CI and docs fully reference new paths.

Exit criteria:
- [ ] No DP-only logic lives under `rl/`; no RL-only logic lives under `dp/`.
- [ ] Shared math/params used by both tracks lives only under `core/`.

## Data Pipeline Ownership (Download + Preprocess)
Canonical locations:
- [ ] Raw data acquisition scripts: `scripts/data/download_dataset.py`.
- [ ] Preprocessing and panel-build scripts: `scripts/data/preprocess_data.py`.
- [ ] Calibration scripts producing MDP params: `scripts/data/calibrate_mdp_params.py`.
- [ ] Raw dumps in `data/raw/`, cleaned feature/panel artifacts in `data/processed/`.

Compatibility path during migration:
- [ ] Keep wrappers at legacy entrypoints (`scripts/download_dataset.py`, `scripts/preprocess_data.py`) until references are updated.

## Data Calibration Plan (Dataset -> MDP Parameters)
Goal: compute proposal parameters from Dunnhumby data before solver training.

- [ ] Build household-time-category panel from transactions (`y_{h,t,j}`, unit price, promo flags, quantity, recency).
- [ ] Select `N` tractable product categories for Version C/B and publish mapping table.
- [ ] Estimate category shelf prices `p_j` as robust non-promo unit-price statistics (median + IQR checks).
- [ ] Fit purchase model on panel:
- [ ] target `y_{h,t,j} = 1` if category `j` purchased at `(h,t)`,
- [ ] logistic with category intercepts plus deal and recency terms,
- [ ] estimate memory decay `alpha` by grid search maximizing validation log-likelihood.
- [ ] Map fitted coefficients to MDP params:
- [ ] `beta_0^{(j)}` from category intercepts,
- [ ] `beta_p` from deal coefficient,
- [ ] `beta_l` from recency coefficient sign-adjusted to proposal convention,
- [ ] `beta_m` from memory/deal coefficient relationship.
- [ ] Calibrate churn dynamics:
- [ ] define operational churn event (example: no purchases for `H` consecutive periods),
- [ ] estimate churn hazard by inactivity level and fit `eta` and `c_0` initialization to match observed hazard curves.
- [ ] Persist finalized params to versioned artifact (`data/processed/mdp_params.yaml` or JSON) used by DP and RL configs.

Exit criteria:
- [ ] Parameter estimation script is reproducible end-to-end from raw/processed data.
- [ ] Validation metrics and diagnostic plots are stored (purchase calibration + churn calibration).
- [ ] One parameter artifact feeds both Version C and Version B implementations.

## Phase 0: Alignment and Interface Freeze
Goal: align repository interfaces and naming to the approved MDP before writing core logic.

- [ ] Freeze parameter schema (`delta`, `gamma`, `beta_0`, `beta_p`, `beta_m`, `beta_l`, `alpha`, `eta`, per-category prices).
- [ ] Confirm shared action convention: `a=0` for no promotion, `a=i` for category `i`.
- [ ] Define two state representations:
- [ ] `ContinuousState` for Version B (`c`, vector `m`, vector `l`).
- [ ] `DiscreteState` for Version C (binned churn/memory/recency).
- [ ] Document transition/reward equations in code-facing form in `specs/`.
- [ ] Add acceptance checks for probability mass and terminal-state behavior.
- [ ] Finalize ownership mapping across `core/`, `dp/`, and `rl/` modules.
- [ ] Finalize script namespaces: `scripts/data/*`, `scripts/dp_*`, `scripts/rl_*`.

Exit criteria:
- [ ] One source-of-truth spec for state/action/transition/reward used by DP and RL paths.
- [ ] No remaining roadmap references to variable discount depth actions.
- [ ] Updated `specs/00_repo_map.md` reflecting new module and script layout.

## Phase 1: Version C Core Dynamics (Initial Implementation Phase 1)
Goal: implement the simplified MDP dynamics needed for exact DP.

- [ ] Implement independent logistic purchase model per category in `src/discount_engine/core/demand.py`.
- [ ] Implement deterministic memory and recency updates from proposal equations.
- [ ] Implement churn transition: churn only on zero-purchase branch, with probability `c_t`.
- [ ] Enumerate purchase subsets (`2^N`, with `N <= 5`) to build transition outcomes.
- [ ] Implement expected immediate reward as realized revenue from purchased subset at effective prices.
- [ ] Add unit tests for:
- [ ] valid probability range and normalization,
- [ ] monotonic effects (higher memory lowers full-price demand),
- [ ] absorbing terminal behavior.

Exit criteria:
- [ ] Transition kernel and reward computation pass tests for `N=2` and `N=5`.
- [ ] Numerical sanity checks match Phase 1 worked-example directionality.

## Phase 2: Version C Solver and Diagnostics (Initial Implementation Phase 2)
Goal: produce the Phase 2 DP deliverable with interpretable policy outputs.

- [ ] Implement Value Iteration / policy extraction in `scripts/dp_solve.py` (or `src/discount_engine/dp/` solver module).
- [ ] Generate policy tables over discretized states.
- [ ] Generate value diagnostics (state-value summaries and at least one heatmap/table view).
- [ ] Add reproducible script entry point with config-driven parameters.
- [ ] Add tests for convergence behavior and deterministic outputs under fixed seeds/configs.

Version C DP test plan (required before Phase 2 sign-off):
- [ ] `test_dp_transitions.py`: every `(s, a)` transition distribution sums to 1 within tolerance and has no negative probabilities.
- [ ] `test_dp_terminal.py`: terminal state is absorbing and yields zero reward for all actions.
- [ ] `test_dp_bellman_backup.py`: Bellman update on a tiny hand-checked fixture matches analytical value.
- [ ] `test_dp_value_iteration.py`: Value Iteration converges under configured `epsilon` and `max_iters`.
- [ ] `test_dp_policy_sanity.py`: policy reacts correctly to canonical states (for example, high-memory state should not always favor repeated discounting).
- [ ] `test_dp_regression.py`: selected state values/policy actions match stored snapshot under fixed parameter artifact.
- [ ] `test_dp_script_e2e.py`: command-line DP solve runs end-to-end and writes expected artifacts.

Test artifacts and thresholds:
- [ ] Store deterministic fixture parameters in `tests/fixtures/dp_params_small.yaml`.
- [ ] Store expected outputs (`V`, policy for selected states) as versioned test snapshots.
- [ ] Enforce numerical tolerances explicitly (`atol`, `rtol`) in all DP numerical tests.

Exit criteria:
- [ ] End-to-end DP run completes from command line.
- [ ] Artifacts show policy trade-off between short-term revenue and future addiction/churn risk.
- [ ] Phase 2 report inputs are generated and versioned.
- [ ] All Version C DP tests pass in CI.

## Phase 3: Version B Simulator + Gym Environment (Initial Implementation Phase 3)
Goal: build the continuous simulator/env needed for model-free RL.

- [ ] Implement continuous-state simulator using same equations (without exposing transition probabilities to agent).
- [ ] Implement Gymnasium-compatible environment with observation `[c, m_1..m_N, l_1..l_N]` and action space `{0..N}`.
- [ ] Implement baseline policies: random, never-promote, heuristic recency-triggered promotion.
- [ ] Add rollout tooling in `scripts/rl_run_simulation.py` and evaluation harness in `scripts/rl_evaluate.py`.
- [ ] Add integration tests for `reset`, `step`, episode termination, and reward accounting.

Simulation step contract (per action `a_t`):
- [ ] Apply action to effective prices (`a_t=0` no promo, `a_t=i` discount category `i` by `delta`).
- [ ] Compute per-category purchase probabilities from calibrated params and current state.
- [ ] Sample purchase outcomes (independent Bernoulli by category for Version B).
- [ ] Compute immediate reward from realized purchased basket at effective prices.
- [ ] Update memory and recency deterministically from action and purchases.
- [ ] Update churn propensity from any-purchase vs no-purchase indicator.
- [ ] Sample churn on zero-purchase branch; if churn, transition to absorbing terminal state.
- [ ] Return `(next_obs, reward, terminated, truncated, info)` with diagnostic fields (`p_buy`, purchases, churn_prob).

Exit criteria:
- [ ] Environment produces stable rollouts and supports baseline evaluation.
- [ ] Baseline metrics are logged for future RL comparisons.

## Phase 4: RL Training, Evaluation, and Final Packaging
Goal: train and evaluate a model-free agent for Version B and finalize deliverables.

- [ ] Train RL agent (initial default: PPO or DQN based on environment diagnostics).
- [ ] Compare against baselines on revenue, promotion frequency, retention proxy, and margin impact.
- [ ] Run targeted hyperparameter sweeps and ablations.
- [ ] Produce final plots/tables and integrate results into report/presentation material.
- [ ] Final code cleanup and reproducibility pass.

Exit criteria:
- [ ] RL policy outperforms non-learning baselines on agreed metrics.
- [ ] Final paper/presentation artifacts are reproducible from repository scripts.

## Immediate Next Sequence (Pending Approval)
1. Execute Data Calibration Plan and publish first parameter artifact.
2. Execute Phase 0 interface freeze against calibrated parameter schema.
3. Start Phase 1 implementation with tests-first for transition and churn logic.
4. Move to Phase 2 only after Phase 1 exit criteria are met.

## Documentation Synchronization Rule
- [ ] After each completed implementation step, update `specs/00_repo_map.md` in the same change set.
- [ ] Treat stale repo-map docs as a blocker for marking a phase step complete.

## Script Naming Convention
- [ ] Adopt `dp_` prefix for DP-focused executable scripts in `scripts/` for clear discoverability.
- [ ] Target names: `scripts/dp_solve.py`, `scripts/dp_validate.py`, `scripts/dp_evaluate.py`.
- [ ] Adopt `rl_` prefix for RL-focused executable scripts.
- [ ] Target names: `scripts/rl_train.py`, `scripts/rl_run_simulation.py`, `scripts/rl_evaluate.py`.
- [ ] Keep temporary compatibility wrappers if needed (for example, `scripts/solve_dp.py` delegating to `scripts/dp_solve.py`) until docs/CI are updated.
