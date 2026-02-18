# Project Roadmap: Item-Level Discount Targeting

This roadmap aligns implementation with the finalized Phase 1 proposal and CME 241 project guidelines.

## Scope Lock
- Objective: maximize discounted long-run expected revenue while balancing churn risk, cannibalization, and discount addiction.
- Discount depth is fixed at `delta`; actions choose one product category to promote (or no promotion).
- Core live state is `(c_t, m_t, l_t)` with terminal churn state `s_empty`.
- Phase 2 target is Version C (tabular DP). Phase 3 target is Version B (model-free RL).
- Course dates: Phase 2 due `February 23, 2026`; Phase 3 presentation `March 13, 2026`; final submission `March 16, 2026`.

## Repository Audit (Current State vs `repo_map_v2.md`)
- [x] Target package split exists: `src/discount_engine/{core,dp,rl,utils}`.
- [x] Target script namespaces exist: `scripts/data/*`, `scripts/dp_*`, `scripts/rl_*`.
- [x] Target DP test file set exists under `tests/dp/`.
- [x] `specs/00_repo_map.md` is aligned with the current tree shape.
- [x] Version C DP implementation is complete:
- [x] `src/discount_engine/dp/value_iteration.py` is implemented and covered by convergence/regression tests.
- [x] `src/discount_engine/dp/{transitions,discretization}.py` implement subset enumeration with bucket-distribution interpolation for next-state smoothing.
- [x] `src/discount_engine/dp/policy.py` is implemented with export/report helpers.
- [x] `scripts/dp_{solve,validate,evaluate}.py` are functional CLI entrypoints.
- [x] DP placeholder tests were replaced with behavior/snapshot/e2e tests.
- [x] Professor scope controls are mostly enforced in implementation:
- [x] Calibration artifact/config is in place (`data/processed/mdp_params.yaml`) with default Phase 2 run mode at `N=3`,
- [x] DP-policy analysis workflow is in place before RL handoff (`notebooks/dp_walkthrough.ipynb` + DP artifact scripts),
- [ ] discrete-action RL method selection (`DQN`) remains pending implementation.

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
- [x] Introduce `src/discount_engine/core`, `src/discount_engine/dp`, `src/discount_engine/rl`.
- [x] Move files from legacy folders (`agents`, `simulators`, `envs`) into new ownership boundaries.
- [ ] Maintain temporary compatibility imports while scripts/tests migrate.
- [ ] Remove compatibility shims once CI and docs fully reference new paths.

Exit criteria:
- [x] No DP-only logic lives under `rl/`; no RL-only logic lives under `dp/`.
- [ ] Shared math/params used by both tracks lives only under `core/`.

## Data Pipeline Ownership (Download + Preprocess)
Canonical locations:
- [x] Raw data acquisition scripts: `scripts/data/download_dataset.py`.
- [x] Preprocessing and panel-build scripts: `scripts/data/preprocess_data.py`.
- [x] Calibration scripts producing MDP params: `scripts/data/calibrate_mdp_params.py`.
- [x] Raw dumps in `data/raw/`, cleaned feature/panel artifacts in `data/processed/`.

Compatibility path during migration:
- [ ] Keep wrappers at legacy entrypoints (`scripts/download_dataset.py`, `scripts/preprocess_data.py`) until references are updated.

## Data Calibration Plan (Dataset -> MDP Parameters)
Goal: compute proposal parameters from Dunnhumby data before solver training.

- [x] Build household-time-category panel from transactions (`y_{h,t,j}`, unit price, promo flags, quantity, recency).
- [x] Select `N` tractable product categories for Version C/B and publish mapping table.
- [x] Estimate category shelf prices `p_j` as robust non-promo unit-price statistics (median + IQR checks).
- [x] Fit purchase model on panel:
- [x] target `y_{h,t,j} = 1` if category `j` purchased at `(h,t)`,
- [x] logistic with category intercepts plus deal and recency terms,
- [x] estimate memory decay `alpha` by grid search maximizing validation log-likelihood.
- [x] Map fitted coefficients to MDP params:
- [x] `beta_0^{(j)}` from category intercepts,
- [x] `beta_p` from deal coefficient,
- [x] `beta_l` from recency coefficient sign-adjusted to proposal convention,
- [x] `beta_m` from memory/deal coefficient relationship.
- [x] Calibrate churn dynamics:
- [x] define operational churn event (example: no purchases for `H` consecutive periods),
- [x] estimate churn hazard by inactivity level and fit `eta` and `c_0` initialization to match observed hazard curves.
- [x] Persist finalized params to versioned artifact (`data/processed/mdp_params.yaml` or JSON) used by DP and RL configs.
- [x] Phase 2 scope-alignment follow-up: produce a DP-targeted calibration artifact/config for default `N=3` runs.

Exit criteria:
- [x] Parameter estimation script is reproducible end-to-end from raw/processed data.
- [x] Validation metrics and diagnostic plots are stored (purchase calibration + churn calibration).
- [x] One parameter artifact feeds both Version C and Version B implementations.

## Phase 0: Alignment and Interface Freeze
Goal: align repository interfaces and naming to the approved MDP before writing core logic.

- [x] Freeze parameter schema (`delta`, `gamma`, `beta_0`, `beta_p`, `beta_m`, `beta_l`, `alpha`, `eta`, per-category prices).
- [x] Confirm shared action convention: `a=0` for no promotion, `a=i` for category `i`.
- [x] Define two state representations:
- [x] `ContinuousState` for Version B (`c`, vector `m`, vector `l`).
- [x] `DiscreteState` for Version C (binned churn/memory/recency).
- [x] Document transition/reward equations in code-facing form in `specs/`.
- [x] Add acceptance checks for probability mass and terminal-state behavior.
- [x] Finalize ownership mapping across `core/`, `dp/`, and `rl/` modules.
- [x] Finalize script namespaces: `scripts/data/*`, `scripts/dp_*`, `scripts/rl_*`.

Exit criteria:
- [x] One source-of-truth spec for state/action/transition/reward used by DP and RL paths.
- [x] No remaining roadmap references to variable discount depth actions.
- [x] Updated `specs/00_repo_map.md` reflecting new module and script layout.

## Phase 1: Version C Core Dynamics (Initial Implementation Phase 1)
Goal: implement the simplified MDP dynamics needed for exact DP.

- [x] Implement independent logistic purchase model per category in `src/discount_engine/core/demand.py`.
- [x] Implement deterministic memory and recency updates from proposal equations.
- [x] Implement churn transition: churn only on zero-purchase branch, with probability `c_t`.
- [x] Enumerate purchase subsets (`2^N`, with Version C support up to `N <= 5`) to build transition outcomes.
- [x] Implement expected immediate reward as realized revenue from purchased subset at effective prices.
- [x] Add unit tests for:
- [x] valid probability range and normalization,
- [x] monotonic effects (higher memory lowers full-price demand),
- [x] absorbing terminal behavior.

Exit criteria:
- [x] Transition kernel and reward computation pass tests for `N=2` and `N=3`.
- [x] Numerical sanity checks match Phase 1 worked-example directionality.

## Phase 2: Version C Solver and Diagnostics (Initial Implementation Phase 2)
Goal: produce the Phase 2 DP deliverable with interpretable policy outputs.

Version C contract to implement (source: `submissions/phase1.ipynb`):
- [x] Cap categories at `N <= 5` (default run setting `N=3`) and support full purchase-subset enumeration (`2^N` outcomes).
- [x] Use discrete state bins exactly for DP:
- [x] `c_t` in calibrated finite churn buckets (legacy 2-bucket support plus metadata-driven 3-bucket runs),
- [x] `m_t^(i) in {Low, Med, High}`,
- [x] `l_t^(i) in {Recent, Stale}`,
- [x] plus absorbing churn terminal state.
- [x] Default DP discretization centers are data-driven from constrained sweep artifacts:
- [x] `MEMORY_GRID=(0.0, 0.9, 2.0)` and `RECENCY_GRID=(2.0, 12.0)` (selected via low-risk constraints in `runs/dp/grid_search/20260218_034643_constrained_40/`, exposed in `configs/dp/solver.yaml`).
- [x] Keep fixed discount-depth action set `A = {0, 1, ..., N}` (`0` = no promotion).
- [x] Compute tabular transitions analytically from the independent logistic purchase model.
- [x] Solve for exact policy with Value Iteration.

Execution plan to complete Phase 2:
- [x] Implement `src/discount_engine/dp/discretization.py`:
- [x] state grids/binning and stable index mapping for `(c, m, l)` and terminal state.
- [x] Implement `src/discount_engine/dp/transitions.py`:
- [x] enumerate all purchase subsets,
- [x] compute branch probabilities and expected immediate rewards,
- [x] map continuous updates into bucket distributions for interpolation-based smoothing,
- [x] include churn branch only on zero-purchase outcome,
- [x] guarantee normalized probability mass per `(s, a)`.
- [x] Implement `src/discount_engine/dp/value_iteration.py`:
- [x] Bellman backup, convergence checks, deterministic stopping/report fields.
- [x] Implement `src/discount_engine/dp/policy.py`:
- [x] greedy extraction and table export helpers.
- [x] Implement `src/discount_engine/dp/quality_checks.py`:
- [x] hard invariants and conceptual diagnostics (warnings by default).
- [x] Replace script scaffolds:
- [x] `scripts/dp_solve.py` for end-to-end solve and artifact writes,
- [x] `scripts/dp_validate.py` for policy/value checks and invariants,
- [x] `scripts/dp_evaluate.py` for report-oriented summaries/plots.
- [x] Ensure validate/evaluate reuse run-resolved solver bucket grids from `config_resolved.yaml` for artifact-consistent diagnostics.
- [x] Replace placeholder DP tests with behavior tests listed below.

Version C DP test plan (required before Phase 2 sign-off):
- [x] `test_dp_transitions.py`: every `(s, a)` transition distribution sums to 1 within tolerance and has no negative probabilities.
- [x] `test_dp_terminal.py`: terminal state is absorbing and yields zero reward for all actions.
- [x] `test_dp_bellman_backup.py`: Bellman update on a tiny hand-checked fixture matches analytical value.
- [x] `test_dp_value_iteration.py`: Value Iteration converges under configured `epsilon` and `max_iters`.
- [x] `test_dp_policy_sanity.py`: policy reacts correctly to canonical states (for example, high-memory state should not always favor repeated discounting).
- [x] `test_dp_regression.py`: selected state values/policy actions match stored snapshot under fixed parameter artifact.
- [x] `test_dp_script_e2e.py`: command-line DP solve runs end-to-end and writes expected artifacts.

Test artifacts and thresholds:
- [x] Store deterministic fixture parameters in `tests/fixtures/dp_params_small.yaml`.
- [x] Store expected outputs (`V`, policy for selected states) as versioned test snapshots.
- [x] Enforce numerical tolerances explicitly (`atol`, `rtol`) in all DP numerical tests.

Exit criteria:
- [x] End-to-end DP run completes from command line.
- [x] DP run outputs are written to `runs/dp/<timestamp>_<tag>/` with complete artifact set.
- [x] Artifacts show policy trade-off between short-term revenue and future addiction/churn risk.
- [x] Phase 2 report inputs are generated and versioned.
- [x] All Version C DP tests pass in CI.
- [x] No DP placeholder assertions remain in `tests/dp/`.
- [x] DP policy analysis is complete before starting any RL implementation work:
- [x] analyze action frequencies across canonical state clusters,
- [x] analyze value function sensitivity to churn/memory bins,
- [x] summarize when/why promotion is preferred vs no-promotion.

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
- [ ] Phase 3 does not start until the Phase 2 DP policy analysis gate is marked complete.

## Phase 4: RL Training, Evaluation, and Final Packaging
Goal: train and evaluate a model-free agent for Version B and finalize deliverables.

- [ ] Train RL agent with a discrete-action method (primary: DQN) for Version B.
- [ ] Compare against baselines on revenue, promotion frequency, retention proxy, and margin impact.
- [ ] Run targeted hyperparameter sweeps and ablations.
- [ ] Produce final plots/tables and integrate results into report/presentation material.
- [ ] Final code cleanup and reproducibility pass.

Exit criteria:
- [ ] RL policy outperforms non-learning baselines on agreed metrics.
- [ ] Final paper/presentation artifacts are reproducible from repository scripts.

## Immediate Next Sequence (Post-Audit)
1. Keep DP regression snapshots and walkthrough outputs in sync when calibration metadata or solver defaults change.
2. Treat `quality_warnings.json` conceptual checks as a report review item before RL training runs.
3. Begin Phase 3 continuous-state environment work only after explicit DP policy sign-off.

## Documentation Synchronization Rule
- [x] After each completed implementation step, update `specs/00_repo_map.md` in the same change set.
- [ ] Treat stale repo-map docs as a blocker for marking a phase step complete.

## Script Naming Convention
- [x] Adopt `dp_` prefix for DP-focused executable scripts in `scripts/` for clear discoverability.
- [x] Target names: `scripts/dp_solve.py`, `scripts/dp_validate.py`, `scripts/dp_evaluate.py`.
- [x] Adopt `rl_` prefix for RL-focused executable scripts.
- [x] Target names: `scripts/rl_train.py`, `scripts/rl_run_simulation.py`, `scripts/rl_evaluate.py`.
- [ ] Keep temporary compatibility wrappers if needed (for example, `scripts/solve_dp.py` delegating to `scripts/dp_solve.py`) until docs/CI are updated.
