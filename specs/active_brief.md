# Active Brief

## Current Status (2026-02-23)

Phase 2 (tabular DP) is **submission-ready**. The submission artifact is `notebooks/dp_walkthrough.ipynb`.
All 6 bugs from the external audit have been fixed. All 57 unit tests pass (3 new `evaluate_policy` tests).
The notebook executes top-to-bottom without errors (kernel: `cme241`, Python 3.12).

### Canonical Run
- **Path:** `runs/dp/20260219_065444_data-driven-final/`
- **Parameters:** data-driven only — no literature overrides
  - `alpha = 0.95`, `beta_m ≈ 0` (1e-6), `deal_signal_mode = positive_centered_anomaly`
  - Product-level deal signals (unscaled): Milk=2.2789, IC=5.8252, FP=5.2508
- **Result:** Policy collapse — action 2 (Promote Ice Cream) in all 648 states
- **Corrected uplift (iterative policy evaluation):** 81.9% overall vs never-promote (160.1% Lapsing)

### Parameter File
- `data/processed/mdp_params.yaml` — source of truth for all MDP parameters
- `configs/dp/solver.yaml` — discretization grids: `memory_grid: [0.0, 0.9, 2.0]`, `recency_grid: [1.0, 4.0]`

---

## BUGS FIXED (2026-02-23)

### Bug 1: Baseline Methodology (CRITICAL) ✓ FIXED

**Location:** `notebooks/dp_walkthrough.ipynb` cell 29 (Section 11: Baseline Comparisons)

**Problem:** Baselines are computed using Q\*(s, a) from the optimal Q-function, not the true value of following a fixed policy. Concretely:

```python
# CURRENT (WRONG): uses Q* — assumes you take action a once, then act optimally forever
vals = [float(q_dict[s].get(action_fn(s, i), 0.0)) for i, s in enumerate(live)]
```

The correct approach is **iterative policy evaluation**: solve V^π(s) = R(s, π(s)) + γ Σ P(s'|s,π(s)) V^π(s') to convergence for each fixed policy. This gives the true long-run value of always following that policy.

**Impact:** Q\* overestimates baseline values (because future states use V\*, not V^π), which **understates** the DP uplift. The reported 1.60% uplift is a lower bound; the true uplift vs. never-promote is larger.

**Fix plan (approved by TDD enforcer, then abandoned for handoff):**

1. Add `evaluate_policy()` to `src/discount_engine/dp/value_iteration.py`:
   ```python
   def evaluate_policy(
       policy: dict[DiscreteState, int],
       params: MDPParams,
       config: ValueIterationConfig,
   ) -> dict[DiscreteState, float]:
   ```
   This reuses the existing `_build_transition_cache` / `_q_from_kernel` infrastructure. For each state, instead of `max_a Q(s,a)`, compute `Q(s, π(s))`.

2. Write a test: `evaluate_policy` with the optimal policy should return the same values as `solve_value_iteration`.

3. Update notebook cell 29: build fixed-policy dicts for Never Promote (all 0), Always ICE CREAM (all 2), etc., call `evaluate_policy`, report mean V^π across live states.

4. Update all downstream uplift claims (cells 13, 22, 40) with the corrected numbers.

**Current (wrong) baseline output from cell 29:**
```
                                       mean_v   uplift_pct
Never Promote                        330.0090       0.0000
Always Promo ICE CREAM/MILK/SHERBTS  335.2751       1.5958
Optimal (DP)                         335.2751       1.5958
Random Uniform                       333.2620       0.9857
```

Note: "Optimal" and "Always ICE CREAM" are identical because the optimal policy IS always ICE CREAM (policy collapse). This equality is expected and correct — the issue is that the Never Promote and Random Uniform values are too high.

### Bug 2: MDP Equations Don't Match Implementation (MEDIUM) ✓ FIXED

**Location:** `notebooks/dp_walkthrough.ipynb` cell 3 (Section 2: MDP Formulation Recap)

Three discrepancies between the notebook equations and `src/discount_engine/dp/transitions.py`:

**(a) Memory update formula**

Notebook says:
```
m_i' = α · m_i + (1 - α) · d_i · 𝟙[a = i]
```

Code does (line 222-223 of transitions.py):
```python
promo_bump = params.delta if promoted else 0.0
mem_val = (params.alpha * decoded.discount_memory[idx]) + promo_bump
```

**Discrepancy:** Code adds `delta` directly when promoted (not `(1-α) · d_i`). The `(1-α)` scaling and the use of `d_i` (deal signal) instead of `δ` (discount depth) are both wrong in the notebook.

**Correct equation:** `m_i' = α · m_i + δ · 𝟙[a = i]`

**(b) Churn update formula**

Notebook says:
```
c' = c + η · (1 - any purchase)
```

Code does (lines 232-235):
```python
if any(purchases):
    churn_val = max(0.0, current_churn - params.eta)
else:
    churn_val = min(1.0, current_churn + params.eta)
```

**Discrepancy:** The notebook formula only captures the "no purchase → increase churn" case. The code also **decreases** churn by η when a purchase occurs, and clamps to [0, 1].

**Correct equation:**
```
c' = clamp(c − η, 0, 1)  if any purchase
c' = clamp(c + η, 0, 1)  if no purchase
```

**(c) Deal signal / reward description**

Notebook says:
> The deal signal d_i uses the `price_delta_dollars` contract: d_i = p_i · δ when promoted.

But the DP actually uses `positive_centered_anomaly` mode (reads `promotion_deal_signal` directly from params). The deal signal description should mention this, or at minimum note that the DP uses a different mode than calibration.

Notebook's reward formula `R(s, a) = Σ P(buy_i) · p_eff_i` is technically the expected immediate revenue, which is correct. But the transition kernel enumerates all 2^N purchase subsets (since next-state depends on which items were purchased). This nuance could be briefly noted.

**Fix:** Rewrite the equations in cell 3 to match the actual implementation.

### Bug 3: Stale Narrative in Section 8 (HIGH) ✓ FIXED

**Location:** `notebooks/dp_walkthrough.ipynb` cell 19 (Section 8: Policy Analysis)

**Problem:** Cell 19 still says:
```
The optimal policy reveals a clear **state-dependent promotion strategy**:
- **Lapsing customers** → promote Milk
- **Engaged / At-Risk** → promote Ice Cream
- **The healthiest Engaged states** → no promotion
```

This is **completely wrong** for the current policy, which is 100% Ice Cream in all 648 states (policy collapse). Cell 22 (two cells later) correctly describes the policy collapse. Cell 19 is stale from an older run with different parameters.

**Fix:** Rewrite cell 19 to introduce the policy collapse finding instead of the old state-dependent narrative.

### Bug 4: Wrong Multiplier Range (MEDIUM) ✓ FIXED

**Location:** Cells 13, 40 (and possibly 9)

**Problem:** The notebook claims product-level signals are "1.9–7.9x" stronger than category-level. The actual multipliers (from the executed cell 12 output) are:
- Milk: 3.0x
- Ice Cream: 7.5x
- Frozen Pizza: 8.7x

So the correct range is **3.0–8.7x**, not 1.9–7.9x.

**Locations to fix:**
- Cell 13: "Product-level signals are **1.9–7.9x** stronger" → "**3–9x** stronger"
- Cell 40 (Key Takeaways): "1.9–7.9x" appears twice

### Bug 5: Causal Caveat Insufficiently Prominent (LOW) ✓ FIXED

**Location:** Cell 13 (Section 5 methodology), cell 39 (Section 14 limitations)

**Problem:** The product-level deal signals are observational logit bumps (products that are on-deal more often may be purchased more for confounding reasons). The limitation IS mentioned in the Section 14 table ("Observational deal signals (not causal)") but should also be noted where the methodology is introduced (cell 13 or nearby).

**Fix:** Add a brief caveat sentence in cell 13 noting that product-level logit bumps are observational and may include selection bias.

### Bug 6: Stale Recency Grid in Spec Files (LOW — not in notebook) ✓ FIXED

**Location:** `specs/active_brief.md` (this file — already fixed by this rewrite), `specs/02_roadmap.md`

**Problem:** References `RECENCY_GRID=(2.0, 12.0)` but `configs/dp/solver.yaml` uses `recency_grid: [1.0, 4.0]` (weekly resolution). The notebook (cell 40) correctly says `[1.0, 4.0]`.

**Fix:** Update `specs/02_roadmap.md` line 132 to say `RECENCY_GRID=(1.0, 4.0)`.

---

## NOTEBOOK CELL INDEX (40 cells total)

| Cell | Type | Section | Content summary |
|------|------|---------|-----------------|
| 0 | md | — | Title + section list |
| 1 | md | §1 | Setup header |
| 2 | code | §1 | Imports, paths, load params, print summary |
| 3 | md | §2 | MDP Formulation Recap (equations corrected) |
| 4 | md | §3 | Calibration methodology |
| 5 | code | §3 | Parameter summary table |
| 6 | code | §3 | Purchase probability visualization |
| 7 | md | §4 | Category question intro |
| 8 | code | §4 | Category comparison viz |
| 9 | md | §5 | Product-level targeting intro + break-even equation |
| 10 | code | §5 | Break-even analysis |
| 11 | code | §5 | Product-level search (finds top products per category) |
| 12 | code | §5 | Category vs product comparison table + chart |
| 13 | md | §5 | Product-level deal signals in DP (multiplier + causal caveat fixed) |
| 14 | md | §6 | Daily vs weekly resolution |
| 15 | md | §7 | DP solver intro |
| 16 | code | §7 | Load solver config, enumerate states |
| 17 | code | §7 | Run value iteration |
| 18 | code | §7 | Convergence plot |
| 19 | md | §8 | Policy analysis intro (rewritten for policy collapse) |
| 20 | code | §8 | Build policy table |
| 21 | code | §8 | Action distribution by churn bucket |
| 22 | md | §8 | Policy structure — data-driven finding (policy collapse) |
| 23 | md | §9 | Value function interpretation |
| 24 | code | §9 | Value distribution by churn bucket |
| 25 | code | §9 | Value heatmap |
| 26 | md | §10 | Q-gap diagnostics intro |
| 27 | code | §10 | Q-gap computation |
| 28 | md | §11 | Baseline comparisons intro |
| 29 | code | §11 | Baseline evaluation (now uses iterative policy evaluation) |
| 30 | code | §11 | Baseline visualization |
| 31 | md | §12 | Memory investigation |
| 32 | code | §12 | Calibration experiment results table |
| 33 | md | §12 | Memory verification header |
| 34 | code | §12 | Q-value variation across memory states |
| 35 | md | §12 | Memory verification conclusion |
| 36 | md | §13 | Quality checks header |
| 37 | code | §13 | Run quality checks |
| 38 | md | §14 | Limitations & Phase 3 Motivation |
| 39 | md | §15 | Key Takeaways (corrected multiplier range + uplift) |

---

## KEY SOURCE FILES

### Transition dynamics (ground truth for equations)
- `src/discount_engine/dp/transitions.py` — purchase probs, deal signal modes, memory/recency/churn updates, reward computation, full 2^N purchase subset enumeration

### Value iteration + policy evaluation
- `src/discount_engine/dp/value_iteration.py` — `solve_value_iteration()`, `evaluate_policy()`, `bellman_action_value()`, `bellman_backup()`, transition cache infrastructure.

### Policy helpers
- `src/discount_engine/dp/policy.py` — `extract_greedy_policy()`, `policy_rows()`, `build_evaluation_summary()`, `state_to_id()` / `state_from_id()`

### Discretization
- `src/discount_engine/dp/discretization.py` — `enumerate_all_states()`, `enumerate_live_states()`, `is_terminal_state()`, `resolve_churn_grid()`, `decode_state()`, `temporary_bucket_grids()`, `MEMORY_GRID`, `RECENCY_GRID` constants

### Parameters
- `src/discount_engine/core/params.py` — `MDPParams`, `CategoryParams`, `load_mdp_params()`
- `src/discount_engine/core/types.py` — `DiscreteState(churn_bucket, memory_buckets, recency_buckets)`, `ContinuousState`

### Tests
- `tests/dp/test_dp_value_iteration.py` — VI tests + 3 `evaluate_policy` tests.
- All 57 tests pass as of last run.

### Configs
- `configs/dp/solver.yaml` — gamma=0.99, epsilon=1e-8, max_iters=10000, memory_grid=[0.0, 0.9, 2.0], recency_grid=[1.0, 4.0]
- `configs/data/calibration.yaml` — calibration settings (beta_m_floor: null)
- `data/processed/mdp_params.yaml` — calibrated MDP parameters

---

## IMPLEMENTATION NOTES

### How value iteration works (for understanding evaluate_policy)
The solver in `value_iteration.py` uses a vectorized approach:
1. `_build_transition_cache()` precomputes `_ActionKernel(next_indices, rewards, probabilities)` for every (state, action) pair
2. `_q_from_kernel()` computes Q(s,a) = dot(probabilities, rewards + gamma * values[next_indices])
3. Value iteration loops: for each state, compute Q for all actions, take max → new V(s)

Policy evaluation is identical except step 3 becomes: for each state, compute Q for the **fixed policy action** only → new V^π(s). The transition cache is reusable.

### Deal signal modes
- `price_delta_dollars`: d = price × delta when promoted (used by calibration)
- `positive_centered_anomaly`: d = `category.promotion_deal_signal` directly (used by DP, reads product-level logit bumps pre-divided by beta_p)
- `binary_delta_indicator`: d = delta when promoted (not used in current runs)

### State representation
- States are `DiscreteState(churn_bucket=int, memory_buckets=tuple[int,...], recency_buckets=tuple[int,...])`
- With 3 categories, 3 churn buckets, 3 memory levels, 2 recency levels: 3 × 3³ × 2³ = 648 live states + 1 terminal
- Q-values in JSON use string keys like `"c:0|m:0,0,0|l:0,0,0"`

### Policy collapse explanation
With beta_m ≈ 0 and only 4 actions, the promotion ROI ranking is state-independent:
- Ice Cream: promotion_deal_signal=5.8252, price=$2.59 → best ROI
- Frozen Pizza: promotion_deal_signal=5.2508, price=$2.00
- Milk: promotion_deal_signal=2.2789, price=$2.49
The Q-gap (best minus second-best) is consistently small (mean 0.49, range 0.27–0.87 over live states).

---

## VERSION C CONTRACT

- Categories: up to N ≤ 5, default N=3
- State: (churn_bucket, memory_buckets, recency_buckets) + terminal
- Actions: {0, 1, ..., N} — 0 = no promotion, i = promote category i
- Discount depth: fixed δ = 0.30
- Solution: exact tabular value iteration
- Gamma: 0.99

## ARCHITECTURE

```
src/discount_engine/
├── core/           # Shared: params, types, demand equations
├── dp/             # Phase 2: discretization, transitions, VI, policy, quality checks
└── rl/             # Phase 3: Gymnasium environment, DQN agent
```

Separation rule: `dp/` never imports from `rl/`, `rl/` never imports from `dp/`. Both share `core/`.

## TDD PROTOCOL

The repo has a `tdd-enforcer` MCP server. For any new source code:
1. `start_task(goal)` → Discovery
2. `submit_plan(plan)` → Plan approval
3. Write failing test → `report_test_failure(output)`
4. Write implementation → `complete_task()`

The Bug 1 fix already went through steps 1-2 (plan approved) before being abandoned for this handoff. The new agent can restart with `start_task()`.
