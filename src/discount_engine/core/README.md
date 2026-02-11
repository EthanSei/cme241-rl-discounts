# Core Calibration README

This document explains how calibration works in `src/discount_engine/core/calibration.py`, including model equations, assumptions, and outputs.

## Purpose
Calibration estimates the MDP parameters used by DP and RL:
- Global: `alpha`, `beta_p`, `beta_l`, `beta_m`, `eta`, `c0`, `delta`, `gamma`
- Per-category: `beta_0` intercept and representative category `price`

The public API is:
- `calibrate_mdp_params(...)`

The output artifact is:
- `data/processed/mdp_params.yaml` (or custom output path)

## Data Inputs
Required processed tables:
- `transaction_data`
- `product`

Expected key columns:
- transactions: `household_key`, `product_id`, `quantity`, `sales_value`, discount columns
- product: `product_id`, category column (default `commodity_desc`)

## End-to-End Pipeline
`calibrate_mdp_params` runs these stages:

1. Load processed tables (Parquet preferred, CSV fallback).
2. Clean transaction and product tables.
3. Compute transaction-level `unit_price` and bounded `discount_rate`.
4. Select top `n_categories` by transaction volume.
5. Build a dense household-time-category panel.
6. Fit purchase model over an `alpha` grid and select alpha by validation NLL.
7. Estimate churn parameters (`c0`, `eta`) from inactivity dynamics.
8. Estimate representative category prices.
9. Save final parameter bundle to YAML.

## Purchase Model
For category `j` at panel row `t`:

`logit_t(j) = beta_0(j) + beta_p * deal_signal_t(j) - beta_l * recency_t(j) - beta_m * memory_t(j)`

`P(purchase_t(j)=1) = sigmoid(logit_t(j))`

Sign constraints are enforced in fitting:
- `beta_p >= 0`
- `beta_l >= 0`
- `beta_m >= 0`

This keeps interpretation monotonic:
- larger deal signal raises purchase propensity
- larger recency/memory lower purchase propensity

## Leakage-Safe Deal Signal Construction
The panel target is:
- `purchased = 1` if household purchased category at that time, else `0`

Deal features are built separately from the target:

1. Category-time aggregate discount stats across households.
2. Household contribution stats at the same category-time.
3. Leave-one-household-out category-time discount rate:
   - excludes the household's own transaction contribution when available.
4. Category baseline discount (median over time).
5. Final signal:
   - `deal_signal = max(0, discount_rate - category_baseline)`
   - fallback to `discount_rate` if anomaly collapses to zero.

This avoids encoding `purchased` directly inside the feature.

## Recency and Memory Features
### Recency
Recency is computed per household-category sequence in chronological order:
- time since previous purchase in that sequence.

### Gap-Aware Memory
Memory is computed as a gap-aware EWMA:

`m_t = alpha^(delta_t) * m_(t-1) + deal_signal_(t-1)`

where `delta_t` is elapsed time between adjacent rows in a household-category sequence.

This prevents irregular time gaps from being treated as equal-length steps.

## Alpha Selection
Alpha is selected from `alpha_grid` using a chronological split:
- Train = earlier rows
- Validation = latest `validation_fraction` rows

Per alpha:
1. Build memory feature.
2. Fit logistic model on train rows.
3. Evaluate train and validation NLL.
4. Select alpha with minimum validation NLL.

Warm-starting and sequence caching are used to improve runtime.

## Churn Calibration
Churn parameters are estimated from household activity run-lengths:

- Build inactivity runs over observed timeline.
- Estimate baseline no-purchase propensity at run length 0 -> `c0` (clipped).
- Fit slope of no-purchase hazard vs run length buckets -> `eta` (clipped).
- Report `churn_rate_at_horizon` in metadata.

This is a coarse operational calibration, not a full survival model.

## Price Estimation
Per category:
- Use median non-promo unit price when available.
- Fallback to median overall unit price.
- Fallback to `1.0` if invalid.

## Output Metadata
Saved metadata includes:
- file/time/category configuration
- selected categories and panel size
- churn stats and inactivity horizon
- alpha selection diagnostics:
  - `fit_train_neg_log_likelihood`
  - `fit_val_neg_log_likelihood`
  - `alpha_selection_metric=validation_nll`
  - `validation_fraction`
- feature/memory mode tags:
  - `deal_signal_mode`
  - `memory_mode`

## Assumptions
1. Category-level modeling is sufficient for tractability.
2. Category-time discount context proxies household promo exposure.
3. Purchase outcomes across categories are modeled independently.
4. One global set of `beta_p`, `beta_l`, `beta_m` applies to all categories.
5. Chronological holdout is enough for alpha model selection.

## Known Limitations
1. No explicit household-level promo-exposure data.
2. No household random effects in the logistic fit.
3. Churn calibration is simplified and bucket-based.
4. Category aggregation can hide product-level heterogeneity.

## Relation to DP/RL Contract
Calibration still produces the same parameter schema consumed by DP/RL:
- `alpha`, `beta_p`, `beta_l`, `beta_m`, `eta`, `c0`, per-category `beta_0`, `price`, plus `delta`, `gamma`.

The changes improve estimation quality and diagnostics; they do not change the DP/RL interface contract.
