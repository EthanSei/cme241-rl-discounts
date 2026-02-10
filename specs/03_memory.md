# Project Memory & ADRs

## 2026-01-31: Phase 1.2 Complete

### SimulatedCustomer Implementation
Implemented `SimulatedCustomer` class modeling discount addiction behavior:
- **Reference price memory decay:** Uses exponential smoothing with α=0.8
- **Purchase probability:** Decays exponentially when price > reference
- **Calibrated beta:** 2.0 for probability model (distinct from elasticity beta)

Key insight: The elasticity β=0.045 from Phase 1.1 measures quantity sensitivity
to price. For the probability model, we need a separate β=2.0 calibrated to
satisfy the 20% probability drop criterion after 5 discounted purchases.

### Validation Results
Ran `make sims` with 500 customers:
- After 5 purchases at 20% discount, avg probability drops 23.6%
- Fresh customers maintain baseline 50% probability at full price
- Results saved to `data/processed/simulation_validation.json`

### Files Added/Modified (legacy paths; superseded by v2 structure)
- `src/discount_engine/core/dynamics.py` - SimulatedCustomer/dynamics scaffold
- `tests/rl/test_rl_env.py` - RL environment test scaffold
- `scripts/rl_run_simulation.py` - Simulation script scaffold

---

## 2026-01-31: Phase 1.1 Complete

### Elasticity Parameters (SOFT DRINKS)
Estimated from 117,209 transactions:
- **α (memory decay):** 0.8
- **β (price sensitivity):** 0.045
- **intercept:** 1.347
- **memory_coef:** -0.218

Parameters saved to `data/processed/elasticity_params.json`.

### Data Pipeline
- Raw CSVs ingested to parquet format
- Filtered to SOFT DRINKS category
- Training sequences built with coupon availability/usage flags
- Training features generated with demographic joins

### Test Coverage
21 tests total covering:
- Data ingestion and loading (3)
- Category filtering (1)
- Sequence building and validation (3)
- Elasticity estimation (1)
- Feature engineering (3)
- Customer simulation (10)
