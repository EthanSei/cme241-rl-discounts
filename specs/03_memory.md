# Project Memory & ADRs

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
11 tests covering:
- Data ingestion and loading (3)
- Category filtering (1)
- Sequence building and validation (3)
- Elasticity estimation (1)
- Feature engineering (3)
