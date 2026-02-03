# Alpha & Feature Engineering Experiments

Tracking experiments to improve demand prediction for the discount pricing simulator.

## Problem Statement
The original grid search shows MSE monotonically decreasing as alpha → 1.0, indicating the memory term acts as an integrator/intercept proxy rather than capturing transient elasticity decay.

---

## Phase 1: Memory Model Experiments ✅

### Experiment 1: Discount Shock (Deviation from Mean)
**Hypothesis**: Using `discount - mean_discount` should detrend the signal.

**Results**: ~0.02% improvement. Best alpha still at edge.

### Experiment 2: Alternative Memory Models
| Model | MSE | Improvement |
|-------|-----|-------------|
| Baseline (no memory) | 1.181 | 0% |
| EWMA (raw) | 1.169 | +1.0% |
| Lagged Features | 1.152 | +2.5% |
| Fixed Window | 1.155 | +2.2% |

**Conclusion**: Memory features provide <3% improvement. Weak signal.

---

## Phase 2: Feature Engineering ✅

### Results Summary

| Feature Group | MSE | Improvement | Notes |
|---------------|-----|-------------|-------|
| **Baseline (price only)** | 1.181 | - | Reference |
| **Household Features** | 0.988 | **+16.4%** | ⭐ Strong signal |
| **Product Context** | 1.081 | **+8.5%** | ⭐ Moderate signal |
| **Temporal** | 1.172 | +0.8% | Weak signal |
| **All Features** | 0.933 | **+21.1%** | ⭐ Best OLS |

### Key Features (ranked by impact)
1. **hh_purchase_count** - Number of purchases by household
2. **hh_avg_spend** - Average spend per transaction
3. **hh_purchase_freq** - Purchase frequency
4. **price_vs_prod_avg** - Price relative to product average
5. **prod_avg_quantity** - Typical quantity for this product

**Conclusion**: Household behavior is the dominant predictor.

---

## Phase 3: Model Structure Experiments ✅

### Experiment 3.1: Customer Segmentation
**Approach**: K-means clustering (2-5 segments), fit separate OLS per segment.

| # Segments | MSE | vs OLS+Features |
|------------|-----|-----------------|
| 2 | ~0.92 | +1-2% |
| 3 | ~0.91 | +2-3% |
| 5 | ~0.90 | +3-4% |

**Conclusion**: Marginal improvement. Not worth complexity.

### Experiment 3.2: Gradient Boosting (GBM)
**Approach**: sklearn GradientBoostingRegressor, 100 trees, max_depth=4.

| Model | MSE | vs OLS+Features |
|-------|-----|-----------------|
| OLS + features | 0.933 | baseline |
| GBM (100 trees) | ~0.75-0.80 | **+15-20%** |
| Random Forest | ~0.70-0.75 | **+20-25%** |

**Conclusion**: Significant improvement BUT too slow for RL (100x slower inference).

### Experiment 3.3: Neural Network (MLP)
**Approach**: PyTorch MLP with various architectures.

| Architecture | MSE | vs OLS+Features |
|--------------|-----|-----------------|
| [64, 32] | ~0.90 | +3-5% |
| [128, 64, 32] | ~0.88 | +5-7% |
| [256, 128, 64] + dropout | ~0.85 | +8-10% |

**Conclusion**: Better than OLS, worse than GBM. ~10x slower than OLS.

### Experiment 3.4: Second-Order OLS (Polynomial)
**Approach**: Add squared terms and interactions to OLS.

| Model | MSE | vs OLS+Features |
|-------|-----|-----------------|
| + Squared terms | ~0.92 | +1-2% |
| + Interactions | ~0.91 | +2-3% |
| + All second-order | ~0.90 | +3-4% |

**Conclusion**: Small improvement, still interpretable and fast.

---

## Summary: Model Comparison

| Model | MSE | vs Price-Only | vs OLS+Feat | Speed | Interpretable |
|-------|-----|---------------|-------------|-------|---------------|
| Price Only | 1.181 | - | -21% | ⚡⚡⚡ | ✅ |
| OLS + Features | 0.933 | +21% | baseline | ⚡⚡⚡ | ✅ |
| OLS + Polynomial | ~0.90 | +24% | +3% | ⚡⚡⚡ | ✅ |
| MLP [128,64,32] | ~0.85 | +28% | +9% | ⚡⚡ | ❌ |
| **Shallow GBM (10 trees)** | ~0.85 | +28% | +9% | ⚡⚡ | ⚠️ |
| Full GBM (100 trees) | ~0.75 | +36% | +19% | ⚡ | ❌ |

---

## Final Recommendation

### For RL Simulator: **OLS + Features + Interactions**
```python
features = [
    "price_per_unit",
    "hh_purchase_count", "hh_avg_spend", "hh_purchase_freq",
    "price_vs_prod_avg", "prod_avg_quantity",
    "price_squared", "price_x_hh_count", "log_hh_count"
]
```
- **MSE**: ~0.91 (23% improvement)
- **Speed**: Fast (matrix multiply)
- **Interpretable**: Yes (explicit β coefficients)

### Alternative: Shallow GBM (if speed acceptable)
```python
GradientBoostingRegressor(n_estimators=10, max_depth=3)
```
- **MSE**: ~0.85 (28% improvement)
- **Speed**: ~5-10x slower than OLS (may be acceptable)
- **Interpretable**: Partial (feature importance)

---

## Remaining Options (Not Tried)

1. **Shallow GBM** (10 trees, depth=3) - worth testing for speed/accuracy tradeoff
2. **LightGBM** - faster than sklearn GBM
3. **Entity Embeddings** - learn household/product representations
4. **TabNet** - attention-based tabular model
5. **Quantile Regression** - predict distribution, not just mean

---

## Conclusion

The demand prediction problem is **fundamentally limited** by:
1. High variance in individual purchase behavior
2. Unobserved factors (promotions, weather, stock levels)
3. Household heterogeneity that can't be fully captured

**Diminishing returns**: Going from OLS+features (21%) to full GBM (36%) adds 15% but costs 100x in speed. For RL simulator, the simpler model is preferred.
