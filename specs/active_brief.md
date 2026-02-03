# Current Task
Phase 1.2: Simulator Implementation

## Phase 1.1 Complete ✅

### 1.1.1 Memory Models (<3% improvement)
- EWMA, lagged features, fixed window all tested
- **Conclusion:** Weak signal, not worth complexity

### 1.1.2 Feature Engineering (+21% improvement ⭐)
| Feature Group | Improvement |
|---------------|-------------|
| Household | +16% |
| Product Context | +8% |
| All Features | +21% |

### 1.1.3 Model Structure (+2% additional)
| Model | vs OLS+Features | Speed |
|-------|-----------------|-------|
| + Interactions | +2% | ⚡⚡⚡ |
| Shallow GBM | +5-8% | ⚡⚡ |
| Full GBM | +15-20% | ⚡ |
| MLP | +3-5% | ⚡⚡ |

**Conclusion:** Interactions worth including (free). GBM/MLP too slow.

---

## Final Model Decision ✅

**OLS + Features + Interactions** (23% improvement)

```python
features = [
    "price_per_unit",
    "hh_purchase_count", "hh_avg_spend", "hh_purchase_freq",
    "price_vs_prod_avg", "prod_avg_quantity",
    "price_squared", "price_x_hh_count", "log_hh_count"
]
```

- **MSE**: ~0.91 (vs 1.18 baseline)
- **Speed**: ⚡⚡⚡ (fast matrix multiply)
- **Interpretable**: ✅ (explicit β coefficients)

---

## Next Steps

1. [ ] Export model coefficients to `data/processed/demand_model.json`
2. [ ] Implement `SimulatedCustomer` in `src/discount_engine/simulators/customer.py`
3. [ ] Validate customer "addiction" behavior
4. [ ] Move to Phase 1.3: Gym Environment
