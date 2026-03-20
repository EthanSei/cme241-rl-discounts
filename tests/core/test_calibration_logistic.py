"""Unit tests for the logistic fitting and evaluation functions in calibration."""

from __future__ import annotations

import numpy as np
import pytest

from discount_engine.core.calibration import (
    _fit_logistic_once,
    _standardize,
    _safe_rescale,
    _evaluate_neg_log_likelihood,
    evaluate_nll,
    fit_logistic,
)


# ---------------------------------------------------------------------------
# _standardize
# ---------------------------------------------------------------------------

class TestStandardize:
    def test_zero_mean_unit_variance(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x_std, mu, sigma = _standardize(x)
        assert mu == pytest.approx(3.0)
        assert sigma == pytest.approx(x.std(), abs=1e-12)
        assert x_std.mean() == pytest.approx(0.0, abs=1e-12)
        assert x_std.std() == pytest.approx(1.0, abs=1e-12)

    def test_constant_feature_returns_unchanged(self):
        x = np.array([7.0, 7.0, 7.0])
        x_std, mu, sigma = _standardize(x)
        assert sigma < 1e-12
        np.testing.assert_array_equal(x_std, x)
        assert mu == pytest.approx(7.0)

    def test_single_element(self):
        x = np.array([42.0])
        x_std, mu, sigma = _standardize(x)
        assert mu == pytest.approx(42.0)
        # std of single element is 0
        assert sigma == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_array_equal(x_std, x)


# ---------------------------------------------------------------------------
# _safe_rescale
# ---------------------------------------------------------------------------

class TestSafeRescale:
    def test_normal_rescale(self):
        assert _safe_rescale(2.0, 0.5) == pytest.approx(4.0)

    def test_near_zero_sigma_returns_coef(self):
        assert _safe_rescale(3.0, 1e-15) == pytest.approx(3.0)

    def test_zero_sigma_returns_coef(self):
        assert _safe_rescale(5.0, 0.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _evaluate_neg_log_likelihood / evaluate_nll
# ---------------------------------------------------------------------------

class TestEvaluateNLL:
    @pytest.fixture()
    def simple_data(self):
        """Perfectly separable tiny dataset."""
        y = np.array([1.0, 1.0, 0.0, 0.0])
        cat_idx = np.array([0, 0, 0, 0])
        deal = np.array([1.0, 1.0, 0.0, 0.0])
        recency = np.zeros(4)
        memory = np.zeros(4)
        return y, cat_idx, deal, recency, memory

    def test_perfect_prediction_has_low_nll(self, simple_data):
        y, cat_idx, deal, recency, memory = simple_data
        # Large positive intercept + large deal coefficient → high prob for y=1
        intercepts = np.array([-5.0])
        nll = _evaluate_neg_log_likelihood(
            y, cat_idx, deal, recency, memory,
            intercepts, deal_coef=10.0, recency_coef=0.0, memory_coef=0.0,
        )
        assert nll < 1.0  # near-perfect predictions → very low NLL

    def test_evaluate_nll_matches_private(self, simple_data):
        y, cat_idx, deal, recency, memory = simple_data
        intercepts = np.array([0.0])
        nll_private = _evaluate_neg_log_likelihood(
            y, cat_idx, deal, recency, memory,
            intercepts, deal_coef=1.0, recency_coef=0.0, memory_coef=0.0,
        )
        nll_public = evaluate_nll(
            y, cat_idx, deal, recency, memory,
            intercepts, 1.0, 0.0, 0.0,
        )
        assert nll_public == pytest.approx(nll_private)

    def test_empty_data_returns_nan(self):
        empty = np.array([], dtype=float)
        empty_int = np.array([], dtype=int)
        result = _evaluate_neg_log_likelihood(
            empty, empty_int, empty, empty, empty,
            np.array([0.0]), 0.0, 0.0, 0.0,
        )
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# _fit_logistic_once / fit_logistic
# ---------------------------------------------------------------------------

def _make_synthetic_data(
    n_per_cat: int = 200,
    n_cats: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic binary-outcome data with known coefficient signs."""
    rng = np.random.RandomState(seed)
    n = n_per_cat * n_cats
    cat_idx = np.repeat(np.arange(n_cats), n_per_cat)
    deal = rng.uniform(0, 1, n)
    recency = rng.uniform(0, 10, n)
    memory = rng.uniform(0, 5, n)

    # True model: higher deal → more purchase, higher recency/memory → less
    intercepts = rng.uniform(-1, 1, n_cats)
    logits = intercepts[cat_idx] + 2.0 * deal - 0.3 * recency - 0.5 * memory
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(0, 1, n) < probs).astype(float)
    return y, cat_idx, deal, recency, memory


class TestFitLogisticOnce:
    def test_recovers_positive_deal_coefficient(self):
        y, cat_idx, deal, recency, memory = _make_synthetic_data()
        fit = _fit_logistic_once(
            y, cat_idx, deal, recency, memory,
            n_categories=3, alpha=0.0,
        )
        assert fit.deal_coef > 0, "deal coefficient should be positive"

    def test_recovers_positive_recency_coefficient(self):
        y, cat_idx, deal, recency, memory = _make_synthetic_data()
        fit = _fit_logistic_once(
            y, cat_idx, deal, recency, memory,
            n_categories=3, alpha=0.0,
        )
        # recency_coef is subtracted in the model, so positive = reducing purchase
        assert fit.recency_coef >= 0, "recency coefficient should be non-negative"

    def test_intercepts_have_correct_shape(self):
        y, cat_idx, deal, recency, memory = _make_synthetic_data(n_cats=5)
        fit = _fit_logistic_once(
            y, cat_idx, deal, recency, memory,
            n_categories=5, alpha=0.0,
        )
        assert fit.intercepts.shape == (5,)

    def test_nll_is_finite(self):
        y, cat_idx, deal, recency, memory = _make_synthetic_data()
        fit = _fit_logistic_once(
            y, cat_idx, deal, recency, memory,
            n_categories=3, alpha=0.0,
        )
        assert np.isfinite(fit.neg_log_likelihood)

    def test_initial_theta_shape_mismatch_raises(self):
        y, cat_idx, deal, recency, memory = _make_synthetic_data(n_cats=3)
        with pytest.raises(ValueError, match="incompatible shape"):
            _fit_logistic_once(
                y, cat_idx, deal, recency, memory,
                n_categories=3, alpha=0.0,
                initial_theta=np.zeros(10),  # wrong size: should be 3+3=6
            )

    def test_feature_length_mismatch_raises(self):
        y = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="length mismatch"):
            _fit_logistic_once(
                y,
                category_idx=np.array([0]),  # too short
                deal=np.array([1.0, 0.0]),
                recency=np.array([0.0, 0.0]),
                memory=np.array([0.0, 0.0]),
                n_categories=1, alpha=0.0,
            )

    def test_constant_features_do_not_crash(self):
        """Constant features (zero variance) should not cause division errors."""
        n = 100
        y = np.random.RandomState(0).binomial(1, 0.5, n).astype(float)
        cat_idx = np.zeros(n, dtype=int)
        deal = np.ones(n) * 0.5  # constant
        recency = np.zeros(n)  # constant
        memory = np.zeros(n)  # constant
        fit = _fit_logistic_once(
            y, cat_idx, deal, recency, memory,
            n_categories=1, alpha=0.0,
        )
        assert np.isfinite(fit.neg_log_likelihood)
        assert fit.intercepts.shape == (1,)


class TestFitLogisticPublic:
    def test_returns_tuple_of_five(self):
        y, cat_idx, deal, recency, memory = _make_synthetic_data(n_cats=2)
        result = fit_logistic(y, cat_idx, deal, recency, memory, n_cats=2)
        assert isinstance(result, tuple)
        assert len(result) == 5
        intercepts, bp, bl, bm, nll = result
        assert intercepts.shape == (2,)
        assert isinstance(bp, float)
        assert isinstance(bl, float)
        assert isinstance(bm, float)
        assert np.isfinite(nll)

    def test_matches_private_function(self):
        y, cat_idx, deal, recency, memory = _make_synthetic_data(n_cats=2)
        fit = _fit_logistic_once(
            y, cat_idx, deal, recency, memory,
            n_categories=2, alpha=0.5,
        )
        intercepts, bp, bl, bm, nll = fit_logistic(
            y, cat_idx, deal, recency, memory,
            n_cats=2, alpha=0.5,
        )
        np.testing.assert_array_almost_equal(intercepts, fit.intercepts)
        assert bp == pytest.approx(fit.deal_coef)
        assert bl == pytest.approx(fit.recency_coef)
        assert bm == pytest.approx(fit.memory_coef)

    def test_fitted_coefficients_improve_nll(self):
        """Fitted model should have lower NLL than a null model."""
        y, cat_idx, deal, recency, memory = _make_synthetic_data()
        intercepts, bp, bl, bm, _ = fit_logistic(
            y, cat_idx, deal, recency, memory, n_cats=3,
        )
        fitted_nll = evaluate_nll(
            y, cat_idx, deal, recency, memory,
            intercepts, bp, bl, bm,
        )
        # Null model: all coefficients zero, intercept at log-odds of base rate
        base_rate = y.mean()
        null_intercepts = np.full(3, np.log(base_rate / (1 - base_rate + 1e-9)))
        null_nll = evaluate_nll(
            y, cat_idx, deal, recency, memory,
            null_intercepts, 0.0, 0.0, 0.0,
        )
        assert fitted_nll < null_nll, "Fitted model should beat the null model"

    def test_scales_to_many_categories(self):
        """Verify the function works with a larger number of categories."""
        y, cat_idx, deal, recency, memory = _make_synthetic_data(
            n_per_cat=50, n_cats=20, seed=99,
        )
        intercepts, bp, bl, bm, nll = fit_logistic(
            y, cat_idx, deal, recency, memory, n_cats=20,
        )
        assert intercepts.shape == (20,)
        assert np.isfinite(nll)
        assert bp > 0


# ---------------------------------------------------------------------------
# Intercept rescaling correctness
# ---------------------------------------------------------------------------

class TestInterceptRescaling:
    def test_rescaled_predictions_match_original(self):
        """Verify that returned original-scale coefficients produce correct probabilities.

        Fit on synthetic data, then confirm that evaluate_nll using the returned
        coefficients gives the same NLL as the fit reports.
        """
        y, cat_idx, deal, recency, memory = _make_synthetic_data(n_cats=3)
        intercepts, bp, bl, bm, fit_nll = fit_logistic(
            y, cat_idx, deal, recency, memory, n_cats=3,
        )
        eval_nll = evaluate_nll(
            y, cat_idx, deal, recency, memory,
            intercepts, bp, bl, bm,
        )
        assert eval_nll == pytest.approx(fit_nll, rel=1e-6), (
            "fit NLL and evaluate_nll should match (both unregularized)"
        )


# ---------------------------------------------------------------------------
# Convergence warning
# ---------------------------------------------------------------------------

class TestConvergenceWarning:
    def test_warns_on_non_convergence(self):
        """Setting maxiter=1 should trigger a convergence warning."""
        y, cat_idx, deal, recency, memory = _make_synthetic_data()
        with pytest.warns(UserWarning, match="did not converge"):
            _fit_logistic_once(
                y, cat_idx, deal, recency, memory,
                n_categories=3, alpha=0.0, maxiter=1,
            )


# ---------------------------------------------------------------------------
# Non-negativity bounds
# ---------------------------------------------------------------------------

class TestNonNegativityBounds:
    def test_deal_coef_nonneg_with_adversarial_data(self):
        """Even when deal negatively correlates with purchase, deal_coef >= 0."""
        rng = np.random.RandomState(7)
        n = 300
        cat_idx = np.zeros(n, dtype=int)
        deal = rng.uniform(0, 1, n)
        recency = np.zeros(n)
        memory = np.zeros(n)
        # Adversarial: higher deal -> LOWER purchase probability
        probs = 1.0 / (1.0 + np.exp(3.0 * deal))
        y = (rng.uniform(0, 1, n) < probs).astype(float)
        fit = _fit_logistic_once(
            y, cat_idx, deal, recency, memory,
            n_categories=1, alpha=0.0,
        )
        assert fit.deal_coef >= 0, "bounds should prevent negative deal_coef"
        assert fit.recency_coef >= 0
        assert fit.memory_coef >= 0


# ---------------------------------------------------------------------------
# Hand-computed NLL
# ---------------------------------------------------------------------------

class TestHandComputedNLL:
    def test_nll_matches_manual_calculation(self):
        """Verify NLL against a hand-computed value on a tiny dataset."""
        y = np.array([1.0, 0.0])
        cat_idx = np.array([0, 0])
        deal = np.array([1.0, 0.0])
        recency = np.zeros(2)
        memory = np.zeros(2)
        # intercept=0, deal_coef=1 → logits = [1.0, 0.0]
        # probs = [sigmoid(1), sigmoid(0)] = [0.7310586, 0.5]
        # NLL = -(log(0.7310586) + log(0.5))
        import math
        p1 = 1.0 / (1.0 + math.exp(-1.0))
        p2 = 0.5
        expected_nll = -(math.log(p1) + math.log(1.0 - p2))
        actual_nll = _evaluate_neg_log_likelihood(
            y, cat_idx, deal, recency, memory,
            np.array([0.0]), deal_coef=1.0, recency_coef=0.0, memory_coef=0.0,
        )
        assert actual_nll == pytest.approx(expected_nll, abs=1e-6)
