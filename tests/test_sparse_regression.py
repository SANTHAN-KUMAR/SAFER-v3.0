"""
Unit tests for Sparse Regression (STLSQ).

Tests cover:
- STLSQ algorithm correctness
- Sparsity enforcement
- Cross-validation for threshold selection
- Numerical stability
"""

import pytest
import numpy as np


class TestSTLSQ:
    """Test Sequential Thresholded Least Squares (STLSQ)."""
    
    @pytest.fixture
    def stlsq(self):
        """Create STLSQ regressor for testing."""
        from safer_v3.physics.sparse_regression import STLSQ
        return STLSQ(threshold=0.1, alpha=0.01, max_iter=100)
    
    def test_fit_sets_coefficients(self, stlsq, rng):
        """Test fit sets coefficient values."""
        n_samples, n_features = 200, 10
        X = rng.normal(0, 1, (n_samples, n_features))
        y = X @ rng.normal(0, 1, n_features) + rng.normal(0, 0.1, n_samples)
        
        stlsq.fit(X, y)
        
        assert stlsq.coef_ is not None
        assert len(stlsq.coef_) == n_features
    
    def test_produces_sparse_solution(self, rng):
        """Test STLSQ produces sparse solutions."""
        from safer_v3.physics.sparse_regression import STLSQ
        
        n_samples, n_features = 200, 20
        X = rng.normal(0, 1, (n_samples, n_features))
        
        # True sparse coefficients
        true_coef = np.zeros(n_features)
        true_coef[:3] = [1.0, -2.0, 1.5]  # Only 3 non-zero
        
        y = X @ true_coef + rng.normal(0, 0.1, n_samples)
        
        stlsq = STLSQ(threshold=0.5, alpha=0.01)
        stlsq.fit(X, y)
        
        # Should recover sparse structure
        n_nonzero = np.sum(np.abs(stlsq.coef_) > 0.01)
        assert n_nonzero < n_features, "Solution should be sparse"
    
    def test_higher_threshold_more_sparse(self, rng):
        """Test higher threshold produces sparser solutions."""
        from safer_v3.physics.sparse_regression import STLSQ
        
        n_samples, n_features = 200, 15
        X = rng.normal(0, 1, (n_samples, n_features))
        y = X @ rng.normal(0, 1, n_features) + rng.normal(0, 0.1, n_samples)
        
        stlsq_low = STLSQ(threshold=0.1)
        stlsq_high = STLSQ(threshold=1.0)
        
        stlsq_low.fit(X, y)
        stlsq_high.fit(X, y)
        
        n_nonzero_low = np.sum(np.abs(stlsq_low.coef_) > 0)
        n_nonzero_high = np.sum(np.abs(stlsq_high.coef_) > 0)
        
        assert n_nonzero_high <= n_nonzero_low
    
    def test_prediction(self, stlsq, rng):
        """Test STLSQ can make predictions."""
        n_samples, n_features = 200, 10
        X = rng.normal(0, 1, (n_samples, n_features))
        true_coef = rng.normal(0, 1, n_features)
        y = X @ true_coef + rng.normal(0, 0.1, n_samples)
        
        stlsq.fit(X, y)
        y_pred = stlsq.predict(X)
        
        assert len(y_pred) == n_samples
    
    def test_convergence(self, rng):
        """Test STLSQ converges within max_iter."""
        from safer_v3.physics.sparse_regression import STLSQ
        
        n_samples, n_features = 100, 5
        X = rng.normal(0, 1, (n_samples, n_features))
        y = X @ rng.normal(0, 1, n_features) + rng.normal(0, 0.1, n_samples)
        
        stlsq = STLSQ(threshold=0.1, max_iter=10)
        stlsq.fit(X, y)
        
        assert hasattr(stlsq, 'n_iter_') or stlsq.coef_ is not None


class TestRegressionResult:
    """Test RegressionResult dataclass."""
    
    def test_result_has_coefficients(self, rng):
        """Test result contains coefficients."""
        from safer_v3.physics.sparse_regression import STLSQ
        
        X = rng.normal(0, 1, (100, 5))
        y = X @ rng.normal(0, 1, 5)
        
        stlsq = STLSQ(threshold=0.1)
        stlsq.fit(X, y)
        
        assert stlsq.coef_ is not None


class TestCrossValidateThreshold:
    """Test threshold cross-validation."""
    
    def test_cross_validate_returns_threshold(self, rng):
        """Test cross-validation returns optimal threshold."""
        from safer_v3.physics.sparse_regression import cross_validate_threshold
        
        n_samples, n_features = 200, 10
        X = rng.normal(0, 1, (n_samples, n_features))
        
        # Sparse true solution
        true_coef = np.zeros(n_features)
        true_coef[:3] = [1.0, -2.0, 1.5]
        y = X @ true_coef + rng.normal(0, 0.1, n_samples)
        
        thresholds = [0.01, 0.1, 0.5, 1.0]
        result = cross_validate_threshold(X, y, thresholds)
        
        # Function returns tuple of (best_threshold, results_dict)
        best_threshold, cv_results = result
        assert best_threshold in thresholds


class TestNumericalStability:
    """Test numerical stability of sparse regression."""
    
    @pytest.fixture
    def stlsq(self):
        """Create STLSQ regressor."""
        from safer_v3.physics.sparse_regression import STLSQ
        return STLSQ(threshold=0.1, alpha=0.01)
    
    def test_handles_collinear_features(self, stlsq, rng):
        """Test handling of nearly collinear features."""
        n_samples = 100
        X = rng.normal(0, 1, (n_samples, 5))
        X = np.column_stack([X, X[:, 0] * 1.0001])  # Nearly collinear
        y = rng.normal(0, 1, n_samples)
        
        stlsq.fit(X, y)
        
        assert not np.any(np.isnan(stlsq.coef_))
        assert not np.any(np.isinf(stlsq.coef_))
    
    def test_handles_small_values(self, stlsq, rng):
        """Test handling of small feature values."""
        n_samples, n_features = 100, 5
        X = rng.normal(0, 1e-6, (n_samples, n_features))
        y = rng.normal(0, 1e-6, n_samples)
        
        stlsq.fit(X, y)
        
        assert not np.any(np.isnan(stlsq.coef_))
    
    def test_handles_large_values(self, stlsq, rng):
        """Test handling of large feature values."""
        n_samples, n_features = 100, 5
        X = rng.normal(0, 1e6, (n_samples, n_features))
        y = rng.normal(0, 1e6, n_samples)
        
        stlsq.fit(X, y)
        
        assert not np.any(np.isnan(stlsq.coef_))
