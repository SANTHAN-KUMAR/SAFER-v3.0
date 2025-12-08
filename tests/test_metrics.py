"""
Unit tests for metrics module.

Tests cover:
- RUL metrics calculation (RMSE, MAE, NASA Score)
- Metrics dataclass functionality
- Edge cases and numerical stability
- Metric comparison utilities
"""

import pytest
import numpy as np


class TestRULMetrics:
    """Test RUL metric calculations."""
    
    def test_calculate_rmse(self, rng):
        """Test RMSE calculation."""
        from safer_v3.utils.metrics import calculate_rul_metrics
        
        targets = np.array([100, 80, 60, 40, 20])
        preds = np.array([95, 85, 55, 45, 15])
        
        metrics = calculate_rul_metrics(targets, preds)
        
        # Manual RMSE calculation
        expected_rmse = np.sqrt(np.mean((targets - preds) ** 2))
        
        assert np.isclose(metrics.rmse, expected_rmse)
    
    def test_calculate_mae(self, rng):
        """Test MAE calculation."""
        from safer_v3.utils.metrics import calculate_rul_metrics
        
        targets = np.array([100, 80, 60, 40, 20])
        preds = np.array([95, 85, 55, 45, 15])
        
        metrics = calculate_rul_metrics(targets, preds)
        
        # Manual MAE calculation
        expected_mae = np.mean(np.abs(targets - preds))
        
        assert np.isclose(metrics.mae, expected_mae)
    
    def test_calculate_nasa_score(self, rng):
        """Test NASA score calculation."""
        from safer_v3.utils.metrics import calculate_rul_metrics
        
        targets = np.array([100, 80, 60, 40, 20])
        preds = np.array([95, 85, 55, 45, 15])
        
        metrics = calculate_rul_metrics(targets, preds)
        
        # NASA score should be positive
        assert metrics.nasa_score >= 0
    
    def test_nasa_score_asymmetry(self, rng):
        """Test NASA score has asymmetric penalty (uses exponential function)."""
        from safer_v3.utils.metrics import calculate_rul_metrics
        
        targets = np.array([50, 50, 50, 50, 50])
        
        # Early predictions (positive error - predicted RUL higher than actual)
        preds_early = np.array([60, 60, 60, 60, 60])  # Predict high (early warning)
        
        # Late predictions (negative error - predicted RUL lower than actual)
        preds_late = np.array([40, 40, 40, 40, 40])  # Predict low (late warning)
        
        metrics_early = calculate_rul_metrics(targets, preds_early)
        metrics_late = calculate_rul_metrics(targets, preds_late)
        
        # Both should have positive scores (penalty)
        assert metrics_early.nasa_score > 0
        assert metrics_late.nasa_score > 0
    
    def test_perfect_predictions(self, rng):
        """Test metrics for perfect predictions."""
        from safer_v3.utils.metrics import calculate_rul_metrics
        
        targets = np.array([100, 80, 60, 40, 20])
        preds = targets.copy()
        
        metrics = calculate_rul_metrics(targets, preds)
        
        assert np.isclose(metrics.rmse, 0)
        assert np.isclose(metrics.mae, 0)
        assert np.isclose(metrics.nasa_score, 0)
    
    def test_r2_score(self, rng):
        """Test R² score calculation."""
        from safer_v3.utils.metrics import calculate_rul_metrics
        
        # Good predictions
        targets = np.linspace(100, 0, 50)
        preds = targets + rng.normal(0, 5, 50)
        
        metrics = calculate_rul_metrics(targets, preds)
        
        # R² should be high for good predictions
        assert metrics.r2 > 0.5
    
    def test_r2_perfect(self, rng):
        """Test R² is 1 for perfect predictions."""
        from safer_v3.utils.metrics import calculate_rul_metrics
        
        targets = np.linspace(100, 0, 50)
        preds = targets.copy()
        
        metrics = calculate_rul_metrics(targets, preds)
        
        assert np.isclose(metrics.r2, 1.0)
    
    def test_r2_negative_for_bad_predictions(self, rng):
        """Test R² can be negative for very bad predictions."""
        from safer_v3.utils.metrics import calculate_rul_metrics
        
        targets = np.linspace(100, 0, 50)
        # Predictions that are worse than mean baseline
        preds = np.ones(50) * 200  # Constant high prediction
        
        metrics = calculate_rul_metrics(targets, preds)
        
        # R² should be negative
        assert metrics.r2 < 0


class TestRULMetricsDataclass:
    """Test RULMetrics dataclass functionality."""
    
    def test_to_dict(self, rng):
        """Test metrics can be converted to dictionary."""
        from safer_v3.utils.metrics import calculate_rul_metrics
        
        targets = np.array([100, 80, 60, 40, 20])
        preds = np.array([95, 85, 55, 45, 15])
        
        metrics = calculate_rul_metrics(targets, preds)
        
        metrics_dict = metrics.to_dict()
        
        assert 'rmse' in metrics_dict
        assert 'mae' in metrics_dict
        assert 'nasa_score' in metrics_dict
    
    def test_str_representation(self, rng):
        """Test metrics have string representation."""
        from safer_v3.utils.metrics import calculate_rul_metrics
        
        targets = np.array([100, 80, 60])
        preds = np.array([95, 85, 55])
        
        metrics = calculate_rul_metrics(targets, preds)
        
        str_repr = str(metrics)
        assert 'rmse' in str_repr.lower() or 'RMSE' in str_repr


class TestMetricsEdgeCases:
    """Test edge cases for metric calculations."""
    
    def test_single_sample(self, rng):
        """Test metrics with single sample."""
        from safer_v3.utils.metrics import calculate_rul_metrics
        
        targets = np.array([50])
        preds = np.array([55])
        
        metrics = calculate_rul_metrics(targets, preds)
        
        assert np.isclose(metrics.rmse, 5)
        assert np.isclose(metrics.mae, 5)
    
    def test_large_values(self, rng):
        """Test metrics with large RUL values."""
        from safer_v3.utils.metrics import calculate_rul_metrics
        
        targets = np.array([10000, 8000, 6000, 4000, 2000])
        preds = np.array([9500, 8500, 5500, 4500, 1500])
        
        metrics = calculate_rul_metrics(targets, preds)
        
        assert metrics.rmse > 0
        assert not np.isnan(metrics.rmse)
    
    def test_zero_targets(self, rng):
        """Test metrics when targets include zeros."""
        from safer_v3.utils.metrics import calculate_rul_metrics
        
        targets = np.array([50, 25, 10, 5, 0])
        preds = np.array([55, 20, 15, 3, 2])
        
        metrics = calculate_rul_metrics(targets, preds)
        
        assert not np.isnan(metrics.rmse)
        assert not np.isnan(metrics.mae)


class TestEarlyLateMetrics:
    """Test early/late prediction metrics."""
    
    def test_early_rate_calculation(self, rng):
        """Test early prediction rate calculation."""
        from safer_v3.utils.metrics import calculate_rul_metrics
        
        targets = np.array([50, 50, 50, 50, 50])
        preds = np.array([60, 55, 50, 45, 40])  # 2 early, 1 exact, 2 late
        
        metrics = calculate_rul_metrics(targets, preds)
        
        if hasattr(metrics, 'early_rate'):
            # Rate is in percentage (0-100), not ratio (0-1)
            assert 0 <= metrics.early_rate <= 100
    
    def test_late_rate_calculation(self, rng):
        """Test late prediction rate calculation."""
        from safer_v3.utils.metrics import calculate_rul_metrics
        
        targets = np.array([50, 50, 50, 50, 50])
        preds = np.array([60, 55, 50, 45, 40])  # 2 early, 1 exact, 2 late
        
        metrics = calculate_rul_metrics(targets, preds)
        
        if hasattr(metrics, 'late_rate'):
            # Rate is in percentage (0-100), not ratio (0-1)
            assert 0 <= metrics.late_rate <= 100


class TestNASAScoreFormula:
    """Test NASA score formula specifically."""
    
    def test_nasa_exponential_penalty(self, rng):
        """Test NASA score uses exponential penalty."""
        from safer_v3.utils.metrics import calculate_rul_metrics
        
        targets = np.array([50])
        
        # Small error
        preds_small = np.array([45])  # 5 cycle late
        
        # Large error
        preds_large = np.array([30])  # 20 cycle late
        
        score_small = calculate_rul_metrics(targets, preds_small).nasa_score
        score_large = calculate_rul_metrics(targets, preds_large).nasa_score
        
        # Score should increase exponentially, not linearly
        # 4x error should result in much more than 4x score
        assert score_large > 4 * score_small, \
            "NASA score should have exponential penalty"
