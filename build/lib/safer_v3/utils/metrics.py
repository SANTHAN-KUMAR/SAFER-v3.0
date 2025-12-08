"""
Metrics for SAFER v3.0 RUL prediction evaluation.

This module provides evaluation metrics including:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- NASA Scoring Function (asymmetric penalty)
- Comprehensive RUL metrics calculation
"""

import numpy as np
from typing import Dict, Tuple, Union, Optional
from dataclasses import dataclass


@dataclass
class RULMetrics:
    """Container for RUL prediction metrics.
    
    Attributes:
        rmse: Root Mean Square Error
        mae: Mean Absolute Error
        nasa_score: NASA scoring function value
        r2: R-squared coefficient
        early_rate: Percentage of early predictions
        late_rate: Percentage of late predictions
        mean_error: Mean prediction error (bias)
        std_error: Standard deviation of error
    """
    rmse: float
    mae: float
    nasa_score: float
    r2: float
    early_rate: float
    late_rate: float
    mean_error: float
    std_error: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'nasa_score': self.nasa_score,
            'r2': self.r2,
            'early_rate': self.early_rate,
            'late_rate': self.late_rate,
            'mean_error': self.mean_error,
            'std_error': self.std_error,
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"RUL Metrics:\n"
            f"  RMSE:       {self.rmse:.2f} cycles\n"
            f"  MAE:        {self.mae:.2f} cycles\n"
            f"  NASA Score: {self.nasa_score:.2f}\n"
            f"  R²:         {self.r2:.4f}\n"
            f"  Early:      {self.early_rate:.1f}%\n"
            f"  Late:       {self.late_rate:.1f}%\n"
            f"  Bias:       {self.mean_error:.2f} cycles\n"
            f"  Std:        {self.std_error:.2f} cycles"
        )


def calculate_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """Calculate Root Mean Square Error.
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        weights: Optional sample weights
        
    Returns:
        RMSE value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if len(y_true) == 0:
        return 0.0
    
    errors = y_true - y_pred
    
    if weights is not None:
        weights = np.asarray(weights).flatten()
        mse = np.average(errors ** 2, weights=weights)
    else:
        mse = np.mean(errors ** 2)
    
    return float(np.sqrt(mse))


def calculate_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """Calculate Mean Absolute Error.
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        weights: Optional sample weights
        
    Returns:
        MAE value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if len(y_true) == 0:
        return 0.0
    
    errors = np.abs(y_true - y_pred)
    
    if weights is not None:
        weights = np.asarray(weights).flatten()
        return float(np.average(errors, weights=weights))
    else:
        return float(np.mean(errors))


def nasa_scoring_function(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    a1: float = 13.0,
    a2: float = 10.0,
) -> float:
    """Calculate NASA scoring function with asymmetric penalty.
    
    The NASA scoring function penalizes late predictions more severely
    than early predictions, reflecting the safety-critical nature of
    prognostics where late predictions could lead to failure.
    
    Score = Σ exp(-d/a1) - 1  for d < 0 (early prediction)
            exp(d/a2) - 1   for d >= 0 (late prediction)
    
    where d = y_true - y_pred (positive means predicted RUL is too low)
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        a1: Scaling factor for early predictions (default: 13)
        a2: Scaling factor for late predictions (default: 10)
        
    Returns:
        Total NASA score (lower is better)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if len(y_true) == 0:
        return 0.0
    
    # d > 0 means prediction is too low (early), d < 0 means too high (late)
    d = y_pred - y_true
    
    scores = np.where(
        d < 0,
        np.exp(-d / a1) - 1,  # Early prediction penalty
        np.exp(d / a2) - 1    # Late prediction penalty (more severe)
    )
    
    return float(np.sum(scores))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared (coefficient of determination).
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        
    Returns:
        R² value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) <= 1:
        return 0.0
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return float(1 - ss_res / ss_tot)


def calculate_rul_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_rul: Optional[int] = None,
) -> RULMetrics:
    """Calculate comprehensive RUL prediction metrics.
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        max_rul: Optional maximum RUL for capping
        
    Returns:
        RULMetrics dataclass with all metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Apply RUL capping if specified
    if max_rul is not None:
        y_true = np.minimum(y_true, max_rul)
        y_pred = np.minimum(y_pred, max_rul)
    
    # Calculate errors
    errors = y_pred - y_true  # positive = predicted too high (late warning)
    
    # Basic metrics
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    nasa_score = nasa_scoring_function(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    
    # Early/late analysis
    n_samples = len(y_true)
    if n_samples > 0:
        early_rate = 100.0 * np.sum(errors < 0) / n_samples
        late_rate = 100.0 * np.sum(errors > 0) / n_samples
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))
    else:
        early_rate = late_rate = mean_error = std_error = 0.0
    
    return RULMetrics(
        rmse=rmse,
        mae=mae,
        nasa_score=nasa_score,
        r2=r2,
        early_rate=early_rate,
        late_rate=late_rate,
        mean_error=mean_error,
        std_error=std_error,
    )


def calculate_per_engine_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    engine_ids: np.ndarray,
) -> Tuple[RULMetrics, Dict[int, RULMetrics]]:
    """Calculate metrics per engine and overall.
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        engine_ids: Engine ID for each sample
        
    Returns:
        Tuple of (overall_metrics, {engine_id: metrics})
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    engine_ids = np.asarray(engine_ids).flatten()
    
    # Overall metrics
    overall = calculate_rul_metrics(y_true, y_pred)
    
    # Per-engine metrics
    per_engine = {}
    unique_engines = np.unique(engine_ids)
    
    for engine_id in unique_engines:
        mask = engine_ids == engine_id
        if np.sum(mask) > 0:
            per_engine[int(engine_id)] = calculate_rul_metrics(
                y_true[mask], y_pred[mask]
            )
    
    return overall, per_engine


def calculate_window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rul_windows: list = [(0, 20), (20, 50), (50, 100), (100, float('inf'))],
) -> Dict[str, RULMetrics]:
    """Calculate metrics for different RUL windows.
    
    This helps analyze model performance at different stages of degradation.
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        rul_windows: List of (min, max) RUL value ranges
        
    Returns:
        Dictionary mapping window names to metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    results = {}
    
    for min_rul, max_rul in rul_windows:
        window_name = f"RUL_{min_rul}-{max_rul if max_rul != float('inf') else 'inf'}"
        mask = (y_true >= min_rul) & (y_true < max_rul)
        
        if np.sum(mask) > 0:
            results[window_name] = calculate_rul_metrics(y_true[mask], y_pred[mask])
    
    return results


class MetricsTracker:
    """Track metrics over time for monitoring during inference."""
    
    def __init__(self, window_size: int = 100):
        """Initialize tracker.
        
        Args:
            window_size: Size of sliding window for recent metrics
        """
        self._window_size = window_size
        self._predictions: list = []
        self._true_values: list = []
        self._timestamps: list = []
    
    def update(self, y_true: float, y_pred: float, timestamp: Optional[float] = None) -> None:
        """Add a new prediction.
        
        Args:
            y_true: True RUL value
            y_pred: Predicted RUL value
            timestamp: Optional timestamp
        """
        self._predictions.append(y_pred)
        self._true_values.append(y_true)
        self._timestamps.append(timestamp)
        
        # Keep only recent values
        if len(self._predictions) > self._window_size:
            self._predictions.pop(0)
            self._true_values.pop(0)
            self._timestamps.pop(0)
    
    def get_recent_metrics(self) -> Optional[RULMetrics]:
        """Get metrics for recent predictions.
        
        Returns:
            RULMetrics or None if no predictions
        """
        if not self._predictions:
            return None
        
        return calculate_rul_metrics(
            np.array(self._true_values),
            np.array(self._predictions)
        )
    
    def get_current_error(self) -> Optional[float]:
        """Get most recent prediction error.
        
        Returns:
            Error value or None
        """
        if not self._predictions:
            return None
        return self._predictions[-1] - self._true_values[-1]
    
    def reset(self) -> None:
        """Reset tracker."""
        self._predictions = []
        self._true_values = []
        self._timestamps = []
