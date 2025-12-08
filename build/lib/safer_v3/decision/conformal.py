"""
Conformal Prediction for SAFER v3.0.

This module implements conformal prediction methods for generating
calibrated uncertainty bounds on RUL predictions.

Conformal prediction provides distribution-free prediction intervals
with guaranteed coverage probability, making it ideal for safety-critical
applications where reliable uncertainty quantification is essential.

Methods:
1. Split Conformal: Simple calibration using holdout set
2. Adaptive Conformal: Online recalibration for non-stationary data

Key Properties:
- Finite-sample coverage guarantee: P(Y ∈ C(X)) ≥ 1 - α
- Distribution-free (no parametric assumptions)
- Works with any underlying predictor (Mamba, LSTM, etc.)
- Computationally efficient (single calibration pass)

For SAFER v3.0:
- Target coverage: 90% (α = 0.1)
- Adaptive window for deployment
- Conservative bounds for safety

References:
    - Vovk et al., "Algorithmic Learning in a Random World" (2005)
    - Romano et al., "Conformalized Quantile Regression" (2019)
    - Barber et al., "Conformal prediction under covariate shift" (2022)
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from collections import deque
import logging


logger = logging.getLogger(__name__)


@dataclass
class ConformalResult:
    """Result of conformal prediction.
    
    Attributes:
        prediction: Point prediction
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval
        coverage: Target coverage probability
        score: Nonconformity score
    """
    prediction: float
    lower: float
    upper: float
    coverage: float = 0.9
    score: float = 0.0
    
    @property
    def width(self) -> float:
        """Width of prediction interval."""
        return self.upper - self.lower
    
    @property
    def is_valid(self) -> bool:
        """Check if interval is valid (lower <= prediction <= upper)."""
        return self.lower <= self.prediction <= self.upper


class ConformalPredictor:
    """Base class for conformal predictors.
    
    Conformal prediction provides prediction intervals with finite-sample
    coverage guarantees. Given a target coverage 1-α, the prediction
    interval C(x) satisfies:
    
        P(Y ∈ C(X)) ≥ 1 - α
    
    This guarantee holds without any distributional assumptions.
    
    The key insight is to use nonconformity scores that measure how
    different a prediction is from the true value:
    
        s(x, y) = |y - ŷ(x)|  (absolute residual)
    
    The prediction interval is then:
    
        C(x) = [ŷ(x) - q, ŷ(x) + q]
    
    where q is the (1-α) quantile of calibration scores.
    """
    
    def __init__(
        self,
        coverage: float = 0.9,
        score_fn: Optional[Callable] = None,
    ):
        """Initialize conformal predictor.
        
        Args:
            coverage: Target coverage probability (1 - α)
            score_fn: Custom nonconformity score function
        """
        if not 0 < coverage < 1:
            raise ValueError("Coverage must be in (0, 1)")
        
        self.coverage = coverage
        self.alpha = 1 - coverage
        self.score_fn = score_fn or self._default_score
        
        # Calibration state
        self._calibrated = False
        self._quantile = None
        self._scores = None
    
    def _default_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """Default nonconformity score (absolute residual).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Nonconformity scores
        """
        return np.abs(y_true - y_pred)
    
    def calibrate(
        self,
        y_cal: np.ndarray,
        y_pred_cal: np.ndarray,
    ) -> float:
        """Calibrate the conformal predictor.
        
        Args:
            y_cal: True values on calibration set
            y_pred_cal: Predictions on calibration set
            
        Returns:
            Calibrated quantile threshold
        """
        y_cal = np.asarray(y_cal).ravel()
        y_pred_cal = np.asarray(y_pred_cal).ravel()
        
        if len(y_cal) != len(y_pred_cal):
            raise ValueError("y_cal and y_pred_cal must have same length")
        
        n = len(y_cal)
        
        # Compute nonconformity scores
        self._scores = self.score_fn(y_cal, y_pred_cal)
        
        # Compute quantile with finite-sample correction
        # q = ceil((n+1)(1-α)) / n -th quantile
        quantile_idx = int(np.ceil((n + 1) * self.coverage)) / n
        quantile_idx = min(quantile_idx, 1.0)  # Cap at 1
        
        self._quantile = np.quantile(self._scores, quantile_idx)
        self._calibrated = True
        
        logger.info(
            f"Conformal predictor calibrated: n={n}, "
            f"coverage={self.coverage:.2%}, quantile={self._quantile:.4f}"
        )
        
        return self._quantile
    
    def predict(
        self,
        y_pred: Union[float, np.ndarray],
    ) -> Union[ConformalResult, List[ConformalResult]]:
        """Generate prediction interval.
        
        Args:
            y_pred: Point prediction(s)
            
        Returns:
            ConformalResult or list of ConformalResults
        """
        if not self._calibrated:
            raise ValueError("Predictor must be calibrated before prediction")
        
        y_pred = np.asarray(y_pred)
        scalar_input = y_pred.ndim == 0
        y_pred = np.atleast_1d(y_pred)
        
        results = []
        for pred in y_pred:
            result = ConformalResult(
                prediction=float(pred),
                lower=float(pred - self._quantile),
                upper=float(pred + self._quantile),
                coverage=self.coverage,
                score=float(self._quantile),
            )
            results.append(result)
        
        if scalar_input:
            return results[0]
        return results
    
    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate coverage on test data.
        
        Args:
            y_true: True values
            y_pred: Predictions
            
        Returns:
            Dictionary with coverage metrics
        """
        if not self._calibrated:
            raise ValueError("Predictor must be calibrated first")
        
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        
        # Generate intervals
        results = self.predict(y_pred)
        
        # Check coverage
        covered = np.array([
            r.lower <= yt <= r.upper
            for r, yt in zip(results, y_true)
        ])
        
        empirical_coverage = np.mean(covered)
        avg_width = np.mean([r.width for r in results])
        
        return {
            'empirical_coverage': float(empirical_coverage),
            'target_coverage': self.coverage,
            'coverage_gap': float(empirical_coverage - self.coverage),
            'average_width': float(avg_width),
            'quantile': float(self._quantile),
            'n_samples': len(y_true),
        }


class SplitConformalPredictor(ConformalPredictor):
    """Split conformal predictor with asymmetric intervals.
    
    Extends basic conformal prediction with:
    - Separate upper/lower quantiles for asymmetric intervals
    - Optional score normalization
    - Batch prediction support
    
    This is useful for RUL prediction where errors may be
    asymmetric (under-prediction is typically worse than over-prediction).
    """
    
    def __init__(
        self,
        coverage: float = 0.9,
        symmetric: bool = True,
        normalize_scores: bool = False,
    ):
        """Initialize split conformal predictor.
        
        Args:
            coverage: Target coverage probability
            symmetric: Use symmetric intervals
            normalize_scores: Normalize scores by prediction magnitude
        """
        super().__init__(coverage)
        self.symmetric = symmetric
        self.normalize_scores = normalize_scores
        
        self._lower_quantile = None
        self._upper_quantile = None
    
    def calibrate(
        self,
        y_cal: np.ndarray,
        y_pred_cal: np.ndarray,
    ) -> Tuple[float, float]:
        """Calibrate with asymmetric intervals.
        
        Args:
            y_cal: True values
            y_pred_cal: Predictions
            
        Returns:
            Tuple of (lower_quantile, upper_quantile)
        """
        y_cal = np.asarray(y_cal).ravel()
        y_pred_cal = np.asarray(y_pred_cal).ravel()
        
        n = len(y_cal)
        residuals = y_cal - y_pred_cal
        
        # Optionally normalize by prediction magnitude
        if self.normalize_scores:
            normalizer = np.maximum(np.abs(y_pred_cal), 1.0)
            residuals = residuals / normalizer
        
        if self.symmetric:
            # Symmetric: use absolute residuals
            scores = np.abs(residuals)
            quantile_idx = np.ceil((n + 1) * self.coverage) / n
            quantile_idx = min(quantile_idx, 1.0)
            
            self._quantile = np.quantile(scores, quantile_idx)
            self._lower_quantile = self._quantile
            self._upper_quantile = self._quantile
        else:
            # Asymmetric: separate quantiles for under/over predictions
            alpha_half = self.alpha / 2
            
            lower_idx = alpha_half
            upper_idx = 1 - alpha_half
            
            # Lower quantile from negative residuals (under-predictions)
            self._lower_quantile = -np.quantile(residuals, lower_idx)
            # Upper quantile from positive residuals (over-predictions)
            self._upper_quantile = np.quantile(residuals, upper_idx)
            
            self._quantile = max(self._lower_quantile, self._upper_quantile)
        
        self._scores = np.abs(residuals)
        self._calibrated = True
        
        logger.info(
            f"Split conformal calibrated: lower={self._lower_quantile:.4f}, "
            f"upper={self._upper_quantile:.4f}"
        )
        
        return self._lower_quantile, self._upper_quantile
    
    def predict(
        self,
        y_pred: Union[float, np.ndarray],
    ) -> Union[ConformalResult, List[ConformalResult]]:
        """Generate prediction interval.
        
        Args:
            y_pred: Point prediction(s)
            
        Returns:
            ConformalResult(s)
        """
        if not self._calibrated:
            raise ValueError("Predictor must be calibrated before prediction")
        
        y_pred = np.asarray(y_pred)
        scalar_input = y_pred.ndim == 0
        y_pred = np.atleast_1d(y_pred)
        
        results = []
        for pred in y_pred:
            # Apply normalization if used during calibration
            if self.normalize_scores:
                normalizer = max(abs(pred), 1.0)
                lower = pred - self._lower_quantile * normalizer
                upper = pred + self._upper_quantile * normalizer
            else:
                lower = pred - self._lower_quantile
                upper = pred + self._upper_quantile
            
            result = ConformalResult(
                prediction=float(pred),
                lower=float(lower),
                upper=float(upper),
                coverage=self.coverage,
                score=float(self._quantile),
            )
            results.append(result)
        
        if scalar_input:
            return results[0]
        return results


class AdaptiveConformalPredictor(ConformalPredictor):
    """Adaptive conformal predictor for online/streaming data.
    
    Maintains a sliding window of recent scores and adapts the
    prediction intervals as new data arrives. This is essential
    for deployment where data distribution may shift over time.
    
    Features:
    - Sliding window of calibration scores
    - Online quantile estimation
    - Forgetting factor for non-stationary data
    - Coverage tracking and alerts
    
    Usage:
        predictor = AdaptiveConformalPredictor(window_size=100)
        
        for x, y_true in stream:
            y_pred = model(x)
            interval = predictor.predict(y_pred)
            predictor.update(y_true, y_pred)
    """
    
    def __init__(
        self,
        coverage: float = 0.9,
        window_size: int = 100,
        min_samples: int = 20,
        forgetting_factor: float = 1.0,
    ):
        """Initialize adaptive conformal predictor.
        
        Args:
            coverage: Target coverage probability
            window_size: Size of sliding window for scores
            min_samples: Minimum samples before prediction
            forgetting_factor: Weight decay for older samples (1.0 = no decay)
        """
        super().__init__(coverage)
        self.window_size = window_size
        self.min_samples = min_samples
        self.forgetting_factor = forgetting_factor
        
        # Sliding window of scores
        self._score_window: deque = deque(maxlen=window_size)
        
        # Coverage tracking
        self._coverage_window: deque = deque(maxlen=window_size)
        self._total_predictions = 0
        self._total_covered = 0
    
    def update(
        self,
        y_true: float,
        y_pred: float,
    ) -> float:
        """Update with new observation.
        
        Args:
            y_true: True value
            y_pred: Predicted value
            
        Returns:
            New nonconformity score
        """
        # Compute score
        score = self.score_fn(np.array([y_true]), np.array([y_pred]))[0]
        
        # Add to window
        self._score_window.append(score)
        
        # Update coverage tracking
        if self._quantile is not None:
            covered = abs(y_true - y_pred) <= self._quantile
            self._coverage_window.append(covered)
            self._total_predictions += 1
            self._total_covered += int(covered)
        
        # Recalibrate quantile
        self._recalibrate()
        
        return score
    
    def _recalibrate(self) -> None:
        """Recalibrate quantile from current window."""
        if len(self._score_window) < self.min_samples:
            return
        
        scores = np.array(self._score_window)
        n = len(scores)
        
        # Apply forgetting factor if specified
        if self.forgetting_factor < 1.0:
            weights = np.power(
                self.forgetting_factor,
                np.arange(n - 1, -1, -1)
            )
            weights /= weights.sum()
            
            # Weighted quantile
            sorted_idx = np.argsort(scores)
            sorted_scores = scores[sorted_idx]
            cumsum = np.cumsum(weights[sorted_idx])
            
            quantile_idx = np.searchsorted(cumsum, self.coverage)
            self._quantile = sorted_scores[min(quantile_idx, n - 1)]
        else:
            # Standard quantile
            quantile_idx = int(np.ceil((n + 1) * self.coverage)) / n
            quantile_idx = min(quantile_idx, 1.0)
            self._quantile = np.quantile(scores, quantile_idx)
        
        self._calibrated = True
    
    def calibrate(
        self,
        y_cal: np.ndarray,
        y_pred_cal: np.ndarray,
    ) -> float:
        """Initial calibration with batch data.
        
        Args:
            y_cal: True values
            y_pred_cal: Predictions
            
        Returns:
            Calibrated quantile
        """
        y_cal = np.asarray(y_cal).ravel()
        y_pred_cal = np.asarray(y_pred_cal).ravel()
        
        # Compute all scores
        scores = self.score_fn(y_cal, y_pred_cal)
        
        # Initialize window with recent scores
        for score in scores[-self.window_size:]:
            self._score_window.append(score)
        
        self._recalibrate()
        
        return self._quantile
    
    def predict(
        self,
        y_pred: Union[float, np.ndarray],
    ) -> Union[ConformalResult, List[ConformalResult]]:
        """Generate prediction interval.
        
        Args:
            y_pred: Point prediction(s)
            
        Returns:
            ConformalResult(s)
        """
        if not self._calibrated:
            # Use conservative default if not calibrated
            logger.warning("Using default quantile (not calibrated)")
            self._quantile = 20.0  # Conservative default for RUL
        
        return super().predict(y_pred)
    
    def get_coverage_stats(self) -> Dict[str, float]:
        """Get coverage statistics.
        
        Returns:
            Dictionary with coverage metrics
        """
        if not self._coverage_window:
            return {
                'window_coverage': 0.0,
                'total_coverage': 0.0,
                'window_size': 0,
                'total_predictions': 0,
            }
        
        window_coverage = np.mean(self._coverage_window)
        total_coverage = (
            self._total_covered / self._total_predictions
            if self._total_predictions > 0 else 0.0
        )
        
        return {
            'window_coverage': float(window_coverage),
            'total_coverage': float(total_coverage),
            'window_size': len(self._coverage_window),
            'total_predictions': self._total_predictions,
            'current_quantile': float(self._quantile) if self._quantile else 0.0,
            'target_coverage': self.coverage,
        }
    
    def is_coverage_acceptable(self, tolerance: float = 0.05) -> bool:
        """Check if recent coverage is within tolerance.
        
        Args:
            tolerance: Acceptable deviation from target
            
        Returns:
            True if coverage is acceptable
        """
        if len(self._coverage_window) < self.min_samples:
            return True  # Not enough data to judge
        
        window_coverage = np.mean(self._coverage_window)
        return abs(window_coverage - self.coverage) <= tolerance


def calibrate_conformal(
    model: torch.nn.Module,
    cal_loader: torch.utils.data.DataLoader,
    coverage: float = 0.9,
    device: torch.device = None,
    asymmetric: bool = False,
) -> Union[ConformalPredictor, SplitConformalPredictor]:
    """Calibrate conformal predictor using model and data.
    
    Convenience function to calibrate a conformal predictor using
    a trained model and calibration data loader.
    
    Args:
        model: Trained model
        cal_loader: Calibration data loader
        coverage: Target coverage probability
        device: Device for inference
        asymmetric: Use asymmetric intervals
        
    Returns:
        Calibrated ConformalPredictor
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, targets in cal_loader:
            sequences = sequences.to(device)
            predictions = model(sequences)
            
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.numpy())
    
    y_pred = np.concatenate(all_preds).ravel()
    y_true = np.concatenate(all_targets).ravel()
    
    # Create and calibrate predictor
    if asymmetric:
        predictor = SplitConformalPredictor(coverage=coverage, symmetric=False)
    else:
        predictor = SplitConformalPredictor(coverage=coverage, symmetric=True)
    
    predictor.calibrate(y_true, y_pred)
    
    # Evaluate coverage
    metrics = predictor.evaluate_coverage(y_true, y_pred)
    logger.info(
        f"Conformal calibration complete: "
        f"coverage={metrics['empirical_coverage']:.2%}, "
        f"width={metrics['average_width']:.2f}"
    )
    
    return predictor


class QuantileRegressionConformal(ConformalPredictor):
    """Conformalized Quantile Regression (CQR).
    
    Combines quantile regression with conformal prediction for
    adaptive prediction intervals that can vary with input.
    
    Unlike standard conformal prediction which uses fixed-width
    intervals, CQR adjusts interval width based on the model's
    uncertainty estimate.
    
    Requires a model that outputs quantile predictions:
    - Lower quantile (e.g., 5th percentile)
    - Median (50th percentile)
    - Upper quantile (e.g., 95th percentile)
    """
    
    def __init__(
        self,
        coverage: float = 0.9,
    ):
        """Initialize CQR predictor.
        
        Args:
            coverage: Target coverage probability
        """
        super().__init__(coverage)
        self._correction = None
    
    def calibrate(
        self,
        y_cal: np.ndarray,
        y_pred_lower: np.ndarray,
        y_pred_upper: np.ndarray,
    ) -> float:
        """Calibrate CQR correction factor.
        
        Args:
            y_cal: True values
            y_pred_lower: Lower quantile predictions
            y_pred_upper: Upper quantile predictions
            
        Returns:
            Calibration correction factor
        """
        y_cal = np.asarray(y_cal).ravel()
        y_pred_lower = np.asarray(y_pred_lower).ravel()
        y_pred_upper = np.asarray(y_pred_upper).ravel()
        
        n = len(y_cal)
        
        # CQR nonconformity score
        scores = np.maximum(
            y_pred_lower - y_cal,
            y_cal - y_pred_upper
        )
        
        # Quantile with finite-sample correction
        quantile_idx = int(np.ceil((n + 1) * self.coverage)) / n
        quantile_idx = min(quantile_idx, 1.0)
        
        self._correction = np.quantile(scores, quantile_idx)
        self._scores = scores
        self._calibrated = True
        
        logger.info(f"CQR calibrated: correction={self._correction:.4f}")
        
        return self._correction
    
    def predict(
        self,
        y_pred_lower: Union[float, np.ndarray],
        y_pred_upper: Union[float, np.ndarray],
        y_pred_median: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[ConformalResult, List[ConformalResult]]:
        """Generate CQR prediction interval.
        
        Args:
            y_pred_lower: Lower quantile prediction(s)
            y_pred_upper: Upper quantile prediction(s)
            y_pred_median: Median prediction(s) for point estimate
            
        Returns:
            ConformalResult(s)
        """
        if not self._calibrated:
            raise ValueError("Predictor must be calibrated before prediction")
        
        y_pred_lower = np.atleast_1d(y_pred_lower)
        y_pred_upper = np.atleast_1d(y_pred_upper)
        
        if y_pred_median is None:
            y_pred_median = (y_pred_lower + y_pred_upper) / 2
        else:
            y_pred_median = np.atleast_1d(y_pred_median)
        
        scalar_input = len(y_pred_lower) == 1
        
        results = []
        for lower, upper, median in zip(y_pred_lower, y_pred_upper, y_pred_median):
            # Apply correction
            corrected_lower = lower - self._correction
            corrected_upper = upper + self._correction
            
            result = ConformalResult(
                prediction=float(median),
                lower=float(corrected_lower),
                upper=float(corrected_upper),
                coverage=self.coverage,
                score=float(self._correction),
            )
            results.append(result)
        
        if scalar_input:
            return results[0]
        return results
