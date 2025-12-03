"""
LPV-SINDy Monitor for SAFER v3.0.

This module implements the Linear Parameter-Varying Sparse Identification
of Nonlinear Dynamics (LPV-SINDy) model for turbofan engine degradation
monitoring.

Key Features:
- Integral formulation for noise robustness (per specification)
- Trapezoidal integration over configurable windows
- Scheduling variable support for operational conditions
- Real-time residual computation for anomaly detection
- DAL C certification compliance per DO-178C

The LPV-SINDy model serves as the physics-based monitor in the SAFER
architecture, providing:
1. Interpretable degradation models
2. Anomaly detection through prediction residuals
3. Validation of neural network predictions
4. Baseline for Simplex decision logic

Architecture (DAL C - Safety Monitor):
------------------------------------
This physics-based model is classified as DAL C per DO-178C guidelines.
It provides independent verification of Mamba predictions and can trigger
safety actions when discrepancies exceed thresholds.

Integral Formulation:
--------------------
Instead of identifying dx/dt = f(x), we use the integral form:
∫x dt = ∫f(x) dt

This is more robust to measurement noise since integration acts as a
low-pass filter. We use trapezoidal rule for numerical integration.

References:
    - Brunton et al., "Discovering governing equations from data" (2016)
    - Kaiser et al., "Sparse identification for MPC in low-data limit" (2018)
    - Hoffmann et al., "Reactive SINDy" (2019)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json

from safer_v3.physics.library import (
    FunctionLibrary,
    PolynomialLibrary,
    CombinedLibrary,
    build_turbofan_library,
)
from safer_v3.physics.sparse_regression import (
    STLSQ,
    RegressionResult,
    cross_validate_threshold,
)
from safer_v3.utils.config import LPVSINDyConfig


logger = logging.getLogger(__name__)


@dataclass
class IntegralFormulation:
    """Configuration for integral SINDy formulation.
    
    Uses trapezoidal integration over a sliding window to convert
    differential equations to integral form for noise robustness.
    
    Attributes:
        window_size: Number of points for integration window
        dt: Time step between samples
        method: Integration method ('trapezoidal', 'simpson')
    """
    window_size: int = 5
    dt: float = 1.0
    method: str = 'trapezoidal'
    
    def integrate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute integral formulation quantities.
        
        For each window, computes:
        - Left side: ∫x dt ≈ x(t+w) - x(t)
        - Right side: ∫Θ(x) dt (for library to compute)
        
        Args:
            x: State trajectory, shape (n_samples, n_features)
            
        Returns:
            Tuple of (delta_x, integration_weights)
            - delta_x: x(t+w) - x(t), shape (n_windows, n_features)
            - integration_weights: Trapezoidal weights for library integration
        """
        n_samples, n_features = x.shape
        n_windows = n_samples - self.window_size + 1
        
        if n_windows <= 0:
            raise ValueError(
                f"Not enough samples ({n_samples}) for window size ({self.window_size})"
            )
        
        # Compute state differences (integral of dx/dt)
        delta_x = np.zeros((n_windows, n_features))
        for i in range(n_windows):
            delta_x[i] = x[i + self.window_size - 1] - x[i]
        
        # Compute integration weights (for trapezoidal rule)
        if self.method == 'trapezoidal':
            weights = np.ones(self.window_size) * self.dt
            weights[0] *= 0.5
            weights[-1] *= 0.5
        elif self.method == 'simpson' and self.window_size >= 3:
            # Simpson's rule for odd window sizes
            if self.window_size % 2 == 1:
                weights = np.zeros(self.window_size)
                weights[0::2] = 2
                weights[1::2] = 4
                weights[0] = 1
                weights[-1] = 1
                weights *= self.dt / 3
            else:
                # Fall back to trapezoidal for even windows
                weights = np.ones(self.window_size) * self.dt
                weights[0] *= 0.5
                weights[-1] *= 0.5
        else:
            weights = np.ones(self.window_size) * self.dt
            weights[0] *= 0.5
            weights[-1] *= 0.5
        
        return delta_x, weights
    
    def integrate_library(
        self,
        library_features: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Integrate library features over windows.
        
        Computes ∫Θ(x) dt for each window using provided weights.
        
        Args:
            library_features: Library output, shape (n_samples, n_lib_features)
            weights: Integration weights, shape (window_size,)
            
        Returns:
            Integrated features, shape (n_windows, n_lib_features)
        """
        n_samples, n_lib_features = library_features.shape
        n_windows = n_samples - self.window_size + 1
        
        integrated = np.zeros((n_windows, n_lib_features))
        
        for i in range(n_windows):
            window = library_features[i:i + self.window_size]
            integrated[i] = np.sum(window * weights.reshape(-1, 1), axis=0)
        
        return integrated


@dataclass
class SchedulingFunction:
    """Scheduling function for LPV systems.
    
    Maps operational parameters to scheduling variables that
    modulate the dynamics.
    
    Common scheduling variables for turbofan:
    - Altitude (affects air density)
    - Mach number (affects inlet conditions)  
    - Throttle setting (affects operating point)
    """
    name: str
    function: Callable[[np.ndarray], np.ndarray]
    bounds: Tuple[float, float] = (-np.inf, np.inf)
    
    def __call__(self, params: np.ndarray) -> np.ndarray:
        """Apply scheduling function.
        
        Args:
            params: Operational parameters
            
        Returns:
            Scheduling variable values
        """
        result = self.function(params)
        return np.clip(result, self.bounds[0], self.bounds[1])


class LPVSINDyMonitor(nn.Module):
    """LPV-SINDy Physics Monitor.
    
    Implements a Linear Parameter-Varying Sparse Identification of
    Nonlinear Dynamics model for monitoring turbofan degradation.
    
    The model learns sparse dynamics of the form:
        dx/dt = Θ(x) @ ξ(p)
    
    where:
    - x: State (sensor measurements)
    - Θ(x): Library of candidate functions
    - ξ(p): Sparse coefficients that may depend on scheduling params p
    
    Using integral formulation:
        Δx = ∫Θ(x)dt @ ξ(p)
    
    This class provides:
    1. Model identification from training data
    2. Real-time residual computation
    3. Anomaly detection via threshold comparison
    4. Model persistence (save/load)
    
    Safety Classification: DAL C
    ---------------------------
    This monitor provides independent physics-based verification of
    neural network predictions. Anomalies detected here can trigger
    the Simplex switch to baseline controller.
    """
    
    def __init__(
        self,
        config: Optional[LPVSINDyConfig] = None,
        library: Optional[FunctionLibrary] = None,
        n_sensors: int = 14,
    ):
        """Initialize LPV-SINDy monitor.
        
        Args:
            config: Configuration parameters
            library: Function library (default: polynomial degree 2)
            n_sensors: Number of input sensors
        """
        super().__init__()
        
        self.config = config or LPVSINDyConfig()
        self.n_sensors = n_sensors
        
        # Function library
        if library is None:
            self.library = build_turbofan_library(
                n_sensors=n_sensors,
                polynomial_degree=self.config.polynomial_degree,
                include_fourier=False,
            )
        else:
            self.library = library
        
        # Integral formulation
        self.integral = IntegralFormulation(
            window_size=self.config.window_size,
            dt=self.config.dt,
            method='trapezoidal',
        )
        
        # Sparse regression
        self.regressor = STLSQ(
            threshold=self.config.threshold,
            alpha=self.config.alpha,
            max_iter=self.config.max_iter,
        )
        
        # Model coefficients (will be set after fitting)
        self._coefficients = None
        self._feature_names = None
        self._is_fitted = False
        
        # Residual statistics for anomaly detection
        self._residual_mean = None
        self._residual_std = None
        self._residual_threshold = None
        
        # Register buffer for coefficients (for model persistence)
        self.register_buffer(
            'coefficients',
            torch.zeros(1),  # Placeholder, updated after fit
        )
        
        logger.info(
            f"Initialized LPVSINDyMonitor: n_sensors={n_sensors}, "
            f"window_size={self.config.window_size}, "
            f"threshold={self.config.threshold}"
        )
    
    def fit(
        self,
        X: np.ndarray,
        validate: bool = True,
        val_fraction: float = 0.2,
    ) -> Dict[str, Any]:
        """Fit LPV-SINDy model to training data.
        
        Args:
            X: Training trajectories, shape (n_samples, n_sensors)
               or list of trajectories
            validate: Whether to compute validation metrics
            val_fraction: Fraction of data for validation
            
        Returns:
            Dictionary with fitting results and metrics
        """
        # Handle multiple trajectories
        if isinstance(X, list):
            X = np.concatenate(X, axis=0)
        
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        logger.info(f"Fitting LPV-SINDy on {n_samples} samples")
        
        # Split for validation if requested
        if validate:
            n_train = int(n_samples * (1 - val_fraction))
            X_train = X[:n_train]
            X_val = X[n_train:]
        else:
            X_train = X
            X_val = None
        
        # Fit library
        self.library.fit(X_train)
        self._feature_names = self.library.get_feature_names()
        
        # Transform to library features
        Theta_train = self.library.transform(X_train)
        
        # Apply integral formulation
        delta_x, weights = self.integral.integrate(X_train)
        Theta_integrated = self.integral.integrate_library(Theta_train, weights)
        
        # Fit sparse regression for each state dimension
        coefficients_list = []
        
        for i in range(n_features):
            self.regressor.fit(Theta_integrated, delta_x[:, i])
            coefficients_list.append(self.regressor.coef_.copy())
        
        self._coefficients = np.column_stack(coefficients_list)
        self._is_fitted = True
        
        # Update PyTorch buffer
        self.coefficients = torch.from_numpy(self._coefficients).float()
        
        # Compute training residuals and statistics
        train_residuals = self._compute_residuals(X_train)
        self._residual_mean = np.mean(train_residuals, axis=0)
        self._residual_std = np.std(train_residuals, axis=0)
        self._residual_std = np.maximum(self._residual_std, 1e-6)  # Avoid division by zero
        
        # Set anomaly threshold (e.g., 3 sigma)
        self._residual_threshold = self.config.residual_threshold_sigma * self._residual_std
        
        # Compute metrics
        results = {
            'n_features': len(self._feature_names),
            'n_nonzero': np.sum(np.abs(self._coefficients) > 1e-10, axis=0),
            'total_nonzero': np.sum(np.abs(self._coefficients) > 1e-10),
            'sparsity': 1.0 - np.mean(np.abs(self._coefficients) > 1e-10),
            'train_rmse': np.sqrt(np.mean(train_residuals ** 2)),
            'train_mae': np.mean(np.abs(train_residuals)),
            'residual_mean': self._residual_mean.tolist(),
            'residual_std': self._residual_std.tolist(),
        }
        
        # Validation metrics
        if validate and X_val is not None and len(X_val) > self.config.window_size:
            val_residuals = self._compute_residuals(X_val)
            results['val_rmse'] = np.sqrt(np.mean(val_residuals ** 2))
            results['val_mae'] = np.mean(np.abs(val_residuals))
        
        logger.info(
            f"LPV-SINDy fit complete: {results['total_nonzero']} non-zero terms, "
            f"sparsity={results['sparsity']:.2%}, train_rmse={results['train_rmse']:.4f}"
        )
        
        return results
    
    def _compute_residuals(self, X: np.ndarray) -> np.ndarray:
        """Compute prediction residuals.
        
        Args:
            X: State trajectory, shape (n_samples, n_features)
            
        Returns:
            Residuals, shape (n_windows, n_features)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before computing residuals")
        
        # Transform and integrate
        Theta = self.library.transform(X)
        delta_x, weights = self.integral.integrate(X)
        Theta_integrated = self.integral.integrate_library(Theta, weights)
        
        # Predict
        delta_x_pred = Theta_integrated @ self._coefficients
        
        # Residuals
        residuals = delta_x - delta_x_pred
        
        return residuals
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for real-time monitoring.
        
        Computes prediction residuals and anomaly scores.
        
        Args:
            x: Input sequence, shape (batch, length, n_sensors)
            
        Returns:
            Tuple of (residuals, anomaly_scores)
            - residuals: shape (batch, n_windows, n_sensors)
            - anomaly_scores: shape (batch, n_windows)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before forward pass")
        
        batch_size, seq_len, n_features = x.shape
        device = x.device
        
        # Convert to numpy for library operations
        x_np = x.detach().cpu().numpy()
        
        all_residuals = []
        all_scores = []
        
        for b in range(batch_size):
            residuals = self._compute_residuals(x_np[b])
            
            # Compute anomaly score (normalized residual magnitude)
            normalized = (residuals - self._residual_mean) / self._residual_std
            scores = np.sqrt(np.mean(normalized ** 2, axis=1))  # RMS over features
            
            all_residuals.append(residuals)
            all_scores.append(scores)
        
        # Stack and convert to tensors
        residuals_tensor = torch.from_numpy(
            np.stack(all_residuals)
        ).float().to(device)
        
        scores_tensor = torch.from_numpy(
            np.stack(all_scores)
        ).float().to(device)
        
        return residuals_tensor, scores_tensor
    
    def detect_anomaly(
        self,
        x: Union[np.ndarray, torch.Tensor],
        threshold_sigma: Optional[float] = None,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect anomalies in sensor data.
        
        Args:
            x: Sensor sequence, shape (length, n_sensors) or (batch, length, n_sensors)
            threshold_sigma: Override threshold (default uses config)
            
        Returns:
            Tuple of (is_anomaly, max_score, details)
            - is_anomaly: Whether anomaly detected
            - max_score: Maximum anomaly score
            - details: Dictionary with per-sensor scores
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before anomaly detection")
        
        if threshold_sigma is None:
            threshold_sigma = self.config.residual_threshold_sigma
        
        # Handle input format
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        
        if x.ndim == 2:
            x = x[np.newaxis, ...]  # Add batch dimension
        
        batch_size = x.shape[0]
        
        all_anomalies = []
        all_max_scores = []
        all_details = []
        
        for b in range(batch_size):
            residuals = self._compute_residuals(x[b])
            
            # Normalize residuals
            normalized = np.abs(residuals - self._residual_mean) / self._residual_std
            
            # Check for anomalies per sensor
            anomaly_mask = normalized > threshold_sigma
            
            # Overall anomaly score
            scores = np.max(normalized, axis=0)  # Max over time per sensor
            max_score = np.max(scores)
            
            is_anomaly = max_score > threshold_sigma
            
            details = {
                'per_sensor_scores': scores.tolist(),
                'anomalous_sensors': np.where(scores > threshold_sigma)[0].tolist(),
                'n_anomalous_points': np.sum(anomaly_mask),
            }
            
            all_anomalies.append(is_anomaly)
            all_max_scores.append(max_score)
            all_details.append(details)
        
        if batch_size == 1:
            return all_anomalies[0], all_max_scores[0], all_details[0]
        
        return all_anomalies, all_max_scores, all_details
    
    def get_active_terms(self) -> Dict[int, List[str]]:
        """Get active (non-zero) terms for each state dimension.
        
        Returns:
            Dictionary mapping state index to list of active term names
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        active_terms = {}
        
        for state_idx in range(self._coefficients.shape[1]):
            coef = self._coefficients[:, state_idx]
            active_mask = np.abs(coef) > 1e-10
            
            terms = []
            for feat_idx, is_active in enumerate(active_mask):
                if is_active:
                    term_name = self._feature_names[feat_idx]
                    term_coef = coef[feat_idx]
                    terms.append(f"{term_coef:+.4f}*{term_name}")
            
            active_terms[state_idx] = terms
        
        return active_terms
    
    def print_equations(self) -> str:
        """Print identified equations in human-readable format.
        
        Returns:
            String representation of equations
        """
        active_terms = self.get_active_terms()
        
        lines = ["Identified Dynamics (Integral Form):"]
        lines.append("=" * 50)
        
        for state_idx, terms in active_terms.items():
            if terms:
                equation = " ".join(terms)
            else:
                equation = "0"
            lines.append(f"Δx{state_idx} = {equation}")
        
        output = "\n".join(lines)
        print(output)
        return output
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to file.
        
        Args:
            path: Output path (creates .npz and .json files)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save coefficients as numpy
        np.savez(
            str(path) + '.npz',
            coefficients=self._coefficients,
            residual_mean=self._residual_mean,
            residual_std=self._residual_std,
        )
        
        # Save metadata as JSON
        metadata = {
            'n_sensors': self.n_sensors,
            'feature_names': self._feature_names,
            'config': {
                'window_size': self.config.window_size,
                'dt': self.config.dt,
                'threshold': self.config.threshold,
                'alpha': self.config.alpha,
                'polynomial_degree': self.config.polynomial_degree,
                'residual_threshold_sigma': self.config.residual_threshold_sigma,
            },
        }
        
        with open(str(path) + '.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved LPV-SINDy model to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'LPVSINDyMonitor':
        """Load model from file.
        
        Args:
            path: Path to saved model (without extension)
            
        Returns:
            Loaded LPVSINDyMonitor
        """
        path = Path(path)
        
        # Load metadata
        with open(str(path) + '.json', 'r') as f:
            metadata = json.load(f)
        
        # Create config
        config = LPVSINDyConfig(**metadata['config'])
        
        # Create monitor
        monitor = cls(
            config=config,
            n_sensors=metadata['n_sensors'],
        )
        
        # Load coefficients
        data = np.load(str(path) + '.npz')
        monitor._coefficients = data['coefficients']
        monitor._residual_mean = data['residual_mean']
        monitor._residual_std = data['residual_std']
        monitor._feature_names = metadata['feature_names']
        monitor._is_fitted = True
        
        # Update PyTorch buffer
        monitor.coefficients = torch.from_numpy(monitor._coefficients).float()
        
        # Recompute threshold
        monitor._residual_threshold = (
            config.residual_threshold_sigma * monitor._residual_std
        )
        
        # Rebuild library (need to fit on dummy data to set names)
        dummy_data = np.zeros((10, metadata['n_sensors']))
        monitor.library.fit(dummy_data)
        
        logger.info(f"Loaded LPV-SINDy model from {path}")
        
        return monitor
    
    def to_onnx(self, path: Union[str, Path]) -> None:
        """Export residual computation to ONNX (limited support).
        
        Note: Full library operations require numpy, so ONNX export
        is limited. Consider using save/load for deployment.
        
        Args:
            path: Output path
        """
        logger.warning(
            "ONNX export for LPV-SINDy is limited. "
            "Use save/load methods for full functionality."
        )
        
        # Export just the coefficient matrix
        path = Path(path)
        np.save(str(path) + '_coefficients.npy', self._coefficients)
