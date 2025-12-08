"""
Sparse Regression Algorithms for LPV-SINDy.

This module implements sparse regression methods for identifying
parsimonious dynamical system models from data.

Algorithms:
- STLSQ: Sequential Thresholded Least Squares (primary SINDy algorithm)
- Ridge Regression: L2-regularized least squares
- Elastic Net: Combined L1 + L2 regularization

The key insight is that most physical systems have sparse representations
in an appropriate basis - i.e., only a few terms from the library are
actually needed to describe the dynamics.

References:
    - Brunton et al., "Discovering governing equations from data" (2016)
    - Zhang & Schaeffer, "On the convergence of the SINDy algorithm" (2019)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class RegressionResult:
    """Result from sparse regression.
    
    Attributes:
        coefficients: Identified coefficients, shape (n_features, n_targets)
        support: Boolean mask of non-zero coefficients
        residuals: Fitting residuals
        iterations: Number of iterations (for iterative methods)
        converged: Whether the algorithm converged
    """
    coefficients: np.ndarray
    support: np.ndarray
    residuals: np.ndarray
    iterations: int
    converged: bool
    
    @property
    def n_nonzero(self) -> int:
        """Number of non-zero coefficients."""
        return np.sum(self.support)
    
    @property
    def sparsity(self) -> float:
        """Sparsity ratio (fraction of zero coefficients)."""
        return 1.0 - self.n_nonzero / self.support.size


def ridge_regression(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Ridge regression (L2-regularized least squares).
    
    Solves: min ||Xw - y||² + α||w||²
    
    Closed-form solution: w = (X'X + αI)^(-1) X'y
    
    Args:
        X: Feature matrix, shape (n_samples, n_features)
        y: Target matrix, shape (n_samples, n_targets)
        alpha: Regularization strength
        
    Returns:
        Coefficients, shape (n_features, n_targets)
    """
    n_features = X.shape[1]
    
    # Ensure y is 2D
    y = np.atleast_2d(y)
    if y.shape[0] != X.shape[0]:
        y = y.T
    
    # Solve normal equations with regularization
    # (X'X + αI)w = X'y
    XtX = X.T @ X
    Xty = X.T @ y
    
    # Add regularization to diagonal
    XtX_reg = XtX + alpha * np.eye(n_features)
    
    # Solve using Cholesky decomposition for numerical stability
    try:
        L = np.linalg.cholesky(XtX_reg)
        coefficients = np.linalg.solve(L.T, np.linalg.solve(L, Xty))
    except np.linalg.LinAlgError:
        # Fall back to pseudoinverse if Cholesky fails
        coefficients = np.linalg.lstsq(XtX_reg, Xty, rcond=None)[0]
    
    return coefficients


def elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> np.ndarray:
    """Elastic Net regression (L1 + L2 regularization).
    
    Solves: min ||Xw - y||² + α * l1_ratio * ||w||₁ + α * (1-l1_ratio) * ||w||²
    
    Uses coordinate descent for optimization.
    
    Args:
        X: Feature matrix, shape (n_samples, n_features)
        y: Target vector, shape (n_samples,) or (n_samples, 1)
        alpha: Overall regularization strength
        l1_ratio: Ratio of L1 to L2 penalty (0 = ridge, 1 = lasso)
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        Coefficients, shape (n_features,)
    """
    n_samples, n_features = X.shape
    y = y.ravel()
    
    # Precompute for efficiency
    X_squared_sum = np.sum(X ** 2, axis=0)
    
    # Initialize coefficients
    coefficients = np.zeros(n_features)
    
    # L1 and L2 penalties
    l1_penalty = alpha * l1_ratio
    l2_penalty = alpha * (1 - l1_ratio)
    
    for iteration in range(max_iter):
        coef_old = coefficients.copy()
        
        # Coordinate descent
        for j in range(n_features):
            # Compute partial residual
            residual = y - X @ coefficients + X[:, j] * coefficients[j]
            
            # Compute update
            rho = X[:, j] @ residual
            
            # Soft thresholding
            if X_squared_sum[j] == 0:
                coefficients[j] = 0
            else:
                z = (X_squared_sum[j] + l2_penalty)
                if rho < -l1_penalty:
                    coefficients[j] = (rho + l1_penalty) / z
                elif rho > l1_penalty:
                    coefficients[j] = (rho - l1_penalty) / z
                else:
                    coefficients[j] = 0
        
        # Check convergence
        if np.max(np.abs(coefficients - coef_old)) < tol:
            break
    
    return coefficients


class STLSQ:
    """Sequential Thresholded Least Squares.
    
    The primary sparse regression algorithm for SINDy. Alternates between:
    1. Least squares fitting
    2. Thresholding small coefficients to zero
    
    This iterative hard thresholding approach promotes sparsity while
    maintaining good fitting accuracy.
    
    Algorithm:
        1. Initialize: w = (X'X)^(-1) X'y (OLS solution)
        2. Threshold: Set w_j = 0 if |w_j| < threshold
        3. Refit: Solve OLS on remaining features
        4. Repeat 2-3 until convergence
    
    References:
        Brunton et al., "Discovering governing equations" (2016)
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        alpha: float = 0.05,
        max_iter: int = 100,
        tol: float = 1e-5,
        normalize_columns: bool = True,
        ridge_kappa: float = 1e-5,
    ):
        """Initialize STLSQ.
        
        Args:
            threshold: Coefficient threshold for sparsification
            alpha: Ridge regularization (for numerical stability)
            max_iter: Maximum iterations
            tol: Convergence tolerance on coefficient change
            normalize_columns: Normalize library columns before fitting
            ridge_kappa: Small ridge penalty for conditioning
        """
        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.normalize_columns = normalize_columns
        self.ridge_kappa = ridge_kappa
        
        # Fitted attributes
        self.coef_ = None
        self.support_ = None
        self.n_iter_ = 0
        self.converged_ = False
        self._column_norms = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> 'STLSQ':
        """Fit STLSQ model.
        
        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target matrix, shape (n_samples,) or (n_samples, n_targets)
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        # Ensure y is 2D
        y = np.atleast_2d(y)
        if y.shape[0] != n_samples:
            y = y.T
        n_targets = y.shape[1]
        
        # Normalize columns if requested
        if self.normalize_columns:
            self._column_norms = np.linalg.norm(X, axis=0)
            self._column_norms[self._column_norms < 1e-10] = 1.0
            X = X / self._column_norms
        else:
            self._column_norms = np.ones(n_features)
        
        # Initialize with ridge regression
        coef = ridge_regression(X, y, alpha=self.ridge_kappa)
        
        # Iterative thresholding
        for iteration in range(self.max_iter):
            coef_old = coef.copy()
            
            # Threshold small coefficients
            small_mask = np.abs(coef) < self.threshold
            coef[small_mask] = 0
            
            # For each target, refit using only non-zero features
            for t in range(n_targets):
                support = np.abs(coef[:, t]) >= self.threshold
                
                if np.sum(support) == 0:
                    # All coefficients thresholded - keep largest
                    largest_idx = np.argmax(np.abs(coef_old[:, t]))
                    support[largest_idx] = True
                
                if np.sum(support) > 0:
                    # Refit on support
                    X_support = X[:, support]
                    coef_support = ridge_regression(
                        X_support, y[:, t:t+1], alpha=self.ridge_kappa
                    )
                    coef[support, t] = coef_support.ravel()
                    coef[~support, t] = 0
            
            self.n_iter_ = iteration + 1
            
            # Check convergence
            coef_change = np.max(np.abs(coef - coef_old))
            if coef_change < self.tol:
                self.converged_ = True
                break
        
        # Rescale coefficients if columns were normalized
        if self.normalize_columns:
            coef = coef / self._column_norms.reshape(-1, 1)
        
        self.coef_ = coef.squeeze() if n_targets == 1 else coef
        self.support_ = np.abs(self.coef_) >= 1e-10
        
        logger.debug(
            f"STLSQ converged={self.converged_} in {self.n_iter_} iterations, "
            f"support size={np.sum(self.support_)}"
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted before predict")
        
        return X @ self.coef_
    
    def get_result(self, y: np.ndarray, X: np.ndarray) -> RegressionResult:
        """Get detailed regression result.
        
        Args:
            y: True targets
            X: Feature matrix
            
        Returns:
            RegressionResult with all details
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted first")
        
        predictions = self.predict(X)
        residuals = y - predictions
        
        return RegressionResult(
            coefficients=self.coef_,
            support=self.support_,
            residuals=residuals,
            iterations=self.n_iter_,
            converged=self.converged_,
        )


# Alias for backwards compatibility
SequentialThresholdedLeastSquares = STLSQ


class STRidge:
    """Sequential Thresholded Ridge Regression.
    
    Variant of STLSQ that uses ridge regression at each step instead of OLS.
    More stable for ill-conditioned problems or highly correlated features.
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        alpha: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-5,
    ):
        """Initialize STRidge.
        
        Args:
            threshold: Coefficient threshold
            alpha: Ridge regularization strength
            max_iter: Maximum iterations
            tol: Convergence tolerance
        """
        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        
        self.coef_ = None
        self.support_ = None
        self.n_iter_ = 0
        self.converged_ = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'STRidge':
        """Fit STRidge model.
        
        Args:
            X: Feature matrix
            y: Target vector/matrix
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        y = np.atleast_2d(y)
        if y.shape[0] != n_samples:
            y = y.T
        n_targets = y.shape[1]
        
        # Initialize with ridge regression
        coef = ridge_regression(X, y, alpha=self.alpha)
        
        for iteration in range(self.max_iter):
            coef_old = coef.copy()
            
            # Threshold
            small_mask = np.abs(coef) < self.threshold
            coef[small_mask] = 0
            
            # Refit for each target on support
            for t in range(n_targets):
                support = np.abs(coef[:, t]) >= self.threshold
                
                if np.sum(support) == 0:
                    largest_idx = np.argmax(np.abs(coef_old[:, t]))
                    support[largest_idx] = True
                
                if np.sum(support) > 0:
                    X_support = X[:, support]
                    coef_support = ridge_regression(
                        X_support, y[:, t:t+1], alpha=self.alpha
                    )
                    coef[support, t] = coef_support.ravel()
                    coef[~support, t] = 0
            
            self.n_iter_ = iteration + 1
            
            if np.max(np.abs(coef - coef_old)) < self.tol:
                self.converged_ = True
                break
        
        self.coef_ = coef.squeeze() if n_targets == 1 else coef
        self.support_ = np.abs(self.coef_) >= 1e-10
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted model."""
        if self.coef_ is None:
            raise ValueError("Model must be fitted before predict")
        return X @ self.coef_


class SparseRelaxedRegularizedRegression:
    """SR3: Sparse Relaxed Regularized Regression.
    
    Advanced sparse regression that relaxes the hard constraint
    w = ξ to a soft penalty, improving optimization landscape.
    
    Solves: min ||Xξ - y||² + λ||w||₀ + (1/2ν)||ξ - w||²
    
    where ξ are the fitting coefficients and w are the sparse coefficients.
    
    References:
        Zheng et al., "A unified framework for sparse relaxed 
        regularized regression" (2019)
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        nu: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-5,
    ):
        """Initialize SR3.
        
        Args:
            threshold: Sparsity threshold
            nu: Relaxation parameter (larger = softer constraint)
            max_iter: Maximum iterations
            tol: Convergence tolerance
        """
        self.threshold = threshold
        self.nu = nu
        self.max_iter = max_iter
        self.tol = tol
        
        self.coef_ = None
        self.support_ = None
        self.n_iter_ = 0
        self.converged_ = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SparseRelaxedRegularizedRegression':
        """Fit SR3 model.
        
        Args:
            X: Feature matrix
            y: Target vector/matrix
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        y = np.atleast_1d(y).ravel()
        
        # Precompute
        XtX = X.T @ X
        Xty = X.T @ y
        
        # Initialize
        xi = np.linalg.lstsq(X, y, rcond=None)[0]
        w = xi.copy()
        
        for iteration in range(self.max_iter):
            xi_old = xi.copy()
            
            # Update ξ (least squares with regularization toward w)
            A = XtX + (1 / self.nu) * np.eye(n_features)
            b = Xty + (1 / self.nu) * w
            xi = np.linalg.solve(A, b)
            
            # Update w (proximal operator / hard thresholding)
            w = xi.copy()
            w[np.abs(w) < self.threshold] = 0
            
            self.n_iter_ = iteration + 1
            
            if np.max(np.abs(xi - xi_old)) < self.tol:
                self.converged_ = True
                break
        
        self.coef_ = w
        self.support_ = np.abs(self.coef_) >= 1e-10
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted model."""
        if self.coef_ is None:
            raise ValueError("Model must be fitted before predict")
        return X @ self.coef_


def cross_validate_threshold(
    X: np.ndarray,
    y: np.ndarray,
    thresholds: Optional[List[float]] = None,
    n_folds: int = 5,
    metric: str = 'mse',
) -> Tuple[float, Dict[str, Any]]:
    """Cross-validate to find optimal sparsity threshold.
    
    Args:
        X: Feature matrix
        y: Target vector
        thresholds: Candidate thresholds to try
        n_folds: Number of CV folds
        metric: Evaluation metric ('mse', 'mae', 'aic', 'bic')
        
    Returns:
        Tuple of (best_threshold, cv_results)
    """
    if thresholds is None:
        # Default threshold range based on initial OLS solution
        coef_init = np.linalg.lstsq(X, y, rcond=None)[0]
        max_coef = np.max(np.abs(coef_init))
        thresholds = np.logspace(-3, 0, 20) * max_coef
    
    n_samples = X.shape[0]
    fold_size = n_samples // n_folds
    
    results = {
        'thresholds': thresholds,
        'cv_scores': [],
        'n_nonzero': [],
    }
    
    for threshold in thresholds:
        fold_scores = []
        fold_nonzero = []
        
        for fold in range(n_folds):
            # Split data
            val_start = fold * fold_size
            val_end = val_start + fold_size
            
            val_mask = np.zeros(n_samples, dtype=bool)
            val_mask[val_start:val_end] = True
            
            X_train, X_val = X[~val_mask], X[val_mask]
            y_train, y_val = y[~val_mask], y[val_mask]
            
            # Fit model
            model = STLSQ(threshold=threshold)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            
            if metric == 'mse':
                score = np.mean((y_val - y_pred) ** 2)
            elif metric == 'mae':
                score = np.mean(np.abs(y_val - y_pred))
            else:
                score = np.mean((y_val - y_pred) ** 2)
            
            fold_scores.append(score)
            fold_nonzero.append(np.sum(model.support_))
        
        results['cv_scores'].append(np.mean(fold_scores))
        results['n_nonzero'].append(np.mean(fold_nonzero))
    
    # Find best threshold (balance fit quality and sparsity)
    best_idx = np.argmin(results['cv_scores'])
    best_threshold = thresholds[best_idx]
    
    logger.info(
        f"Best threshold: {best_threshold:.4f}, "
        f"CV score: {results['cv_scores'][best_idx]:.6f}, "
        f"Non-zero: {results['n_nonzero'][best_idx]:.1f}"
    )
    
    return best_threshold, results


def information_criterion(
    X: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    criterion: str = 'aic',
) -> float:
    """Compute information criterion for model selection.
    
    Args:
        X: Feature matrix
        y: Target vector
        coefficients: Model coefficients
        criterion: 'aic' (Akaike) or 'bic' (Bayesian)
        
    Returns:
        Information criterion value (lower is better)
    """
    n_samples = X.shape[0]
    
    # Compute residual sum of squares
    y_pred = X @ coefficients
    rss = np.sum((y - y_pred) ** 2)
    
    # Number of non-zero parameters
    k = np.sum(np.abs(coefficients) > 1e-10)
    
    # Log-likelihood (assuming Gaussian errors)
    sigma2 = rss / n_samples
    log_likelihood = -n_samples / 2 * (np.log(2 * np.pi * sigma2) + 1)
    
    if criterion == 'aic':
        # AIC = 2k - 2ln(L)
        return 2 * k - 2 * log_likelihood
    elif criterion == 'bic':
        # BIC = k*ln(n) - 2ln(L)
        return k * np.log(n_samples) - 2 * log_likelihood
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
