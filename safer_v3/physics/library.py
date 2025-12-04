"""
Function Library for LPV-SINDy.

This module provides building blocks for constructing candidate function
libraries used in sparse identification of nonlinear dynamics.

Libraries transform raw state measurements into feature matrices that
can be used for regression. The choice of library functions determines
what kinds of dynamics can be identified.

Available Libraries:
- PolynomialLibrary: Polynomial terms up to specified degree
- FourierLibrary: Sine/cosine basis functions
- CustomLibrary: User-defined functions
- CombinedLibrary: Composition of multiple libraries

References:
    - Brunton et al., "Discovering governing equations from data by sparse 
      identification of nonlinear dynamical systems" (2016)
    - Kaiser et al., "Sparse identification of nonlinear dynamics for model
      predictive control in the low-data limit" (2018)
"""

import numpy as np
from typing import List, Callable, Optional, Tuple, Union, Dict, Any
from itertools import combinations_with_replacement
from abc import ABC, abstractmethod
import logging


logger = logging.getLogger(__name__)


class FunctionLibrary(ABC):
    """Abstract base class for function libraries.
    
    A function library transforms state vectors into feature vectors
    that can be used for sparse regression in system identification.
    
    Subclasses must implement:
    - fit(): Learn any necessary parameters from data
    - transform(): Apply library functions to data
    - get_feature_names(): Return human-readable names for features
    """
    
    def __init__(self, include_bias: bool = True):
        """Initialize library.
        
        Args:
            include_bias: Whether to include a constant (bias) term
        """
        self.include_bias = include_bias
        self.n_input_features_ = None
        self.n_output_features_ = None
        self.feature_names_ = None
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'FunctionLibrary':
        """Fit the library to data.
        
        Args:
            X: Input data, shape (n_samples, n_features)
            feature_names: Optional names for input features
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using library functions.
        
        Args:
            X: Input data, shape (n_samples, n_features)
            
        Returns:
            Transformed features, shape (n_samples, n_library_features)
        """
        pass
    
    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            X: Input data
            feature_names: Optional feature names
            
        Returns:
            Transformed features
        """
        return self.fit(X, feature_names).transform(X)
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get names of output features.
        
        Returns:
            List of feature names
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(include_bias={self.include_bias})"


class PolynomialLibrary(FunctionLibrary):
    """Polynomial function library.
    
    Generates polynomial features up to a specified degree.
    For degree=2 and features [x, y], generates:
    [1, x, y, x², xy, y²]
    
    This is the most common library for SINDy as many physical
    systems have polynomial nonlinearities.
    """
    
    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        include_interaction: bool = True,
    ):
        """Initialize polynomial library.
        
        Args:
            degree: Maximum polynomial degree
            include_bias: Include constant term
            include_interaction: Include cross-terms (xy, xyz, etc.)
        """
        super().__init__(include_bias)
        self.degree = degree
        self.include_interaction = include_interaction
        self._powers = None
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'PolynomialLibrary':
        """Fit polynomial library.
        
        Args:
            X: Input data, shape (n_samples, n_features)
            feature_names: Optional input feature names
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        self.n_input_features_ = n_features
        
        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f'x{i}' for i in range(n_features)]
        self._input_feature_names = feature_names
        
        # Generate all power combinations
        self._powers = []
        self.feature_names_ = []
        
        # Bias term
        if self.include_bias:
            self._powers.append(tuple([0] * n_features))
            self.feature_names_.append('1')
        
        # Generate polynomial terms
        for deg in range(1, self.degree + 1):
            if self.include_interaction:
                # All combinations with replacement
                for combo in combinations_with_replacement(range(n_features), deg):
                    powers = [0] * n_features
                    for idx in combo:
                        powers[idx] += 1
                    self._powers.append(tuple(powers))
                    
                    # Generate feature name
                    name_parts = []
                    for i, p in enumerate(powers):
                        if p == 1:
                            name_parts.append(feature_names[i])
                        elif p > 1:
                            name_parts.append(f'{feature_names[i]}^{p}')
                    self.feature_names_.append(' '.join(name_parts) if name_parts else '1')
            else:
                # Only pure powers, no interactions
                for i in range(n_features):
                    powers = [0] * n_features
                    powers[i] = deg
                    self._powers.append(tuple(powers))
                    
                    if deg == 1:
                        self.feature_names_.append(feature_names[i])
                    else:
                        self.feature_names_.append(f'{feature_names[i]}^{deg}')
        
        self._powers = np.array(self._powers)
        self.n_output_features_ = len(self._powers)
        self._is_fitted = True
        
        logger.debug(f"PolynomialLibrary fitted: {self.n_output_features_} features")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to polynomial features.
        
        Args:
            X: Input data, shape (n_samples, n_features)
            
        Returns:
            Polynomial features, shape (n_samples, n_output_features)
        """
        if not self._is_fitted:
            raise ValueError("Library must be fitted before transform")
        
        n_samples, n_features = X.shape
        
        if n_features != self.n_input_features_:
            raise ValueError(
                f"Expected {self.n_input_features_} features, got {n_features}"
            )
        
        # Compute polynomial features using broadcasting
        # X_poly[i, j] = prod(X[i, k] ** powers[j, k] for k in features)
        X_poly = np.ones((n_samples, self.n_output_features_))
        
        for j, powers in enumerate(self._powers):
            for k, p in enumerate(powers):
                if p > 0:
                    X_poly[:, j] *= X[:, k] ** p
        
        return X_poly
    
    def get_feature_names(self) -> List[str]:
        """Get polynomial feature names."""
        if not self._is_fitted:
            raise ValueError("Library must be fitted first")
        return self.feature_names_


class FourierLibrary(FunctionLibrary):
    """Fourier (trigonometric) function library.
    
    Generates sine and cosine features at multiple frequencies.
    Useful for periodic dynamics or as a universal approximator.
    
    For each input feature x and frequency k, generates:
    sin(k * x), cos(k * x)
    """
    
    def __init__(
        self,
        n_frequencies: int = 3,
        include_bias: bool = True,
        include_sin: bool = True,
        include_cos: bool = True,
    ):
        """Initialize Fourier library.
        
        Args:
            n_frequencies: Number of frequency components (1, 2, ..., n)
            include_bias: Include constant term
            include_sin: Include sine terms
            include_cos: Include cosine terms
        """
        super().__init__(include_bias)
        self.n_frequencies = n_frequencies
        self.include_sin = include_sin
        self.include_cos = include_cos
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'FourierLibrary':
        """Fit Fourier library.
        
        Args:
            X: Input data, shape (n_samples, n_features)
            feature_names: Optional input feature names
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        self.n_input_features_ = n_features
        
        if feature_names is None:
            feature_names = [f'x{i}' for i in range(n_features)]
        self._input_feature_names = feature_names
        
        # Build feature names
        self.feature_names_ = []
        
        if self.include_bias:
            self.feature_names_.append('1')
        
        for k in range(1, self.n_frequencies + 1):
            for i, name in enumerate(feature_names):
                if self.include_sin:
                    self.feature_names_.append(f'sin({k}*{name})')
                if self.include_cos:
                    self.feature_names_.append(f'cos({k}*{name})')
        
        self.n_output_features_ = len(self.feature_names_)
        self._is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to Fourier features.
        
        Args:
            X: Input data, shape (n_samples, n_features)
            
        Returns:
            Fourier features
        """
        if not self._is_fitted:
            raise ValueError("Library must be fitted before transform")
        
        n_samples, n_features = X.shape
        
        features = []
        
        if self.include_bias:
            features.append(np.ones((n_samples, 1)))
        
        for k in range(1, self.n_frequencies + 1):
            for i in range(n_features):
                if self.include_sin:
                    features.append(np.sin(k * X[:, i:i+1]))
                if self.include_cos:
                    features.append(np.cos(k * X[:, i:i+1]))
        
        return np.hstack(features)
    
    def get_feature_names(self) -> List[str]:
        """Get Fourier feature names."""
        if not self._is_fitted:
            raise ValueError("Library must be fitted first")
        return self.feature_names_


class CustomLibrary(FunctionLibrary):
    """Custom function library with user-defined functions.
    
    Allows specification of arbitrary functions for the library.
    Each function should take a 2D array and return a 1D or 2D array.
    
    Example:
        library = CustomLibrary(
            functions=[
                lambda X: X[:, 0] * X[:, 1],  # x * y
                lambda X: np.exp(-X[:, 0]),   # exp(-x)
            ],
            function_names=['x*y', 'exp(-x)']
        )
    """
    
    def __init__(
        self,
        functions: List[Callable[[np.ndarray], np.ndarray]],
        function_names: Optional[List[str]] = None,
        include_bias: bool = True,
    ):
        """Initialize custom library.
        
        Args:
            functions: List of functions, each taking (n_samples, n_features)
                      and returning (n_samples,) or (n_samples, k)
            function_names: Optional names for each function
            include_bias: Include constant term
        """
        super().__init__(include_bias)
        self.functions = functions
        
        if function_names is None:
            function_names = [f'f{i}' for i in range(len(functions))]
        
        if len(function_names) != len(functions):
            raise ValueError("Number of function names must match number of functions")
        
        self._function_names = function_names
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'CustomLibrary':
        """Fit custom library.
        
        Args:
            X: Input data
            feature_names: Ignored for custom library
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        self.n_input_features_ = n_features
        
        # Determine output dimensions by applying functions
        self.feature_names_ = []
        self._output_dims = []
        
        if self.include_bias:
            self.feature_names_.append('1')
        
        for func, name in zip(self.functions, self._function_names):
            result = func(X)
            if result.ndim == 1:
                self._output_dims.append(1)
                self.feature_names_.append(name)
            else:
                k = result.shape[1]
                self._output_dims.append(k)
                for i in range(k):
                    self.feature_names_.append(f'{name}[{i}]')
        
        self.n_output_features_ = len(self.feature_names_)
        self._is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using custom functions.
        
        Args:
            X: Input data
            
        Returns:
            Custom features
        """
        if not self._is_fitted:
            raise ValueError("Library must be fitted before transform")
        
        n_samples = X.shape[0]
        features = []
        
        if self.include_bias:
            features.append(np.ones((n_samples, 1)))
        
        for func, dim in zip(self.functions, self._output_dims):
            result = func(X)
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            features.append(result)
        
        return np.hstack(features)
    
    def get_feature_names(self) -> List[str]:
        """Get custom feature names."""
        if not self._is_fitted:
            raise ValueError("Library must be fitted first")
        return self.feature_names_


class CombinedLibrary(FunctionLibrary):
    """Combined library from multiple component libraries.
    
    Concatenates features from multiple libraries into a single
    feature matrix. Useful for combining different types of
    basis functions (e.g., polynomials + Fourier).
    
    Example:
        combined = CombinedLibrary([
            PolynomialLibrary(degree=2),
            FourierLibrary(n_frequencies=2),
        ])
    """
    
    def __init__(
        self,
        libraries: List[FunctionLibrary],
        include_bias: bool = True,
    ):
        """Initialize combined library.
        
        Args:
            libraries: List of component libraries
            include_bias: Include single bias term (individual library
                         bias terms are disabled)
        """
        super().__init__(include_bias)
        
        # Disable individual library bias terms
        self.libraries = []
        for lib in libraries:
            lib.include_bias = False
            self.libraries.append(lib)
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'CombinedLibrary':
        """Fit all component libraries.
        
        Args:
            X: Input data
            feature_names: Optional input feature names
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        self.n_input_features_ = n_features
        
        # Fit all libraries
        for lib in self.libraries:
            lib.fit(X, feature_names)
        
        # Combine feature names
        self.feature_names_ = []
        
        if self.include_bias:
            self.feature_names_.append('1')
        
        for lib in self.libraries:
            self.feature_names_.extend(lib.get_feature_names())
        
        self.n_output_features_ = len(self.feature_names_)
        self._is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using all libraries.
        
        Args:
            X: Input data
            
        Returns:
            Combined features
        """
        if not self._is_fitted:
            raise ValueError("Library must be fitted before transform")
        
        n_samples = X.shape[0]
        features = []
        
        if self.include_bias:
            features.append(np.ones((n_samples, 1)))
        
        for lib in self.libraries:
            features.append(lib.transform(X))
        
        return np.hstack(features)
    
    def get_feature_names(self) -> List[str]:
        """Get combined feature names."""
        if not self._is_fitted:
            raise ValueError("Library must be fitted first")
        return self.feature_names_


class ScheduledLibrary(FunctionLibrary):
    """Library with scheduling-variable-dependent coefficients.
    
    Implements Linear Parameter-Varying (LPV) structure where
    library coefficients depend on scheduling variables.
    
    For each base library function f_i and scheduling function g_j:
    Generates features: f_i(x) * g_j(p)
    
    where x is the state and p is the scheduling parameter.
    """
    
    def __init__(
        self,
        base_library: FunctionLibrary,
        scheduling_functions: List[Callable[[np.ndarray], np.ndarray]],
        scheduling_names: Optional[List[str]] = None,
        include_bias: bool = True,
    ):
        """Initialize scheduled library.
        
        Args:
            base_library: Base function library for state variables
            scheduling_functions: Functions of scheduling parameters
            scheduling_names: Names for scheduling functions
            include_bias: Include constant term
        """
        super().__init__(include_bias)
        
        self.base_library = base_library
        self.base_library.include_bias = False
        
        self.scheduling_functions = scheduling_functions
        
        if scheduling_names is None:
            scheduling_names = [f'p{i}' for i in range(len(scheduling_functions))]
        self._scheduling_names = scheduling_names
    
    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        P: Optional[np.ndarray] = None,
    ) -> 'ScheduledLibrary':
        """Fit scheduled library.
        
        Args:
            X: State data, shape (n_samples, n_state_features)
            feature_names: Optional state feature names
            P: Scheduling parameters (unused for fit, just for interface)
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        self.n_input_features_ = n_features
        
        # Fit base library
        self.base_library.fit(X, feature_names)
        base_names = self.base_library.get_feature_names()
        
        # Generate scheduled feature names
        self.feature_names_ = []
        
        if self.include_bias:
            self.feature_names_.append('1')
        
        for sched_name in self._scheduling_names:
            for base_name in base_names:
                self.feature_names_.append(f'{sched_name}*{base_name}')
        
        self.n_output_features_ = len(self.feature_names_)
        self._is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Transform data with scheduling parameters.
        
        Args:
            X: State data, shape (n_samples, n_state_features)
            P: Scheduling parameters, shape (n_samples, n_scheduling)
            
        Returns:
            Scheduled features
        """
        if not self._is_fitted:
            raise ValueError("Library must be fitted before transform")
        
        n_samples = X.shape[0]
        
        # Get base features
        base_features = self.base_library.transform(X)  # (n_samples, n_base)
        
        features = []
        
        if self.include_bias:
            features.append(np.ones((n_samples, 1)))
        
        # Apply scheduling functions and multiply
        for i, sched_func in enumerate(self.scheduling_functions):
            sched_values = sched_func(P)  # (n_samples,) or (n_samples, 1)
            if sched_values.ndim == 1:
                sched_values = sched_values.reshape(-1, 1)
            
            # Multiply each base feature by scheduling value
            scheduled = base_features * sched_values  # Broadcasting
            features.append(scheduled)
        
        return np.hstack(features)
    
    def get_feature_names(self) -> List[str]:
        """Get scheduled feature names."""
        if not self._is_fitted:
            raise ValueError("Library must be fitted first")
        return self.feature_names_


class LPVAugmentedLibrary(FunctionLibrary):
    """Augmented library with health-dependent scheduling interaction terms.
    
    Generates both standard polynomial terms and p-weighted interaction terms,
    where p(t) is a scheduling parameter representing system health.
    
    For a state x and health parameter p(t), generates:
    - Standard: [1, x, x², x³, ...]
    - Augmented: [p·x, p·x², p·x³, ...]
    
    This captures health-dependent dynamics where the system response
    changes as it degrades. Used in LPV-SINDy for adaptive degradation
    monitoring.
    
    Mathematical formulation:
        Ξ_augmented = [1, x, x², ..., p·x, p·x², p·x³, ...]
        
    The health parameter p ∈ [0,1] represents:
        p = 1.0: Healthy (start of life)
        p = 0.0: End of life
    
    References:
        - Hofmann et al., "Reactive SINDy" (2019)
        - Goebel et al., "Prognostics and Health Management of Electronics" (2017)
    """
    
    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        include_interaction: bool = True,
    ):
        """Initialize augmented LPV library.
        
        Args:
            degree: Maximum polynomial degree for state terms
            include_bias: Include constant term (1)
            include_interaction: Include cross-terms (xy, xyz, etc.)
        """
        super().__init__(include_bias)
        self.degree = degree
        self.include_interaction = include_interaction
        self._base_library = PolynomialLibrary(
            degree=degree,
            include_bias=include_bias,
            include_interaction=include_interaction,
        )
        self._base_powers = None
        self._augmented_powers = None
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'LPVAugmentedLibrary':
        """Fit augmented library.
        
        Args:
            X: Input state data, shape (n_samples, n_features)
            feature_names: Optional input feature names
            
        Returns:
            self
        """
        # Fit base polynomial library
        self._base_library.fit(X, feature_names)
        self.n_input_features_ = self._base_library.n_input_features_
        
        base_names = self._base_library.get_feature_names()
        self._base_powers = self._base_library._powers.copy()
        
        # Generate augmented feature names (base + p-weighted)
        self.feature_names_ = base_names.copy()
        
        # Add p-weighted terms (skip the bias term '1')
        for name in base_names:
            if name != '1':  # Skip constant term
                self.feature_names_.append(f'p·{name}')
        
        self.n_output_features_ = len(self.feature_names_)
        self._is_fitted = True
        
        logger.debug(
            f"LPVAugmentedLibrary fitted: {len(base_names)} base features, "
            f"{len(base_names)-1} augmented features, "
            f"total {self.n_output_features_}"
        )
        
        return self
    
    def transform(self, X: np.ndarray, p: Optional[np.ndarray] = None) -> np.ndarray:
        """Transform data with augmented LPV features.
        
        Args:
            X: Input state data, shape (n_samples, n_features)
            p: Health scheduling parameter, shape (n_samples,) or scalar
               If None, assumes p=1 (fully healthy)
            
        Returns:
            Augmented features, shape (n_samples, n_output_features)
        """
        if not self._is_fitted:
            raise ValueError("Library must be fitted before transform")
        
        n_samples = X.shape[0]
        
        # Handle scheduling parameter
        if p is None:
            p = np.ones(n_samples)
        else:
            p = np.asarray(p)
            if p.ndim == 0:  # Scalar
                p = np.full(n_samples, p)
            if len(p) != n_samples:
                raise ValueError(
                    f"Scheduling parameter size {len(p)} doesn't match "
                    f"data size {n_samples}"
                )
        
        # Get base polynomial features
        base_features = self._base_library.transform(X)  # (n_samples, n_base)
        
        # Create augmented features: [base_features | p * base_features_no_bias]
        # Skip the bias term when multiplying by p
        augmented_features = np.hstack([
            base_features,
            base_features[:, 1:] * p.reshape(-1, 1),  # Multiply by p, skip bias
        ])
        
        return augmented_features
    
    def get_feature_names(self) -> List[str]:
        """Get augmented feature names."""
        if not self._is_fitted:
            raise ValueError("Library must be fitted first")
        return self.feature_names_


def build_turbofan_library(
    n_sensors: int = 14,
    polynomial_degree: int = 2,
    include_fourier: bool = False,
    n_frequencies: int = 2,
) -> FunctionLibrary:
    """Build a function library suitable for turbofan degradation modeling.
    
    Creates a library with:
    - Polynomial terms (captures nonlinear degradation)
    - Optional Fourier terms (captures periodic effects)
    
    Args:
        n_sensors: Number of sensor inputs
        polynomial_degree: Maximum polynomial degree
        include_fourier: Whether to include Fourier terms
        n_frequencies: Number of Fourier frequencies
        
    Returns:
        Configured function library
    """
    libraries = [PolynomialLibrary(degree=polynomial_degree, include_bias=False)]
    
    if include_fourier:
        libraries.append(
            FourierLibrary(n_frequencies=n_frequencies, include_bias=False)
        )
    
    if len(libraries) == 1:
        lib = libraries[0]
        lib.include_bias = True
        return lib
    
    return CombinedLibrary(libraries, include_bias=True)


def build_lpv_library(
    n_sensors: int = 14,
    n_op_settings: int = 3,
    polynomial_degree: int = 2,
) -> ScheduledLibrary:
    """Build an LPV library for turbofan with operational scheduling.
    
    Creates a library where coefficients depend on operational settings
    (altitude, Mach number, throttle), capturing how dynamics change
    across the flight envelope.
    
    Args:
        n_sensors: Number of sensor inputs
        n_op_settings: Number of operational settings (scheduling vars)
        polynomial_degree: Polynomial degree for state terms
        
    Returns:
        Configured LPV library
    """
    # Base library for sensor states
    base_library = PolynomialLibrary(
        degree=polynomial_degree,
        include_bias=False,
    )
    
    # Scheduling functions: polynomial in operational settings
    # p0: constant (no scheduling)
    # p1, p2, p3: linear in each op setting
    scheduling_functions = [
        lambda P: np.ones(P.shape[0]),  # Constant
    ]
    scheduling_names = ['1']
    
    for i in range(n_op_settings):
        scheduling_functions.append(
            lambda P, idx=i: P[:, idx]  # Linear in op setting
        )
        scheduling_names.append(f'op{i+1}')
    
    return ScheduledLibrary(
        base_library=base_library,
        scheduling_functions=scheduling_functions,
        scheduling_names=scheduling_names,
        include_bias=True,
    )
