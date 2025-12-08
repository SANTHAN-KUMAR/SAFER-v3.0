"""
Physics Module Initialization for SAFER v3.0.

This subpackage implements the LPV-SINDy (Linear Parameter-Varying Sparse
Identification of Nonlinear Dynamics) physics monitor for turbofan engines.

Modules:
- library.py: Function library for building candidate functions
- sparse_regression.py: Sequential Thresholded Least Squares (STLSQ)
- lpv_sindy.py: Complete LPV-SINDy model with integral formulation

The physics module provides:
1. Interpretable degradation models
2. Anomaly detection through residual monitoring
3. Physics-based validation of neural network predictions
4. DAL C certification compliance per DO-178C

Key Features:
- Integral formulation for noise robustness
- Scheduling variables from operational conditions
- Sparse regression for parsimonious models
- Real-time residual computation
"""

from safer_v3.physics.library import (
    FunctionLibrary,
    PolynomialLibrary,
    FourierLibrary,
    CustomLibrary,
    CombinedLibrary,
)
from safer_v3.physics.sparse_regression import (
    STLSQ,
    SequentialThresholdedLeastSquares,
    ridge_regression,
    elastic_net,
)
from safer_v3.physics.lpv_sindy import (
    LPVSINDyMonitor,
    IntegralFormulation,
    SchedulingFunction,
)

__all__ = [
    # Library
    'FunctionLibrary',
    'PolynomialLibrary',
    'FourierLibrary',
    'CustomLibrary',
    'CombinedLibrary',
    # Sparse regression
    'STLSQ',
    'SequentialThresholdedLeastSquares',
    'ridge_regression',
    'elastic_net',
    # LPV-SINDy
    'LPVSINDyMonitor',
    'IntegralFormulation',
    'SchedulingFunction',
]
