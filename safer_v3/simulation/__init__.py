"""
SAFER v3.0 Simulation Package.

This package provides simulation and data generation capabilities
for testing and validating the SAFER prognostics system.

Components:
    - engine_sim: Turbofan engine degradation simulation
    - data_generator: Synthetic data generation for testing

The simulation framework generates realistic degradation trajectories
that mimic C-MAPSS data characteristics for:
- Unit testing and integration testing
- Stress testing with edge cases
- Demonstration and benchmarking
- Sensitivity analysis
"""

from .engine_sim import (
    EngineSimulator,
    DegradationModel,
    LinearDegradation,
    ExponentialDegradation,
    PiecewiseDegradation,
    SensorNoise,
    FaultInjector,
)

from .data_generator import (
    CMAPSSGenerator,
    StreamingDataGenerator,
    SyntheticFleet,
    generate_trajectory,
    generate_fleet_data,
)


__all__ = [
    # Engine simulation
    'EngineSimulator',
    'DegradationModel',
    'LinearDegradation',
    'ExponentialDegradation',
    'PiecewiseDegradation',
    'SensorNoise',
    'FaultInjector',
    # Data generation
    'CMAPSSGenerator',
    'StreamingDataGenerator',
    'SyntheticFleet',
    'generate_trajectory',
    'generate_fleet_data',
]
