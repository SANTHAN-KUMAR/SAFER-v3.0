"""
Shared test fixtures and configuration for SAFER v3.0 test suite.

This module provides common fixtures used across all test modules,
ensuring consistent test setup and teardown.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Random State Fixtures
# =============================================================================

@pytest.fixture
def random_seed():
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def rng(random_seed):
    """NumPy random generator with fixed seed."""
    return np.random.default_rng(random_seed)


# =============================================================================
# Data Shape Constants
# =============================================================================

@pytest.fixture
def n_sensors():
    """Standard number of sensors for C-MAPSS dataset."""
    return 14


@pytest.fixture
def window_size():
    """Standard window size for sequence models."""
    return 50


@pytest.fixture
def batch_size():
    """Standard batch size for testing."""
    return 4


@pytest.fixture
def max_rul():
    """Maximum RUL cap value."""
    return 125


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_sensor_data(rng, n_sensors):
    """Generate sample sensor data for testing.
    
    Returns:
        Array of shape (1000, n_sensors) with realistic sensor ranges.
    """
    n_samples = 1000
    # Generate data with different scales per sensor
    data = np.zeros((n_samples, n_sensors))
    for i in range(n_sensors):
        # Create trending data + noise
        trend = np.linspace(0, 1, n_samples) * (i + 1) * 10
        noise = rng.normal(0, 0.1, n_samples)
        data[:, i] = trend + noise
    return data


@pytest.fixture
def sample_sequence_batch(rng, batch_size, window_size, n_sensors):
    """Generate batch of sequences for model testing.
    
    Returns:
        Array of shape (batch_size, window_size, n_sensors)
    """
    return rng.normal(0, 1, (batch_size, window_size, n_sensors)).astype(np.float32)


@pytest.fixture
def sample_rul_targets(rng, batch_size, max_rul):
    """Generate sample RUL targets.
    
    Returns:
        Array of shape (batch_size,) with values in [0, max_rul]
    """
    return rng.integers(0, max_rul, size=batch_size).astype(np.float32)


@pytest.fixture
def sample_trajectory(rng, n_sensors, max_rul):
    """Generate a single degradation trajectory.
    
    Simulates engine degradation with:
    - Decreasing health trend
    - Sensor correlations
    - Realistic noise
    
    Returns:
        Dictionary with 'sensors' and 'rul' arrays
    """
    n_cycles = max_rul + 50
    rul = np.maximum(max_rul - np.arange(n_cycles), 0)
    
    # Generate correlated sensor readings
    sensors = np.zeros((n_cycles, n_sensors))
    base_degradation = 1 - rul / max_rul  # 0 to 1 degradation
    
    for i in range(n_sensors):
        sensor_sensitivity = rng.uniform(0.5, 2.0)
        baseline = rng.uniform(100, 1000)
        noise_scale = rng.uniform(1, 10)
        
        sensors[:, i] = (
            baseline + 
            sensor_sensitivity * base_degradation * 50 +
            rng.normal(0, noise_scale, n_cycles)
        )
    
    return {
        'sensors': sensors,
        'rul': rul,
        'n_cycles': n_cycles,
    }


# =============================================================================
# PyTorch Fixtures (conditional on availability)
# =============================================================================

@pytest.fixture
def torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.fixture
def torch_device(torch_available):
    """Get PyTorch device for testing."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def torch_tensor_batch(sample_sequence_batch, torch_available, torch_device):
    """Convert numpy batch to PyTorch tensor."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    import torch
    return torch.from_numpy(sample_sequence_batch).to(torch_device)


@pytest.fixture
def torch_rul_targets(sample_rul_targets, torch_available, torch_device):
    """Convert numpy RUL targets to PyTorch tensor."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    import torch
    return torch.from_numpy(sample_rul_targets).to(torch_device)


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def mamba_config():
    """Create MambaConfig for testing."""
    from safer_v3.utils.config import MambaConfig
    return MambaConfig(
        d_input=14,
        d_model=32,  # Smaller for fast tests
        d_state=8,
        n_layers=2,
        dropout=0.1,
        max_rul=125,
        sequence_length=50,
        use_jit=False,  # Disable JIT for testing
    )


@pytest.fixture
def lpv_sindy_config():
    """Create LPVSINDyConfig for testing."""
    from safer_v3.utils.config import LPVSINDyConfig
    return LPVSINDyConfig(
        n_features=14,
        polynomial_degree=2,
        threshold=0.5,
        alpha=0.01,
        window_size=5,
        dt=1.0,
        use_adaptive_scheduling=True,
    )


@pytest.fixture
def decision_config():
    """Create DecisionConfig for testing."""
    from safer_v3.utils.config import DecisionConfig
    return DecisionConfig(
        alpha=0.05,
        tau_conflict=20.0,
        critical_rul=20,
        degradation_rul=50,
    )


# =============================================================================
# Test Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


# =============================================================================
# Skip Helpers
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests when no GPU is available."""
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    
    if not has_gpu:
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
