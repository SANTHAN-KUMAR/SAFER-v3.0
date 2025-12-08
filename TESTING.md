# SAFER v3.0 Testing Guide

This guide explains how to run tests and validate the SAFER v3.0 system.

## Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio

# Install the package in development mode
pip install -e .
```

## Running Tests

### Run All Tests

```bash
# From project root
pytest tests/ -v
```

### Run Tests by Module

```bash
# Test Mamba model
pytest tests/test_mamba.py -v

# Test baselines (LSTM, Transformer, CNN-LSTM)
pytest tests/test_baselines.py -v

# Test LPV-SINDy physics monitor
pytest tests/test_lpv_sindy.py -v

# Test decision modules
pytest tests/test_simplex.py -v
pytest tests/test_conformal.py -v
pytest tests/test_alerts.py -v

# Test utilities
pytest tests/test_metrics.py -v
pytest tests/test_library.py -v
pytest tests/test_sparse_regression.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=safer_v3 --cov-report=html
# Open htmlcov/index.html in browser
```

### Skip Slow Tests

```bash
pytest tests/ -v -m "not slow"
```

### Skip GPU Tests (for CPU-only environments)

```bash
pytest tests/ -v -m "not gpu"
```

## Test Categories

| Category | Description | Files |
|----------|-------------|-------|
| **Core Models** | Neural network architectures | `test_mamba.py`, `test_baselines.py` |
| **Physics** | LPV-SINDy, libraries, regression | `test_lpv_sindy.py`, `test_library.py`, `test_sparse_regression.py` |
| **Decision** | Simplex, conformal, alerts | `test_simplex.py`, `test_conformal.py`, `test_alerts.py` |
| **Utilities** | Metrics, config | `test_metrics.py` |
| **Integration** | Existing integral formulation tests | `test_integral_sindy.py` |

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `random_seed` - Fixed seed (42) for reproducibility
- `rng` - NumPy random generator
- `n_sensors` - Standard sensor count (14)
- `window_size` - Standard window (50)
- `sample_sensor_data` - Generated sensor data
- `sample_trajectory` - Full degradation trajectory
- `mamba_config` - MambaConfig for testing
- `lpv_sindy_config` - LPVSINDyConfig for testing
- `torch_device` - PyTorch device (CPU/CUDA)

## Writing New Tests

### Basic Test Structure

```python
import pytest
import numpy as np

class TestMyFeature:
    """Test suite for my feature."""
    
    @pytest.fixture
    def my_object(self, torch_available):
        """Create object for testing."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.my_module import MyClass
        return MyClass()
    
    def test_basic_functionality(self, my_object):
        """Test basic functionality."""
        result = my_object.do_something()
        assert result is not None
    
    def test_edge_case(self, my_object, rng):
        """Test edge case with random data."""
        data = rng.normal(0, 1, (100, 14))
        result = my_object.process(data)
        assert result.shape == (100,)
```

### Using Fixtures

```python
def test_with_sample_data(self, sample_sensor_data, n_sensors):
    """Test using sample data fixture."""
    assert sample_sensor_data.shape[1] == n_sensors
```

### Testing PyTorch Models

```python
def test_model_forward(self, model, torch_tensor_batch, batch_size):
    """Test model forward pass."""
    output = model(torch_tensor_batch)
    assert output.shape == (batch_size, 1)
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .
      - name: Run tests
        run: pytest tests/ -v --cov=safer_v3
```

## Troubleshooting

### PyTorch Not Found

Tests will skip automatically if PyTorch is not installed:
```
SKIPPED [1] test_mamba.py: PyTorch not available
```

Solution: `pip install torch`

### Import Errors

If tests fail with import errors:
```bash
pip install -e .  # Reinstall package
```

### GPU Memory Issues

For GPU memory issues:
```bash
pytest tests/ -v -m "not gpu"  # Skip GPU tests
```

Or limit GPU memory:
```python
import torch
torch.cuda.set_per_process_memory_fraction(0.5)
```

## Expected Test Results

After all fixes are applied, tests should pass with:
- **Core Models**: All pass (when PyTorch installed)
- **Physics**: All pass
- **Decision**: All pass
- **Utilities**: All pass

Target coverage: **>80%**
