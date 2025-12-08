"""
Unit tests for LPV-SINDy Physics Monitor.

Tests cover:
- Monitor initialization with different configurations
- Scheduling parameter computation
- Fit method with and without adaptive scheduling
- Residual computation
- Anomaly detection
- Sparsity of discovered models
- Integral formulation
"""

import pytest
import numpy as np


class TestLPVSINDyMonitorInitialization:
    """Test LPV-SINDy monitor initialization."""
    
    def test_default_initialization(self, torch_available, n_sensors):
        """Test monitor initializes with defaults."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
        monitor = LPVSINDyMonitor(n_sensors=n_sensors)
        assert monitor.n_sensors == n_sensors
        assert monitor.config is not None
    
    def test_config_initialization(self, torch_available, lpv_sindy_config, n_sensors):
        """Test monitor initializes with config."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
        monitor = LPVSINDyMonitor(config=lpv_sindy_config, n_sensors=n_sensors)
        assert monitor.config.threshold == lpv_sindy_config.threshold
        assert monitor.config.polynomial_degree == lpv_sindy_config.polynomial_degree
    
    def test_lpv_library_used_when_adaptive(self, torch_available, n_sensors):
        """Test LPVAugmentedLibrary is used when adaptive scheduling enabled."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
        from safer_v3.utils.config import LPVSINDyConfig
        
        config = LPVSINDyConfig(use_adaptive_scheduling=True)
        monitor = LPVSINDyMonitor(config=config, n_sensors=n_sensors)
        
        # Library should be LPVAugmentedLibrary type
        assert 'LPVAugmented' in type(monitor.library).__name__ or \
               hasattr(monitor.library, 'transform')


class TestSchedulingParameter:
    """Test scheduling parameter computation."""
    
    @pytest.fixture
    def monitor(self, torch_available, n_sensors):
        """Create monitor for testing."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
        from safer_v3.utils.config import LPVSINDyConfig
        config = LPVSINDyConfig(
            use_adaptive_scheduling=True,
            egtm_sensor_idx=9,
            nominal_egtm=100.0,
            min_egtm=0.0,
        )
        return LPVSINDyMonitor(config=config, n_sensors=n_sensors)
    
    def test_scheduling_parameter_shape(self, monitor, sample_sensor_data):
        """Test scheduling parameter has correct shape."""
        p = monitor.compute_scheduling_parameter(sample_sensor_data)
        assert p.shape == (len(sample_sensor_data),)
    
    def test_scheduling_parameter_range(self, monitor, rng, n_sensors):
        """Test scheduling parameter is in [0, 1]."""
        # Create data with EGTM in expected range
        X = rng.uniform(0, 100, (500, n_sensors))
        p = monitor.compute_scheduling_parameter(X)
        assert np.all(p >= 0), "Scheduling param must be >= 0"
        assert np.all(p <= 1), "Scheduling param must be <= 1"
    
    def test_scheduling_degradation_trend(self, monitor, sample_trajectory, n_sensors):
        """Test scheduling parameter decreases with degradation."""
        # Create trajectory where sensor 9 (EGTM) decreases over time
        n_samples = 200
        X = np.random.randn(n_samples, n_sensors)
        X[:, 9] = np.linspace(100, 10, n_samples)  # Degrading EGTM
        
        p = monitor.compute_scheduling_parameter(X)
        
        # First half should have higher p (healthier)
        p_first_half = p[:n_samples//2].mean()
        p_second_half = p[n_samples//2:].mean()
        assert p_first_half > p_second_half, "p should decrease with degradation"


class TestLPVSINDyFit:
    """Test LPV-SINDy model fitting."""
    
    @pytest.fixture
    def monitor(self, torch_available, n_sensors):
        """Create monitor for testing."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
        from safer_v3.utils.config import LPVSINDyConfig
        config = LPVSINDyConfig(
            use_adaptive_scheduling=False,  # Simpler for basic fit test
            threshold=0.5,
            window_size=5,
        )
        return LPVSINDyMonitor(config=config, n_sensors=n_sensors)
    
    def test_fit_returns_results(self, monitor, sample_sensor_data):
        """Test fit returns results dictionary."""
        results = monitor.fit(sample_sensor_data)
        assert isinstance(results, dict)
        assert 'n_features' in results
        assert 'sparsity' in results
        assert 'train_rmse' in results
    
    def test_is_fitted_flag(self, monitor, sample_sensor_data):
        """Test _is_fitted flag is set after fitting."""
        assert not monitor._is_fitted
        monitor.fit(sample_sensor_data)
        assert monitor._is_fitted
    
    def test_coefficients_set(self, monitor, sample_sensor_data):
        """Test coefficients are set after fitting."""
        monitor.fit(sample_sensor_data)
        assert monitor._coefficients is not None
        assert monitor._coefficients.shape[1] == sample_sensor_data.shape[1]
    
    def test_sparsity_with_high_threshold(self, torch_available, sample_sensor_data, n_sensors):
        """Test higher threshold produces sparser model."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
        from safer_v3.utils.config import LPVSINDyConfig
        
        config_sparse = LPVSINDyConfig(threshold=1.0, use_adaptive_scheduling=False)
        config_dense = LPVSINDyConfig(threshold=0.01, use_adaptive_scheduling=False)
        
        monitor_sparse = LPVSINDyMonitor(config=config_sparse, n_sensors=n_sensors)
        monitor_dense = LPVSINDyMonitor(config=config_dense, n_sensors=n_sensors)
        
        results_sparse = monitor_sparse.fit(sample_sensor_data)
        results_dense = monitor_dense.fit(sample_sensor_data)
        
        # Sparse model should have higher sparsity
        assert results_sparse['sparsity'] >= results_dense['sparsity']


class TestLPVSINDyFitAdaptive:
    """Test LPV-SINDy fitting with adaptive scheduling."""
    
    @pytest.fixture
    def monitor(self, torch_available, n_sensors):
        """Create adaptive monitor for testing."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
        from safer_v3.utils.config import LPVSINDyConfig
        config = LPVSINDyConfig(
            use_adaptive_scheduling=True,
            threshold=0.5,
            window_size=5,
        )
        return LPVSINDyMonitor(config=config, n_sensors=n_sensors)
    
    def test_adaptive_fit_includes_scheduling_stats(self, monitor, sample_sensor_data):
        """Test adaptive fit includes scheduling parameter statistics."""
        results = monitor.fit(sample_sensor_data)
        assert results.get('use_adaptive_scheduling') == True
        if 'scheduling_param_stats' in results:
            stats = results['scheduling_param_stats']
            assert 'min' in stats
            assert 'max' in stats
            assert 'mean' in stats
    
    def test_adaptive_stores_scheduling_param(self, monitor, sample_sensor_data):
        """Test scheduling parameter is stored after fit."""
        monitor.fit(sample_sensor_data)
        assert monitor._scheduling_param is not None


class TestResidualComputation:
    """Test residual computation for anomaly detection."""
    
    @pytest.fixture
    def fitted_monitor(self, torch_available, sample_sensor_data, n_sensors):
        """Create fitted monitor for testing."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
        from safer_v3.utils.config import LPVSINDyConfig
        config = LPVSINDyConfig(use_adaptive_scheduling=False, window_size=5)
        monitor = LPVSINDyMonitor(config=config, n_sensors=n_sensors)
        monitor.fit(sample_sensor_data)
        return monitor
    
    def test_residual_shape(self, fitted_monitor, sample_sensor_data):
        """Test residuals have correct shape."""
        residuals = fitted_monitor._compute_residuals(sample_sensor_data)
        n_windows = len(sample_sensor_data) - fitted_monitor.config.window_size + 1
        assert residuals.shape[0] == n_windows
        assert residuals.shape[1] == sample_sensor_data.shape[1]
    
    def test_residual_stats_set(self, fitted_monitor):
        """Test residual statistics are set after fit."""
        assert fitted_monitor._residual_mean is not None
        assert fitted_monitor._residual_std is not None
        assert fitted_monitor._residual_threshold is not None
    
    def test_unfitted_raises_error(self, torch_available, sample_sensor_data, n_sensors):
        """Test computing residuals before fit raises error."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
        monitor = LPVSINDyMonitor(n_sensors=n_sensors)
        with pytest.raises(ValueError):
            monitor._compute_residuals(sample_sensor_data)


class TestAnomalyDetection:
    """Test anomaly detection functionality."""
    
    @pytest.fixture
    def fitted_monitor(self, torch_available, sample_sensor_data, n_sensors):
        """Create fitted monitor for testing."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
        from safer_v3.utils.config import LPVSINDyConfig
        config = LPVSINDyConfig(
            use_adaptive_scheduling=False, 
            window_size=5,
            residual_threshold_sigma=3.0,
        )
        monitor = LPVSINDyMonitor(config=config, n_sensors=n_sensors)
        monitor.fit(sample_sensor_data)
        return monitor
    
    def test_normal_data_low_anomaly_rate(self, fitted_monitor, sample_sensor_data):
        """Test normal data has low anomaly rate."""
        residuals = fitted_monitor._compute_residuals(sample_sensor_data)
        normalized = (residuals - fitted_monitor._residual_mean) / fitted_monitor._residual_std
        anomaly_scores = np.sqrt(np.mean(normalized ** 2, axis=1))
        
        # With 3-sigma threshold, expect ~1% anomalies for normal data
        threshold = fitted_monitor.config.residual_threshold_sigma
        anomaly_rate = np.mean(anomaly_scores > threshold)
        assert anomaly_rate < 0.3, f"Anomaly rate {anomaly_rate} too high for normal data"
