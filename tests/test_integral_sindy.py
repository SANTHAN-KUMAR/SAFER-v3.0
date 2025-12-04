"""
Unit tests for integral SINDy formulation.

Tests the IntegralFormulation class to ensure:
1. Trapezoidal integration accuracy
2. Window-based processing correctness
3. Edge case handling (small windows, noisy data, etc.)
4. Numerical stability with different scales

References:
    - Numerical recipes for trapezoidal rule
    - Kaiser et al., "Sparse identification for MPC in low-data limit" (2018)
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from safer_v3.physics.lpv_sindy import IntegralFormulation


class TestIntegralFormulation:
    """Test suite for IntegralFormulation class."""
    
    @pytest.fixture
    def integral(self):
        """Create a standard integral formulation for testing."""
        return IntegralFormulation(
            window_size=5,
            dt=1.0,
            method='trapezoidal',
        )
    
    def test_initialization(self):
        """Test IntegralFormulation initialization."""
        integral = IntegralFormulation(window_size=10, dt=0.5, method='trapezoidal')
        assert integral.window_size == 10
        assert integral.dt == 0.5
        assert integral.method == 'trapezoidal'
    
    def test_integrate_linear_trajectory(self, integral):
        """Test integration of linear trajectory (constant velocity).
        
        For a linear trajectory x(t) = at + b with constant derivative,
        the integral ∫x dt should be (a/2)t² + bt + c.
        
        For discrete case with window_size=5:
        Δx = x(t+5) - x(t) should be correctly computed.
        """
        # Linear trajectory: x(t) = 2*t
        t = np.arange(0, 20)
        x = 2 * t
        x = x.reshape(-1, 1)  # (20, 1)
        
        delta_x, weights = integral.integrate(x)
        
        # For linear trajectory, delta_x should be constant = 2 * window_size = 10
        expected_delta = 2 * integral.window_size
        
        np.testing.assert_allclose(
            delta_x[:, 0],
            expected_delta,
            rtol=1e-10,
            err_msg="Linear trajectory integration failed"
        )
    
    def test_integrate_quadratic_trajectory(self, integral):
        """Test integration of quadratic trajectory.
        
        For x(t) = t² (parabolic), verify that delta_x = x(t+w) - x(t)
        is computed correctly.
        """
        # Quadratic trajectory: x(t) = t²
        t = np.arange(0, 20, dtype=np.float64)
        x = t ** 2
        x = x.reshape(-1, 1)
        
        delta_x, weights = integral.integrate(x)
        
        # Compute expected values: Δx = (t+w)² - t²
        for i in range(delta_x.shape[0]):
            t_start = i
            t_end = t_start + integral.window_size
            expected = t_end ** 2 - t_start ** 2
            
            np.testing.assert_allclose(
                delta_x[i, 0],
                expected,
                rtol=1e-10,
                err_msg=f"Quadratic integration failed at window {i}"
            )
    
    def test_weights_trapezoidal(self, integral):
        """Test that trapezoidal weights are correct.
        
        For trapezoidal rule: w = [0.5, 1, 1, ..., 1, 0.5] * dt / window_size
        """
        t = np.random.randn(100, 1)
        delta_x, weights = integral.integrate(t)
        
        # Verify weight structure for trapezoidal rule
        assert len(weights) == integral.window_size
        
        # First and last weights should be 0.5 (trapezoidal rule)
        # Middle weights should be 1.0
        expected_weights = np.ones(integral.window_size)
        expected_weights[0] *= 0.5
        expected_weights[-1] *= 0.5
        
        np.testing.assert_allclose(
            weights,
            expected_weights,
            rtol=1e-10,
            err_msg="Trapezoidal weights incorrect"
        )
    
    def test_window_size_constraint(self, integral):
        """Test that data must be longer than window size."""
        # Data shorter than window
        X_short = np.random.randn(3, 2)
        
        with pytest.raises(ValueError):
            integral.integrate(X_short)
    
    def test_multi_dimensional_state(self):
        """Test integration with multi-dimensional state."""
        integral = IntegralFormulation(window_size=5, dt=1.0)
        
        # 3D state: [x, y, z]
        n_samples = 50
        n_dims = 3
        X = np.random.randn(n_samples, n_dims)
        
        delta_x, weights = integral.integrate(X)
        
        # Check shape
        expected_n_windows = n_samples - integral.window_size + 1
        assert delta_x.shape == (expected_n_windows, n_dims)
        
        # Check each dimension independently
        for d in range(n_dims):
            x_d = X[:, d:d+1]
            delta_x_d, _ = integral.integrate(x_d)
            np.testing.assert_allclose(delta_x[:, d], delta_x_d[:, 0])
    
    def test_noisy_data_smoothing(self):
        """Test that integral formulation smooths high-frequency noise.
        
        Integration acts as a low-pass filter. High-frequency noise
        should be attenuated more than low-frequency signals.
        """
        n_samples = 1000
        t = np.linspace(0, 10, n_samples)
        
        # Low-frequency signal (slowly varying)
        signal = 10 * np.sin(2 * np.pi * 0.2 * t)
        
        # Add high-frequency noise
        noise = np.random.randn(n_samples) * 2
        x = signal + noise
        
        integral = IntegralFormulation(window_size=10, dt=0.01)
        delta_x, weights = integral.integrate(x.reshape(-1, 1))
        
        # Integrated signal should follow low-frequency component better
        # (this is qualitative, but we can check that variance reduces)
        noise_variance = np.var(noise)
        integrated_variance = np.var(delta_x)
        
        # Integration should reduce high-frequency content
        assert integrated_variance < noise_variance, \
            "Integration should reduce noise variance"
    
    def test_integrate_library(self, integral):
        """Test integration of library features."""
        n_samples = 100
        n_features = 5
        
        # Simulate library features (e.g., polynomial basis)
        X = np.random.randn(n_samples, n_features)
        
        delta_x, weights = integral.integrate(X)
        
        # Integrate library features using the weights
        Theta_integrated = integral.integrate_library(X, weights)
        
        # Check shape
        expected_n_windows = n_samples - integral.window_size + 1
        assert Theta_integrated.shape == (expected_n_windows, n_features)
        
        # Check numerical values
        # Manual computation: sum(Theta[i:i+w] * weights) for each window
        for i in range(expected_n_windows):
            window = X[i:i+integral.window_size]
            expected = np.sum(window * weights.reshape(-1, 1), axis=0)
            np.testing.assert_allclose(
                Theta_integrated[i],
                expected,
                rtol=1e-10,
                err_msg=f"Library integration failed at window {i}"
            )
    
    def test_small_window_edge_case(self):
        """Test with minimum valid window size (2)."""
        integral = IntegralFormulation(window_size=2, dt=1.0)
        
        X = np.random.randn(10, 3)
        delta_x, weights = integral.integrate(X)
        
        # Should produce n_samples - 1 windows
        assert delta_x.shape[0] == 9
        assert len(weights) == 2
    
    def test_large_window_size(self):
        """Test with large window size (most of data)."""
        n_samples = 50
        window_size = n_samples - 5
        integral = IntegralFormulation(window_size=window_size, dt=1.0)
        
        X = np.random.randn(n_samples, 3)
        delta_x, weights = integral.integrate(X)
        
        # Should produce only 5 windows
        assert delta_x.shape[0] == 5
        assert len(weights) == window_size
    
    def test_zero_data_handling(self):
        """Test handling of zero/constant data."""
        integral = IntegralFormulation(window_size=5, dt=1.0)
        
        # Constant data (no change)
        X = np.ones((20, 2)) * 5.0
        
        delta_x, weights = integral.integrate(X)
        
        # Δx should be zeros (no change in constant trajectory)
        np.testing.assert_allclose(delta_x, 0.0, atol=1e-14)
    
    def test_numerical_stability_large_scale(self):
        """Test numerical stability with large-scale data."""
        integral = IntegralFormulation(window_size=5, dt=1.0)
        
        # Very large values
        X_large = np.random.randn(50, 3) * 1e6
        delta_x_large, _ = integral.integrate(X_large)
        
        # Very small values
        X_small = np.random.randn(50, 3) * 1e-6
        delta_x_small, _ = integral.integrate(X_small)
        
        # Scaling should be preserved
        ratio = np.abs(delta_x_large[0, 0] / delta_x_small[0, 0])
        expected_ratio = 1e12  # 1e6 / 1e-6
        
        np.testing.assert_allclose(
            ratio,
            expected_ratio,
            rtol=0.1,  # Slightly relaxed for numerical errors
            err_msg="Scale preservation failed"
        )
    
    def test_deterministic_output(self):
        """Test that same input always produces same output (deterministic)."""
        integral = IntegralFormulation(window_size=5, dt=1.0)
        X = np.random.randn(30, 3)
        
        # Run multiple times
        results = []
        for _ in range(5):
            delta_x, weights = integral.integrate(X)
            results.append(delta_x.copy())
        
        # All should be identical
        for i in range(1, 5):
            np.testing.assert_array_equal(results[0], results[i])
    
    def test_monotonicity_monotonic_data(self):
        """Test with monotonically increasing data.
        
        For monotonically increasing x, delta_x should be positive.
        """
        integral = IntegralFormulation(window_size=5, dt=1.0)
        
        # Strictly increasing
        X = np.arange(0, 50).reshape(-1, 1) * 1.0
        delta_x, _ = integral.integrate(X)
        
        # All elements should be positive
        assert np.all(delta_x > 0), "Delta_x should be positive for increasing trajectory"
    
    def test_negative_trajectory(self):
        """Test with decreasing (negative derivative) trajectory."""
        integral = IntegralFormulation(window_size=5, dt=1.0)
        
        # Strictly decreasing
        X = np.arange(50, 0, -1).reshape(-1, 1) * 1.0
        delta_x, _ = integral.integrate(X)
        
        # All elements should be negative
        assert np.all(delta_x < 0), "Delta_x should be negative for decreasing trajectory"


class TestIntegralFormulationPerformance:
    """Performance tests for IntegralFormulation."""
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        integral = IntegralFormulation(window_size=10, dt=1.0)
        
        # Large dataset: 1M samples, 14 sensors
        X = np.random.randn(int(1e6), 14)
        
        import time
        start = time.time()
        delta_x, weights = integral.integrate(X)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 5 seconds on modern hardware)
        assert elapsed < 5.0, f"Integration took {elapsed:.2f}s (expected < 5s)"
        
        print(f"\n✓ 1M sample integration completed in {elapsed:.3f}s")
    
    def test_memory_efficiency(self):
        """Test memory usage with large datasets."""
        integral = IntegralFormulation(window_size=5, dt=1.0)
        
        # 100k samples
        n_samples = 100000
        n_features = 14
        X = np.random.randn(n_samples, n_features)
        
        # Memory should be reasonable
        import sys
        x_size_mb = sys.getsizeof(X) / (1024**2)
        
        delta_x, weights = integral.integrate(X)
        delta_size_mb = sys.getsizeof(delta_x) / (1024**2)
        
        # Output should be similar size to input (window_size is small)
        assert delta_size_mb < x_size_mb * 2, "Memory usage seems excessive"
        
        print(f"\n✓ Input: {x_size_mb:.1f} MB, Output: {delta_size_mb:.1f} MB")


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
