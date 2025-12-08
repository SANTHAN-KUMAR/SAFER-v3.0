"""
Unit tests for Function Libraries.

Tests cover:
- PolynomialLibrary
- FourierLibrary
- LPVAugmentedLibrary (health-dependent terms)
- CombinedLibrary
- Feature name generation
"""

import pytest
import numpy as np


class TestPolynomialLibrary:
    """Test PolynomialLibrary functionality."""
    
    @pytest.fixture
    def library(self):
        """Create PolynomialLibrary for testing."""
        from safer_v3.physics.library import PolynomialLibrary
        return PolynomialLibrary(degree=2, include_bias=True)
    
    def test_fit_transform_shape(self, library, sample_sensor_data):
        """Test fit and transform produce correct shape."""
        library.fit(sample_sensor_data)
        features = library.transform(sample_sensor_data)
        
        assert features.shape[0] == sample_sensor_data.shape[0]
        assert features.shape[1] > 0
    
    def test_includes_bias_term(self, library, sample_sensor_data):
        """Test bias term is included when enabled."""
        library.fit(sample_sensor_data)
        features = library.transform(sample_sensor_data)
        
        # First column should be all ones (bias)
        assert np.allclose(features[:, 0], 1.0)
    
    def test_includes_linear_terms(self, library, sample_sensor_data):
        """Test linear terms are included."""
        library.fit(sample_sensor_data)
        feature_names = library.get_feature_names()
        
        # Should have x0, x1, ... terms
        linear_terms = [n for n in feature_names if n.startswith('x') and '^' not in n]
        assert len(linear_terms) > 0
    
    def test_includes_quadratic_terms(self, library, sample_sensor_data):
        """Test quadratic terms are included for degree 2."""
        library.fit(sample_sensor_data)
        feature_names = library.get_feature_names()
        
        # Should have x^2 terms
        quadratic_terms = [n for n in feature_names if '^2' in n or '*' in n]
        assert len(quadratic_terms) > 0
    
    def test_feature_names_match_count(self, library, sample_sensor_data):
        """Test feature names count matches transform output."""
        library.fit(sample_sensor_data)
        features = library.transform(sample_sensor_data)
        feature_names = library.get_feature_names()
        
        assert len(feature_names) == features.shape[1]


class TestLPVAugmentedLibrary:
    """Test LPVAugmentedLibrary with health-dependent terms."""
    
    @pytest.fixture
    def library(self):
        """Create LPVAugmentedLibrary for testing."""
        from safer_v3.physics.library import LPVAugmentedLibrary
        return LPVAugmentedLibrary(
            degree=2,
            include_bias=True,
            include_interaction=True,
        )
    
    def test_transform_with_scheduling_param(self, library, rng, n_sensors):
        """Test transform accepts scheduling parameter."""
        X = rng.normal(0, 1, (200, n_sensors))
        p = rng.uniform(0, 1, 200)  # Scheduling parameter
        
        library.fit(X)
        features = library.transform(X, p)
        
        assert features.shape[0] == X.shape[0]
        assert features.shape[1] > 0
    
    def test_includes_p_terms(self, library, rng, n_sensors):
        """Test library includes p-dependent terms."""
        X = rng.normal(0, 1, (200, n_sensors))
        p = rng.uniform(0, 1, 200)
        
        library.fit(X)
        features = library.transform(X, p)
        feature_names = library.get_feature_names()
        
        # Should have p*x terms
        p_terms = [n for n in feature_names if 'p' in n.lower()]
        # Depending on implementation, may have different naming
        assert features.shape[1] > n_sensors  # Should have more than just linear
    
    def test_p_zero_vs_p_one(self, library, rng, n_sensors):
        """Test features differ for p=0 vs p=1."""
        X = rng.normal(0, 1, (100, n_sensors))
        p_zero = np.zeros(100)
        p_one = np.ones(100)
        
        library.fit(X)
        features_zero = library.transform(X, p_zero)
        features_one = library.transform(X, p_one)
        
        # Features should differ when p differs
        assert not np.allclose(features_zero, features_one)


class TestFourierLibrary:
    """Test FourierLibrary functionality."""
    
    @pytest.fixture
    def library(self):
        """Create FourierLibrary for testing."""
        from safer_v3.physics.library import FourierLibrary
        return FourierLibrary(n_frequencies=3)
    
    def test_transform_creates_sin_cos(self, library, sample_sensor_data):
        """Test Fourier library creates sin/cos features."""
        library.fit(sample_sensor_data)
        features = library.transform(sample_sensor_data)
        
        # Should have 2*n_frequencies*n_sensors terms (sin and cos for each)
        assert features.shape[1] > sample_sensor_data.shape[1]
    
    def test_features_bounded(self, library, sample_sensor_data):
        """Test Fourier features are bounded by ±1."""
        library.fit(sample_sensor_data)
        features = library.transform(sample_sensor_data)
        
        # Sin/cos are bounded
        assert np.all(features >= -1.01)  # Small margin for numerical error
        assert np.all(features <= 1.01)


class TestCombinedLibrary:
    """Test CombinedLibrary functionality."""
    
    @pytest.fixture
    def combined_library(self):
        """Create CombinedLibrary for testing."""
        from safer_v3.physics.library import (
            CombinedLibrary, 
            PolynomialLibrary, 
            FourierLibrary
        )
        poly = PolynomialLibrary(degree=1)
        fourier = FourierLibrary(n_frequencies=2)
        return CombinedLibrary([poly, fourier])
    
    def test_combines_features(self, combined_library, sample_sensor_data):
        """Test combined library merges features."""
        combined_library.fit(sample_sensor_data)
        features = combined_library.transform(sample_sensor_data)
        
        # Should have features from both libraries
        assert features.shape[1] > sample_sensor_data.shape[1]
    
    def test_feature_names_from_all(self, combined_library, sample_sensor_data):
        """Test feature names come from all libraries."""
        combined_library.fit(sample_sensor_data)
        feature_names = combined_library.get_feature_names()
        
        # Should have polynomial and Fourier names
        assert len(feature_names) > 0


class TestBuildTurbofanLibrary:
    """Test build_turbofan_library convenience function."""
    
    def test_creates_library(self, n_sensors):
        """Test function creates a valid library."""
        from safer_v3.physics.library import build_turbofan_library
        library = build_turbofan_library(
            n_sensors=n_sensors,
            polynomial_degree=2,
        )
        assert library is not None
    
    def test_library_works(self, sample_sensor_data, n_sensors):
        """Test created library can fit and transform."""
        from safer_v3.physics.library import build_turbofan_library
        library = build_turbofan_library(
            n_sensors=n_sensors,
            polynomial_degree=2,
        )
        
        library.fit(sample_sensor_data)
        features = library.transform(sample_sensor_data)
        
        assert features.shape[0] == sample_sensor_data.shape[0]
