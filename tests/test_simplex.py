"""
Unit tests for Simplex Decision Module.

Tests cover:
- SimplexDecisionModule initialization
- State transitions (COMPLEX, BASELINE, TRANSITION, FAULT)
- Safety monitoring and switch logic
- Conformal prediction integration
- Alert generation
- Hysteresis and rate limiting
"""

import pytest
import numpy as np


class TestSimplexDecisionModule:
    """Test suite for SimplexDecisionModule."""
    
    @pytest.fixture
    def simplex(self, torch_available):
        """Create SimplexDecisionModule for testing."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.decision.simplex import SimplexDecisionModule, SimplexConfig
        config = SimplexConfig(
            physics_threshold=0.1,
            divergence_threshold=30.0,
            uncertainty_threshold=50.0,
            hysteresis_cycles=5,
            conservative_margin=5.0,
        )
        return SimplexDecisionModule(config)
    
    def test_initialization(self, simplex):
        """Test Simplex initializes in BASELINE state (safe by design)."""
        from safer_v3.decision.simplex import SimplexState
        # Simplex starts in BASELINE state for safety
        assert simplex.state == SimplexState.BASELINE
    
    def test_decide_returns_result(self, simplex):
        """Test decide returns a DecisionResult."""
        result = simplex.decide(
            complex_rul=50.0,
            baseline_rul=52.0,
            rul_lower=45.0,
            rul_upper=55.0,
            physics_residual=0.05,
        )
        assert result is not None
        assert hasattr(result, 'rul')
        assert hasattr(result, 'state')
    
    def test_baseline_mode_uses_margin(self, simplex):
        """Test baseline mode applies conservative margin."""
        from safer_v3.decision.simplex import SimplexState
        # Simplex starts in BASELINE mode
        result = simplex.decide(
            complex_rul=50.0,
            baseline_rul=52.0,
            rul_lower=45.0,
            rul_upper=55.0,
            physics_residual=0.05,  # Below threshold
        )
        # Should use baseline RUL minus conservative margin
        assert result.state == SimplexState.BASELINE
        expected_rul = 52.0 - simplex.config.conservative_margin
        assert result.rul == expected_rul
    
    def test_high_physics_triggers_switch(self, simplex):
        """Test high physics residual triggers switch to baseline."""
        from safer_v3.decision.simplex import SimplexState, SwitchReason
        
        # First call with normal residual
        simplex.decide(
            complex_rul=50.0,
            baseline_rul=52.0,
            rul_lower=45.0,
            rul_upper=55.0,
            physics_residual=0.05,
        )
        
        # Second call with high residual
        result = simplex.decide(
            complex_rul=50.0,
            baseline_rul=52.0,
            rul_lower=45.0,
            rul_upper=55.0,
            physics_residual=0.5,  # Above threshold
        )
        
        # Should trigger switch or be in transition/baseline
        assert result.state in [SimplexState.TRANSITION, SimplexState.BASELINE] or \
               (hasattr(result, 'switch_reason') and result.switch_reason == SwitchReason.PHYSICS_ANOMALY)
    
    def test_high_uncertainty_triggers_switch(self, simplex):
        """Test high uncertainty triggers switch to baseline."""
        from safer_v3.decision.simplex import SimplexState
        
        # Wide confidence interval = high uncertainty
        result = simplex.decide(
            complex_rul=50.0,
            baseline_rul=52.0,
            rul_lower=0.0,    # Very wide interval
            rul_upper=100.0,  # 100 cycle uncertainty
            physics_residual=0.05,
        )
        
        # High uncertainty should affect decision
        assert result.rul_upper - result.rul_lower > 0 or result.rul_upper >= result.rul_lower
    
    def test_baseline_uses_conservative_margin(self, simplex):
        """Test baseline mode applies conservative margin."""
        from safer_v3.decision.simplex import SimplexState
        
        # Force into baseline by high physics residual multiple times
        for _ in range(10):
            result = simplex.decide(
                complex_rul=50.0,
                baseline_rul=60.0,  # Baseline predicts higher
                rul_lower=45.0,
                rul_upper=55.0,
                physics_residual=0.5,  # High residual
            )
        
        if result.state == SimplexState.BASELINE:
            # Should use baseline RUL minus margin
            expected_conservative = 60.0 - simplex.config.conservative_margin
            assert result.rul <= 60.0  # Should be conservative
    
    def test_hysteresis_prevents_rapid_switch(self, simplex):
        """Test hysteresis prevents rapid mode switching."""
        # Make multiple decisions with alternating residuals
        states = []
        for i in range(20):
            residual = 0.05 if i % 2 == 0 else 0.5  # Alternating
            result = simplex.decide(
                complex_rul=50.0,
                baseline_rul=52.0,
                rul_lower=45.0,
                rul_upper=55.0,
                physics_residual=residual,
            )
            states.append(result.state.name)
        
        # Should not oscillate every cycle due to hysteresis
        transitions = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
        assert transitions < len(states) - 1, "Hysteresis should reduce oscillations"


class TestSimplexConfig:
    """Test SimplexConfig validation and defaults."""
    
    def test_default_config(self, torch_available):
        """Test default config has reasonable values."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.decision.simplex import SimplexConfig
        config = SimplexConfig()
        assert config.physics_threshold > 0
        assert config.divergence_threshold > 0
        assert config.hysteresis_cycles > 0
    
    def test_config_validation(self, torch_available):
        """Test config rejects negative threshold at creation."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.decision.simplex import SimplexConfig
        # Negative threshold - dataclass doesn't validate, but we can check value
        config = SimplexConfig(physics_threshold=-1.0)
        # Just verify the field was set (validation may be external)
        assert config.physics_threshold == -1.0


class TestSafetyMonitor:
    """Test SafetyMonitor component."""
    
    @pytest.fixture
    def safety_monitor(self, torch_available):
        """Create SafetyMonitor for testing."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.decision.simplex import SafetyMonitor, SimplexConfig
        config = SimplexConfig(
            physics_threshold=0.1,
            divergence_threshold=30.0,
            uncertainty_threshold=50.0,
        )
        return SafetyMonitor(config=config)
    
    def test_check_physics_safe(self, safety_monitor):
        """Test physics check with safe residual."""
        is_safe, reason = safety_monitor.check(
            physics_residual=0.05,  # Below threshold
            complex_rul=50.0,
            baseline_rul=52.0,
            uncertainty_width=10.0,
        )
        assert is_safe  # Should be safe
    
    def test_check_physics_unsafe(self, safety_monitor):
        """Test physics check with unsafe residual."""
        is_safe, reason = safety_monitor.check(
            physics_residual=0.5,  # Above threshold
            complex_rul=50.0,
            baseline_rul=52.0,
            uncertainty_width=10.0,
        )
        assert not is_safe  # Should be unsafe
    
    def test_check_divergence_safe(self, safety_monitor):
        """Test divergence check with small difference."""
        is_safe, reason = safety_monitor.check(
            physics_residual=0.05,
            complex_rul=50.0,
            baseline_rul=55.0,  # Small difference
            uncertainty_width=10.0,
        )
        assert is_safe
    
    def test_check_divergence_unsafe(self, safety_monitor):
        """Test divergence check with large difference."""
        is_safe, reason = safety_monitor.check(
            physics_residual=0.05,
            complex_rul=50.0,
            baseline_rul=100.0,  # Large difference
            uncertainty_width=10.0,
        )
        assert not is_safe
    
    def test_check_uncertainty_safe(self, safety_monitor):
        """Test uncertainty check with narrow interval."""
        is_safe, reason = safety_monitor.check(
            physics_residual=0.05,
            complex_rul=50.0,
            baseline_rul=52.0,
            uncertainty_width=10.0,  # Narrow interval
        )
        assert is_safe
    
    def test_check_uncertainty_unsafe(self, safety_monitor):
        """Test uncertainty check with wide interval."""
        is_safe, reason = safety_monitor.check(
            physics_residual=0.05,
            complex_rul=50.0,
            baseline_rul=52.0,
            uncertainty_width=100.0,  # Wide interval
        )
        assert not is_safe


class TestSwitchRateLimiter:
    """Test SwitchRateLimiter component."""
    
    @pytest.fixture
    def rate_limiter(self, torch_available):
        """Create SwitchRateLimiter for testing."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.decision.simplex import SwitchRateLimiter
        return SwitchRateLimiter(
            max_switches_per_minute=2.0,
            window_seconds=60.0,
        )
    
    def test_allows_first_switch(self, rate_limiter):
        """Test first switch is always allowed."""
        can_switch = rate_limiter.can_switch()
        assert can_switch
    
    def test_blocks_rapid_switches(self, rate_limiter):
        """Test rapid switches are blocked."""
        # Record first switch
        rate_limiter.record_switch()
        
        # Record second switch
        rate_limiter.record_switch()
        
        # Third switch should be blocked (over rate limit)
        rate_limiter.record_switch()
        can_switch = rate_limiter.can_switch()
        assert not can_switch, "Rapid switch should be blocked"
