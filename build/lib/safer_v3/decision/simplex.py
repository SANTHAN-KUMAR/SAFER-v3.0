"""
Simplex Decision Module for SAFER v3.0.

This module implements the Simplex architecture for runtime safety
switching between high-performance and high-assurance components.

The Simplex approach provides formal safety guarantees by:
1. Running a high-performance complex controller (Mamba) in parallel
2. Monitoring via a safety checker (LPV-SINDy physics monitor)
3. Switching to a safe baseline if anomalies are detected

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │                    Decision Module                    │
    │  ┌─────────────┐   ┌─────────────┐   ┌────────────┐ │
    │  │   Mamba     │   │  Physics    │   │  Baseline  │ │
    │  │   (DAL E)   │──▶│  Monitor    │──▶│  (DAL C)   │ │
    │  │             │   │  (DAL C)    │   │            │ │
    │  └─────────────┘   └─────────────┘   └────────────┘ │
    │          │              │                   │        │
    │          ▼              ▼                   ▼        │
    │      ┌───────────────────────────────────────┐      │
    │      │           Simplex Arbiter             │      │
    │      │    (decides which output to use)      │      │
    │      └───────────────────────────────────────┘      │
    └──────────────────────────────────────────────────────┘

Safety Properties:
- Fail-safe: Defaults to baseline on any error
- Bounded switching: Maximum switch frequency limit
- Hysteresis: Prevents oscillation between modes
- Audit trail: All decisions logged for post-analysis

References:
    - Sha et al., "Using Simplicity to Control Complexity" (2001)
    - Seto et al., "Simplex Architecture for Safe Online Control" (1998)
"""

import time
import logging
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple, Callable, List
from dataclasses import dataclass, field
from collections import deque
import threading
import numpy as np


logger = logging.getLogger(__name__)


class SimplexState(Enum):
    """Current state of Simplex controller."""
    COMPLEX = auto()     # Using Mamba (high performance)
    BASELINE = auto()    # Using baseline (high assurance)
    TRANSITION = auto()  # Switching between modes
    FAULT = auto()       # Fault detected, fail-safe mode


class SwitchReason(Enum):
    """Reason for Simplex mode switch."""
    PHYSICS_ANOMALY = auto()      # Physics monitor detected anomaly
    UNCERTAINTY_HIGH = auto()     # Confidence interval too wide
    PREDICTION_DIVERGENCE = auto() # Mamba/baseline diverge significantly
    TIMEOUT = auto()              # Complex controller timeout
    MANUAL = auto()               # Manual override
    RECOVERY = auto()             # Recovering to complex mode
    INITIALIZATION = auto()       # Initial state


@dataclass
class DecisionResult:
    """Result of Simplex decision.
    
    Attributes:
        rul: Final RUL output
        rul_lower: Lower confidence bound
        rul_upper: Upper confidence bound
        state: Current Simplex state
        complex_rul: Mamba RUL prediction
        baseline_rul: Baseline RUL prediction
        used_source: Which predictor was used
        switch_reason: Reason if switch occurred
        physics_residual: Physics monitor residual
        decision_latency_ms: Decision computation time
        timestamp: Decision timestamp
    """
    rul: float
    rul_lower: float
    rul_upper: float
    state: SimplexState
    complex_rul: Optional[float] = None
    baseline_rul: Optional[float] = None
    used_source: str = 'baseline'
    switch_reason: Optional[SwitchReason] = None
    physics_residual: float = 0.0
    decision_latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def is_using_complex(self) -> bool:
        """Check if using complex (Mamba) predictor."""
        return self.state == SimplexState.COMPLEX
    
    @property
    def confidence_width(self) -> float:
        """Width of confidence interval."""
        return self.rul_upper - self.rul_lower
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rul': self.rul,
            'rul_lower': self.rul_lower,
            'rul_upper': self.rul_upper,
            'state': self.state.name,
            'complex_rul': self.complex_rul,
            'baseline_rul': self.baseline_rul,
            'used_source': self.used_source,
            'switch_reason': self.switch_reason.name if self.switch_reason else None,
            'physics_residual': self.physics_residual,
            'decision_latency_ms': self.decision_latency_ms,
            'timestamp': self.timestamp,
        }


@dataclass
class SimplexConfig:
    """Configuration for Simplex decision module.
    
    Attributes:
        physics_threshold: Threshold for physics anomaly detection
        divergence_threshold: Max allowed Mamba-baseline difference
        uncertainty_threshold: Max allowed confidence interval width
        recovery_window: Cycles before attempting recovery to complex
        max_switch_rate: Maximum switches per minute
        hysteresis_cycles: Cycles to wait before re-switching
        timeout_ms: Timeout for complex predictor
        conservative_margin: Extra safety margin for baseline
    """
    physics_threshold: float = 0.1
    divergence_threshold: float = 30.0
    uncertainty_threshold: float = 50.0
    recovery_window: int = 10
    max_switch_rate: float = 2.0  # switches per minute
    hysteresis_cycles: int = 5
    timeout_ms: float = 20.0
    conservative_margin: float = 5.0


class SafetyMonitor:
    """Monitors multiple safety conditions.
    
    Aggregates signals from physics monitor, uncertainty
    quantification, and other safety checks.
    """
    
    def __init__(
        self,
        config: SimplexConfig,
        window_size: int = 20,
    ):
        """Initialize safety monitor.
        
        Args:
            config: Simplex configuration
            window_size: Size of history window
        """
        self.config = config
        self.window_size = window_size
        
        # History buffers
        self._physics_residuals: deque = deque(maxlen=window_size)
        self._divergences: deque = deque(maxlen=window_size)
        self._uncertainties: deque = deque(maxlen=window_size)
        
        # State
        self._anomaly_count = 0
        self._total_checks = 0
    
    def check(
        self,
        physics_residual: float,
        complex_rul: float,
        baseline_rul: float,
        uncertainty_width: float,
    ) -> Tuple[bool, Optional[SwitchReason]]:
        """Check all safety conditions.
        
        Args:
            physics_residual: Residual from physics monitor
            complex_rul: Mamba prediction
            baseline_rul: Baseline prediction
            uncertainty_width: Width of confidence interval
            
        Returns:
            Tuple of (is_safe, switch_reason if unsafe)
        """
        self._total_checks += 1
        
        # Update history
        divergence = abs(complex_rul - baseline_rul)
        self._physics_residuals.append(physics_residual)
        self._divergences.append(divergence)
        self._uncertainties.append(uncertainty_width)
        
        # Check conditions
        reasons = []
        
        # Physics anomaly
        if physics_residual > self.config.physics_threshold:
            reasons.append(SwitchReason.PHYSICS_ANOMALY)
            logger.warning(
                f"Physics anomaly detected: residual={physics_residual:.4f} "
                f"(threshold={self.config.physics_threshold})"
            )
        
        # Prediction divergence
        if divergence > self.config.divergence_threshold:
            reasons.append(SwitchReason.PREDICTION_DIVERGENCE)
            logger.warning(
                f"Prediction divergence: {divergence:.1f} cycles "
                f"(threshold={self.config.divergence_threshold})"
            )
        
        # High uncertainty
        if uncertainty_width > self.config.uncertainty_threshold:
            reasons.append(SwitchReason.UNCERTAINTY_HIGH)
            logger.warning(
                f"High uncertainty: width={uncertainty_width:.1f} "
                f"(threshold={self.config.uncertainty_threshold})"
            )
        
        if reasons:
            self._anomaly_count += 1
            # Return highest priority reason
            return False, reasons[0]
        
        return True, None
    
    def check_recovery(self) -> bool:
        """Check if recovery to complex mode is safe.
        
        Returns:
            True if recovery is recommended
        """
        if len(self._physics_residuals) < self.config.recovery_window:
            return False
        
        recent = list(self._physics_residuals)[-self.config.recovery_window:]
        
        # Check if recent residuals are all below threshold
        return all(r < self.config.physics_threshold * 0.8 for r in recent)
    
    def check_health_trend(
        self,
        health_parameter: np.ndarray,
        min_samples: int = 5,
        improvement_threshold: float = 0.02,
    ) -> bool:
        """Check if health parameter is improving (trending towards health).
        
        For safe recovery from baseline to complex mode, we should verify
        that the engine health is actually improving, not just that residuals
        are temporarily low.
        
        The health parameter p(t) should be increasing (moving towards 1.0)
        for recovery to be justified. This prevents recovery during transient
        good periods in a degrading system.
        
        Args:
            health_parameter: Health parameter trajectory p(t), typically
                            computed from EGT margin. Should be shape (n_samples,)
                            with values in [0, 1] where 1.0 = healthy.
            min_samples: Minimum number of samples to analyze
            improvement_threshold: Minimum required health improvement rate
                                 (p_{t} - p_{t-w}) / w for recovery permission
            
        Returns:
            True if health is improving (dp/dt > threshold), False otherwise
            
        Example:
            >>> p = np.linspace(0.5, 0.6, 100)  # Improving health
            >>> is_improving = monitor.check_health_trend(p)
            >>> assert is_improving  # Health improving, recovery safe
        """
        # Validate input
        if len(health_parameter) < min_samples:
            logger.debug(
                f"Insufficient health samples ({len(health_parameter)} < {min_samples})"
            )
            return False
        
        # Get recent health parameter values
        recent_p = health_parameter[-min_samples:]
        
        # Compute health trend: average derivative dp/dt
        # Use linear regression for robustness to noise
        t = np.arange(len(recent_p))
        
        # Simple trend: compare first vs last
        p_start = recent_p[0]
        p_end = recent_p[-1]
        p_trend = (p_end - p_start) / len(recent_p)
        
        is_improving = p_trend > improvement_threshold
        
        logger.debug(
            f"Health trend check: p_trend={p_trend:.6f}, "
            f"threshold={improvement_threshold}, improving={is_improving}"
        )
        
        return is_improving
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_checks': self._total_checks,
            'anomaly_count': self._anomaly_count,
            'anomaly_rate': (
                self._anomaly_count / self._total_checks
                if self._total_checks > 0 else 0.0
            ),
            'avg_physics_residual': (
                np.mean(self._physics_residuals)
                if self._physics_residuals else 0.0
            ),
            'avg_divergence': (
                np.mean(self._divergences)
                if self._divergences else 0.0
            ),
            'avg_uncertainty': (
                np.mean(self._uncertainties)
                if self._uncertainties else 0.0
            ),
        }


class SwitchRateLimiter:
    """Limits mode switching frequency.
    
    Prevents rapid oscillation between complex and baseline
    modes which could indicate instability.
    """
    
    def __init__(
        self,
        max_switches_per_minute: float = 2.0,
        window_seconds: float = 60.0,
    ):
        """Initialize rate limiter.
        
        Args:
            max_switches_per_minute: Maximum allowed switch rate
            window_seconds: Time window for rate calculation
        """
        self.max_switches = max_switches_per_minute
        self.window_seconds = window_seconds
        self._switch_times: deque = deque()
    
    def can_switch(self) -> bool:
        """Check if switching is allowed.
        
        Returns:
            True if switch is allowed
        """
        current_time = time.time()
        
        # Remove old entries
        while (self._switch_times and 
               current_time - self._switch_times[0] > self.window_seconds):
            self._switch_times.popleft()
        
        # Check rate
        return len(self._switch_times) < self.max_switches
    
    def record_switch(self) -> None:
        """Record a switch event."""
        self._switch_times.append(time.time())
    
    def get_current_rate(self) -> float:
        """Get current switch rate per minute.
        
        Returns:
            Switches per minute
        """
        current_time = time.time()
        
        # Remove old entries
        while (self._switch_times and 
               current_time - self._switch_times[0] > self.window_seconds):
            self._switch_times.popleft()
        
        if not self._switch_times:
            return 0.0
        
        return len(self._switch_times) * (60.0 / self.window_seconds)


class SimplexDecisionModule:
    """Main Simplex decision module for SAFER.
    
    Implements the Simplex architecture to safely arbitrate
    between Mamba (complex) and baseline (safe) predictions.
    
    Key features:
    - Multi-signal safety monitoring
    - Rate-limited switching with hysteresis
    - Automatic recovery attempts
    - Comprehensive logging and audit trail
    - Thread-safe operation
    
    Usage:
        simplex = SimplexDecisionModule(config)
        
        result = simplex.decide(
            complex_rul=25.3,
            baseline_rul=28.1,
            rul_lower=20.0,
            rul_upper=35.0,
            physics_residual=0.05,
        )
        
        print(f"RUL: {result.rul}, State: {result.state.name}")
    """
    
    def __init__(
        self,
        config: Optional[SimplexConfig] = None,
    ):
        """Initialize Simplex decision module.
        
        Args:
            config: Simplex configuration
        """
        self.config = config or SimplexConfig()
        
        # Components
        self._safety_monitor = SafetyMonitor(self.config)
        self._rate_limiter = SwitchRateLimiter(
            self.config.max_switch_rate
        )
        
        # State
        self._state = SimplexState.BASELINE  # Start safe
        self._last_switch_time = 0.0
        self._cycles_since_switch = 0
        self._lock = threading.RLock()
        
        # History for audit
        self._decision_history: deque = deque(maxlen=1000)
        self._switch_history: deque = deque(maxlen=100)
        
        # Statistics
        self._total_decisions = 0
        self._complex_decisions = 0
        self._baseline_decisions = 0
        self._switch_count = 0
        
        logger.info(
            f"Simplex initialized: physics_thresh={self.config.physics_threshold}, "
            f"divergence_thresh={self.config.divergence_threshold}"
        )
    
    def decide(
        self,
        complex_rul: Optional[float],
        baseline_rul: float,
        rul_lower: float,
        rul_upper: float,
        physics_residual: float = 0.0,
        force_baseline: bool = False,
    ) -> DecisionResult:
        """Make Simplex decision.
        
        Args:
            complex_rul: Mamba RUL prediction (None if unavailable)
            baseline_rul: Baseline RUL prediction
            rul_lower: Lower confidence bound
            rul_upper: Upper confidence bound
            physics_residual: Physics monitor residual
            force_baseline: Force baseline mode (manual override)
            
        Returns:
            DecisionResult with final RUL and metadata
        """
        start_time = time.time()
        
        with self._lock:
            self._total_decisions += 1
            self._cycles_since_switch += 1
            
            switch_reason = None
            new_state = self._state
            
            # Force baseline if requested or complex unavailable
            if force_baseline:
                new_state = SimplexState.BASELINE
                switch_reason = SwitchReason.MANUAL
            elif complex_rul is None:
                new_state = SimplexState.BASELINE
                switch_reason = SwitchReason.TIMEOUT
            else:
                # Run safety checks
                uncertainty_width = rul_upper - rul_lower
                is_safe, reason = self._safety_monitor.check(
                    physics_residual=physics_residual,
                    complex_rul=complex_rul,
                    baseline_rul=baseline_rul,
                    uncertainty_width=uncertainty_width,
                )
                
                if self._state == SimplexState.COMPLEX:
                    # Currently using complex - check for switch to baseline
                    if not is_safe:
                        if self._can_switch():
                            new_state = SimplexState.BASELINE
                            switch_reason = reason
                else:
                    # Currently using baseline - check for recovery
                    if (is_safe and 
                        self._cycles_since_switch >= self.config.hysteresis_cycles and
                        self._safety_monitor.check_recovery()):
                        if self._can_switch():
                            new_state = SimplexState.COMPLEX
                            switch_reason = SwitchReason.RECOVERY
            
            # Handle state transition
            if new_state != self._state:
                self._handle_switch(new_state, switch_reason)
            
            # Select output
            if self._state == SimplexState.COMPLEX and complex_rul is not None:
                final_rul = complex_rul
                used_source = 'mamba'
                self._complex_decisions += 1
            else:
                # Use baseline with conservative margin
                final_rul = baseline_rul - self.config.conservative_margin
                used_source = 'baseline'
                self._baseline_decisions += 1
            
            # Ensure non-negative RUL
            final_rul = max(0.0, final_rul)
            
            # Compute latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Create result
            result = DecisionResult(
                rul=final_rul,
                rul_lower=max(0.0, rul_lower - self.config.conservative_margin),
                rul_upper=rul_upper,
                state=self._state,
                complex_rul=complex_rul,
                baseline_rul=baseline_rul,
                used_source=used_source,
                switch_reason=switch_reason,
                physics_residual=physics_residual,
                decision_latency_ms=latency_ms,
            )
            
            # Record for audit
            self._decision_history.append(result)
            
            return result
    
    def _can_switch(self) -> bool:
        """Check if mode switch is allowed.
        
        Returns:
            True if switch is permitted
        """
        # Check hysteresis
        if self._cycles_since_switch < self.config.hysteresis_cycles:
            return False
        
        # Check rate limit
        if not self._rate_limiter.can_switch():
            logger.warning("Switch rate limit exceeded")
            return False
        
        return True
    
    def _handle_switch(
        self,
        new_state: SimplexState,
        reason: Optional[SwitchReason],
    ) -> None:
        """Handle state transition.
        
        Args:
            new_state: New state
            reason: Reason for switch
        """
        old_state = self._state
        self._state = new_state
        self._last_switch_time = time.time()
        self._cycles_since_switch = 0
        self._switch_count += 1
        
        # Record switch
        self._rate_limiter.record_switch()
        self._switch_history.append({
            'timestamp': self._last_switch_time,
            'from': old_state.name,
            'to': new_state.name,
            'reason': reason.name if reason else 'unknown',
        })
        
        logger.info(
            f"Simplex switch: {old_state.name} -> {new_state.name} "
            f"(reason: {reason.name if reason else 'unknown'})"
        )
    
    @property
    def state(self) -> SimplexState:
        """Current Simplex state."""
        return self._state
    
    @property
    def is_using_complex(self) -> bool:
        """Check if currently using complex predictor."""
        return self._state == SimplexState.COMPLEX
    
    def force_baseline(self) -> None:
        """Force switch to baseline mode."""
        with self._lock:
            if self._state != SimplexState.BASELINE:
                self._handle_switch(SimplexState.BASELINE, SwitchReason.MANUAL)
    
    def force_complex(self) -> None:
        """Force switch to complex mode (use with caution)."""
        with self._lock:
            if self._state != SimplexState.COMPLEX:
                self._handle_switch(SimplexState.COMPLEX, SwitchReason.MANUAL)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get decision statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                'total_decisions': self._total_decisions,
                'complex_decisions': self._complex_decisions,
                'baseline_decisions': self._baseline_decisions,
                'complex_ratio': (
                    self._complex_decisions / self._total_decisions
                    if self._total_decisions > 0 else 0.0
                ),
                'switch_count': self._switch_count,
                'current_state': self._state.name,
                'switch_rate': self._rate_limiter.get_current_rate(),
                'safety_stats': self._safety_monitor.get_statistics(),
            }
    
    def get_recent_decisions(self, n: int = 10) -> List[DecisionResult]:
        """Get recent decisions.
        
        Args:
            n: Number of decisions to return
            
        Returns:
            List of recent DecisionResults
        """
        with self._lock:
            return list(self._decision_history)[-n:]
    
    def get_switch_history(self) -> List[Dict[str, Any]]:
        """Get switch history.
        
        Returns:
            List of switch events
        """
        with self._lock:
            return list(self._switch_history)
    
    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._total_decisions = 0
            self._complex_decisions = 0
            self._baseline_decisions = 0
            self._switch_count = 0
            self._decision_history.clear()
            self._switch_history.clear()


class SimplexEnsemble:
    """Ensemble of Simplex modules for multi-engine scenarios.
    
    Manages multiple Simplex decision modules for fleet-level
    prognostics where decisions may be correlated across engines.
    """
    
    def __init__(
        self,
        engine_ids: List[str],
        config: Optional[SimplexConfig] = None,
    ):
        """Initialize ensemble.
        
        Args:
            engine_ids: List of engine identifiers
            config: Shared configuration
        """
        self.engine_ids = engine_ids
        self.config = config or SimplexConfig()
        
        self._modules: Dict[str, SimplexDecisionModule] = {
            engine_id: SimplexDecisionModule(self.config)
            for engine_id in engine_ids
        }
    
    def decide(
        self,
        engine_id: str,
        complex_rul: Optional[float],
        baseline_rul: float,
        rul_lower: float,
        rul_upper: float,
        physics_residual: float = 0.0,
    ) -> DecisionResult:
        """Make decision for specific engine.
        
        Args:
            engine_id: Engine identifier
            complex_rul: Mamba prediction
            baseline_rul: Baseline prediction
            rul_lower: Lower bound
            rul_upper: Upper bound
            physics_residual: Physics residual
            
        Returns:
            DecisionResult
        """
        if engine_id not in self._modules:
            raise KeyError(f"Unknown engine: {engine_id}")
        
        return self._modules[engine_id].decide(
            complex_rul=complex_rul,
            baseline_rul=baseline_rul,
            rul_lower=rul_lower,
            rul_upper=rul_upper,
            physics_residual=physics_residual,
        )
    
    def get_fleet_status(self) -> Dict[str, Any]:
        """Get fleet-wide status.
        
        Returns:
            Dictionary with fleet statistics
        """
        status = {
            'total_engines': len(self.engine_ids),
            'engines_in_complex': 0,
            'engines_in_baseline': 0,
            'total_switches': 0,
        }
        
        for engine_id, module in self._modules.items():
            stats = module.get_statistics()
            status['total_switches'] += stats['switch_count']
            
            if module.is_using_complex:
                status['engines_in_complex'] += 1
            else:
                status['engines_in_baseline'] += 1
        
        return status
    
    def force_all_baseline(self) -> None:
        """Force all engines to baseline mode."""
        for module in self._modules.values():
            module.force_baseline()
    
    def get_module(self, engine_id: str) -> SimplexDecisionModule:
        """Get module for specific engine.
        
        Args:
            engine_id: Engine identifier
            
        Returns:
            SimplexDecisionModule
        """
        if engine_id not in self._modules:
            raise KeyError(f"Unknown engine: {engine_id}")
        return self._modules[engine_id]
