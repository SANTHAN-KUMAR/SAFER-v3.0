"""
Turbofan Engine Degradation Simulation for SAFER v3.0.

This module provides realistic engine degradation simulation
based on physics-informed models and C-MAPSS data characteristics.

The simulator generates synthetic sensor readings that follow
realistic degradation patterns, enabling:
- Testing without real engine data
- Edge case and stress testing
- Algorithm development and debugging
- Training data augmentation

Physics Background:
- Turbofan engines degrade due to wear, fouling, erosion
- Degradation affects efficiency, temperatures, pressures
- Multiple operating conditions create complex patterns
- Faults cause abrupt changes in degradation trajectory

References:
    - Saxena et al., "Damage propagation modeling for aircraft
      engine run-to-failure simulation" (2008)
    - Frederick et al., "User's Guide for the Commercial Modular
      Aero-Propulsion System Simulation (C-MAPSS)" (2007)
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging


logger = logging.getLogger(__name__)


# C-MAPSS sensor definitions
SENSOR_NAMES = [
    'T2',    # Total temperature at fan inlet (°R)
    'T24',   # Total temperature at LPC outlet (°R)
    'T30',   # Total temperature at HPC outlet (°R)
    'T50',   # Total temperature at LPT outlet (°R)
    'P2',    # Pressure at fan inlet (psia)
    'P15',   # Total pressure in bypass-duct (psia)
    'P30',   # Total pressure at HPC outlet (psia)
    'Nf',    # Physical fan speed (rpm)
    'Nc',    # Physical core speed (rpm)
    'epr',   # Engine pressure ratio (P50/P2)
    'Ps30',  # Static pressure at HPC outlet (psia)
    'phi',   # Ratio of fuel flow to Ps30 (pps/psi)
    'NRf',   # Corrected fan speed (rpm)
    'NRc',   # Corrected core speed (rpm)
    'BPR',   # Bypass Ratio
    'farB',  # Burner fuel-air ratio
    'htBleed',  # Bleed Enthalpy
    'Nf_dmd',   # Demanded fan speed (rpm)
    'PCNfR_dmd', # Demanded corrected fan speed (rpm)
    'W31',   # HPT coolant bleed (lbm/s)
    'W32',   # LPT coolant bleed (lbm/s)
]

# Operating condition indices
OP_COND_NAMES = ['altitude', 'mach', 'TRA']

# Prognostic sensors (14 sensors with degradation sensitivity)
PROGNOSTIC_INDICES = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]


@dataclass
class SensorSpec:
    """Specification for a single sensor.
    
    Attributes:
        name: Sensor identifier
        nominal: Nominal value at healthy state
        degradation_rate: Rate of change per cycle (% of nominal)
        noise_std: Standard deviation of noise (% of nominal)
        min_val: Minimum physical limit
        max_val: Maximum physical limit
        op_sensitivity: Sensitivity to operating conditions
    """
    name: str
    nominal: float
    degradation_rate: float = 0.0
    noise_std: float = 0.01
    min_val: float = -np.inf
    max_val: float = np.inf
    op_sensitivity: Dict[str, float] = field(default_factory=dict)


class DegradationModel(ABC):
    """Abstract base class for degradation models."""
    
    @abstractmethod
    def __call__(
        self,
        cycle: int,
        total_cycles: int,
        health_index: float,
    ) -> float:
        """Compute degradation factor.
        
        Args:
            cycle: Current cycle number
            total_cycles: Total cycles until failure
            health_index: Current health state [0, 1]
            
        Returns:
            Degradation factor (0 = healthy, 1 = failed)
        """
        pass
    
    @abstractmethod
    def health_evolution(
        self,
        n_cycles: int,
    ) -> np.ndarray:
        """Generate health evolution over cycles.
        
        Args:
            n_cycles: Number of cycles
            
        Returns:
            Array of health indices
        """
        pass


class LinearDegradation(DegradationModel):
    """Linear degradation model.
    
    Health decreases linearly from 1 to 0 over the lifetime.
    Simple but provides baseline behavior.
    """
    
    def __init__(
        self,
        rate: float = 1.0,
        initial_health: float = 1.0,
    ):
        """Initialize linear degradation.
        
        Args:
            rate: Degradation rate multiplier
            initial_health: Starting health index
        """
        self.rate = rate
        self.initial_health = initial_health
    
    def __call__(
        self,
        cycle: int,
        total_cycles: int,
        health_index: float,
    ) -> float:
        """Compute linear degradation."""
        progress = cycle / total_cycles
        return min(1.0, progress * self.rate)
    
    def health_evolution(self, n_cycles: int) -> np.ndarray:
        """Generate linear health trajectory."""
        return np.linspace(self.initial_health, 0.0, n_cycles)


class ExponentialDegradation(DegradationModel):
    """Exponential degradation model.
    
    Health decreases slowly initially, then rapidly near end of life.
    More realistic for wear-out failure modes.
    
    h(t) = exp(-λ * (t/T)^β)
    
    where λ is rate, β controls shape (β > 1 for wear-out).
    """
    
    def __init__(
        self,
        rate: float = 3.0,
        shape: float = 2.0,
        initial_health: float = 1.0,
    ):
        """Initialize exponential degradation.
        
        Args:
            rate: Rate parameter λ
            shape: Shape parameter β
            initial_health: Starting health
        """
        self.rate = rate
        self.shape = shape
        self.initial_health = initial_health
    
    def __call__(
        self,
        cycle: int,
        total_cycles: int,
        health_index: float,
    ) -> float:
        """Compute exponential degradation."""
        progress = cycle / total_cycles
        degradation = 1.0 - np.exp(-self.rate * (progress ** self.shape))
        return min(1.0, degradation)
    
    def health_evolution(self, n_cycles: int) -> np.ndarray:
        """Generate exponential health trajectory."""
        t = np.linspace(0, 1, n_cycles)
        health = self.initial_health * np.exp(-self.rate * (t ** self.shape))
        return health


class PiecewiseDegradation(DegradationModel):
    """Piecewise degradation with multiple phases.
    
    Models realistic degradation with:
    1. Break-in phase (mild improvement or stable)
    2. Normal wear phase (slow degradation)
    3. Accelerated wear phase (rapid degradation)
    
    This matches observed turbofan behavior.
    """
    
    def __init__(
        self,
        phase_boundaries: List[float] = None,
        phase_rates: List[float] = None,
        initial_health: float = 1.0,
    ):
        """Initialize piecewise degradation.
        
        Args:
            phase_boundaries: Normalized time for phase transitions
            phase_rates: Degradation rates for each phase
            initial_health: Starting health
        """
        self.phase_boundaries = phase_boundaries or [0.1, 0.7, 1.0]
        self.phase_rates = phase_rates or [0.1, 0.5, 2.0]
        self.initial_health = initial_health
        
        if len(self.phase_rates) != len(self.phase_boundaries):
            raise ValueError("Phase boundaries and rates must match")
    
    def __call__(
        self,
        cycle: int,
        total_cycles: int,
        health_index: float,
    ) -> float:
        """Compute piecewise degradation."""
        progress = cycle / total_cycles
        
        cumulative_degradation = 0.0
        prev_boundary = 0.0
        
        for boundary, rate in zip(self.phase_boundaries, self.phase_rates):
            if progress <= boundary:
                # In this phase
                phase_progress = progress - prev_boundary
                cumulative_degradation += phase_progress * rate
                break
            else:
                # Past this phase
                phase_duration = boundary - prev_boundary
                cumulative_degradation += phase_duration * rate
                prev_boundary = boundary
        
        return min(1.0, cumulative_degradation)
    
    def health_evolution(self, n_cycles: int) -> np.ndarray:
        """Generate piecewise health trajectory."""
        health = np.zeros(n_cycles)
        
        for i in range(n_cycles):
            degradation = self(i, n_cycles, 1.0)
            health[i] = self.initial_health * (1.0 - degradation)
        
        return health


@dataclass
class SensorNoise:
    """Sensor noise model.
    
    Attributes:
        gaussian_std: Gaussian noise standard deviation
        bias: Constant sensor bias
        drift_rate: Linear drift rate per cycle
        spike_prob: Probability of noise spike
        spike_magnitude: Magnitude of noise spikes
    """
    gaussian_std: float = 0.01
    bias: float = 0.0
    drift_rate: float = 0.0
    spike_prob: float = 0.0
    spike_magnitude: float = 0.1
    
    def apply(
        self,
        values: np.ndarray,
        cycle: int,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """Apply noise to sensor values.
        
        Args:
            values: Clean sensor values
            cycle: Current cycle number
            rng: Random generator
            
        Returns:
            Noisy sensor values
        """
        rng = rng or np.random.default_rng()
        
        # Gaussian noise
        noise = rng.normal(0, self.gaussian_std, values.shape)
        
        # Bias
        noisy = values + self.bias
        
        # Drift
        noisy = noisy + self.drift_rate * cycle
        
        # Spikes
        if self.spike_prob > 0:
            spikes = rng.random(values.shape) < self.spike_prob
            spike_values = rng.normal(0, self.spike_magnitude, values.shape)
            noisy = noisy + spikes * spike_values
        
        # Add Gaussian noise
        noisy = noisy + noise * np.abs(values)
        
        return noisy


class FaultInjector:
    """Injects faults into sensor data.
    
    Simulates various fault modes:
    - Sensor bias shifts
    - Sensor drift
    - Stuck sensors
    - Intermittent faults
    - Complete failures
    """
    
    def __init__(
        self,
        fault_type: str = 'none',
        fault_start: int = 0,
        fault_magnitude: float = 0.1,
        affected_sensors: List[int] = None,
    ):
        """Initialize fault injector.
        
        Args:
            fault_type: Type of fault ('none', 'bias', 'drift', 'stuck', 'intermittent')
            fault_start: Cycle when fault begins
            fault_magnitude: Magnitude of fault effect
            affected_sensors: Indices of affected sensors
        """
        self.fault_type = fault_type
        self.fault_start = fault_start
        self.fault_magnitude = fault_magnitude
        self.affected_sensors = affected_sensors or []
        
        self._stuck_values: Optional[np.ndarray] = None
    
    def apply(
        self,
        values: np.ndarray,
        cycle: int,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """Apply fault to sensor values.
        
        Args:
            values: Sensor values
            cycle: Current cycle
            rng: Random generator
            
        Returns:
            Faulted sensor values
        """
        if cycle < self.fault_start:
            return values
        
        rng = rng or np.random.default_rng()
        faulty = values.copy()
        
        if self.fault_type == 'none':
            return faulty
        
        cycles_since_fault = cycle - self.fault_start
        
        for sensor_idx in self.affected_sensors:
            if sensor_idx >= len(values):
                continue
            
            if self.fault_type == 'bias':
                faulty[sensor_idx] += self.fault_magnitude * values[sensor_idx]
            
            elif self.fault_type == 'drift':
                drift = self.fault_magnitude * cycles_since_fault * 0.001
                faulty[sensor_idx] += drift * values[sensor_idx]
            
            elif self.fault_type == 'stuck':
                if self._stuck_values is None:
                    self._stuck_values = values.copy()
                faulty[sensor_idx] = self._stuck_values[sensor_idx]
            
            elif self.fault_type == 'intermittent':
                if rng.random() < 0.3:  # 30% of cycles affected
                    faulty[sensor_idx] += self.fault_magnitude * values[sensor_idx]
        
        return faulty


class EngineSimulator:
    """Turbofan engine degradation simulator.
    
    Generates realistic sensor data trajectories based on
    configurable degradation models and noise characteristics.
    
    Usage:
        simulator = EngineSimulator(
            total_cycles=200,
            degradation_model=ExponentialDegradation(),
        )
        
        trajectory = simulator.generate_trajectory()
        
        # Or step-by-step
        simulator.reset()
        for cycle in range(200):
            sensors = simulator.step()
    """
    
    # Default sensor specifications (based on C-MAPSS characteristics)
    DEFAULT_SENSOR_SPECS = {
        'T2': SensorSpec('T2', 642.0, 0.0, 0.005),      # Fan inlet temp
        'T24': SensorSpec('T24', 1590.0, 0.001, 0.01),  # LPC outlet temp
        'T30': SensorSpec('T30', 1400.0, 0.002, 0.01),  # HPC outlet temp
        'T50': SensorSpec('T50', 1100.0, 0.001, 0.01),  # LPT outlet temp
        'P2': SensorSpec('P2', 14.6, 0.0, 0.005),       # Fan inlet pressure
        'P15': SensorSpec('P15', 21.6, 0.0, 0.005),     # Bypass duct pressure
        'P30': SensorSpec('P30', 550.0, 0.002, 0.01),   # HPC outlet pressure
        'Nf': SensorSpec('Nf', 2388.0, 0.001, 0.01),    # Fan speed
        'Nc': SensorSpec('Nc', 9000.0, 0.002, 0.01),    # Core speed
        'epr': SensorSpec('epr', 1.3, 0.001, 0.01),     # Engine pressure ratio
        'Ps30': SensorSpec('Ps30', 47.5, 0.001, 0.01),  # HPC static pressure
        'phi': SensorSpec('phi', 520.0, 0.002, 0.02),   # Fuel flow ratio
        'NRf': SensorSpec('NRf', 2388.0, 0.001, 0.01),  # Corrected fan speed
        'NRc': SensorSpec('NRc', 8200.0, 0.002, 0.01),  # Corrected core speed
        'BPR': SensorSpec('BPR', 8.4, 0.001, 0.01),     # Bypass ratio
        'farB': SensorSpec('farB', 0.03, 0.001, 0.02),  # Fuel-air ratio
        'htBleed': SensorSpec('htBleed', 390.0, 0.001, 0.01),  # Bleed enthalpy
        'Nf_dmd': SensorSpec('Nf_dmd', 2388.0, 0.0, 0.005),    # Demanded fan speed
        'PCNfR_dmd': SensorSpec('PCNfR_dmd', 100.0, 0.0, 0.005),  # Demanded corrected
        'W31': SensorSpec('W31', 38.0, 0.001, 0.01),    # HPT bleed
        'W32': SensorSpec('W32', 23.0, 0.001, 0.01),    # LPT bleed
    }
    
    # Operating condition ranges (based on C-MAPSS)
    OP_COND_RANGES = {
        'altitude': (0, 42000),      # feet
        'mach': (0.0, 0.84),         # Mach number
        'TRA': (20, 100),            # Throttle resolver angle
    }
    
    def __init__(
        self,
        total_cycles: int = 200,
        degradation_model: Optional[DegradationModel] = None,
        sensor_noise: Optional[SensorNoise] = None,
        fault_injector: Optional[FaultInjector] = None,
        operating_conditions: str = 'single',
        seed: Optional[int] = None,
    ):
        """Initialize engine simulator.
        
        Args:
            total_cycles: Total cycles until failure
            degradation_model: Degradation model to use
            sensor_noise: Noise model for sensors
            fault_injector: Optional fault injection
            operating_conditions: 'single' (FD001/FD003) or 'multiple' (FD002/FD004)
            seed: Random seed for reproducibility
        """
        self.total_cycles = total_cycles
        self.degradation_model = degradation_model or ExponentialDegradation()
        self.sensor_noise = sensor_noise or SensorNoise()
        self.fault_injector = fault_injector or FaultInjector()
        self.operating_conditions = operating_conditions
        
        self.rng = np.random.default_rng(seed)
        
        # State
        self._cycle = 0
        self._health = 1.0
        self._current_op_cond = None
        
        # Generate base operating condition if single
        if operating_conditions == 'single':
            self._base_op_cond = self._generate_op_condition()
        else:
            self._op_cond_sequence = self._generate_op_sequence()
    
    def _generate_op_condition(self) -> np.ndarray:
        """Generate a single operating condition."""
        op_cond = np.zeros(3)
        for i, (name, (low, high)) in enumerate(self.OP_COND_RANGES.items()):
            op_cond[i] = self.rng.uniform(low, high)
        return op_cond
    
    def _generate_op_sequence(self) -> np.ndarray:
        """Generate sequence of operating conditions."""
        # Generate a few distinct operating conditions
        n_conditions = self.rng.integers(3, 7)
        conditions = np.array([
            self._generate_op_condition()
            for _ in range(n_conditions)
        ])
        
        # Create sequence by sampling
        sequence = np.zeros((self.total_cycles, 3))
        current_idx = 0
        change_prob = 0.05
        
        for i in range(self.total_cycles):
            if self.rng.random() < change_prob:
                current_idx = self.rng.integers(0, n_conditions)
            sequence[i] = conditions[current_idx]
        
        return sequence
    
    def reset(self) -> None:
        """Reset simulator to initial state."""
        self._cycle = 0
        self._health = 1.0
        
        if self.operating_conditions == 'single':
            self._base_op_cond = self._generate_op_condition()
        else:
            self._op_cond_sequence = self._generate_op_sequence()
    
    def get_operating_condition(self, cycle: int) -> np.ndarray:
        """Get operating condition for given cycle.
        
        Args:
            cycle: Cycle number
            
        Returns:
            Operating condition array [altitude, mach, TRA]
        """
        if self.operating_conditions == 'single':
            # Add small variations
            variation = self.rng.normal(0, 0.01, 3)
            return self._base_op_cond * (1 + variation)
        else:
            return self._op_cond_sequence[min(cycle, self.total_cycles - 1)]
    
    def step(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Advance simulation by one cycle.
        
        Returns:
            Tuple of (operating_conditions, sensor_readings, RUL)
        """
        if self._cycle >= self.total_cycles:
            raise StopIteration("Simulation complete")
        
        # Get operating condition
        op_cond = self.get_operating_condition(self._cycle)
        
        # Compute degradation
        degradation = self.degradation_model(
            self._cycle, self.total_cycles, self._health
        )
        self._health = 1.0 - degradation
        
        # Generate sensor readings
        sensors = self._generate_sensors(op_cond, degradation)
        
        # Apply noise
        sensors = self.sensor_noise.apply(sensors, self._cycle, self.rng)
        
        # Apply faults
        sensors = self.fault_injector.apply(sensors, self._cycle, self.rng)
        
        # Compute RUL
        rul = self.total_cycles - self._cycle - 1
        
        self._cycle += 1
        
        return op_cond, sensors, rul
    
    def _generate_sensors(
        self,
        op_cond: np.ndarray,
        degradation: float,
    ) -> np.ndarray:
        """Generate sensor readings based on state.
        
        Args:
            op_cond: Operating conditions
            degradation: Current degradation level [0, 1]
            
        Returns:
            Array of sensor readings
        """
        sensors = np.zeros(21)
        
        for i, name in enumerate(SENSOR_NAMES):
            spec = self.DEFAULT_SENSOR_SPECS.get(
                name,
                SensorSpec(name, 1.0)
            )
            
            # Base value
            value = spec.nominal
            
            # Apply operating condition effects
            # Simplified: altitude affects temperature and pressure
            alt_factor = 1.0 - 0.001 * op_cond[0] / 1000  # Altitude effect
            mach_factor = 1.0 + 0.1 * op_cond[1]          # Mach effect
            tra_factor = 0.8 + 0.004 * op_cond[2]         # Throttle effect
            
            if 'T' in name:  # Temperature sensors
                value *= tra_factor * mach_factor
            elif 'P' in name:  # Pressure sensors
                value *= alt_factor * tra_factor
            elif 'N' in name:  # Speed sensors
                value *= tra_factor
            
            # Apply degradation (increases temps, decreases efficiency)
            if i in PROGNOSTIC_INDICES:
                if 'T' in name:
                    # Temperature increases with degradation
                    value *= (1.0 + spec.degradation_rate * degradation * 50)
                elif 'BPR' in name or 'epr' in name:
                    # Efficiency decreases
                    value *= (1.0 - spec.degradation_rate * degradation * 30)
                else:
                    # General degradation effect
                    value *= (1.0 + spec.degradation_rate * degradation * 20 * 
                             (1 if self.rng.random() > 0.5 else -1))
            
            sensors[i] = value
        
        return sensors
    
    def generate_trajectory(self) -> Dict[str, np.ndarray]:
        """Generate complete trajectory.
        
        Returns:
            Dictionary with 'op_cond', 'sensors', 'rul' arrays
        """
        self.reset()
        
        op_conds = []
        sensors = []
        ruls = []
        
        for _ in range(self.total_cycles):
            op_cond, sensor_reading, rul = self.step()
            op_conds.append(op_cond)
            sensors.append(sensor_reading)
            ruls.append(rul)
        
        return {
            'op_cond': np.array(op_conds),
            'sensors': np.array(sensors),
            'rul': np.array(ruls),
            'cycle': np.arange(self.total_cycles),
        }
    
    def get_prognostic_sensors(
        self,
        sensor_data: np.ndarray,
    ) -> np.ndarray:
        """Extract prognostic sensors from full sensor array.
        
        Args:
            sensor_data: Full sensor array (N, 21) or (21,)
            
        Returns:
            Prognostic sensors (N, 14) or (14,)
        """
        return sensor_data[..., PROGNOSTIC_INDICES]


def create_fleet_simulators(
    n_engines: int,
    lifetime_range: Tuple[int, int] = (100, 300),
    seed: Optional[int] = None,
) -> List[EngineSimulator]:
    """Create fleet of engine simulators.
    
    Args:
        n_engines: Number of engines
        lifetime_range: Range of total cycles (min, max)
        seed: Random seed
        
    Returns:
        List of EngineSimulator instances
    """
    rng = np.random.default_rng(seed)
    
    simulators = []
    for i in range(n_engines):
        total_cycles = rng.integers(lifetime_range[0], lifetime_range[1])
        
        # Vary degradation model
        model_type = rng.choice(['linear', 'exponential', 'piecewise'])
        if model_type == 'linear':
            model = LinearDegradation()
        elif model_type == 'exponential':
            model = ExponentialDegradation(
                rate=rng.uniform(2.0, 4.0),
                shape=rng.uniform(1.5, 2.5),
            )
        else:
            model = PiecewiseDegradation()
        
        # Vary noise
        noise = SensorNoise(
            gaussian_std=rng.uniform(0.005, 0.02),
            bias=rng.uniform(-0.01, 0.01),
        )
        
        sim = EngineSimulator(
            total_cycles=total_cycles,
            degradation_model=model,
            sensor_noise=noise,
            seed=rng.integers(0, 2**31),
        )
        simulators.append(sim)
    
    return simulators
