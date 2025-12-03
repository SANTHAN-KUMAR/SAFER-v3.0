"""
Data Generation for SAFER v3.0.

This module provides data generation utilities for creating
synthetic C-MAPSS-like datasets and streaming data for testing.

Features:
- CMAPSSGenerator: Creates datasets matching C-MAPSS format
- StreamingDataGenerator: Real-time data streaming simulation
- SyntheticFleet: Fleet-level data generation
- Utilities for trajectory generation and augmentation

The generators are designed to produce data that:
1. Matches C-MAPSS statistical characteristics
2. Includes realistic degradation patterns
3. Supports various operating conditions
4. Can be used for training, testing, and demonstrations
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Generator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import time
import threading
import queue
import logging

from .engine_sim import (
    EngineSimulator,
    DegradationModel,
    ExponentialDegradation,
    PiecewiseDegradation,
    SensorNoise,
    SENSOR_NAMES,
    OP_COND_NAMES,
    PROGNOSTIC_INDICES,
    create_fleet_simulators,
)


logger = logging.getLogger(__name__)


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation.
    
    Attributes:
        min_cycles: Minimum trajectory length
        max_cycles: Maximum trajectory length
        degradation_type: Type of degradation model
        noise_level: Noise standard deviation
        operating_mode: 'single' or 'multiple' operating conditions
    """
    min_cycles: int = 100
    max_cycles: int = 300
    degradation_type: str = 'exponential'
    noise_level: float = 0.01
    operating_mode: str = 'single'


def generate_trajectory(
    config: Optional[TrajectoryConfig] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Generate a single engine trajectory.
    
    Args:
        config: Trajectory configuration
        seed: Random seed
        
    Returns:
        Dictionary with trajectory data
    """
    config = config or TrajectoryConfig()
    rng = np.random.default_rng(seed)
    
    # Random lifetime within range
    total_cycles = rng.integers(config.min_cycles, config.max_cycles + 1)
    
    # Create degradation model
    if config.degradation_type == 'linear':
        from .engine_sim import LinearDegradation
        model = LinearDegradation()
    elif config.degradation_type == 'exponential':
        model = ExponentialDegradation(
            rate=rng.uniform(2.5, 3.5),
            shape=rng.uniform(1.8, 2.2),
        )
    else:
        model = PiecewiseDegradation()
    
    # Create simulator
    simulator = EngineSimulator(
        total_cycles=total_cycles,
        degradation_model=model,
        sensor_noise=SensorNoise(gaussian_std=config.noise_level),
        operating_conditions=config.operating_mode,
        seed=rng.integers(0, 2**31),
    )
    
    return simulator.generate_trajectory()


def generate_fleet_data(
    n_engines: int,
    config: Optional[TrajectoryConfig] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, np.ndarray]]:
    """Generate data for fleet of engines.
    
    Args:
        n_engines: Number of engines
        config: Trajectory configuration
        seed: Random seed
        
    Returns:
        List of trajectory dictionaries
    """
    config = config or TrajectoryConfig()
    rng = np.random.default_rng(seed)
    
    trajectories = []
    for i in range(n_engines):
        traj = generate_trajectory(
            config=config,
            seed=rng.integers(0, 2**31),
        )
        traj['engine_id'] = i + 1
        trajectories.append(traj)
    
    logger.info(f"Generated {n_engines} engine trajectories")
    return trajectories


class CMAPSSGenerator:
    """Generator for C-MAPSS format datasets.
    
    Creates synthetic datasets that match the format and
    characteristics of NASA C-MAPSS benchmark data.
    
    Output Format:
    - Training data: unit, cycle, op1, op2, op3, sensor1-21
    - Test data: Same format, truncated before failure
    - RUL file: One RUL value per test unit
    
    Usage:
        generator = CMAPSSGenerator(n_train=100, n_test=100)
        train_df, test_df, rul_array = generator.generate()
        generator.save('output_dir')
    """
    
    def __init__(
        self,
        n_train: int = 100,
        n_test: int = 100,
        config: Optional[TrajectoryConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize C-MAPSS generator.
        
        Args:
            n_train: Number of training units
            n_test: Number of test units
            config: Trajectory configuration
            seed: Random seed
        """
        self.n_train = n_train
        self.n_test = n_test
        self.config = config or TrajectoryConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self._train_data: Optional[pd.DataFrame] = None
        self._test_data: Optional[pd.DataFrame] = None
        self._rul_values: Optional[np.ndarray] = None
    
    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Generate complete dataset.
        
        Returns:
            Tuple of (train_df, test_df, rul_array)
        """
        logger.info(f"Generating C-MAPSS dataset: {self.n_train} train, {self.n_test} test")
        
        # Generate training data
        train_trajectories = generate_fleet_data(
            self.n_train,
            self.config,
            self.rng.integers(0, 2**31),
        )
        self._train_data = self._trajectories_to_dataframe(train_trajectories)
        
        # Generate test data (truncated)
        test_trajectories = generate_fleet_data(
            self.n_test,
            self.config,
            self.rng.integers(0, 2**31),
        )
        
        # Truncate test trajectories and compute RUL
        truncated, rul_values = self._truncate_trajectories(test_trajectories)
        self._test_data = self._trajectories_to_dataframe(truncated)
        self._rul_values = np.array(rul_values)
        
        return self._train_data, self._test_data, self._rul_values
    
    def _trajectories_to_dataframe(
        self,
        trajectories: List[Dict[str, np.ndarray]],
    ) -> pd.DataFrame:
        """Convert trajectories to C-MAPSS format DataFrame.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            DataFrame in C-MAPSS format
        """
        rows = []
        
        for traj in trajectories:
            engine_id = traj['engine_id']
            n_cycles = len(traj['cycle'])
            
            for i in range(n_cycles):
                row = [engine_id, i + 1]  # Unit, cycle (1-indexed)
                
                # Operating conditions
                row.extend(traj['op_cond'][i].tolist())
                
                # Sensor readings
                row.extend(traj['sensors'][i].tolist())
                
                rows.append(row)
        
        # Column names
        columns = ['unit', 'cycle'] + OP_COND_NAMES + SENSOR_NAMES
        
        return pd.DataFrame(rows, columns=columns)
    
    def _truncate_trajectories(
        self,
        trajectories: List[Dict[str, np.ndarray]],
    ) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
        """Truncate trajectories for test data.
        
        Args:
            trajectories: Full trajectories
            
        Returns:
            Tuple of (truncated trajectories, RUL values)
        """
        truncated = []
        rul_values = []
        
        for traj in trajectories:
            total_len = len(traj['cycle'])
            
            # Random truncation point (keep 50-90% of trajectory)
            keep_fraction = self.rng.uniform(0.5, 0.9)
            truncate_at = int(total_len * keep_fraction)
            truncate_at = max(20, truncate_at)  # Keep at least 20 cycles
            
            # Compute RUL at truncation point
            rul = total_len - truncate_at
            rul_values.append(rul)
            
            # Truncate
            truncated_traj = {
                'engine_id': traj['engine_id'],
                'cycle': traj['cycle'][:truncate_at],
                'op_cond': traj['op_cond'][:truncate_at],
                'sensors': traj['sensors'][:truncate_at],
                'rul': traj['rul'][:truncate_at],
            }
            truncated.append(truncated_traj)
        
        return truncated, rul_values
    
    def save(self, output_dir: str, prefix: str = 'FD_SYN') -> None:
        """Save dataset to files.
        
        Args:
            output_dir: Output directory path
            prefix: File prefix (e.g., 'FD001')
        """
        if self._train_data is None:
            raise ValueError("Must call generate() before save()")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training data
        train_file = output_path / f'train_{prefix}.txt'
        self._train_data.to_csv(
            train_file, sep=' ', header=False, index=False
        )
        logger.info(f"Saved training data: {train_file}")
        
        # Save test data
        test_file = output_path / f'test_{prefix}.txt'
        self._test_data.to_csv(
            test_file, sep=' ', header=False, index=False
        )
        logger.info(f"Saved test data: {test_file}")
        
        # Save RUL values
        rul_file = output_path / f'RUL_{prefix}.txt'
        np.savetxt(rul_file, self._rul_values, fmt='%d')
        logger.info(f"Saved RUL values: {rul_file}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        if self._train_data is None:
            return {}
        
        train_cycles = self._train_data.groupby('unit')['cycle'].max()
        test_cycles = self._test_data.groupby('unit')['cycle'].max()
        
        return {
            'n_train': self.n_train,
            'n_test': self.n_test,
            'train_cycles_mean': float(train_cycles.mean()),
            'train_cycles_std': float(train_cycles.std()),
            'test_cycles_mean': float(test_cycles.mean()),
            'test_cycles_std': float(test_cycles.std()),
            'rul_mean': float(self._rul_values.mean()),
            'rul_std': float(self._rul_values.std()),
            'rul_min': int(self._rul_values.min()),
            'rul_max': int(self._rul_values.max()),
        }


class StreamingDataGenerator:
    """Real-time streaming data generator.
    
    Simulates real-time sensor data streaming for testing
    the SAFER fabric and decision modules.
    
    Features:
    - Configurable data rate
    - Multiple engine support
    - Thread-safe queue-based output
    - Pausable/resumable streaming
    
    Usage:
        generator = StreamingDataGenerator(n_engines=5)
        generator.start()
        
        while True:
            packet = generator.get_next()
            if packet:
                process(packet)
    """
    
    def __init__(
        self,
        n_engines: int = 5,
        sample_rate_hz: float = 1.0,
        config: Optional[TrajectoryConfig] = None,
        buffer_size: int = 100,
        seed: Optional[int] = None,
    ):
        """Initialize streaming generator.
        
        Args:
            n_engines: Number of engines to simulate
            sample_rate_hz: Samples per second
            config: Trajectory configuration
            buffer_size: Output buffer size
            seed: Random seed
        """
        self.n_engines = n_engines
        self.sample_rate_hz = sample_rate_hz
        self.config = config or TrajectoryConfig()
        self.buffer_size = buffer_size
        
        self.rng = np.random.default_rng(seed)
        
        # Create simulators
        self._simulators = create_fleet_simulators(
            n_engines,
            (self.config.min_cycles, self.config.max_cycles),
            seed,
        )
        
        # Threading
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Statistics
        self._packets_generated = 0
        self._start_time = 0.0
    
    def start(self) -> None:
        """Start streaming."""
        if self._running:
            return
        
        self._running = True
        self._paused = False
        self._start_time = time.time()
        
        self._thread = threading.Thread(target=self._generate_loop)
        self._thread.daemon = True
        self._thread.start()
        
        logger.info(f"Streaming started: {self.n_engines} engines @ {self.sample_rate_hz} Hz")
    
    def stop(self) -> None:
        """Stop streaming."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        
        logger.info(f"Streaming stopped: {self._packets_generated} packets generated")
    
    def pause(self) -> None:
        """Pause streaming."""
        self._paused = True
    
    def resume(self) -> None:
        """Resume streaming."""
        self._paused = False
    
    def _generate_loop(self) -> None:
        """Main generation loop."""
        interval = 1.0 / self.sample_rate_hz
        
        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue
            
            start = time.time()
            
            # Generate packet for each engine
            for engine_idx, simulator in enumerate(self._simulators):
                try:
                    op_cond, sensors, rul = simulator.step()
                    
                    packet = {
                        'engine_id': engine_idx + 1,
                        'timestamp': time.time(),
                        'cycle': simulator._cycle,
                        'op_cond': op_cond,
                        'sensors': sensors,
                        'rul_true': rul,
                    }
                    
                    try:
                        self._queue.put_nowait(packet)
                        self._packets_generated += 1
                    except queue.Full:
                        pass  # Drop if buffer full
                
                except StopIteration:
                    # Engine reached end of life, reset
                    simulator.reset()
            
            # Maintain sample rate
            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)
    
    def get_next(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get next data packet.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Data packet or None if timeout
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_all_available(self) -> List[Dict[str, Any]]:
        """Get all available packets.
        
        Returns:
            List of data packets
        """
        packets = []
        while True:
            try:
                packets.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return packets
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics.
        
        Returns:
            Dictionary with statistics
        """
        elapsed = time.time() - self._start_time if self._start_time else 0
        
        return {
            'running': self._running,
            'paused': self._paused,
            'packets_generated': self._packets_generated,
            'elapsed_seconds': elapsed,
            'actual_rate_hz': (
                self._packets_generated / elapsed / self.n_engines
                if elapsed > 0 else 0
            ),
            'buffer_size': self._queue.qsize(),
        }
    
    def __enter__(self) -> 'StreamingDataGenerator':
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()


class SyntheticFleet:
    """Synthetic fleet for comprehensive testing.
    
    Manages a fleet of engines with various degradation
    states for testing fleet-level prognostics.
    
    Features:
    - Engines at different degradation stages
    - Configurable fleet composition
    - Batch prediction support
    - Fleet statistics tracking
    """
    
    def __init__(
        self,
        n_healthy: int = 10,
        n_degrading: int = 5,
        n_critical: int = 2,
        seed: Optional[int] = None,
    ):
        """Initialize synthetic fleet.
        
        Args:
            n_healthy: Number of healthy engines (>50% RUL)
            n_degrading: Number of degrading engines (20-50% RUL)
            n_critical: Number of critical engines (<20% RUL)
            seed: Random seed
        """
        self.rng = np.random.default_rng(seed)
        
        self.engines: Dict[str, Dict[str, Any]] = {}
        
        # Create engines at different degradation stages
        engine_id = 1
        
        # Healthy engines
        for _ in range(n_healthy):
            progress = self.rng.uniform(0.0, 0.5)  # 0-50% through life
            self._create_engine(f'ENG-{engine_id:03d}', progress)
            engine_id += 1
        
        # Degrading engines
        for _ in range(n_degrading):
            progress = self.rng.uniform(0.5, 0.8)  # 50-80% through life
            self._create_engine(f'ENG-{engine_id:03d}', progress)
            engine_id += 1
        
        # Critical engines
        for _ in range(n_critical):
            progress = self.rng.uniform(0.8, 0.95)  # 80-95% through life
            self._create_engine(f'ENG-{engine_id:03d}', progress)
            engine_id += 1
        
        logger.info(
            f"Synthetic fleet created: {n_healthy} healthy, "
            f"{n_degrading} degrading, {n_critical} critical"
        )
    
    def _create_engine(self, engine_id: str, progress: float) -> None:
        """Create engine at specific degradation progress.
        
        Args:
            engine_id: Engine identifier
            progress: Degradation progress (0-1)
        """
        total_cycles = self.rng.integers(150, 300)
        current_cycle = int(total_cycles * progress)
        remaining_cycles = total_cycles - current_cycle
        
        simulator = EngineSimulator(
            total_cycles=total_cycles,
            degradation_model=ExponentialDegradation(),
            seed=self.rng.integers(0, 2**31),
        )
        
        # Advance to current position
        for _ in range(current_cycle):
            try:
                simulator.step()
            except StopIteration:
                break
        
        self.engines[engine_id] = {
            'simulator': simulator,
            'total_cycles': total_cycles,
            'current_cycle': current_cycle,
            'rul_true': remaining_cycles,
            'status': self._get_status(progress),
        }
    
    def _get_status(self, progress: float) -> str:
        """Get status label for progress level."""
        if progress < 0.5:
            return 'healthy'
        elif progress < 0.8:
            return 'degrading'
        else:
            return 'critical'
    
    def get_current_readings(self) -> Dict[str, Dict[str, Any]]:
        """Get current sensor readings for all engines.
        
        Returns:
            Dictionary mapping engine_id to readings
        """
        readings = {}
        
        for engine_id, engine in self.engines.items():
            simulator = engine['simulator']
            
            try:
                op_cond, sensors, rul = simulator.step()
                engine['current_cycle'] += 1
                engine['rul_true'] = rul
                
                readings[engine_id] = {
                    'sensors': sensors,
                    'op_cond': op_cond,
                    'rul_true': rul,
                    'status': engine['status'],
                }
                
                # Update status
                progress = engine['current_cycle'] / engine['total_cycles']
                engine['status'] = self._get_status(progress)
                
            except StopIteration:
                # Engine failed
                readings[engine_id] = {
                    'sensors': None,
                    'op_cond': None,
                    'rul_true': 0,
                    'status': 'failed',
                }
                engine['status'] = 'failed'
        
        return readings
    
    def get_prognostic_sensors(
        self,
        readings: Dict[str, Dict[str, Any]],
    ) -> Dict[str, np.ndarray]:
        """Extract prognostic sensors from readings.
        
        Args:
            readings: Fleet readings
            
        Returns:
            Dictionary mapping engine_id to prognostic sensors
        """
        prognostic = {}
        
        for engine_id, data in readings.items():
            if data['sensors'] is not None:
                prognostic[engine_id] = data['sensors'][PROGNOSTIC_INDICES]
        
        return prognostic
    
    def get_fleet_status(self) -> Dict[str, Any]:
        """Get fleet-level status summary.
        
        Returns:
            Dictionary with fleet statistics
        """
        statuses = [e['status'] for e in self.engines.values()]
        ruls = [e['rul_true'] for e in self.engines.values()]
        
        return {
            'total_engines': len(self.engines),
            'healthy': statuses.count('healthy'),
            'degrading': statuses.count('degrading'),
            'critical': statuses.count('critical'),
            'failed': statuses.count('failed'),
            'mean_rul': float(np.mean(ruls)),
            'min_rul': int(min(ruls)),
            'max_rul': int(max(ruls)),
        }
    
    def get_engines_by_status(
        self,
        status: str,
    ) -> List[str]:
        """Get engine IDs by status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List of engine IDs
        """
        return [
            eid for eid, e in self.engines.items()
            if e['status'] == status
        ]
    
    def reset_engine(self, engine_id: str) -> None:
        """Reset engine to initial state (simulates replacement).
        
        Args:
            engine_id: Engine to reset
        """
        if engine_id not in self.engines:
            raise KeyError(f"Unknown engine: {engine_id}")
        
        engine = self.engines[engine_id]
        engine['simulator'].reset()
        engine['current_cycle'] = 0
        engine['rul_true'] = engine['total_cycles']
        engine['status'] = 'healthy'
        
        logger.info(f"Engine {engine_id} reset (replaced)")


def create_test_batch(
    batch_size: int = 32,
    seq_length: int = 50,
    n_sensors: int = 14,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create test batch for model evaluation.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        n_sensors: Number of sensors
        seed: Random seed
        
    Returns:
        Tuple of (sequences, rul_targets)
    """
    rng = np.random.default_rng(seed)
    
    sequences = []
    targets = []
    
    config = TrajectoryConfig(
        min_cycles=seq_length + 20,
        max_cycles=seq_length + 100,
    )
    
    for _ in range(batch_size):
        traj = generate_trajectory(config, rng.integers(0, 2**31))
        
        # Get prognostic sensors
        sensors = traj['sensors'][:, PROGNOSTIC_INDICES]
        
        # Random starting point
        max_start = len(sensors) - seq_length
        start = rng.integers(0, max_start)
        
        # Extract sequence
        seq = sensors[start:start + seq_length]
        rul = traj['rul'][start + seq_length - 1]
        
        sequences.append(seq)
        targets.append(rul)
    
    return np.array(sequences), np.array(targets)
