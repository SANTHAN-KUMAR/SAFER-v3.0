"""
Shared Memory Transport for SAFER v3.0.

This module implements the shared memory transport layer for
high-performance inter-process communication.

Key Features:
- Zero-copy data transfer via shared memory
- Structured packet formats for sensors and predictions
- Multiple channel support (sensor, prediction, control)
- Automatic serialization/deserialization
- Timestamp and sequence number tracking

The transport layer sits on top of ring buffers and provides:
1. Type-safe packet handling
2. Channel multiplexing
3. Latency measurement
4. Connection management

Architecture:
    Sensor Process --> [SensorPacket] --> SharedMemory --> Mamba Process
    Mamba Process --> [PredictionPacket] --> SharedMemory --> Decision Process

Performance Targets:
- End-to-end latency: <1ms
- Throughput: >10,000 packets/second
- Memory overhead: <1MB per channel
"""

import numpy as np
import struct
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List, Union
from pathlib import Path
from enum import IntEnum
import logging

from safer_v3.fabric.ring_buffer import RingBuffer, RingBufferConfig


logger = logging.getLogger(__name__)


class PacketType(IntEnum):
    """Packet type identifiers."""
    SENSOR = 1
    PREDICTION = 2
    CONTROL = 3
    HEARTBEAT = 4
    SHUTDOWN = 5


@dataclass
class TransportConfig:
    """Configuration for shared memory transport.
    
    Attributes:
        sensor_buffer_capacity: Capacity of sensor data buffer
        prediction_buffer_capacity: Capacity of prediction buffer
        n_sensors: Number of sensor channels
        use_timestamps: Include timestamps in packets
        use_sequence_numbers: Include sequence numbers
        shm_prefix: Prefix for shared memory names
    """
    sensor_buffer_capacity: int = 4096
    prediction_buffer_capacity: int = 1024
    n_sensors: int = 14
    use_timestamps: bool = True
    use_sequence_numbers: bool = True
    shm_prefix: str = "safer_v3"


@dataclass
class SensorPacket:
    """Packet structure for sensor data.
    
    Layout (148 bytes total for 14 sensors):
        - packet_type: uint8 (1 byte)
        - flags: uint8 (1 byte)
        - sequence: uint32 (4 bytes)
        - timestamp: float64 (8 bytes)
        - unit_id: uint16 (2 bytes)
        - time_cycle: uint16 (2 bytes)
        - n_sensors: uint16 (2 bytes)
        - reserved: uint16 (2 bytes)
        - op_settings: float32[3] (12 bytes)
        - sensors: float32[n_sensors] (n_sensors * 4 bytes)
        
    Total for n_sensors=14: 1+1+4+8+2+2+2+2+12+56 = 90 bytes
    Padded to 128 bytes for alignment.
    """
    HEADER_FORMAT = '<BBIQHHHHfff'  # Little-endian
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 32 bytes
    
    packet_type: int = PacketType.SENSOR
    flags: int = 0
    sequence: int = 0
    timestamp: float = 0.0
    unit_id: int = 0
    time_cycle: int = 0
    n_sensors: int = 14
    op_settings: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    sensors: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=np.float32))
    
    @classmethod
    def packet_size(cls, n_sensors: int = 14) -> int:
        """Calculate packet size for given sensor count."""
        # Header + sensors, padded to 8-byte boundary
        size = cls.HEADER_SIZE + n_sensors * 4
        return ((size + 7) // 8) * 8  # Align to 8 bytes
    
    def to_bytes(self) -> bytes:
        """Serialize packet to bytes."""
        header = struct.pack(
            self.HEADER_FORMAT,
            self.packet_type,
            self.flags,
            self.sequence,
            int(self.timestamp * 1e9),  # Nanoseconds
            self.unit_id,
            self.time_cycle,
            self.n_sensors,
            0,  # Reserved
            self.op_settings[0],
            self.op_settings[1],
            self.op_settings[2],
        )
        
        sensor_bytes = self.sensors.astype(np.float32).tobytes()
        
        # Pad to packet size
        total = header + sensor_bytes
        packet_size = self.packet_size(self.n_sensors)
        
        if len(total) < packet_size:
            total += b'\x00' * (packet_size - len(total))
        
        return total
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'SensorPacket':
        """Deserialize packet from bytes."""
        # Parse header
        header_values = struct.unpack_from(cls.HEADER_FORMAT, data, 0)
        
        packet_type, flags, sequence, timestamp_ns, unit_id, time_cycle, n_sensors, _, \
            op1, op2, op3 = header_values
        
        # Parse sensors
        sensor_offset = cls.HEADER_SIZE
        sensors = np.frombuffer(
            data[sensor_offset:sensor_offset + n_sensors * 4],
            dtype=np.float32,
        ).copy()
        
        return cls(
            packet_type=packet_type,
            flags=flags,
            sequence=sequence,
            timestamp=timestamp_ns / 1e9,
            unit_id=unit_id,
            time_cycle=time_cycle,
            n_sensors=n_sensors,
            op_settings=np.array([op1, op2, op3], dtype=np.float32),
            sensors=sensors,
        )
    
    @classmethod
    def create(
        cls,
        sensors: np.ndarray,
        unit_id: int = 0,
        time_cycle: int = 0,
        op_settings: Optional[np.ndarray] = None,
        sequence: int = 0,
    ) -> 'SensorPacket':
        """Create sensor packet.
        
        Args:
            sensors: Sensor values
            unit_id: Engine unit ID
            time_cycle: Current time cycle
            op_settings: Operational settings [3]
            sequence: Sequence number
            
        Returns:
            SensorPacket instance
        """
        if op_settings is None:
            op_settings = np.zeros(3, dtype=np.float32)
        
        return cls(
            packet_type=PacketType.SENSOR,
            sequence=sequence,
            timestamp=time.time(),
            unit_id=unit_id,
            time_cycle=time_cycle,
            n_sensors=len(sensors),
            op_settings=np.asarray(op_settings, dtype=np.float32),
            sensors=np.asarray(sensors, dtype=np.float32),
        )


@dataclass
class PredictionPacket:
    """Packet structure for RUL predictions.
    
    Layout (64 bytes):
        - packet_type: uint8 (1 byte)
        - flags: uint8 (1 byte)
        - source: uint8 (1 byte) - 0=Mamba, 1=Baseline, 2=SINDy
        - reserved: uint8 (1 byte)
        - sequence: uint32 (4 bytes)
        - timestamp: float64 (8 bytes)
        - unit_id: uint16 (2 bytes)
        - time_cycle: uint16 (2 bytes)
        - rul_prediction: float32 (4 bytes)
        - confidence_lower: float32 (4 bytes)
        - confidence_upper: float32 (4 bytes)
        - anomaly_score: float32 (4 bytes)
        - latency_us: uint32 (4 bytes)
        - padding: 24 bytes
    """
    FORMAT = '<BBBBIQHHffffI'
    SIZE = struct.calcsize(FORMAT)  # 40 bytes, pad to 64
    PACKET_SIZE = 64
    
    # Source identifiers
    SOURCE_MAMBA = 0
    SOURCE_BASELINE = 1
    SOURCE_SINDY = 2
    
    packet_type: int = PacketType.PREDICTION
    flags: int = 0
    source: int = SOURCE_MAMBA
    sequence: int = 0
    timestamp: float = 0.0
    unit_id: int = 0
    time_cycle: int = 0
    rul_prediction: float = 0.0
    confidence_lower: float = 0.0
    confidence_upper: float = 0.0
    anomaly_score: float = 0.0
    latency_us: int = 0
    
    def to_bytes(self) -> bytes:
        """Serialize packet to bytes."""
        data = struct.pack(
            self.FORMAT,
            self.packet_type,
            self.flags,
            self.source,
            0,  # Reserved
            self.sequence,
            int(self.timestamp * 1e9),
            self.unit_id,
            self.time_cycle,
            self.rul_prediction,
            self.confidence_lower,
            self.confidence_upper,
            self.anomaly_score,
            self.latency_us,
        )
        
        # Pad to packet size
        return data + b'\x00' * (self.PACKET_SIZE - len(data))
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'PredictionPacket':
        """Deserialize packet from bytes."""
        values = struct.unpack_from(cls.FORMAT, data, 0)
        
        packet_type, flags, source, _, sequence, timestamp_ns, unit_id, time_cycle, \
            rul, conf_lower, conf_upper, anomaly, latency = values
        
        return cls(
            packet_type=packet_type,
            flags=flags,
            source=source,
            sequence=sequence,
            timestamp=timestamp_ns / 1e9,
            unit_id=unit_id,
            time_cycle=time_cycle,
            rul_prediction=rul,
            confidence_lower=conf_lower,
            confidence_upper=conf_upper,
            anomaly_score=anomaly,
            latency_us=latency,
        )
    
    @classmethod
    def create(
        cls,
        rul: float,
        unit_id: int = 0,
        time_cycle: int = 0,
        confidence: Optional[Tuple[float, float]] = None,
        anomaly_score: float = 0.0,
        source: int = SOURCE_MAMBA,
        sequence: int = 0,
        latency_us: int = 0,
    ) -> 'PredictionPacket':
        """Create prediction packet.
        
        Args:
            rul: RUL prediction value
            unit_id: Engine unit ID
            time_cycle: Current time cycle
            confidence: (lower, upper) confidence bounds
            anomaly_score: Anomaly detection score
            source: Prediction source (Mamba, Baseline, SINDy)
            sequence: Sequence number
            latency_us: Inference latency in microseconds
            
        Returns:
            PredictionPacket instance
        """
        if confidence is None:
            confidence = (rul - 10, rul + 10)
        
        return cls(
            packet_type=PacketType.PREDICTION,
            source=source,
            sequence=sequence,
            timestamp=time.time(),
            unit_id=unit_id,
            time_cycle=time_cycle,
            rul_prediction=rul,
            confidence_lower=confidence[0],
            confidence_upper=confidence[1],
            anomaly_score=anomaly_score,
            latency_us=latency_us,
        )


class SharedMemoryTransport:
    """Shared memory transport for inter-process communication.
    
    Provides high-level interface for sending and receiving
    typed packets over shared memory channels.
    
    Channels:
        - sensor: Sensor data from simulator to predictors
        - prediction: RUL predictions from predictors to decision
        - control: Control commands (shutdown, reconfigure)
    
    Usage:
        # Producer side
        transport = SharedMemoryTransport(config, role='producer')
        transport.send_sensors(sensor_data)
        
        # Consumer side
        transport = SharedMemoryTransport(config, role='consumer')
        packet = transport.receive_sensors()
    """
    
    def __init__(
        self,
        config: TransportConfig,
        role: str = 'producer',
        use_mmap: bool = False,
        mmap_dir: Optional[str] = None,
    ):
        """Initialize shared memory transport.
        
        Args:
            config: Transport configuration
            role: 'producer' or 'consumer'
            use_mmap: Use memory-mapped files
            mmap_dir: Directory for mmap files
        """
        self.config = config
        self.role = role
        self.use_mmap = use_mmap
        self.mmap_dir = Path(mmap_dir) if mmap_dir else None
        
        # Determine if we create or attach to buffers
        create = (role == 'producer')
        
        # Sensor packet size
        sensor_packet_size = SensorPacket.packet_size(config.n_sensors)
        
        # Create ring buffers for each channel
        self._sensor_buffer = self._create_buffer(
            name=f"{config.shm_prefix}_sensors",
            capacity=config.sensor_buffer_capacity,
            element_size=sensor_packet_size,
            create=create,
        )
        
        self._prediction_buffer = self._create_buffer(
            name=f"{config.shm_prefix}_predictions",
            capacity=config.prediction_buffer_capacity,
            element_size=PredictionPacket.PACKET_SIZE,
            create=create,
        )
        
        # Sequence counters
        self._sensor_seq = 0
        self._prediction_seq = 0
        
        # Latency tracking
        self._latencies: List[float] = []
        self._max_latency_samples = 1000
        
        logger.info(
            f"SharedMemoryTransport initialized: role={role}, "
            f"sensor_capacity={config.sensor_buffer_capacity}, "
            f"prediction_capacity={config.prediction_buffer_capacity}"
        )
    
    def _create_buffer(
        self,
        name: str,
        capacity: int,
        element_size: int,
        create: bool,
    ) -> RingBuffer:
        """Create or attach to ring buffer.
        
        Args:
            name: Buffer name
            capacity: Buffer capacity
            element_size: Element size
            create: Whether to create new buffer
            
        Returns:
            RingBuffer instance
        """
        mmap_path = None
        if self.use_mmap and self.mmap_dir:
            mmap_path = str(self.mmap_dir / f"{name}.buf")
        
        config = RingBufferConfig(
            capacity=capacity,
            element_size=element_size,
            name=name,
            use_mmap=self.use_mmap,
            mmap_path=mmap_path,
        )
        
        return RingBuffer(config, create=create)
    
    def send_sensors(
        self,
        sensors: np.ndarray,
        unit_id: int = 0,
        time_cycle: int = 0,
        op_settings: Optional[np.ndarray] = None,
    ) -> bool:
        """Send sensor data.
        
        Args:
            sensors: Sensor values array
            unit_id: Engine unit ID
            time_cycle: Current time cycle
            op_settings: Operational settings
            
        Returns:
            True if send succeeded
        """
        packet = SensorPacket.create(
            sensors=sensors,
            unit_id=unit_id,
            time_cycle=time_cycle,
            op_settings=op_settings,
            sequence=self._sensor_seq,
        )
        
        success = self._sensor_buffer.write(packet.to_bytes())
        
        if success:
            self._sensor_seq += 1
        
        return success
    
    def receive_sensors(self, timeout_ms: float = 0) -> Optional[SensorPacket]:
        """Receive sensor data.
        
        Args:
            timeout_ms: Timeout in milliseconds (0 = no wait)
            
        Returns:
            SensorPacket if available, None otherwise
        """
        data = self._sensor_buffer.try_read(timeout_ms)
        
        if data is None:
            return None
        
        packet = SensorPacket.from_bytes(data)
        
        # Track latency
        latency = time.time() - packet.timestamp
        self._track_latency(latency)
        
        return packet
    
    def send_prediction(
        self,
        rul: float,
        unit_id: int = 0,
        time_cycle: int = 0,
        confidence: Optional[Tuple[float, float]] = None,
        anomaly_score: float = 0.0,
        source: int = PredictionPacket.SOURCE_MAMBA,
        latency_us: int = 0,
    ) -> bool:
        """Send RUL prediction.
        
        Args:
            rul: RUL prediction value
            unit_id: Engine unit ID
            time_cycle: Current time cycle
            confidence: Confidence interval (lower, upper)
            anomaly_score: Anomaly detection score
            source: Prediction source
            latency_us: Inference latency
            
        Returns:
            True if send succeeded
        """
        packet = PredictionPacket.create(
            rul=rul,
            unit_id=unit_id,
            time_cycle=time_cycle,
            confidence=confidence,
            anomaly_score=anomaly_score,
            source=source,
            sequence=self._prediction_seq,
            latency_us=latency_us,
        )
        
        success = self._prediction_buffer.write(packet.to_bytes())
        
        if success:
            self._prediction_seq += 1
        
        return success
    
    def receive_prediction(self, timeout_ms: float = 0) -> Optional[PredictionPacket]:
        """Receive RUL prediction.
        
        Args:
            timeout_ms: Timeout in milliseconds
            
        Returns:
            PredictionPacket if available, None otherwise
        """
        data = self._prediction_buffer.try_read(timeout_ms)
        
        if data is None:
            return None
        
        packet = PredictionPacket.from_bytes(data)
        
        # Track latency
        latency = time.time() - packet.timestamp
        self._track_latency(latency)
        
        return packet
    
    def _track_latency(self, latency: float) -> None:
        """Track latency for statistics.
        
        Args:
            latency: Latency value in seconds
        """
        self._latencies.append(latency * 1000)  # Convert to ms
        
        if len(self._latencies) > self._max_latency_samples:
            self._latencies = self._latencies[-self._max_latency_samples:]
    
    @property
    def sensors_available(self) -> int:
        """Number of sensor packets available."""
        return self._sensor_buffer.available
    
    @property
    def predictions_available(self) -> int:
        """Number of prediction packets available."""
        return self._prediction_buffer.available
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics.
        
        Returns:
            Dictionary with latency statistics in milliseconds
        """
        if not self._latencies:
            return {
                'mean_ms': 0.0,
                'min_ms': 0.0,
                'max_ms': 0.0,
                'p50_ms': 0.0,
                'p95_ms': 0.0,
                'p99_ms': 0.0,
            }
        
        latencies = np.array(self._latencies)
        
        return {
            'mean_ms': float(np.mean(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'samples': len(latencies),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transport statistics.
        
        Returns:
            Dictionary with all statistics
        """
        return {
            'sensor_buffer': self._sensor_buffer.get_stats(),
            'prediction_buffer': self._prediction_buffer.get_stats(),
            'latency': self.get_latency_stats(),
        }
    
    def close(self) -> None:
        """Close transport and release resources."""
        self._sensor_buffer.close()
        self._prediction_buffer.close()
        logger.info("SharedMemoryTransport closed")
    
    def __enter__(self) -> 'SharedMemoryTransport':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
