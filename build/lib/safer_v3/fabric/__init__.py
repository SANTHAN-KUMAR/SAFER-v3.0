"""
Fabric Module Initialization for SAFER v3.0.

This subpackage implements the shared-memory communication fabric for
inter-process data exchange in the SAFER architecture.

Modules:
- ring_buffer.py: Lock-free SPSC ring buffer for sensor data
- shm_transport.py: Shared memory transport layer
- process_manager.py: Process orchestration and lifecycle management

The fabric provides:
1. Zero-copy data transfer between processes
2. Lock-free synchronization for real-time performance
3. Deterministic latency guarantees
4. Graceful degradation on failure

Architecture:
    [Simulator] --ring_buffer--> [Mamba Process]
         |                            |
         +--ring_buffer--> [SINDy Process]
                                      |
                          [Decision Process] <--+

Performance Targets:
- Transport latency: <1ms
- Memory bandwidth: >1GB/s
- Lock contention: Zero (SPSC design)
"""

from safer_v3.fabric.ring_buffer import (
    RingBuffer,
    RingBufferConfig,
    create_ring_buffer,
)
from safer_v3.fabric.shm_transport import (
    SharedMemoryTransport,
    TransportConfig,
    SensorPacket,
    PredictionPacket,
)
from safer_v3.fabric.process_manager import (
    ProcessManager,
    ProcessConfig,
    WorkerProcess,
    ProcessState,
)

__all__ = [
    # Ring buffer
    'RingBuffer',
    'RingBufferConfig',
    'create_ring_buffer',
    # Transport
    'SharedMemoryTransport',
    'TransportConfig',
    'SensorPacket',
    'PredictionPacket',
    # Process manager
    'ProcessManager',
    'ProcessConfig',
    'WorkerProcess',
    'ProcessState',
]
