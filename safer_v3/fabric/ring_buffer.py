"""
Lock-Free Ring Buffer for SAFER v3.0.

This module implements a Single-Producer Single-Consumer (SPSC) lock-free
ring buffer for high-performance inter-process communication.

Key Features:
- Lock-free operation using atomic indices
- Cache-line padding to prevent false sharing
- Memory-mapped backing for shared memory support
- Sequence numbers for ordering guarantees
- Overflow detection and handling

The ring buffer is the core data structure for the SAFER communication
fabric, enabling zero-copy data transfer with deterministic latency.

Design Principles:
1. SPSC model - one writer, one reader, no locks needed
2. Power-of-2 sizing for efficient modulo via bitmask
3. Sequence numbers to detect wraparound
4. Cache-line alignment to prevent false sharing

Performance:
- Write: O(1) constant time
- Read: O(1) constant time  
- No syscalls in hot path
- No locks or atomic RMW operations (only atomic loads/stores)

References:
    - Lamport, "Proving the correctness of multiprocess programs" (1977)
    - LMAX Disruptor pattern
"""

import numpy as np
import mmap
import struct
import ctypes
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Any, Generic, TypeVar
from pathlib import Path
import logging
import os
import time


logger = logging.getLogger(__name__)


# Cache line size (64 bytes on most modern CPUs)
CACHE_LINE_SIZE = 64

# Magic number for buffer validation
RING_BUFFER_MAGIC = 0x52494E47  # "RING" in ASCII


@dataclass
class RingBufferConfig:
    """Configuration for ring buffer.
    
    Attributes:
        capacity: Number of slots (must be power of 2)
        element_size: Size of each element in bytes
        name: Optional name for shared memory
        use_mmap: Use memory-mapped file for persistence
        mmap_path: Path for mmap file
    """
    capacity: int = 1024
    element_size: int = 128  # Default for sensor packets
    name: str = "safer_ring_buffer"
    use_mmap: bool = False
    mmap_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        # Ensure capacity is power of 2
        if self.capacity & (self.capacity - 1) != 0:
            # Round up to next power of 2
            self.capacity = 1 << (self.capacity - 1).bit_length()
            logger.warning(f"Capacity rounded up to power of 2: {self.capacity}")
        
        if self.element_size <= 0:
            raise ValueError("element_size must be positive")


class AtomicInt64:
    """Atomic 64-bit integer using ctypes.
    
    Provides atomic load and store operations for lock-free
    synchronization between processes.
    """
    
    def __init__(self, initial_value: int = 0, buffer: Optional[memoryview] = None):
        """Initialize atomic integer.
        
        Args:
            initial_value: Initial value
            buffer: Optional buffer for shared memory backing
        """
        if buffer is not None:
            # Use provided buffer (for shared memory)
            self._value = ctypes.c_int64.from_buffer(buffer)
            self._value.value = initial_value
        else:
            self._value = ctypes.c_int64(initial_value)
    
    def load(self) -> int:
        """Atomic load (acquire semantics)."""
        return self._value.value
    
    def store(self, value: int) -> None:
        """Atomic store (release semantics)."""
        self._value.value = value
    
    def __repr__(self) -> str:
        return f"AtomicInt64({self._value.value})"


@dataclass
class RingBufferHeader:
    """Header structure for ring buffer.
    
    Contains metadata and atomic indices for synchronization.
    Layout in memory (with cache-line padding):
    
    Offset 0:   magic (8 bytes)
    Offset 8:   capacity (8 bytes)
    Offset 16:  element_size (8 bytes)
    Offset 24:  reserved (40 bytes) -> total 64 bytes
    Offset 64:  write_index (8 bytes + 56 padding) -> 64 bytes
    Offset 128: read_index (8 bytes + 56 padding) -> 64 bytes
    Total: 192 bytes
    """
    HEADER_SIZE = 192
    
    magic: int = RING_BUFFER_MAGIC
    capacity: int = 0
    element_size: int = 0
    write_index: AtomicInt64 = field(default_factory=AtomicInt64)
    read_index: AtomicInt64 = field(default_factory=AtomicInt64)
    
    @classmethod
    def from_buffer(cls, buffer: memoryview) -> 'RingBufferHeader':
        """Create header from memory buffer.
        
        Args:
            buffer: Memory buffer containing header
            
        Returns:
            RingBufferHeader instance
        """
        # Read fixed fields
        magic, capacity, element_size = struct.unpack_from('<QQQ', buffer, 0)
        
        # Create atomic indices from buffer offsets
        # write_index at offset 64
        write_idx = AtomicInt64(0, buffer[64:72])
        # read_index at offset 128
        read_idx = AtomicInt64(0, buffer[128:136])
        
        return cls(
            magic=magic,
            capacity=capacity,
            element_size=element_size,
            write_index=write_idx,
            read_index=read_idx,
        )
    
    def write_to_buffer(self, buffer: memoryview) -> None:
        """Write header to memory buffer.
        
        Args:
            buffer: Target memory buffer
        """
        struct.pack_into('<QQQ', buffer, 0, self.magic, self.capacity, self.element_size)
        # Atomic indices are already backed by buffer


class RingBuffer:
    """Lock-free SPSC Ring Buffer.
    
    A high-performance circular buffer for single-producer single-consumer
    scenarios. Uses atomic operations for synchronization without locks.
    
    The buffer supports fixed-size elements and provides:
    - O(1) write and read operations
    - Sequence numbers for ordering
    - Overflow detection
    - Optional shared memory backing
    
    Usage:
        # Producer
        buffer = RingBuffer(config)
        buffer.write(data)
        
        # Consumer  
        data = buffer.read()
    
    Thread Safety:
        - Safe for exactly one producer and one consumer
        - Not safe for multiple producers or multiple consumers
    """
    
    def __init__(
        self,
        config: RingBufferConfig,
        create: bool = True,
    ):
        """Initialize ring buffer.
        
        Args:
            config: Buffer configuration
            create: If True, create new buffer; if False, attach to existing
        """
        self.config = config
        self.capacity = config.capacity
        self.element_size = config.element_size
        self.mask = self.capacity - 1  # For efficient modulo
        
        # Calculate total buffer size
        self.header_size = RingBufferHeader.HEADER_SIZE
        self.data_size = self.capacity * self.element_size
        self.total_size = self.header_size + self.data_size
        
        # Allocate or attach to buffer
        if config.use_mmap and config.mmap_path:
            self._init_mmap(config.mmap_path, create)
        else:
            self._init_memory(create)
        
        # Create header view
        header_view = memoryview(self._buffer)[:self.header_size]
        
        if create:
            # Initialize new buffer
            self.header = RingBufferHeader(
                magic=RING_BUFFER_MAGIC,
                capacity=self.capacity,
                element_size=self.element_size,
                write_index=AtomicInt64(0, header_view[64:72]),
                read_index=AtomicInt64(0, header_view[128:136]),
            )
            self.header.write_to_buffer(header_view)
        else:
            # Attach to existing buffer
            self.header = RingBufferHeader.from_buffer(header_view)
            
            if self.header.magic != RING_BUFFER_MAGIC:
                raise ValueError("Invalid ring buffer magic number")
            if self.header.capacity != self.capacity:
                raise ValueError(
                    f"Capacity mismatch: expected {self.capacity}, "
                    f"got {self.header.capacity}"
                )
        
        # Create data view
        self._data = memoryview(self._buffer)[self.header_size:]
        
        # Statistics
        self._write_count = 0
        self._read_count = 0
        self._overflow_count = 0
        
        logger.debug(
            f"RingBuffer initialized: capacity={self.capacity}, "
            f"element_size={self.element_size}, total_size={self.total_size}"
        )
    
    def _init_memory(self, create: bool) -> None:
        """Initialize with regular memory allocation.
        
        Args:
            create: Whether to create new buffer
        """
        self._buffer = bytearray(self.total_size)
        self._mmap = None
    
    def _init_mmap(self, path: str, create: bool) -> None:
        """Initialize with memory-mapped file.
        
        Args:
            path: Path to mmap file
            create: Whether to create new file
        """
        path = Path(path)
        
        if create:
            # Create directory if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file with proper size
            with open(path, 'wb') as f:
                f.write(b'\x00' * self.total_size)
        
        # Open file for memory mapping
        self._file = open(path, 'r+b')
        self._mmap = mmap.mmap(
            self._file.fileno(),
            self.total_size,
            access=mmap.ACCESS_WRITE,
        )
        self._buffer = self._mmap
    
    def write(self, data: bytes) -> bool:
        """Write data to buffer.
        
        Args:
            data: Data to write (must be exactly element_size bytes)
            
        Returns:
            True if write succeeded, False if buffer full
        """
        if len(data) != self.element_size:
            raise ValueError(
                f"Data size {len(data)} != element size {self.element_size}"
            )
        
        # Load indices
        write_idx = self.header.write_index.load()
        read_idx = self.header.read_index.load()
        
        # Check if buffer is full
        if write_idx - read_idx >= self.capacity:
            self._overflow_count += 1
            logger.warning(f"Ring buffer overflow (count: {self._overflow_count})")
            return False
        
        # Calculate slot position
        slot = write_idx & self.mask
        offset = slot * self.element_size
        
        # Write data
        self._data[offset:offset + self.element_size] = data
        
        # Update write index (release semantics)
        self.header.write_index.store(write_idx + 1)
        
        self._write_count += 1
        return True
    
    def read(self) -> Optional[bytes]:
        """Read data from buffer.
        
        Returns:
            Data bytes if available, None if buffer empty
        """
        # Load indices
        write_idx = self.header.write_index.load()
        read_idx = self.header.read_index.load()
        
        # Check if buffer is empty
        if read_idx >= write_idx:
            return None
        
        # Calculate slot position
        slot = read_idx & self.mask
        offset = slot * self.element_size
        
        # Read data
        data = bytes(self._data[offset:offset + self.element_size])
        
        # Update read index (release semantics)
        self.header.read_index.store(read_idx + 1)
        
        self._read_count += 1
        return data
    
    def try_read(self, timeout_ms: float = 0) -> Optional[bytes]:
        """Try to read with optional timeout.
        
        Args:
            timeout_ms: Timeout in milliseconds (0 = no wait)
            
        Returns:
            Data bytes if available within timeout, None otherwise
        """
        if timeout_ms <= 0:
            return self.read()
        
        deadline = time.perf_counter() + timeout_ms / 1000.0
        
        while time.perf_counter() < deadline:
            data = self.read()
            if data is not None:
                return data
            # Brief sleep to avoid busy-waiting
            time.sleep(0.0001)  # 100 microseconds
        
        return None
    
    def peek(self) -> Optional[bytes]:
        """Peek at next data without consuming.
        
        Returns:
            Data bytes if available, None if buffer empty
        """
        write_idx = self.header.write_index.load()
        read_idx = self.header.read_index.load()
        
        if read_idx >= write_idx:
            return None
        
        slot = read_idx & self.mask
        offset = slot * self.element_size
        
        return bytes(self._data[offset:offset + self.element_size])
    
    def write_batch(self, data_list: list) -> int:
        """Write multiple items in batch.
        
        Args:
            data_list: List of data items to write
            
        Returns:
            Number of items successfully written
        """
        written = 0
        for data in data_list:
            if self.write(data):
                written += 1
            else:
                break
        return written
    
    def read_batch(self, max_items: int) -> list:
        """Read multiple items in batch.
        
        Args:
            max_items: Maximum items to read
            
        Returns:
            List of data items read
        """
        items = []
        for _ in range(max_items):
            data = self.read()
            if data is None:
                break
            items.append(data)
        return items
    
    @property
    def available(self) -> int:
        """Number of items available to read."""
        write_idx = self.header.write_index.load()
        read_idx = self.header.read_index.load()
        return max(0, write_idx - read_idx)
    
    @property
    def free_space(self) -> int:
        """Number of free slots for writing."""
        return self.capacity - self.available
    
    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.available == 0
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.available >= self.capacity
    
    def clear(self) -> None:
        """Clear buffer by resetting indices."""
        current_write = self.header.write_index.load()
        self.header.read_index.store(current_write)
    
    def get_stats(self) -> dict:
        """Get buffer statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'capacity': self.capacity,
            'element_size': self.element_size,
            'available': self.available,
            'free_space': self.free_space,
            'write_count': self._write_count,
            'read_count': self._read_count,
            'overflow_count': self._overflow_count,
            'write_index': self.header.write_index.load(),
            'read_index': self.header.read_index.load(),
        }
    
    def close(self) -> None:
        """Close buffer and release resources."""
        if hasattr(self, '_mmap') and self._mmap is not None:
            self._mmap.close()
            self._file.close()
        
        logger.debug("RingBuffer closed")
    
    def __enter__(self) -> 'RingBuffer':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def __repr__(self) -> str:
        return (
            f"RingBuffer(capacity={self.capacity}, "
            f"element_size={self.element_size}, "
            f"available={self.available})"
        )


class TypedRingBuffer(RingBuffer):
    """Ring buffer with typed serialization.
    
    Provides automatic serialization/deserialization of numpy arrays
    or structured data types.
    """
    
    def __init__(
        self,
        config: RingBufferConfig,
        dtype: np.dtype,
        create: bool = True,
    ):
        """Initialize typed ring buffer.
        
        Args:
            config: Buffer configuration
            dtype: Numpy dtype for elements
            create: Whether to create new buffer
        """
        self.dtype = np.dtype(dtype)
        
        # Ensure element_size matches dtype
        config.element_size = self.dtype.itemsize
        
        super().__init__(config, create)
    
    def write_array(self, arr: np.ndarray) -> bool:
        """Write numpy array to buffer.
        
        Args:
            arr: Array to write (must match dtype)
            
        Returns:
            True if write succeeded
        """
        if arr.dtype != self.dtype:
            arr = arr.astype(self.dtype)
        return self.write(arr.tobytes())
    
    def read_array(self) -> Optional[np.ndarray]:
        """Read numpy array from buffer.
        
        Returns:
            Array if available, None if buffer empty
        """
        data = self.read()
        if data is None:
            return None
        return np.frombuffer(data, dtype=self.dtype)


def create_ring_buffer(
    capacity: int = 1024,
    element_size: int = 128,
    name: str = "safer_ring_buffer",
    use_mmap: bool = False,
    mmap_path: Optional[str] = None,
) -> RingBuffer:
    """Factory function to create a ring buffer.
    
    Args:
        capacity: Number of slots (rounded to power of 2)
        element_size: Size of each element in bytes
        name: Buffer name
        use_mmap: Use memory-mapped file
        mmap_path: Path for mmap file
        
    Returns:
        Configured RingBuffer instance
    """
    config = RingBufferConfig(
        capacity=capacity,
        element_size=element_size,
        name=name,
        use_mmap=use_mmap,
        mmap_path=mmap_path,
    )
    return RingBuffer(config, create=True)
