"""
Deterministic logging configuration for SAFER v3.0.

This module provides a structured logging system with:
- Component-specific loggers
- Performance metrics logging
- Deterministic timestamps
- File and console handlers
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import threading


# Thread-safe logger cache
_loggers: Dict[str, logging.Logger] = {}
_lock = threading.Lock()


class DeterministicFormatter(logging.Formatter):
    """Custom formatter with deterministic timestamp formatting."""
    
    def __init__(self, include_thread: bool = False):
        """Initialize formatter.
        
        Args:
            include_thread: Whether to include thread name in output
        """
        self.include_thread = include_thread
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with deterministic structure.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log string
        """
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        level = record.levelname.ljust(8)
        name = record.name.ljust(20)
        
        if self.include_thread:
            thread = f"[{record.threadName}]".ljust(15)
            return f"{timestamp} | {level} | {thread} | {name} | {record.getMessage()}"
        else:
            return f"{timestamp} | {level} | {name} | {record.getMessage()}"


class PerformanceLogger:
    """Logger for performance metrics with structured output."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize performance logger.
        
        Args:
            logger: Base logger instance
        """
        self._logger = logger
        self._metrics: Dict[str, list] = {}
    
    def log_latency(self, component: str, latency_ms: float) -> None:
        """Log component latency.
        
        Args:
            component: Component name
            latency_ms: Latency in milliseconds
        """
        key = f"{component}_latency"
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append(latency_ms)
        
        self._logger.debug(f"PERF | {component} | latency={latency_ms:.3f}ms")
    
    def log_throughput(self, component: str, samples_per_sec: float) -> None:
        """Log component throughput.
        
        Args:
            component: Component name
            samples_per_sec: Throughput in samples per second
        """
        self._logger.debug(f"PERF | {component} | throughput={samples_per_sec:.1f} samples/s")
    
    def log_memory(self, component: str, memory_mb: float) -> None:
        """Log memory usage.
        
        Args:
            component: Component name
            memory_mb: Memory usage in megabytes
        """
        self._logger.debug(f"PERF | {component} | memory={memory_mb:.2f}MB")
    
    def get_statistics(self, component: str) -> Dict[str, float]:
        """Get statistics for a component.
        
        Args:
            component: Component name
            
        Returns:
            Dictionary with min, max, mean, std latencies
        """
        key = f"{component}_latency"
        if key not in self._metrics or not self._metrics[key]:
            return {}
        
        values = self._metrics[key]
        import statistics
        
        return {
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'count': len(values),
        }


def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
    include_thread: bool = False,
) -> None:
    """Set up the logging system.
    
    Args:
        log_dir: Directory for log files (created if not exists)
        level: Logging level
        console: Enable console output
        file: Enable file output
        include_thread: Include thread name in output
    """
    # Create root logger for SAFER
    root_logger = logging.getLogger('safer_v3')
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Create formatter
    formatter = DeterministicFormatter(include_thread=include_thread)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"safer_v3_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific component.
    
    Args:
        name: Component name (e.g., 'mamba', 'lpv_sindy', 'fabric')
        
    Returns:
        Logger instance
    """
    full_name = f"safer_v3.{name}"
    
    with _lock:
        if full_name not in _loggers:
            logger = logging.getLogger(full_name)
            _loggers[full_name] = logger
        return _loggers[full_name]


def get_performance_logger(name: str) -> PerformanceLogger:
    """Get a performance logger for a specific component.
    
    Args:
        name: Component name
        
    Returns:
        PerformanceLogger instance
    """
    base_logger = get_logger(f"{name}.perf")
    return PerformanceLogger(base_logger)


class LogContext:
    """Context manager for structured logging of operations."""
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        """Initialize log context.
        
        Args:
            logger: Logger instance
            operation: Operation name
            **context: Additional context to log
        """
        self._logger = logger
        self._operation = operation
        self._context = context
        self._start_time: Optional[float] = None
    
    def __enter__(self) -> 'LogContext':
        """Enter context and log start."""
        import time
        self._start_time = time.perf_counter()
        
        context_str = " | ".join(f"{k}={v}" for k, v in self._context.items())
        if context_str:
            self._logger.info(f"START | {self._operation} | {context_str}")
        else:
            self._logger.info(f"START | {self._operation}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context and log completion."""
        import time
        elapsed = (time.perf_counter() - self._start_time) * 1000
        
        if exc_type is None:
            self._logger.info(f"END   | {self._operation} | elapsed={elapsed:.2f}ms")
        else:
            self._logger.error(f"FAIL  | {self._operation} | error={exc_val} | elapsed={elapsed:.2f}ms")
        
        return False  # Don't suppress exceptions
