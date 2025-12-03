"""
Process Manager for SAFER v3.0.

This module provides process orchestration and lifecycle management
for the SAFER multi-process architecture.

Components:
1. WorkerProcess - Base class for worker processes
2. ProcessManager - Orchestrates multiple workers
3. ProcessConfig - Configuration for process setup

The process manager handles:
- Process spawning and monitoring
- Graceful shutdown with cleanup
- Failure detection and restart
- Resource management
- Inter-process communication setup

Architecture:
    ProcessManager
        |
        +-- SimulatorProcess (generates sensor data)
        +-- MambaProcess (neural network inference)
        +-- SINDyProcess (physics monitor)
        +-- DecisionProcess (Simplex logic)

Safety Considerations:
- Watchdog monitoring for process health
- Automatic restart on failure
- Graceful degradation modes
- Clean shutdown protocol
"""

import multiprocessing as mp
from multiprocessing import Process, Queue, Event
import threading
import time
import signal
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Type
from enum import Enum, auto
from pathlib import Path
import logging
import traceback


logger = logging.getLogger(__name__)


class ProcessState(Enum):
    """Process lifecycle states."""
    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()
    RESTARTING = auto()


@dataclass
class ProcessConfig:
    """Configuration for worker processes.
    
    Attributes:
        name: Process name for identification
        target: Target function or class to run
        args: Positional arguments for target
        kwargs: Keyword arguments for target
        restart_on_failure: Auto-restart on crash
        max_restarts: Maximum restart attempts
        restart_delay: Delay between restarts (seconds)
        startup_timeout: Time to wait for startup (seconds)
        shutdown_timeout: Time to wait for graceful shutdown (seconds)
        priority: Process priority (0 = normal, <0 = higher)
        cpu_affinity: List of CPU cores to bind to
    """
    name: str
    target: Optional[Callable] = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    restart_on_failure: bool = True
    max_restarts: int = 3
    restart_delay: float = 1.0
    startup_timeout: float = 10.0
    shutdown_timeout: float = 5.0
    priority: int = 0
    cpu_affinity: Optional[List[int]] = None


class WorkerProcess:
    """Base class for SAFER worker processes.
    
    Provides common functionality for all worker processes:
    - Lifecycle management (start, stop, restart)
    - Signal handling
    - Health monitoring
    - Clean shutdown
    
    Subclasses should implement:
    - setup(): Initialize resources
    - run_iteration(): Main loop iteration
    - cleanup(): Release resources
    
    Usage:
        class MyWorker(WorkerProcess):
            def setup(self):
                self.model = load_model()
            
            def run_iteration(self):
                data = self.receive()
                result = self.model(data)
                self.send(result)
            
            def cleanup(self):
                del self.model
        
        worker = MyWorker(config)
        worker.start()
    """
    
    def __init__(
        self,
        config: ProcessConfig,
        stop_event: Optional[Event] = None,
        command_queue: Optional[Queue] = None,
        status_queue: Optional[Queue] = None,
    ):
        """Initialize worker process.
        
        Args:
            config: Process configuration
            stop_event: Event to signal shutdown
            command_queue: Queue for receiving commands
            status_queue: Queue for sending status updates
        """
        self.config = config
        self.name = config.name
        
        # Synchronization primitives
        self._stop_event = stop_event or Event()
        self._command_queue = command_queue or Queue()
        self._status_queue = status_queue or Queue()
        
        # State tracking
        self._state = ProcessState.CREATED
        self._process: Optional[Process] = None
        self._start_time: Optional[float] = None
        self._restart_count = 0
        self._last_heartbeat: Optional[float] = None
        
        # Statistics
        self._iterations = 0
        self._errors = 0
    
    @property
    def state(self) -> ProcessState:
        """Current process state."""
        return self._state
    
    @property
    def is_running(self) -> bool:
        """Check if process is running."""
        return (
            self._state == ProcessState.RUNNING and
            self._process is not None and
            self._process.is_alive()
        )
    
    @property
    def uptime(self) -> float:
        """Process uptime in seconds."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    def start(self) -> None:
        """Start the worker process."""
        if self._state not in (ProcessState.CREATED, ProcessState.STOPPED):
            raise RuntimeError(f"Cannot start process in state {self._state}")
        
        self._state = ProcessState.STARTING
        
        # Create process
        self._process = Process(
            target=self._run_wrapper,
            name=self.name,
            daemon=False,
        )
        
        self._stop_event.clear()
        self._process.start()
        self._start_time = time.time()
        
        # Wait for startup confirmation
        try:
            startup_confirmed = self._wait_for_startup()
            if startup_confirmed:
                self._state = ProcessState.RUNNING
                logger.info(f"Process '{self.name}' started (PID: {self._process.pid})")
            else:
                self._state = ProcessState.FAILED
                logger.error(f"Process '{self.name}' failed to start")
        except Exception as e:
            self._state = ProcessState.FAILED
            logger.error(f"Process '{self.name}' startup error: {e}")
    
    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop the worker process gracefully.
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        if self._process is None:
            return
        
        timeout = timeout or self.config.shutdown_timeout
        self._state = ProcessState.STOPPING
        
        # Signal shutdown
        self._stop_event.set()
        
        # Wait for graceful shutdown
        self._process.join(timeout=timeout)
        
        if self._process.is_alive():
            # Force terminate
            logger.warning(f"Force terminating process '{self.name}'")
            self._process.terminate()
            self._process.join(timeout=1.0)
            
            if self._process.is_alive():
                # Last resort: kill
                self._process.kill()
        
        self._state = ProcessState.STOPPED
        logger.info(f"Process '{self.name}' stopped")
    
    def restart(self) -> None:
        """Restart the worker process."""
        if self._restart_count >= self.config.max_restarts:
            logger.error(
                f"Process '{self.name}' exceeded max restarts "
                f"({self.config.max_restarts})"
            )
            self._state = ProcessState.FAILED
            return
        
        self._state = ProcessState.RESTARTING
        self._restart_count += 1
        
        logger.info(
            f"Restarting process '{self.name}' "
            f"(attempt {self._restart_count}/{self.config.max_restarts})"
        )
        
        # Stop if running
        if self._process is not None and self._process.is_alive():
            self.stop()
        
        # Wait before restart
        time.sleep(self.config.restart_delay)
        
        # Reset state
        self._state = ProcessState.CREATED
        self._stop_event.clear()
        
        # Start again
        self.start()
    
    def send_command(self, command: str, **kwargs) -> None:
        """Send command to worker process.
        
        Args:
            command: Command name
            **kwargs: Command arguments
        """
        self._command_queue.put({'command': command, **kwargs})
    
    def get_status(self) -> Dict[str, Any]:
        """Get process status.
        
        Returns:
            Status dictionary
        """
        return {
            'name': self.name,
            'state': self._state.name,
            'pid': self._process.pid if self._process else None,
            'uptime': self.uptime,
            'iterations': self._iterations,
            'errors': self._errors,
            'restart_count': self._restart_count,
            'is_alive': self._process.is_alive() if self._process else False,
        }
    
    def _wait_for_startup(self) -> bool:
        """Wait for startup confirmation.
        
        Returns:
            True if startup succeeded
        """
        deadline = time.time() + self.config.startup_timeout
        
        while time.time() < deadline:
            try:
                status = self._status_queue.get(timeout=0.1)
                if status.get('event') == 'started':
                    return True
                if status.get('event') == 'failed':
                    return False
            except:
                pass
            
            if not self._process.is_alive():
                return False
        
        return False
    
    def _run_wrapper(self) -> None:
        """Wrapper for running the process main loop."""
        try:
            # Setup signal handlers
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            
            # Set CPU affinity if specified
            if self.config.cpu_affinity:
                try:
                    os.sched_setaffinity(0, self.config.cpu_affinity)
                except (AttributeError, OSError):
                    pass  # Not available on all platforms
            
            # Initialize
            self.setup()
            
            # Signal successful startup
            self._status_queue.put({'event': 'started', 'pid': os.getpid()})
            
            # Main loop
            while not self._stop_event.is_set():
                try:
                    # Check for commands
                    self._process_commands()
                    
                    # Run iteration
                    self.run_iteration()
                    self._iterations += 1
                    
                    # Send heartbeat periodically
                    if self._iterations % 100 == 0:
                        self._status_queue.put({
                            'event': 'heartbeat',
                            'iterations': self._iterations,
                        })
                    
                except Exception as e:
                    self._errors += 1
                    logger.error(f"Error in '{self.name}': {e}")
                    traceback.print_exc()
                    
                    if self._errors > 10:
                        logger.error(f"Too many errors in '{self.name}', stopping")
                        break
            
        except Exception as e:
            logger.error(f"Fatal error in '{self.name}': {e}")
            traceback.print_exc()
            self._status_queue.put({'event': 'failed', 'error': str(e)})
        
        finally:
            # Cleanup
            try:
                self.cleanup()
            except Exception as e:
                logger.error(f"Cleanup error in '{self.name}': {e}")
            
            self._status_queue.put({'event': 'stopped'})
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Process '{self.name}' received signal {signum}")
        self._stop_event.set()
    
    def _process_commands(self) -> None:
        """Process incoming commands."""
        try:
            while not self._command_queue.empty():
                cmd = self._command_queue.get_nowait()
                command = cmd.pop('command', None)
                
                if command == 'stop':
                    self._stop_event.set()
                elif command == 'status':
                    self._status_queue.put(self.get_status())
                else:
                    self.handle_command(command, **cmd)
        except:
            pass
    
    # Methods to override in subclasses
    
    def setup(self) -> None:
        """Initialize process resources. Override in subclass."""
        pass
    
    def run_iteration(self) -> None:
        """Run one iteration of main loop. Override in subclass."""
        time.sleep(0.01)  # Default: idle
    
    def cleanup(self) -> None:
        """Clean up process resources. Override in subclass."""
        pass
    
    def handle_command(self, command: str, **kwargs) -> None:
        """Handle custom commands. Override in subclass."""
        logger.warning(f"Unknown command: {command}")


class ProcessManager:
    """Orchestrates multiple worker processes.
    
    Provides centralized management for all SAFER processes:
    - Start/stop all processes
    - Monitor health and restart on failure
    - Graceful shutdown coordination
    - Resource cleanup
    
    Usage:
        manager = ProcessManager()
        manager.add_process(simulator_config)
        manager.add_process(mamba_config)
        manager.start_all()
        
        # Run until shutdown
        manager.wait()
        
        manager.stop_all()
    """
    
    def __init__(self, watchdog_interval: float = 1.0):
        """Initialize process manager.
        
        Args:
            watchdog_interval: Interval for health checks (seconds)
        """
        self._workers: Dict[str, WorkerProcess] = {}
        self._watchdog_interval = watchdog_interval
        self._watchdog_thread: Optional[threading.Thread] = None
        self._running = False
        self._shutdown_event = Event()
        
        # Setup signal handlers for main process
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        signal.signal(signal.SIGINT, self._shutdown_handler)
        
        logger.info("ProcessManager initialized")
    
    def add_process(
        self,
        config: ProcessConfig,
        worker_class: Type[WorkerProcess] = WorkerProcess,
    ) -> None:
        """Add a worker process.
        
        Args:
            config: Process configuration
            worker_class: Worker class to instantiate
        """
        if config.name in self._workers:
            raise ValueError(f"Process '{config.name}' already exists")
        
        worker = worker_class(
            config=config,
            stop_event=Event(),
            command_queue=Queue(),
            status_queue=Queue(),
        )
        
        self._workers[config.name] = worker
        logger.info(f"Added process '{config.name}'")
    
    def remove_process(self, name: str) -> None:
        """Remove a worker process.
        
        Args:
            name: Process name
        """
        if name not in self._workers:
            raise ValueError(f"Process '{name}' not found")
        
        worker = self._workers[name]
        
        if worker.is_running:
            worker.stop()
        
        del self._workers[name]
        logger.info(f"Removed process '{name}'")
    
    def start_all(self) -> None:
        """Start all worker processes."""
        logger.info("Starting all processes...")
        
        self._running = True
        self._shutdown_event.clear()
        
        # Start workers
        for name, worker in self._workers.items():
            try:
                worker.start()
            except Exception as e:
                logger.error(f"Failed to start '{name}': {e}")
        
        # Start watchdog
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
        )
        self._watchdog_thread.start()
        
        logger.info("All processes started")
    
    def stop_all(self, timeout: float = 10.0) -> None:
        """Stop all worker processes.
        
        Args:
            timeout: Maximum time to wait for each process
        """
        logger.info("Stopping all processes...")
        
        self._running = False
        self._shutdown_event.set()
        
        # Stop watchdog
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=2.0)
        
        # Stop workers in reverse order
        for name in reversed(list(self._workers.keys())):
            worker = self._workers[name]
            try:
                worker.stop(timeout=timeout)
            except Exception as e:
                logger.error(f"Error stopping '{name}': {e}")
        
        logger.info("All processes stopped")
    
    def restart_process(self, name: str) -> None:
        """Restart a specific process.
        
        Args:
            name: Process name
        """
        if name not in self._workers:
            raise ValueError(f"Process '{name}' not found")
        
        self._workers[name].restart()
    
    def get_process(self, name: str) -> WorkerProcess:
        """Get worker by name.
        
        Args:
            name: Process name
            
        Returns:
            WorkerProcess instance
        """
        if name not in self._workers:
            raise ValueError(f"Process '{name}' not found")
        return self._workers[name]
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all processes.
        
        Returns:
            Dictionary of process statuses
        """
        return {
            name: worker.get_status()
            for name, worker in self._workers.items()
        }
    
    def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for shutdown signal.
        
        Args:
            timeout: Maximum time to wait (None = forever)
        """
        self._shutdown_event.wait(timeout=timeout)
    
    def is_healthy(self) -> bool:
        """Check if all processes are healthy.
        
        Returns:
            True if all processes running
        """
        return all(
            worker.is_running
            for worker in self._workers.values()
        )
    
    def _watchdog_loop(self) -> None:
        """Watchdog thread for monitoring process health."""
        while self._running:
            try:
                for name, worker in self._workers.items():
                    # Check if process died unexpectedly
                    if worker.state == ProcessState.RUNNING and not worker.is_running:
                        logger.warning(f"Process '{name}' died unexpectedly")
                        
                        if worker.config.restart_on_failure:
                            worker.restart()
                        else:
                            worker._state = ProcessState.FAILED
                    
                    # Drain status queue
                    while True:
                        try:
                            status = worker._status_queue.get_nowait()
                            event = status.get('event')
                            
                            if event == 'heartbeat':
                                worker._last_heartbeat = time.time()
                            elif event == 'stopped':
                                if worker._state == ProcessState.RUNNING:
                                    worker._state = ProcessState.STOPPED
                            elif event == 'failed':
                                worker._state = ProcessState.FAILED
                                if worker.config.restart_on_failure:
                                    worker.restart()
                        except:
                            break
                
                time.sleep(self._watchdog_interval)
                
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
    
    def _shutdown_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"ProcessManager received signal {signum}")
        self._shutdown_event.set()
        self._running = False
    
    def __enter__(self) -> 'ProcessManager':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop_all()


# Convenience function for creating process configs

def create_process_config(
    name: str,
    target: Callable,
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    restart: bool = True,
) -> ProcessConfig:
    """Create process configuration.
    
    Args:
        name: Process name
        target: Target function
        args: Positional arguments
        kwargs: Keyword arguments
        restart: Enable auto-restart
        
    Returns:
        ProcessConfig instance
    """
    return ProcessConfig(
        name=name,
        target=target,
        args=args,
        kwargs=kwargs or {},
        restart_on_failure=restart,
    )
