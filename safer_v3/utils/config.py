"""
Configuration management for SAFER v3.0.

This module provides dataclass-based configuration for all system components
with validation and default values for deterministic operation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import json
import yaml


@dataclass
class MambaConfig:
    """Configuration for the Mamba Prognostic Core.
    
    Attributes:
        d_input: Input dimension (number of sensors)
        d_model: Model dimension for internal representations
        d_state: State dimension for SSM
        n_layers: Number of Mamba layers
        dropout: Dropout rate for regularization
        max_rul: Maximum RUL value for capping predictions
        sequence_length: Input sequence length for training
        use_jit: Enable torch.compile JIT compilation
        onnx_export: Enable ONNX export capability
    """
    d_input: int = 14
    d_model: int = 64
    d_state: int = 16
    n_layers: int = 4
    dropout: float = 0.1
    max_rul: int = 125
    sequence_length: int = 50
    use_jit: bool = True
    onnx_export: bool = False
    
    # Training hyperparameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 100
    patience: int = 15
    
    # Sensor indices for C-MAPSS (14 prognostic sensors)
    # Based on domain knowledge: sensors 2,3,4,7,8,9,11,12,13,14,15,17,20,21
    sensor_indices: List[int] = field(default_factory=lambda: [
        2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21
    ])
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.d_input > 0, "Input dimension must be positive"
        assert self.d_model > 0 and self.d_model % 2 == 0, "Model dimension must be positive and even"
        assert self.d_state > 0, "State dimension must be positive"
        assert self.n_layers > 0, "Number of layers must be positive"
        assert 0 <= self.dropout < 1, "Dropout must be in [0, 1)"
        assert len(self.sensor_indices) == self.d_input, "Sensor indices must match input dimension"


@dataclass
class LPVSINDyConfig:
    """Configuration for the LPV-SINDy Physics Monitor.
    
    Attributes:
        n_features: Number of input features (sensors)
        polynomial_degree: Polynomial degree for library (alias: poly_degree)
        include_interactions: Include interaction terms
        threshold: Sparsity threshold for STLSQ
        alpha: Regularization strength for Lasso
        window_size: Window size for integral formulation (alias: integration_window)
        dt: Time step between samples
        residual_threshold_sigma: Number of sigma for anomaly threshold
        max_iter: Maximum iterations for sparse regression
    """
    n_features: int = 14
    polynomial_degree: int = 2  # Renamed from poly_degree
    include_interactions: bool = True
    threshold: float = 0.1
    alpha: float = 0.01
    window_size: int = 5  # Renamed from integration_window
    dt: float = 1.0  # Added: time step
    residual_threshold_sigma: float = 3.0  # Renamed from residual_threshold
    max_iter: int = 100
    
    # LPV scheduling parameter (normalized EGT margin)
    egtm_sensor_idx: int = 9  # Index of EGT margin sensor
    
    # Aliases for backward compatibility
    @property
    def poly_degree(self) -> int:
        return self.polynomial_degree
    
    @property
    def integration_window(self) -> int:
        return self.window_size
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.n_features > 0, "Number of features must be positive"
        assert self.polynomial_degree >= 1, "Polynomial degree must be at least 1"
        assert self.threshold > 0, "Threshold must be positive"
        assert self.window_size >= 2, "Window size must be at least 2"


@dataclass
class FabricConfig:
    """Configuration for the Simulation Fabric.
    
    Attributes:
        buffer_size: Number of frames in ring buffer
        frame_size: Size of each frame in bytes
        shm_name: Name for shared memory segment
        poll_interval_ms: Polling interval in milliseconds
    """
    buffer_size: int = 1024
    frame_size: int = 56  # 14 sensors * 4 bytes (float32)
    shm_name: str = "safer_v3_shm"
    poll_interval_ms: float = 1.0
    
    # Memory layout constants
    header_size: int = 64  # Cache-line aligned header
    
    def total_size(self) -> int:
        """Calculate total shared memory size."""
        return self.header_size + (self.buffer_size * self.frame_size)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.buffer_size > 0 and (self.buffer_size & (self.buffer_size - 1)) == 0, \
            "Buffer size must be a positive power of 2"
        assert self.frame_size > 0, "Frame size must be positive"


@dataclass
class DecisionConfig:
    """Configuration for the Simplex Decision Module.
    
    Attributes:
        alpha: Target coverage for conformal prediction (1 - confidence)
        lambda_cp: Learning rate for conformal prediction update
        tau_conflict: Threshold for model conflict detection
        critical_rul: RUL threshold for critical failure alert
        degradation_rul: RUL threshold for degradation warning
        epsilon: Small value to avoid division by zero
    """
    alpha: float = 0.05  # 95% confidence
    lambda_cp: float = 0.01
    tau_conflict: float = 20.0
    critical_rul: int = 20
    degradation_rul: int = 50
    epsilon: float = 1e-6
    smoothing_window: int = 10
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert 0 < self.alpha < 1, "Alpha must be in (0, 1)"
        assert self.lambda_cp > 0, "Lambda must be positive"
        assert self.tau_conflict > 0, "Conflict threshold must be positive"


@dataclass
class SAFERConfig:
    """Master configuration for SAFER v3.0 system.
    
    This configuration class aggregates all component configurations
    and provides methods for saving/loading from files.
    """
    mamba: MambaConfig = field(default_factory=MambaConfig)
    lpv_sindy: LPVSINDyConfig = field(default_factory=LPVSINDyConfig)
    fabric: FabricConfig = field(default_factory=FabricConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    
    # Global settings
    data_dir: Path = field(default_factory=lambda: Path("CMAPSSData"))
    model_dir: Path = field(default_factory=lambda: Path("models"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    
    # Deterministic settings
    random_seed: int = 42
    deterministic: bool = True
    
    # Performance targets
    max_inference_latency_ms: float = 20.0
    target_rmse: float = 15.0
    
    def __post_init__(self):
        """Ensure paths are Path objects."""
        self.data_dir = Path(self.data_dir)
        self.model_dir = Path(self.model_dir)
        self.log_dir = Path(self.log_dir)
    
    def validate(self) -> None:
        """Validate all configurations."""
        self.mamba.validate()
        self.lpv_sindy.validate()
        self.fabric.validate()
        self.decision.validate()
    
    def save(self, path: Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Path to save configuration
        """
        path = Path(path)
        config_dict = self._to_dict()
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: Path) -> 'SAFERConfig':
        """Load configuration from YAML file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            SAFERConfig instance
        """
        path = Path(path)
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls._from_dict(config_dict)
    
    def _to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'mamba': {
                'd_input': self.mamba.d_input,
                'd_model': self.mamba.d_model,
                'd_state': self.mamba.d_state,
                'n_layers': self.mamba.n_layers,
                'dropout': self.mamba.dropout,
                'max_rul': self.mamba.max_rul,
                'sequence_length': self.mamba.sequence_length,
                'use_jit': self.mamba.use_jit,
                'learning_rate': self.mamba.learning_rate,
                'weight_decay': self.mamba.weight_decay,
                'batch_size': self.mamba.batch_size,
                'epochs': self.mamba.epochs,
                'patience': self.mamba.patience,
                'sensor_indices': self.mamba.sensor_indices,
            },
            'lpv_sindy': {
                'n_features': self.lpv_sindy.n_features,
                'poly_degree': self.lpv_sindy.poly_degree,
                'include_interactions': self.lpv_sindy.include_interactions,
                'threshold': self.lpv_sindy.threshold,
                'alpha': self.lpv_sindy.alpha,
                'integration_window': self.lpv_sindy.integration_window,
                'residual_threshold': self.lpv_sindy.residual_threshold,
            },
            'fabric': {
                'buffer_size': self.fabric.buffer_size,
                'frame_size': self.fabric.frame_size,
                'shm_name': self.fabric.shm_name,
                'poll_interval_ms': self.fabric.poll_interval_ms,
            },
            'decision': {
                'alpha': self.decision.alpha,
                'lambda_cp': self.decision.lambda_cp,
                'tau_conflict': self.decision.tau_conflict,
                'critical_rul': self.decision.critical_rul,
                'degradation_rul': self.decision.degradation_rul,
            },
            'global': {
                'data_dir': str(self.data_dir),
                'model_dir': str(self.model_dir),
                'log_dir': str(self.log_dir),
                'random_seed': self.random_seed,
                'deterministic': self.deterministic,
                'max_inference_latency_ms': self.max_inference_latency_ms,
                'target_rmse': self.target_rmse,
            }
        }
    
    @classmethod
    def _from_dict(cls, d: dict) -> 'SAFERConfig':
        """Create configuration from dictionary."""
        mamba_cfg = MambaConfig(**d.get('mamba', {}))
        lpv_cfg = LPVSINDyConfig(**d.get('lpv_sindy', {}))
        fabric_cfg = FabricConfig(**d.get('fabric', {}))
        decision_cfg = DecisionConfig(**d.get('decision', {}))
        
        global_cfg = d.get('global', {})
        
        return cls(
            mamba=mamba_cfg,
            lpv_sindy=lpv_cfg,
            fabric=fabric_cfg,
            decision=decision_cfg,
            data_dir=Path(global_cfg.get('data_dir', 'CMAPSSData')),
            model_dir=Path(global_cfg.get('model_dir', 'models')),
            log_dir=Path(global_cfg.get('log_dir', 'logs')),
            random_seed=global_cfg.get('random_seed', 42),
            deterministic=global_cfg.get('deterministic', True),
            max_inference_latency_ms=global_cfg.get('max_inference_latency_ms', 20.0),
            target_rmse=global_cfg.get('target_rmse', 15.0),
        )


# Sensor names for C-MAPSS dataset documentation
CMAPSS_SENSOR_NAMES = {
    0: "T2",      # Total temperature at fan inlet
    1: "T24",     # Total temperature at LPC outlet
    2: "T30",     # Total temperature at HPC outlet
    3: "T50",     # Total temperature at LPT outlet
    4: "P2",      # Pressure at fan inlet
    5: "P15",     # Total pressure in bypass-duct
    6: "P30",     # Total pressure at HPC outlet
    7: "Nf",      # Physical fan speed
    8: "Nc",      # Physical core speed
    9: "epr",     # Engine pressure ratio
    10: "Ps30",   # Static pressure at HPC outlet
    11: "phi",    # Ratio of fuel flow to Ps30
    12: "NRf",    # Corrected fan speed
    13: "NRc",    # Corrected core speed
    14: "BPR",    # Bypass ratio
    15: "farB",   # Burner fuel-air ratio
    16: "htBleed", # Bleed Enthalpy
    17: "Nf_dmd", # Demanded fan speed
    18: "PCNfR_dmd", # Demanded corrected fan speed
    19: "W31",    # HPT coolant bleed
    20: "W32",    # LPT coolant bleed
}

# Operational settings column indices
OP_SETTING_COLS = [2, 3, 4]  # operational settings 1-3

# All sensor column indices (after unit number, time, and op settings)
SENSOR_COLS = list(range(5, 26))

# Selected prognostic sensor indices (14 most informative)
PROGNOSTIC_SENSOR_INDICES = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
