"""
Core modules for SAFER v3.0 Prognostic System.

This package contains:
- SSM operations (discretization, parallel scan)
- Mamba architecture implementation
- Baseline models (LSTM, Transformer) for benchmarking
- Training infrastructure
"""

from safer_v3.core.ssm_ops import (
    discretize_zoh,
    parallel_selective_scan,
    recurrent_step,
    SSMKernel,
)
from safer_v3.core.mamba import (
    RMSNorm,
    MambaBlock,
    MambaRULPredictor,
)
from safer_v3.core.baselines import (
    LSTMPredictor,
    TransformerPredictor,
    CNNLSTMPredictor,
)
from safer_v3.core.trainer import (
    CMAPSSDataset,
    Trainer,
)

__all__ = [
    # SSM Operations
    "discretize_zoh",
    "parallel_selective_scan",
    "recurrent_step",
    "SSMKernel",
    # Mamba
    "RMSNorm",
    "MambaBlock",
    "MambaRULPredictor",
    # Baselines
    "LSTMPredictor",
    "TransformerPredictor",
    "CNNLSTMPredictor",
    # Training
    "CMAPSSDataset",
    "Trainer",
]
