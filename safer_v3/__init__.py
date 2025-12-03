"""
SAFER v3.0: Safety-Aware Flexible Emulation for Reliability

A tri-partite architecture for aerospace turbofan engine prognostics:
1. Prognostic Core: Mamba-based Selective State Space Model for RUL prediction
2. Physics Monitor: Adaptive LPV-SINDy subsystem for physics validation
3. Simulation Fabric: Shared-memory multiprocessing transport

Copyright (c) 2024 SAFER Development Team
Licensed under the MIT License
"""

__version__ = "3.0.0"
__author__ = "SAFER Development Team"

from safer_v3.utils.config import SAFERConfig
from safer_v3.core.mamba import MambaRULPredictor
from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
from safer_v3.decision.simplex import SimplexDecisionModule

__all__ = [
    "SAFERConfig",
    "MambaRULPredictor",
    "LPVSINDyMonitor",
    "SimplexDecisionModule",
]
