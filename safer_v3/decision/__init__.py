"""
Decision Module Initialization for SAFER v3.0.

This subpackage implements the decision-making components for the
SAFER architecture, including uncertainty quantification and the
Simplex safety switch.

Modules:
- conformal.py: Conformal prediction for uncertainty quantification
- alerts.py: Alert generation and management
- simplex.py: Simplex decision module for safety switching

The decision module provides:
1. Calibrated uncertainty bounds via conformal prediction
2. Multi-level alert system with severity classification
3. Simplex architecture for safety-critical decisions
4. DAL C certification compliance

Architecture:
    Mamba Prediction --> Conformal Calibration --> Uncertainty Bounds
                                |
    SINDy Residuals ----------->+
                                |
                                v
                    Alert Generator --> Severity Classification
                                |
                                v
                    Simplex Switch --> Safe Output
                                |
                         +------+------+
                         |             |
                    [Mamba RUL]  [Baseline RUL]

Safety Classification (DAL C):
- Decision logic is safety-critical per DO-178C
- Conservative switching criteria
- Fail-safe defaults
"""

from safer_v3.decision.conformal import (
    ConformalPredictor,
    SplitConformalPredictor,
    AdaptiveConformalPredictor,
    calibrate_conformal,
)
from safer_v3.decision.alerts import (
    AlertLevel,
    Alert,
    AlertManager,
    AlertRule,
    create_rul_alert_rules,
)
from safer_v3.decision.simplex import (
    SimplexDecisionModule,
    SimplexState,
    DecisionResult,
    SafetyMonitor,
)

__all__ = [
    # Conformal prediction
    'ConformalPredictor',
    'SplitConformalPredictor',
    'AdaptiveConformalPredictor',
    'calibrate_conformal',
    # Alerts
    'AlertLevel',
    'Alert',
    'AlertManager',
    'AlertRule',
    'create_rul_alert_rules',
    # Simplex
    'SimplexDecisionModule',
    'SimplexState',
    'DecisionResult',
    'SafetyMonitor',
]
