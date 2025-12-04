#!/usr/bin/env python3
'''
SAFER v3.0 Inference Example

This script demonstrates how to use the deployed SAFER models
for RUL prediction in production.
'''

import sys
import json
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# Assuming SAFER package is installed
try:
    from safer_v3.core.mamba import MambaRULPredictor
    from safer_v3.core.baselines import LSTMPredictor
    from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
    from safer_v3.decision.conformal import SplitConformalPredictor
    from safer_v3.decision.simplex import SimplexDecisionModule, SimplexConfig
    from safer_v3.decision.alerts import AlertManager, create_rul_alert_rules
except ImportError:
    print("Error: SAFER v3.0 package not found. Install it or run from project root.")
    sys.exit(1)


@dataclass
class SAFERResult:
    '''Result from SAFER inference.'''
    rul: float
    rul_lower: float
    rul_upper: float
    state: str
    alerts: List[str]
    complex_rul: float
    baseline_rul: float
    physics_score: float


class SAFERInference:
    '''
    SAFER v3.0 Inference Wrapper.
    
    Loads all models and provides simple predict() interface.
    '''
    
    def __init__(
        self,
        mamba_checkpoint: str,
        baseline_checkpoint: str,
        physics_model: str,
        conformal_params: str,
        device: str = None,
    ):
        '''Initialize SAFER inference.
        
        Args:
            mamba_checkpoint: Path to Mamba PyTorch checkpoint
            baseline_checkpoint: Path to LSTM baseline checkpoint
            physics_model: Path to LPV-SINDy model (without extension)
            conformal_params: Path to conformal params JSON
            device: Compute device (cuda/cpu)
        '''
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        print(f"Initializing SAFER v3.0 on {device}...")
        
        # Load Mamba
        ckpt = torch.load(mamba_checkpoint, map_location=device, weights_only=False)
        mamba_config = ckpt.get('config', {})
        self.mamba = MambaRULPredictor(
            d_input=mamba_config.get('d_input', 14),
            d_model=mamba_config.get('d_model', 128),
            d_state=mamba_config.get('d_state', 16),
            n_layers=mamba_config.get('n_layers', 6),
        )
        self.mamba.load_state_dict(ckpt['model_state_dict'])
        self.mamba.to(device).eval()
        print("✓ Mamba loaded")
        
        # Load LSTM baseline
        ckpt = torch.load(baseline_checkpoint, map_location=device, weights_only=False)
        baseline_config = ckpt['config']
        self.baseline = LSTMPredictor(
            d_input=baseline_config['d_input'],
            d_model=baseline_config['d_model'],
            n_layers=baseline_config['n_layers'],
            dropout=baseline_config['dropout'],
            bidirectional=baseline_config['bidirectional'],
            max_rul=baseline_config['max_rul'],
        )
        self.baseline.load_state_dict(ckpt['model_state_dict'])
        self.baseline.to(device).eval()
        print("✓ LSTM baseline loaded")
        
        # Load physics monitor
        self.physics = LPVSINDyMonitor.load(physics_model)
        print("✓ LPV-SINDy loaded")
        
        # Load conformal predictor
        with open(conformal_params) as f:
            params = json.load(f)
        self.conformal = SplitConformalPredictor(coverage=0.9, symmetric=True)
        self.conformal._quantile = params['quantile']
        self.conformal._lower_quantile = params.get('lower_quantile', params['quantile'])
        self.conformal._upper_quantile = params.get('upper_quantile', params['quantile'])
        self.conformal._calibrated = True
        print("✓ Conformal predictor loaded")
        
        # Initialize Simplex
        simplex_config = SimplexConfig(
            physics_threshold=3.0,
            divergence_threshold=50.0,
            uncertainty_threshold=100.0,
        )
        self.simplex = SimplexDecisionModule(simplex_config)
        self.simplex.force_complex()  # Start with Mamba
        print("✓ Simplex initialized")
        
        # Initialize alerts
        self.alert_manager = AlertManager()
        self.alert_manager.add_rules(create_rul_alert_rules(
            critical_threshold=10,
            warning_threshold=25,
            caution_threshold=50,
            advisory_threshold=100,
        ))
        print("✓ Alert manager initialized")
        
        print("SAFER v3.0 ready for inference\n")
    
    def predict(self, sequence: np.ndarray) -> SAFERResult:
        '''
        Predict RUL with full SAFER pipeline.
        
        Args:
            sequence: Sensor data, shape (window_size, n_sensors)
                     Expected: (30, 14)
        
        Returns:
            SAFERResult with RUL and metadata
        '''
        # Validate input
        if sequence.shape != (30, 14):
            raise ValueError(f"Expected shape (30, 14), got {sequence.shape}")
        
        # Prepare input
        x = torch.from_numpy(sequence).float().unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            mamba_rul = self.mamba(x).item()
            baseline_rul = self.baseline(x).item()
        
        # Physics check
        try:
            _, physics_score, _ = self.physics.detect_anomaly(sequence)
        except:
            physics_score = 0.0
        
        # Conformal interval
        interval = self.conformal.predict(mamba_rul)
        
        # Simplex decision
        decision = self.simplex.decide(
            complex_rul=mamba_rul,
            baseline_rul=baseline_rul,
            rul_lower=interval.lower,
            rul_upper=interval.upper,
            physics_residual=physics_score,
        )
        
        # Alerts
        alerts = self.alert_manager.process(decision.rul)
        alert_levels = [a.level.name for a in alerts]
        
        return SAFERResult(
            rul=decision.rul,
            rul_lower=interval.lower,
            rul_upper=interval.upper,
            state=decision.state.name,
            alerts=alert_levels,
            complex_rul=mamba_rul,
            baseline_rul=baseline_rul,
            physics_score=physics_score,
        )


def main():
    '''Example usage.'''
    import argparse
    
    parser = argparse.ArgumentParser(description="SAFER v3.0 Inference Example")
    parser.add_argument("--test", action="store_true",
                        help="Run with dummy test data")
    parser.add_argument("--mamba", type=str, default="../models/mamba_rul.pt")
    parser.add_argument("--baseline", type=str, default="../models/lstm_baseline.pt")
    parser.add_argument("--physics", type=str, default="../models/lpv_sindy_model")
    parser.add_argument("--conformal", type=str, default="../config/conformal_params.json")
    
    args = parser.parse_args()
    
    # Initialize
    safer = SAFERInference(
        mamba_checkpoint=args.mamba,
        baseline_checkpoint=args.baseline,
        physics_model=args.physics,
        conformal_params=args.conformal,
    )
    
    if args.test:
        print("Running test with dummy data...\n")
        # Generate dummy sensor data
        test_sequence = np.random.randn(30, 14).astype(np.float32)
        
        result = safer.predict(test_sequence)
        
        print("=" * 60)
        print("SAFER v3.0 Prediction Result")
        print("=" * 60)
        print(f"RUL Prediction: {result.rul:.1f} cycles")
        print(f"90% Confidence: [{result.rul_lower:.1f}, {result.rul_upper:.1f}]")
        print(f"Simplex State: {result.state}")
        print(f"Mamba RUL: {result.complex_rul:.1f}")
        print(f"LSTM RUL: {result.baseline_rul:.1f}")
        print(f"Physics Score: {result.physics_score:.3f}")
        print(f"Alerts: {result.alerts if result.alerts else 'None'}")
        print("=" * 60)


if __name__ == "__main__":
    main()
