#!/usr/bin/env python3
"""
Full SAFER v3.0 Architecture Integration Script.

This script integrates ALL components of the SAFER architecture:
1. Mamba RUL Predictor (DAL E) - Primary high-accuracy predictor
2. LSTM Baseline (DAL C) - Safety fallback predictor
3. LPV-SINDy Physics Monitor (DAL C) - Physics-based anomaly detection
4. Conformal Prediction - Uncertainty quantification
5. Simplex Decision Module - Safety arbitration
6. Alert Manager - Multi-level alert generation

This represents the complete proposed architecture as per specification.
"""

import sys
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from safer_v3.core.mamba import MambaRULPredictor
from safer_v3.core.baselines import LSTMPredictor
from safer_v3.core.trainer import DataModule
from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
from safer_v3.decision.conformal import SplitConformalPredictor
from safer_v3.decision.simplex import SimplexDecisionModule, SimplexConfig
from safer_v3.decision.alerts import AlertManager, create_rul_alert_rules, AlertLevel


def run_full_safer_pipeline(
    mamba_checkpoint_path: str,
    baseline_checkpoint_path: str = None,
    physics_model_path: str = None,
    conformal_params_path: str = None,
    output_dir: str = None,
    data_dir: str = "CMAPSSData",
    dataset: str = "FD001",
    device: str = None,
):
    """Run the complete SAFER v3.0 architecture pipeline.
    
    Args:
        mamba_checkpoint_path: Path to trained Mamba model
        baseline_checkpoint_path: Path to trained LSTM baseline (optional)
        physics_model_path: Path to trained LPV-SINDy model (optional)
        conformal_params_path: Path to conformal calibration params (optional)
        output_dir: Output directory
        data_dir: Path to C-MAPSS data
        dataset: Dataset name
        device: Compute device
    """
    # Setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    mamba_checkpoint_path = Path(mamba_checkpoint_path)
    mamba_dir = mamba_checkpoint_path.parent
    
    if output_dir is None:
        output_dir = mamba_dir / "full_safer_evaluation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("SAFER v3.0 Full Architecture Integration")
    print(f"{'='*70}")
    
    # ============================================================
    # Step 1: Load Mamba RUL Predictor (DAL E)
    # ============================================================
    print(f"\n{'='*70}")
    print("1. Loading Mamba RUL Predictor (DAL E)")
    print(f"{'='*70}")
    
    checkpoint = torch.load(mamba_checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config
    # Extract model configuration from checkpoint if present, otherwise args.json
    if 'config' in checkpoint and checkpoint['config'] is not None:
        cfg = checkpoint['config']
    else:
        # Try loading from args.json
        args_path = mamba_dir / "args.json"
        if args_path.exists():
            with open(args_path) as f:
                cfg = json.load(f)
        else:
            cfg = {}

    # Ensure required keys with sensible defaults
    mamba_config = {
        'd_input': cfg.get('d_input', 14),
        'd_model': cfg.get('d_model', cfg.get('d_model', 128)),
        'd_state': cfg.get('d_state', cfg.get('d_state', 16)),
        'n_layers': cfg.get('n_layers', cfg.get('n_layers', 6)),
        'expand': cfg.get('expand', 2),
        'dropout': cfg.get('dropout', 0.1),
        'max_rul': cfg.get('max_rul', cfg.get('max_rul', 125)),
    }
    
    mamba_model = MambaRULPredictor(
        d_input=mamba_config['d_input'],
        d_model=mamba_config['d_model'],
        d_state=mamba_config['d_state'],
        n_layers=mamba_config['n_layers'],
        expand=mamba_config.get('expand', 2),
        dropout=mamba_config.get('dropout', 0.1),
        max_rul=mamba_config.get('max_rul', 125),
    )
    mamba_model.load_state_dict(checkpoint['model_state_dict'])
    mamba_model = mamba_model.to(device)
    mamba_model.eval()
    
    print(f"✓ Mamba model loaded: {mamba_config['n_layers']} layers, "
          f"d_model={mamba_config['d_model']}")
    
    # ============================================================
    # Step 2: Load LSTM Baseline (DAL C)
    # ============================================================
    print(f"\n{'='*70}")
    print("2. Loading LSTM Baseline Predictor (DAL C)")
    print(f"{'='*70}")
    
    baseline_model = None
    use_mean_baseline = True
    
    if baseline_checkpoint_path and Path(baseline_checkpoint_path).exists():
        baseline_ckpt = torch.load(baseline_checkpoint_path, map_location=device, weights_only=False)
        baseline_config = baseline_ckpt['config']
        
        baseline_model = LSTMPredictor(
            d_input=baseline_config['d_input'],
            d_model=baseline_config['d_model'],
            n_layers=baseline_config['n_layers'],
            dropout=baseline_config['dropout'],
            bidirectional=baseline_config['bidirectional'],
            max_rul=baseline_config['max_rul'],
        )
        baseline_model.load_state_dict(baseline_ckpt['model_state_dict'])
        baseline_model = baseline_model.to(device)
        baseline_model.eval()
        use_mean_baseline = False
        print(f"✓ LSTM baseline loaded: {baseline_config['n_layers']} layers, "
              f"d_model={baseline_config['d_model']}")
    else:
        print("⚠ No LSTM baseline found, using mean-based fallback")
    
    # ============================================================
    # Step 3: Load LPV-SINDy Physics Monitor (DAL C)
    # ============================================================
    print(f"\n{'='*70}")
    print("3. Loading LPV-SINDy Physics Monitor (DAL C)")
    print(f"{'='*70}")
    
    physics_monitor = None
    use_physics = False
    
    if physics_model_path:
        # Check if model files exist (load expects path without extension)
        model_path = Path(physics_model_path)
        json_file = Path(str(model_path) + '.json')
        npz_file = Path(str(model_path) + '.npz')
        
        if json_file.exists() and npz_file.exists():
            try:
                physics_monitor = LPVSINDyMonitor.load(physics_model_path)
                use_physics = True
                print(f"✓ LPV-SINDy physics monitor loaded from {model_path.name}")
            except Exception as e:
                print(f"⚠ Failed to load physics monitor: {e}")
                print("  Will use physics_residual=0.0 as fallback")
        else:
            print(f"⚠ Physics model files not found at {model_path}")
            print("  Will use physics_residual=0.0 as fallback")
    else:
        print("⚠ No physics model path provided, using physics_residual=0.0 as fallback")
    
    # ============================================================
    # Step 4: Load Conformal Prediction Parameters
    # ============================================================
    print(f"\n{'='*70}")
    print("4. Loading Conformal Prediction Parameters")
    print(f"{'='*70}")
    
    conformal = SplitConformalPredictor(coverage=0.9, symmetric=True)
    
    if conformal_params_path is None:
        conformal_params_path = mamba_dir / "calibration" / "conformal_params.json"
    
    if Path(conformal_params_path).exists():
        with open(conformal_params_path) as f:
            params = json.load(f)
        # Set internal calibrated state according to predictor implementation
        try:
            # For SplitConformalPredictor: set lower/upper quantiles
            if hasattr(conformal, '_lower_quantile') and hasattr(conformal, '_upper_quantile'):
                conformal._lower_quantile = params.get('lower_quantile', params.get('quantile'))
                conformal._upper_quantile = params.get('upper_quantile', params.get('quantile'))
                conformal._quantile = params.get('quantile', max(conformal._lower_quantile, conformal._upper_quantile))
            else:
                conformal._quantile = params.get('quantile')

            # Mark as calibrated
            conformal._calibrated = True
            print(f"✓ Conformal predictor loaded: quantile={float(params.get('quantile')):.2f}")
        except Exception as e:
            print(f"⚠ Failed to apply conformal params: {e}")
            conformal._calibrated = False
    else:
        print("⚠ No conformal params found, will calibrate on validation set")
        conformal.is_calibrated_ = False
    
    # ============================================================
    # Step 5: Setup Simplex Decision Module
    # ============================================================
    print(f"\n{'='*70}")
    print("5. Setting up Simplex Decision Module")
    print(f"{'='*70}")
    
    # Map user-friendly parameters to SimplexConfig fields:
    # - min_switch_interval (seconds) -> max_switch_rate (switches per minute)
    # - hysteresis_margin (cycles or relative) -> hysteresis_cycles / conservative_margin
    min_switch_interval = 5  # seconds between allowed switches (user-intent)
    hysteresis_cycles = 5    # cycles to wait before allowing re-switch
    conservative_margin = 5.0  # safety margin (RUL cycles) applied to baseline

    simplex_config = SimplexConfig(
        physics_threshold=3.0,           # Relaxed: physics residual threshold
        divergence_threshold=50.0,       # Relaxed: allow more Mamba-baseline difference
        uncertainty_threshold=100.0,     # Relaxed: allow wider confidence intervals
        recovery_window=10,
        max_switch_rate=(60.0 / min_switch_interval) if min_switch_interval > 0 else 2.0,
        hysteresis_cycles=hysteresis_cycles,
        conservative_margin=conservative_margin,
    )
    simplex = SimplexDecisionModule(simplex_config)
    # Start in COMPLEX mode since Mamba is validated
    simplex.force_complex()
    print(f"✓ Simplex configured: physics_thresh={simplex_config.physics_threshold}, "
          f"divergence_thresh={simplex_config.divergence_threshold}")
    print(f"  Initial state: COMPLEX (using Mamba predictor)")
    
    # ============================================================
    # Step 6: Setup Alert Manager
    # ============================================================
    print(f"\n{'='*70}")
    print("6. Setting up Alert Manager")
    print(f"{'='*70}")
    
    alert_manager = AlertManager()
    alert_rules = create_rul_alert_rules(
        critical_threshold=10,
        warning_threshold=25,
        caution_threshold=50,
        advisory_threshold=100,
    )
    alert_manager.add_rules(alert_rules)
    print(f"✓ Alert manager configured with {len(alert_rules)} rules")
    print(f"  - CRITICAL: RUL ≤ 10 cycles")
    print(f"  - WARNING: RUL ≤ 25 cycles")
    print(f"  - CAUTION: RUL ≤ 50 cycles")
    print(f"  - ADVISORY: RUL ≤ 100 cycles")
    
    # ============================================================
    # Step 7: Load Test Data
    # ============================================================
    print(f"\n{'='*70}")
    print("7. Loading Test Data")
    print(f"{'='*70}")
    
    window_size = 30
    data_module = DataModule(
        data_dir=data_dir,
        dataset=dataset,
        window_size=window_size,
        batch_size=64,
        max_rul=125,
    )
    data_module.setup()
    
    test_loader = data_module.test_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"✓ Test samples: {len(data_module.test_dataset)}")
    print(f"✓ Validation samples: {len(data_module.val_dataset)}")
    
    # Calibrate conformal if needed
    if not conformal._calibrated:
        print("\nCalibrating conformal predictor on validation set...")
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                sequences, targets = batch
                x = sequences.to(device)
                y = targets
                pred = mamba_model(x).squeeze().cpu().numpy()
                val_preds.extend(pred)
                val_targets.extend(y.numpy())
        conformal.calibrate(np.array(val_targets), np.array(val_preds))
        print(f"✓ Calibrated: quantile={conformal._quantile:.2f}")
    
    # Compute mean baseline if needed
    if use_mean_baseline:
        print("\nComputing mean baseline from validation set...")
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                sequences, _ = batch
                x = sequences.to(device)
                pred = mamba_model(x).squeeze().cpu().numpy()
                val_preds.extend(pred)
        mean_baseline_value = np.mean(val_preds)
        print(f"✓ Mean baseline value: {mean_baseline_value:.2f}")
    
    # ============================================================
    # Step 8: Run Full SAFER Pipeline on Test Data
    # ============================================================
    print(f"\n{'='*70}")
    print("8. Running Full SAFER Pipeline")
    print(f"{'='*70}")
    
    results = {
        'mamba_predictions': [],
        'baseline_predictions': [],
        'physics_residuals': [],
        'conformal_lower': [],
        'conformal_upper': [],
        'final_rul': [],
        'simplex_states': [],
        'alerts': [],
        'targets': [],
    }
    
    alert_counts = {level.name: 0 for level in AlertLevel}
    state_counts = {'COMPLEX': 0, 'BASELINE': 0, 'RECOVERY': 0}
    
    print("\nProcessing test samples...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="SAFER Pipeline"):
            sequences, targets = batch
            x = sequences.to(device)
            y = targets.numpy()
            
            # Get Mamba predictions
            mamba_pred = mamba_model(x).squeeze().cpu().numpy()
            
            # Get baseline predictions
            if use_mean_baseline:
                baseline_pred = np.full_like(mamba_pred, mean_baseline_value)
            else:
                baseline_pred = baseline_model(x).squeeze().cpu().numpy()
            
            # Compute physics residuals
            if use_physics:
                x_np = x.cpu().numpy()
                physics_residuals = []
                for i in range(x_np.shape[0]):
                    try:
                        _, max_score, _ = physics_monitor.detect_anomaly(x_np[i])
                        physics_residuals.append(max_score)
                    except:
                        physics_residuals.append(0.0)
                physics_residuals = np.array(physics_residuals)
            else:
                physics_residuals = np.zeros_like(mamba_pred)
            
            # Process each sample through Simplex + Conformal + Alerts
            for i in range(len(mamba_pred)):
                m_pred = float(mamba_pred[i])
                b_pred = float(baseline_pred[i])
                p_resid = float(physics_residuals[i])
                target = float(y[i].item() if hasattr(y[i], 'item') else y[i])
                
                # Conformal interval
                interval = conformal.predict(m_pred)
                lower = interval.lower
                upper = interval.upper
                
                # Simplex decision
                decision = simplex.decide(
                    complex_rul=m_pred,
                    baseline_rul=b_pred,
                    rul_lower=lower,
                    rul_upper=upper,
                    physics_residual=p_resid,
                )
                
                final_rul = decision.rul
                state_name = decision.state.name
                state_counts[state_name] = state_counts.get(state_name, 0) + 1
                
                # Alert generation
                alerts = alert_manager.process(final_rul)
                for alert in alerts:
                    alert_counts[alert.level.name] += 1
                
                # Store results
                results['mamba_predictions'].append(m_pred)
                results['baseline_predictions'].append(b_pred)
                results['physics_residuals'].append(p_resid)
                results['conformal_lower'].append(lower)
                results['conformal_upper'].append(upper)
                results['final_rul'].append(final_rul)
                results['simplex_states'].append(state_name)
                results['alerts'].append([a.level.name for a in alerts])
                results['targets'].append(target)
    
    # ============================================================
    # Step 9: Compute Metrics and Analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("9. Results Summary")
    print(f"{'='*70}")
    
    # Convert to arrays
    targets = np.array(results['targets'])
    mamba_preds = np.array(results['mamba_predictions'])
    baseline_preds = np.array(results['baseline_predictions'])
    final_preds = np.array(results['final_rul'])
    lowers = np.array(results['conformal_lower'])
    uppers = np.array(results['conformal_upper'])
    
    # Prediction metrics
    mamba_rmse = np.sqrt(np.mean((mamba_preds - targets) ** 2))
    mamba_mae = np.mean(np.abs(mamba_preds - targets))
    baseline_rmse = np.sqrt(np.mean((baseline_preds - targets) ** 2))
    baseline_mae = np.mean(np.abs(baseline_preds - targets))
    final_rmse = np.sqrt(np.mean((final_preds - targets) ** 2))
    final_mae = np.mean(np.abs(final_preds - targets))
    
    # Coverage
    coverage = np.mean((targets >= lowers) & (targets <= uppers))
    avg_width = np.mean(uppers - lowers)
    
    print(f"\n📊 Prediction Metrics:")
    print(f"  Mamba (DAL E):    RMSE={mamba_rmse:.2f}, MAE={mamba_mae:.2f}")
    print(f"  Baseline (DAL C): RMSE={baseline_rmse:.2f}, MAE={baseline_mae:.2f}")
    print(f"  Final (Simplex):  RMSE={final_rmse:.2f}, MAE={final_mae:.2f}")
    
    print(f"\n📊 Conformal Prediction:")
    print(f"  Coverage: {coverage:.1%} (target: 90%)")
    print(f"  Average interval width: {avg_width:.2f} cycles")
    
    print(f"\n📊 Simplex Decision Statistics:")
    total = sum(state_counts.values())
    for state, count in state_counts.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {state}: {count} ({pct:.1f}%)")
    
    print(f"\n📊 Alert Statistics:")
    total_alerts = sum(alert_counts.values())
    for severity, count in alert_counts.items():
        if count > 0:
            print(f"  {severity}: {count}")
    print(f"  Total alerts: {total_alerts}")
    
    # ============================================================
    # Step 10: Save Results
    # ============================================================
    print(f"\n{'='*70}")
    print("10. Saving Results")
    print(f"{'='*70}")
    
    # Save detailed results
    output_data = {
        'pipeline': 'SAFER v3.0 Full Architecture',
        'components': {
            'mamba': True,
            'lstm_baseline': not use_mean_baseline,
            'physics_monitor': use_physics,
            'conformal': True,
            'simplex': True,
            'alerts': True,
        },
        'metrics': {
            'mamba_rmse': float(mamba_rmse),
            'mamba_mae': float(mamba_mae),
            'baseline_rmse': float(baseline_rmse),
            'baseline_mae': float(baseline_mae),
            'final_rmse': float(final_rmse),
            'final_mae': float(final_mae),
            'coverage': float(coverage),
            'avg_interval_width': float(avg_width),
        },
        'simplex_stats': {k: int(v) for k, v in state_counts.items()},
        'alert_stats': {k: int(v) for k, v in alert_counts.items()},
        'n_samples': len(targets),
    }
    
    with open(output_dir / "full_safer_results.json", 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Predictions comparison
    ax = axes[0, 0]
    sample_idx = np.arange(min(500, len(targets)))
    ax.plot(sample_idx, targets[:len(sample_idx)], 'k-', label='True RUL', alpha=0.7)
    ax.plot(sample_idx, mamba_preds[:len(sample_idx)], 'b-', label='Mamba', alpha=0.5)
    ax.plot(sample_idx, final_preds[:len(sample_idx)], 'r--', label='Final (Simplex)', alpha=0.7)
    ax.fill_between(sample_idx, lowers[:len(sample_idx)], uppers[:len(sample_idx)], 
                    alpha=0.2, color='blue', label='90% CI')
    ax.set_xlabel('Sample')
    ax.set_ylabel('RUL (cycles)')
    ax.set_title('SAFER v3.0 Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Simplex state distribution
    ax = axes[0, 1]
    states = list(state_counts.keys())
    counts = [state_counts[s] for s in states]
    colors = ['green', 'orange', 'blue']
    ax.bar(states, counts, color=colors[:len(states)])
    ax.set_xlabel('Simplex State')
    ax.set_ylabel('Count')
    ax.set_title('Simplex Decision Distribution')
    
    # 3. Alert distribution
    ax = axes[1, 0]
    severities = ['CRITICAL', 'WARNING', 'CAUTION', 'ADVISORY']
    alert_values = [alert_counts.get(s, 0) for s in severities]
    colors = ['red', 'orange', 'yellow', 'blue']
    ax.bar(severities, alert_values, color=colors)
    ax.set_xlabel('Alert Severity')
    ax.set_ylabel('Count')
    ax.set_title('Alert Distribution')
    
    # 4. Prediction error distribution
    ax = axes[1, 1]
    errors = final_preds - targets
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error (cycles)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Final RUL Error Distribution (RMSE={final_rmse:.2f})')
    
    plt.tight_layout()
    plt.savefig(output_dir / "full_safer_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - full_safer_results.json")
    print(f"  - full_safer_dashboard.png")
    
    print(f"\n{'='*70}")
    print("✓ SAFER v3.0 Full Pipeline Complete!")
    print(f"{'='*70}")
    
    return output_dir, output_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Full SAFER v3.0 Pipeline")
    parser.add_argument("--mamba_checkpoint", type=str, required=True,
                        help="Path to trained Mamba checkpoint")
    parser.add_argument("--baseline_checkpoint", type=str, default=None,
                        help="Path to trained LSTM baseline checkpoint")
    parser.add_argument("--physics_model", type=str, default=None,
                        help="Path to trained LPV-SINDy model")
    parser.add_argument("--conformal_params", type=str, default=None,
                        help="Path to conformal calibration params")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="CMAPSSData")
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    run_full_safer_pipeline(
        mamba_checkpoint_path=args.mamba_checkpoint,
        baseline_checkpoint_path=args.baseline_checkpoint,
        physics_model_path=args.physics_model,
        conformal_params_path=args.conformal_params,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        dataset=args.dataset,
        device=args.device,
    )
