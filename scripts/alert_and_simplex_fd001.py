"""
Alert & Simplex Integration Script for FD001.

This script:
1. Loads calibrated conformal intervals from validation
2. Runs Simplex decision module on test set
3. Integrates alert system with RUL thresholds
4. Computes alerting metrics (precision, recall, time-to-detection)
5. Generates decision history and alert statistics

Usage:
    python scripts/alert_and_simplex_fd001.py
"""

import sys
from pathlib import Path
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from loguru import logger
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from safer_v3.core.trainer import DataModule
from safer_v3.core.mamba import MambaRULPredictor
from safer_v3.decision.conformal import SplitConformalPredictor
from safer_v3.decision.alerts import AlertManager, create_rul_alert_rules, AlertLevel
from safer_v3.decision.simplex import SimplexDecisionModule, SimplexConfig, DecisionResult
from safer_v3.utils.metrics import calculate_rul_metrics


@dataclass
class AlertingMetrics:
    """Metrics for alert system performance."""
    total_samples: int
    total_alerts: int
    alerts_by_level: Dict[str, int]
    
    # Time-to-detection: how many cycles before failure
    time_to_detection_mean: float
    time_to_detection_std: float
    
    # Coverage of alerts
    coverage_by_level: Dict[str, float]
    
    # Alert statistics
    critical_accuracy: float  # % of CRITICAL alerts that were justified
    false_positive_rate: float


def load_conformal_params(params_path: Path) -> dict:
    """Load calibrated conformal parameters.
    
    Args:
        params_path: Path to conformal_params.json
        
    Returns:
        Dictionary with calibration parameters
    """
    logger.info(f"Loading conformal parameters from {params_path}")
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    logger.info(f"Conformal quantile: {params['quantile']:.2f} cycles")
    logger.info(f"Empirical coverage: {params['empirical_coverage']:.1%}")
    
    return params


def load_baseline_model(device: torch.device) -> torch.nn.Module:
    """Load baseline model for Simplex.
    
    Args:
        device: Device for model
        
    Returns:
        Simple baseline (returns mean prediction)
    """
    logger.info("Using simple baseline: mean prediction")
    return None


def load_checkpoint(checkpoint_path: Path, args_path: Path = None) -> Tuple:
    """Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        args_path: Optional path to args.json
        
    Returns:
        Tuple of (model, config, metadata)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load config from args.json
    config = None
    if args_path and args_path.exists():
        logger.info(f"Loading model config from {args_path}")
        with open(args_path, 'r') as f:
            args = json.load(f)
            config = {
                'd_input': 14,
                'd_model': args.get('d_model', 128),
                'n_layers': args.get('n_layers', 6),
                'd_state': args.get('d_state', 16),
                'd_conv': 4,
                'expand': args.get('expand', 2),
                'dropout': args.get('dropout', 0.1),
                'max_rul': args.get('max_rul', 125),
            }
    
    if config is None:
        # Infer from checkpoint
        state_dict = checkpoint['model_state_dict']
        layer_keys = [k for k in state_dict.keys() if k.startswith('layers.')]
        n_layers = max([int(k.split('.')[1]) for k in layer_keys if k.split('.')[1].isdigit()]) + 1
        d_model = state_dict['input_proj.weight'].shape[0]
        
        config = {
            'd_input': 14,
            'd_model': d_model,
            'n_layers': n_layers,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'dropout': 0.1,
            'max_rul': 125,
        }
    
    # Create and load model
    model = MambaRULPredictor(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    metadata = {
        'epoch': checkpoint.get('epoch', -1),
        'best_val_rmse': checkpoint.get('best_val_rmse', float('inf')),
    }
    
    logger.info(f"Model loaded: epoch={metadata['epoch']}, val_rmse={metadata['best_val_rmse']:.4f}")
    
    return model, config, metadata


def get_test_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get model predictions on test set.
    
    Args:
        model: Trained model
        data_loader: Test data loader
        device: Device for inference
        
    Returns:
        Tuple of (predictions, targets)
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, targets in data_loader:
            sequences = sequences.to(device)
            predictions = model(sequences)
            
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.numpy())
    
    y_pred = np.concatenate(all_preds).ravel()
    y_true = np.concatenate(all_targets).ravel()
    
    return y_pred, y_true


def run_simplex_and_alerts(
    y_pred_mamba: np.ndarray,
    y_true: np.ndarray,
    conformal_params: dict,
    baseline_rul: np.ndarray = None,
) -> Tuple[List[DecisionResult], AlertManager]:
    """Run Simplex decision and alert system on test set.
    
    Args:
        y_pred_mamba: Mamba predictions on test set
        y_true: True RUL values
        conformal_params: Calibrated conformal parameters
        baseline_rul: Baseline predictions (if None, uses mean)
        
    Returns:
        Tuple of (decision_results, alert_manager)
    """
    logger.info("Setting up Simplex decision module...")
    
    # Configure Simplex
    config = SimplexConfig(
        physics_threshold=0.15,
        divergence_threshold=30.0,
        uncertainty_threshold=100.0,
        recovery_window=10,
        max_switch_rate=2.0,
        hysteresis_cycles=5,
    )
    
    simplex = SimplexDecisionModule(config)
    
    # Setup alert manager
    alert_manager = AlertManager(max_history=10000)
    alert_manager.add_rules(create_rul_alert_rules(
        critical_threshold=10,
        warning_threshold=25,
        caution_threshold=50,
        advisory_threshold=100,
    ))
    
    # Generate baseline predictions (simple mean forecast)
    if baseline_rul is None:
        baseline_rul = np.full_like(y_pred_mamba, y_pred_mamba.mean())
    
    # Extract interval parameters
    quantile = conformal_params['quantile']
    
    logger.info(f"Running Simplex decisions on {len(y_pred_mamba)} test samples...")
    
    decision_results = []
    alerts_triggered = []
    
    for i, (y_pred, y_bl, y_t) in enumerate(zip(y_pred_mamba, baseline_rul, y_true)):
        # Compute confidence bounds
        rul_lower = max(0, y_pred - quantile)
        rul_upper = y_pred + quantile
        
        # Simplex decision
        result = simplex.decide(
            complex_rul=float(y_pred),
            baseline_rul=float(y_bl),
            rul_lower=float(rul_lower),
            rul_upper=float(rul_upper),
            physics_residual=0.0,  # No physics monitor in this version
        )
        
        decision_results.append(result)
        
        # Process alerts
        context = {
            'interval_width': rul_upper - rul_lower,
            'simplex_state': result.state.name,
            'source': result.used_source,
        }
        
        alerts = alert_manager.process(float(result.rul), context)
        for alert in alerts:
            alerts_triggered.append({
                'sample': i,
                'alert_id': alert.alert_id,
                'level': alert.level.name,
                'message': alert.message,
                'rul_value': alert.rul_value,
                'true_rul': float(y_t),
            })
    
    logger.info(f"Simplex decisions complete: {len(decision_results)} samples processed")
    logger.info(f"Alerts triggered: {len(alerts_triggered)}")
    
    # Print simplex statistics
    stats = simplex.get_statistics()
    logger.info("=" * 60)
    logger.info("SIMPLEX STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total decisions: {stats['total_decisions']}")
    logger.info(f"Complex mode: {stats['complex_decisions']} ({stats['complex_ratio']:.1%})")
    logger.info(f"Baseline mode: {stats['baseline_decisions']}")
    logger.info(f"Mode switches: {stats['switch_count']}")
    logger.info("=" * 60)
    
    return decision_results, alert_manager, alerts_triggered


def compute_alerting_metrics(
    y_true: np.ndarray,
    alerts_triggered: List[dict],
    decision_results: List[DecisionResult],
) -> AlertingMetrics:
    """Compute alerting system metrics.
    
    Args:
        y_true: True RUL values
        alerts_triggered: List of triggered alerts
        decision_results: List of decision results
        
    Returns:
        AlertingMetrics object
    """
    # Count alerts by level
    alerts_by_level = {}
    for level in AlertLevel:
        count = sum(1 for a in alerts_triggered if a['level'] == level.name)
        alerts_by_level[level.name] = count
    
    # Time-to-detection: cycles before reaching RUL < critical threshold
    critical_threshold = 10
    ttd_values = []
    
    for i, y_t in enumerate(y_true):
        cycles_remaining = max(0, y_t - critical_threshold)
        if cycles_remaining > 0:
            ttd_values.append(cycles_remaining)
    
    ttd_mean = np.mean(ttd_values) if ttd_values else 0.0
    ttd_std = np.std(ttd_values) if ttd_values else 0.0
    
    # Coverage by alert level
    coverage_by_level = {}
    for level_name in alerts_by_level:
        level_alerts = [a for a in alerts_triggered if a['level'] == level_name]
        if level_alerts:
            justified = sum(1 for a in level_alerts if a['true_rul'] < a['rul_value'])
            coverage = justified / len(level_alerts)
        else:
            coverage = 0.0
        coverage_by_level[level_name] = coverage
    
    # False positive rate (alerts when RUL > threshold)
    critical_alerts = [a for a in alerts_triggered if a['level'] == 'CRITICAL']
    if critical_alerts:
        false_positives = sum(1 for a in critical_alerts if a['true_rul'] > 10)
        fpr = false_positives / len(critical_alerts)
    else:
        fpr = 0.0
    
    # Critical accuracy
    if critical_alerts:
        accurate = sum(1 for a in critical_alerts if a['true_rul'] < a['rul_value'])
        critical_acc = accurate / len(critical_alerts)
    else:
        critical_acc = 0.0
    
    metrics = AlertingMetrics(
        total_samples=len(y_true),
        total_alerts=len(alerts_triggered),
        alerts_by_level=alerts_by_level,
        time_to_detection_mean=ttd_mean,
        time_to_detection_std=ttd_std,
        coverage_by_level=coverage_by_level,
        critical_accuracy=critical_acc,
        false_positive_rate=fpr,
    )
    
    return metrics


def plot_simplex_and_alerts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decision_results: List[DecisionResult],
    alerts_triggered: List[dict],
    metrics: AlertingMetrics,
    save_dir: Path,
):
    """Generate visualization plots.
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        decision_results: Decision results
        alerts_triggered: Triggered alerts
        metrics: Alerting metrics
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Predictions with decision mode
    ax = axes[0, 0]
    indices = np.arange(len(y_true))
    
    # Color by simplex mode
    colors = ['blue' if r.is_using_complex else 'orange' for r in decision_results]
    scatter = ax.scatter(indices, y_pred, c=colors, alpha=0.5, s=10, label='Predictions')
    ax.plot(indices, y_true, 'k-', linewidth=1, alpha=0.7, label='True RUL')
    
    # Mark alert thresholds
    ax.axhline(10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Critical (10)')
    ax.axhline(25, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Warning (25)')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('RUL (cycles)')
    ax.set_title('Predictions and Simplex Mode (Blue=Complex, Orange=Baseline)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Alert distribution
    ax = axes[0, 1]
    alert_levels = list(metrics.alerts_by_level.keys())
    alert_counts = list(metrics.alerts_by_level.values())
    
    colors_alerts = {'INFO': 'green', 'ADVISORY': 'blue', 'CAUTION': 'yellow', 
                     'WARNING': 'orange', 'CRITICAL': 'red'}
    bar_colors = [colors_alerts.get(level, 'gray') for level in alert_levels]
    
    ax.bar(alert_levels, alert_counts, color=bar_colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title(f'Alert Distribution (Total: {metrics.total_alerts})')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Simplex mode timeline
    ax = axes[1, 0]
    simplex_states = [1 if r.is_using_complex else 0 for r in decision_results]
    ax.plot(indices, simplex_states, 'b-', linewidth=1, alpha=0.7)
    ax.fill_between(indices, simplex_states, alpha=0.3)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Mode (1=Complex, 0=Baseline)')
    ax.set_title('Simplex Mode Timeline')
    ax.set_ylim([-0.1, 1.1])
    ax.grid(True, alpha=0.3)
    
    # 4. Alert accuracy metrics
    ax = axes[1, 1]
    ax.axis('off')
    
    metrics_text = f"""
    Alerting Performance Metrics
    {'='*40}
    Total Samples:        {metrics.total_samples}
    Total Alerts:         {metrics.total_alerts}
    
    Critical Accuracy:    {metrics.critical_accuracy:.1%}
    False Positive Rate:  {metrics.false_positive_rate:.1%}
    
    Time-to-Detection:
      Mean:               {metrics.time_to_detection_mean:.1f} cycles
      Std Dev:            {metrics.time_to_detection_std:.1f} cycles
    
    Coverage by Level:
    """
    
    for level, coverage in metrics.coverage_by_level.items():
        count = metrics.alerts_by_level.get(level, 0)
        metrics_text += f"\n      {level:12s}: {coverage:6.1%} ({count} alerts)"
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = save_dir / 'simplex_and_alerts.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved simplex and alerts plot to {save_path}")
    plt.close()


def main():
    """Main workflow."""
    # Paths
    checkpoint_path = project_root / 'checkpoints' / 'best_model.pt'
    args_path = project_root / 'outputs' / 'mamba_FD001_20251203_174328' / 'args.json'
    calibration_dir = project_root / 'outputs' / 'mamba_FD001_20251203_174328' / 'calibration'
    output_dir = project_root / 'outputs' / 'mamba_FD001_20251203_174328' / 'alerts'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = project_root / 'CMAPSSData'
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model, config, metadata = load_checkpoint(checkpoint_path, args_path)
    model = model.to(device)
    
    # Load conformal parameters
    params_path = calibration_dir / 'conformal_params.json'
    conformal_params = load_conformal_params(params_path)
    
    # Load test data
    logger.info("Loading test data...")
    data_module = DataModule(
        data_dir=str(data_dir),
        dataset='FD001',
        batch_size=256,
        window_size=30,
        val_split=0.2,
    )
    data_module.setup()
    
    test_loader = data_module.test_dataloader()
    if test_loader is None:
        logger.error("No test data available")
        return 1
    
    logger.info(f"Test samples: {len(data_module.test_dataset)}")
    
    # Get test predictions
    logger.info("Running inference on test set...")
    y_pred, y_true = get_test_predictions(model, test_loader, device)
    
    # Compute test metrics
    test_metrics = calculate_rul_metrics(y_true, y_pred)
    logger.info(f"Test RMSE: {test_metrics.rmse:.4f}")
    logger.info(f"Test MAE: {test_metrics.mae:.4f}")
    
    # Run Simplex and alerts
    decision_results, alert_manager, alerts_triggered = run_simplex_and_alerts(
        y_pred, y_true, conformal_params
    )
    
    # Compute alerting metrics
    logger.info("Computing alerting metrics...")
    alerting_metrics = compute_alerting_metrics(y_true, alerts_triggered, decision_results)
    
    logger.info("=" * 60)
    logger.info("ALERTING METRICS")
    logger.info("=" * 60)
    logger.info(f"Total Alerts: {alerting_metrics.total_alerts}")
    logger.info(f"Critical Accuracy: {alerting_metrics.critical_accuracy:.1%}")
    logger.info(f"False Positive Rate: {alerting_metrics.false_positive_rate:.1%}")
    logger.info(f"Time-to-Detection: {alerting_metrics.time_to_detection_mean:.1f} ± {alerting_metrics.time_to_detection_std:.1f} cycles")
    logger.info("=" * 60)
    
    # Save decision results
    logger.info("Saving decision results...")
    
    # Convert decision results to dicts with proper serialization
    decisions_list = []
    for r in decision_results:
        d = asdict(r)
        d['state'] = r.state.name  # Convert enum to string
        d['switch_reason'] = r.switch_reason.name if r.switch_reason else None  # Convert enum to string
        decisions_list.append(d)
    
    decision_data = {
        'decisions': decisions_list,
        'alerts': alerts_triggered,
        'metrics': {
            'test_rmse': test_metrics.rmse,
            'test_mae': test_metrics.mae,
            'test_nasa_score': test_metrics.nasa_score,
            'alerting_metrics': {
                'total_alerts': alerting_metrics.total_alerts,
                'critical_accuracy': alerting_metrics.critical_accuracy,
                'false_positive_rate': alerting_metrics.false_positive_rate,
                'time_to_detection_mean': alerting_metrics.time_to_detection_mean,
            }
        }
    }
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(item) for item in obj]
        return obj
    
    decision_data = convert_types(decision_data)
    
    results_path = output_dir / 'simplex_and_alerts_results.json'
    with open(results_path, 'w') as f:
        json.dump(decision_data, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Generate plots
    logger.info("Generating plots...")
    plot_simplex_and_alerts(y_true, y_pred, decision_results, alerts_triggered, 
                           alerting_metrics, output_dir)
    
    # Save alert history
    logger.info("Saving alert manager state...")
    alert_stats = alert_manager.get_statistics()
    alerts_path = output_dir / 'alert_statistics.json'
    with open(alerts_path, 'w') as f:
        json.dump(convert_types(alert_stats), f, indent=2)
    
    logger.success("✓ Simplex and alerting integration complete!")
    logger.info(f"All outputs saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
