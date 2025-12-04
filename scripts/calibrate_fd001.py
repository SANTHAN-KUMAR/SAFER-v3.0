"""
Conformal Calibration Script for FD001 Model.

This script:
1. Loads the trained checkpoint
2. Runs inference on validation split
3. Calibrates conformal prediction intervals
4. Evaluates coverage and saves calibration parameters
5. Generates diagnostic plots

Usage:
    python scripts/calibrate_fd001.py
"""

import sys
from pathlib import Path
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from safer_v3.core.trainer import DataModule
from safer_v3.core.mamba import MambaRULPredictor
from safer_v3.decision.conformal import SplitConformalPredictor, ConformalResult
from safer_v3.utils.metrics import calculate_rul_metrics


def load_checkpoint(checkpoint_path: Path, args_path: Path = None) -> tuple:
    """Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        args_path: Optional path to args.json with training config
        
    Returns:
        Tuple of (model, config, metadata)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to load model config from args.json first
    config = None
    if args_path and args_path.exists():
        logger.info(f"Loading model config from {args_path}")
        with open(args_path, 'r') as f:
            args = json.load(f)
            config = {
                'd_input': 14,  # CMAPSS has 14 sensors
                'd_model': args.get('d_model', 128),
                'n_layers': args.get('n_layers', 6),
                'd_state': args.get('d_state', 16),
                'd_conv': 4,
                'expand': args.get('expand', 2),
                'dropout': args.get('dropout', 0.1),
                'max_rul': args.get('max_rul', 125),
            }
    
    # Fallback: infer from checkpoint state_dict
    if config is None:
        logger.warning("No args.json found, inferring config from checkpoint")
        state_dict = checkpoint['model_state_dict']
        
        # Count layers
        layer_keys = [k for k in state_dict.keys() if k.startswith('layers.')]
        n_layers = max([int(k.split('.')[1]) for k in layer_keys if k.split('.')[1].isdigit()]) + 1
        
        # Get d_model from input_proj
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
        logger.info(f"Inferred config: d_model={d_model}, n_layers={n_layers}")
    
    # Create model
    model = MambaRULPredictor(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    metadata = {
        'epoch': checkpoint.get('epoch', -1),
        'best_val_rmse': checkpoint.get('best_val_rmse', float('inf')),
    }
    
    logger.info(f"Model loaded: epoch={metadata['epoch']}, val_rmse={metadata['best_val_rmse']:.4f}")
    
    return model, config, metadata


def get_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    """Get model predictions on dataset.
    
    Args:
        model: Trained model
        data_loader: Data loader
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


def plot_calibration_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    results: list,
    coverage_metrics: dict,
    save_dir: Path,
):
    """Generate calibration diagnostic plots.
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        results: List of ConformalResult objects
        coverage_metrics: Coverage metrics dictionary
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Prediction intervals vs true values
    ax = axes[0, 0]
    indices = np.arange(len(y_true))
    sample_indices = np.linspace(0, len(y_true)-1, min(500, len(y_true)), dtype=int)
    
    ax.scatter(sample_indices, y_true[sample_indices], alpha=0.5, s=20, label='True RUL', color='black')
    ax.scatter(sample_indices, y_pred[sample_indices], alpha=0.5, s=20, label='Predicted RUL', color='blue')
    
    lower = np.array([r.lower for r in results])
    upper = np.array([r.upper for r in results])
    ax.fill_between(sample_indices, lower[sample_indices], upper[sample_indices], 
                     alpha=0.2, color='blue', label='90% Prediction Interval')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('RUL (cycles)')
    ax.set_title('Prediction Intervals on Validation Set')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Coverage by RUL bins
    ax = axes[0, 1]
    rul_bins = np.linspace(0, y_true.max(), 10)
    bin_coverage = []
    bin_centers = []
    
    for i in range(len(rul_bins)-1):
        mask = (y_true >= rul_bins[i]) & (y_true < rul_bins[i+1])
        if mask.sum() > 0:
            covered = np.array([
                r.lower <= yt <= r.upper
                for r, yt in zip(np.array(results)[mask], y_true[mask])
            ])
            bin_coverage.append(covered.mean())
            bin_centers.append((rul_bins[i] + rul_bins[i+1]) / 2)
    
    ax.plot(bin_centers, bin_coverage, marker='o', linewidth=2)
    ax.axhline(0.9, color='red', linestyle='--', label='Target Coverage (90%)')
    ax.axhline(coverage_metrics['empirical_coverage'], color='green', 
               linestyle='--', label=f"Overall Coverage ({coverage_metrics['empirical_coverage']:.1%})")
    ax.set_xlabel('RUL Bin Center (cycles)')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title('Coverage by RUL Range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.7, 1.0])
    
    # 3. Interval width distribution
    ax = axes[1, 0]
    widths = np.array([r.width for r in results])
    ax.hist(widths, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(coverage_metrics['average_width'], color='red', 
               linestyle='--', linewidth=2, label=f"Mean Width: {coverage_metrics['average_width']:.2f}")
    ax.set_xlabel('Interval Width (cycles)')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Interval Width Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Residual vs interval width
    ax = axes[1, 1]
    residuals = np.abs(y_true - y_pred)
    ax.scatter(widths, residuals, alpha=0.3, s=10)
    ax.plot([widths.min(), widths.max()], [widths.min()/2, widths.max()/2], 
            'r--', label='Half-width line')
    ax.set_xlabel('Interval Width (cycles)')
    ax.set_ylabel('Absolute Residual (cycles)')
    ax.set_title('Residual vs Interval Width')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / 'calibration_diagnostics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved calibration plots to {save_path}")
    plt.close()


def main():
    """Main calibration workflow."""
    # Configuration
    checkpoint_path = project_root / 'checkpoints' / 'best_model.pt'
    args_path = project_root / 'outputs' / 'mamba_FD001_20251203_174328' / 'args.json'
    output_dir = project_root / 'outputs' / 'mamba_FD001_20251203_174328' / 'calibration'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = project_root / 'CMAPSSData'
    dataset = 'FD001'
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1
    
    model, config, metadata = load_checkpoint(checkpoint_path, args_path)
    model = model.to(device)
    
    # Load data
    logger.info("Loading validation data...")
    data_module = DataModule(
        data_dir=str(data_dir),
        dataset=dataset,
        batch_size=256,
        window_size=30,
        val_split=0.2,
    )
    data_module.setup()
    
    val_loader = data_module.val_dataloader()
    logger.info(f"Validation samples: {len(data_module.val_dataset)}")
    
    # Get predictions
    logger.info("Running inference on validation set...")
    start_time = time.time()
    y_pred, y_true = get_predictions(model, val_loader, device)
    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.2f}s")
    
    # Compute point metrics
    point_metrics = calculate_rul_metrics(y_true, y_pred)
    logger.info(f"Validation RMSE: {point_metrics.rmse:.4f}")
    logger.info(f"Validation MAE: {point_metrics.mae:.4f}")
    logger.info(f"NASA Score: {point_metrics.nasa_score:.2f}")
    
    # Calibrate conformal predictor
    logger.info("Calibrating conformal predictor (90% coverage)...")
    predictor = SplitConformalPredictor(
        coverage=0.9,
        symmetric=True,  # Use symmetric intervals
        normalize_scores=False,
    )
    
    predictor.calibrate(y_true, y_pred)
    
    # Evaluate coverage
    logger.info("Evaluating coverage...")
    coverage_metrics = predictor.evaluate_coverage(y_true, y_pred)
    
    logger.info("=" * 60)
    logger.info("CALIBRATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Target Coverage:     {coverage_metrics['target_coverage']:.1%}")
    logger.info(f"Empirical Coverage:  {coverage_metrics['empirical_coverage']:.1%}")
    logger.info(f"Coverage Gap:        {coverage_metrics['coverage_gap']:.2%}")
    logger.info(f"Average Width:       {coverage_metrics['average_width']:.2f} cycles")
    logger.info(f"Quantile:            {coverage_metrics['quantile']:.2f} cycles")
    logger.info(f"Validation Samples:  {coverage_metrics['n_samples']}")
    logger.info("=" * 60)
    
    # Generate prediction intervals for all validation samples
    results = predictor.predict(y_pred)
    
    # Save calibration parameters
    calibration_params = {
        'coverage': predictor.coverage,
        'quantile': float(predictor._quantile),
        'lower_quantile': float(predictor._lower_quantile),
        'upper_quantile': float(predictor._upper_quantile),
        'symmetric': predictor.symmetric,
        'normalize_scores': predictor.normalize_scores,
        'calibration_samples': len(y_true),
        'empirical_coverage': coverage_metrics['empirical_coverage'],
        'average_width': coverage_metrics['average_width'],
        'point_metrics': {
            'rmse': point_metrics.rmse,
            'mae': point_metrics.mae,
            'nasa_score': point_metrics.nasa_score,
            'r2': point_metrics.r2,
        },
        'metadata': metadata,
    }
    
    params_path = output_dir / 'conformal_params.json'
    with open(params_path, 'w') as f:
        json.dump(calibration_params, f, indent=2)
    logger.info(f"Saved calibration parameters to {params_path}")
    
    # Save example predictions with intervals
    n_examples = min(100, len(results))
    examples = []
    for i in range(n_examples):
        examples.append({
            'true_rul': float(y_true[i]),
            'predicted_rul': float(y_pred[i]),
            'lower_bound': float(results[i].lower),
            'upper_bound': float(results[i].upper),
            'interval_width': float(results[i].width),
            'covered': bool(results[i].lower <= y_true[i] <= results[i].upper),
        })
    
    examples_path = output_dir / 'example_intervals.json'
    with open(examples_path, 'w') as f:
        json.dump(examples, f, indent=2)
    logger.info(f"Saved {n_examples} example intervals to {examples_path}")
    
    # Generate diagnostic plots
    logger.info("Generating calibration plots...")
    plot_calibration_diagnostics(y_true, y_pred, results, coverage_metrics, output_dir)
    
    # Save all intervals for downstream use
    intervals_data = {
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'lower_bounds': [r.lower for r in results],
        'upper_bounds': [r.upper for r in results],
    }
    
    intervals_path = output_dir / 'validation_intervals.json'
    with open(intervals_path, 'w') as f:
        json.dump(intervals_data, f)
    logger.info(f"Saved all validation intervals to {intervals_path}")
    
    logger.success("✓ Conformal calibration complete!")
    logger.info(f"All outputs saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
