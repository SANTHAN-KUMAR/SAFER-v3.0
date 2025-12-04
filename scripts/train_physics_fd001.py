#!/usr/bin/env python3
"""
Train LPV-SINDy Physics Monitor for SAFER v3.0.

This script trains the physics-based anomaly detection model that provides
independent verification of neural network predictions.

DAL C Classification: Physics monitor is part of the safety-critical path.
"""

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
from safer_v3.physics.library import PolynomialLibrary, build_turbofan_library
from safer_v3.core.trainer import DataModule


@dataclass
class LPVSINDyTrainConfig:
    """Configuration for LPV-SINDy training."""
    # Data settings
    window_size: int = 5
    dt: float = 1.0
    
    # Library settings
    polynomial_degree: int = 2
    include_fourier: bool = False
    
    # Sparse regression settings
    threshold: float = 0.1
    alpha: float = 0.01
    max_iter: int = 100
    
    # Anomaly detection
    residual_threshold_sigma: float = 3.0


def train_physics_monitor(
    data_dir: str = "CMAPSSData",
    dataset: str = "FD001",
    output_dir: str = None,
    config: LPVSINDyTrainConfig = None,
    batch_size: int = 64,
):
    """Train LPV-SINDy physics monitor.
    
    Args:
        data_dir: Path to C-MAPSS data directory
        dataset: Dataset name (FD001, FD002, etc.)
        output_dir: Output directory (auto-generated if None)
        config: Training configuration
        batch_size: Batch size for data loading
    """
    if config is None:
        config = LPVSINDyTrainConfig()
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "outputs" / f"lpv_sindy_{dataset}_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load data
    print(f"\n{'='*60}")
    print(f"Loading {dataset} dataset for physics model training...")
    print(f"{'='*60}")
    
    data_module = DataModule(
        data_dir=data_dir,
        dataset=dataset,
        window_size=30,  # Use longer window for data loading
        batch_size=batch_size,
        max_rul=125,
    )
    data_module.setup()
    
    # Extract sensor sequences for training physics model
    print("\nExtracting sensor sequences...")
    
    train_sequences = []
    for batch in tqdm(data_module.train_dataloader(), desc="Processing train data"):
        sequences, _ = batch
        sequences = sequences.numpy()  # (batch, length, n_sensors)
        for seq in sequences:
            train_sequences.append(seq)
    
    val_sequences = []
    for batch in tqdm(data_module.val_dataloader(), desc="Processing val data"):
        sequences, _ = batch
        sequences = sequences.numpy()
        for seq in sequences:
            val_sequences.append(seq)
    
    test_sequences = []
    for batch in tqdm(data_module.test_dataloader(), desc="Processing test data"):
        sequences, _ = batch
        sequences = sequences.numpy()
        for seq in sequences:
            test_sequences.append(seq)
    
    print(f"\nTrain sequences: {len(train_sequences)}")
    print(f"Val sequences: {len(val_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")
    
    # Concatenate all training data for physics model
    train_data = np.concatenate(train_sequences, axis=0)
    val_data = np.concatenate(val_sequences, axis=0)
    
    print(f"Total train samples: {train_data.shape[0]}")
    print(f"Total val samples: {val_data.shape[0]}")
    print(f"Number of sensors: {train_data.shape[1]}")
    
    # Create custom config for LPVSINDyMonitor
    @dataclass
    class MonitorConfig:
        window_size: int = config.window_size
        dt: float = config.dt
        polynomial_degree: int = config.polynomial_degree
        threshold: float = config.threshold
        alpha: float = config.alpha
        max_iter: int = config.max_iter
        residual_threshold_sigma: float = config.residual_threshold_sigma
    
    monitor_config = MonitorConfig()
    
    # Create function library
    print(f"\n{'='*60}")
    print("Building function library...")
    print(f"{'='*60}")
    
    library = build_turbofan_library(
        n_sensors=14,
        polynomial_degree=config.polynomial_degree,
        include_fourier=config.include_fourier,
    )
    
    # Create LPV-SINDy monitor
    print(f"\n{'='*60}")
    print("Creating LPV-SINDy Physics Monitor (DAL C)")
    print(f"{'='*60}")
    
    monitor = LPVSINDyMonitor(
        config=monitor_config,
        library=library,
        n_sensors=14,
    )
    
    # Fit the model
    print(f"\n{'='*60}")
    print("Fitting LPV-SINDy model...")
    print(f"{'='*60}")
    
    fit_results = monitor.fit(
        X=train_data,
        validate=True,
        val_fraction=0.2,
    )
    
    print(f"\nFitting Results:")
    print(f"  Number of library features: {fit_results['n_features']}")
    print(f"  Total non-zero terms: {fit_results['total_nonzero']}")
    print(f"  Sparsity: {fit_results['sparsity']:.2%}")
    print(f"  Train RMSE: {fit_results['train_rmse']:.6f}")
    print(f"  Train MAE: {fit_results['train_mae']:.6f}")
    if 'val_rmse' in fit_results:
        print(f"  Val RMSE: {fit_results['val_rmse']:.6f}")
        print(f"  Val MAE: {fit_results['val_mae']:.6f}")
    
    # Evaluate on validation sequences
    print(f"\n{'='*60}")
    print("Evaluating anomaly detection on validation data...")
    print(f"{'='*60}")
    
    val_scores = []
    for seq in tqdm(val_sequences[:1000], desc="Evaluating"):  # Sample for speed
        try:
            is_anomaly, max_score, details = monitor.detect_anomaly(seq)
            val_scores.append(max_score)
        except Exception as e:
            continue
        # Convert to numpy array only if we collected scores
        if len(val_scores) > 0:
            val_scores_arr = np.array(val_scores)
            print(f"\nValidation Anomaly Scores:")
            print(f"  Mean: {np.mean(val_scores_arr):.4f}")
            print(f"  Std: {np.std(val_scores_arr):.4f}")
            print(f"  Min: {np.min(val_scores_arr):.4f}")
            print(f"  Max: {np.max(val_scores_arr):.4f}")
            print(f"  Anomaly rate (>{config.residual_threshold_sigma}σ): "
                  f"{np.mean(val_scores_arr > config.residual_threshold_sigma):.2%}")
        else:
            val_scores_arr = None
    
    # Print identified equations (sample)
    print(f"\n{'='*60}")
    print("Sample Identified Dynamics:")
    print(f"{'='*60}")
    
    active_terms = monitor.get_active_terms()
    for state_idx in range(min(3, len(active_terms))):  # Show first 3 states
        terms = active_terms.get(state_idx, [])
        if terms:
            print(f"\nΔx{state_idx} = {' '.join(terms[:5])}...")  # Show first 5 terms
        else:
            print(f"\nΔx{state_idx} = 0")
    
    # Save model
    print(f"\n{'='*60}")
    print("Saving model...")
    print(f"{'='*60}")
    
    monitor.save(output_dir / "lpv_sindy_model")
    
    # Save training configuration and results
    results = {
        'model': 'LPV-SINDy Physics Monitor (DAL C)',
        'dataset': dataset,
        'config': asdict(config),
        'fit_results': {
            'n_features': fit_results['n_features'],
            'total_nonzero': int(fit_results['total_nonzero']),
            'sparsity': float(fit_results['sparsity']),
            'train_rmse': float(fit_results['train_rmse']),
            'train_mae': float(fit_results['train_mae']),
        },
        'validation': {
            'mean_score': float(np.mean(val_scores_arr)) if val_scores_arr is not None else None,
            'std_score': float(np.std(val_scores_arr)) if val_scores_arr is not None else None,
            'anomaly_rate': float(np.mean(val_scores_arr > config.residual_threshold_sigma)) if val_scores_arr is not None else None,
        },
    }
    
    with open(output_dir / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*60}")
    
    return output_dir, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LPV-SINDy Physics Monitor")
    parser.add_argument("--data_dir", type=str, default="CMAPSSData")
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for data loading (passed to DataModule)")
    parser.add_argument("--polynomial_degree", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--window_size", type=int, default=5)
    
    args = parser.parse_args()
    
    config = LPVSINDyTrainConfig(
        polynomial_degree=args.polynomial_degree,
        threshold=args.threshold,
        window_size=args.window_size,
    )
    
    train_physics_monitor(
        data_dir=args.data_dir,
        dataset=args.dataset,
        output_dir=args.output_dir,
        config=config,
        batch_size=args.batch_size,
    )
