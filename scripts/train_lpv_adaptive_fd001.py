"""
Enhanced LPV-SINDy training script with adaptive scheduling parameter.

This script demonstrates the new features added to SAFER v3.0:
1. Automatic scheduling parameter computation from EGT margin
2. LPV augmented library with p-weighted interaction terms
3. LPV decomposition to extract health-independent vs health-dependent dynamics
4. Comparison of standard vs augmented LPV performance

Usage:
    python train_lpv_adaptive_fd001.py
    
Output:
    Trained models and comparison metrics in outputs/
    
References:
    - Hoffmann et al., "Reactive SINDy" (2019)
    - Brunton et al., "Discovering governing equations from data" (2016)
"""

import numpy as np
import torch
import logging
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from safer_v3.physics.lpv_sindy import LPVSINDyMonitor, IntegralFormulation
from safer_v3.physics.library import (
    build_turbofan_library,
    LPVAugmentedLibrary,
)
from safer_v3.utils.config import LPVSINDyConfig, MambaConfig
from safer_v3.utils.dataset import load_cmapss


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_and_prepare_data(data_dir: Path, dataset: str = 'FD001'):
    """Load and prepare C-MAPSS dataset.
    
    Args:
        data_dir: Path to CMAPSSData directory
        dataset: Dataset name (FD001, FD002, etc.)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Loading {dataset} dataset...")
    
    try:
        # Try using the dataset utility
        X_train, y_train, X_test, y_test = load_cmapss(
            data_dir=data_dir,
            dataset=dataset,
            normalize=True,
        )
    except Exception as e:
        logger.warning(f"Could not load with utility: {e}")
        logger.info("Loading raw data instead...")
        
        # Load raw files
        train_file = data_dir / f'train_{dataset}.txt'
        test_file = data_dir / f'test_{dataset}.txt'
        rul_file = data_dir / f'RUL_{dataset}.txt'
        
        if not train_file.exists():
            raise FileNotFoundError(f"Could not find {train_file}")
        
        # Load training data
        data_train = np.loadtxt(train_file)
        X_train = data_train[:, 2:]  # Skip unit and time
        
        # Load test data
        data_test = np.loadtxt(test_file)
        X_test = data_test[:, 2:]
        
        # Load RUL values (for test set)
        y_test = np.loadtxt(rul_file)
        y_train = None
        
        logger.info(f"Loaded {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test


def train_standard_lpv(
    X_train: np.ndarray,
    config: LPVSINDyConfig,
    output_dir: Path,
) -> dict:
    """Train standard LPV-SINDy model.
    
    Args:
        X_train: Training data
        config: Configuration
        output_dir: Output directory
        
    Returns:
        Dictionary with results
    """
    logger.info("\n" + "="*60)
    logger.info("STANDARD LPV-SINDy TRAINING")
    logger.info("="*60)
    
    # Create monitor with standard library
    monitor = LPVSINDyMonitor(
        config=config,
        library=build_turbofan_library(
            n_sensors=X_train.shape[1],
            polynomial_degree=config.polynomial_degree,
        ),
        n_sensors=X_train.shape[1],
    )
    
    # Fit model
    fit_results = monitor.fit(X_train, validate=True, val_fraction=0.2)
    
    # Compute scheduling parameter
    p = monitor.compute_scheduling_parameter(X_train)
    
    # Decompose
    decomp_results = monitor.fit_lpv_decomposition(p)
    
    # Combine results
    results = {
        'model_type': 'standard_lpv',
        'fit_metrics': fit_results,
        'decomposition': {
            'rmse': decomp_results['decomposition_rmse'],
            'r2': decomp_results['explained_variance'],
            'xi_0_norm': float(np.linalg.norm(decomp_results['coefficients_0'])),
            'xi_1_norm': float(np.linalg.norm(decomp_results['coefficients_1'])),
        },
        'scheduling_param_stats': {
            'min': float(p.min()),
            'max': float(p.max()),
            'mean': float(p.mean()),
            'std': float(p.std()),
        },
        'model': monitor,
    }
    
    logger.info(f"✓ Standard model training complete")
    logger.info(f"  - Train RMSE: {fit_results['train_rmse']:.4f}")
    logger.info(f"  - Val RMSE: {fit_results.get('val_rmse', 'N/A')}")
    logger.info(f"  - Non-zero terms: {fit_results['total_nonzero']}")
    logger.info(f"  - Sparsity: {fit_results['sparsity']:.2%}")
    
    return results


def train_augmented_lpv(
    X_train: np.ndarray,
    config: LPVSINDyConfig,
    output_dir: Path,
) -> dict:
    """Train augmented LPV-SINDy model with p-weighted terms.
    
    Args:
        X_train: Training data
        config: Configuration
        output_dir: Output directory
        
    Returns:
        Dictionary with results
    """
    logger.info("\n" + "="*60)
    logger.info("AUGMENTED LPV-SINDy TRAINING (with p-weighted terms)")
    logger.info("="*60)
    
    # Create monitor with augmented library
    augmented_lib = LPVAugmentedLibrary(
        degree=config.polynomial_degree,
        include_bias=True,
        include_interaction=config.include_interactions,
    )
    
    monitor = LPVSINDyMonitor(
        config=config,
        library=augmented_lib,
        n_sensors=X_train.shape[1],
    )
    
    # Compute scheduling parameter first
    p = monitor.compute_scheduling_parameter(X_train)
    
    # Fit library with augmented terms
    augmented_lib.fit(X_train)
    
    # Custom fit with augmented library
    X_train = np.asarray(X_train, dtype=np.float64)
    n_samples, n_features = X_train.shape
    
    # Split for validation
    n_train = int(n_samples * 0.8)
    X_fit = X_train[:n_train]
    X_val = X_train[n_train:]
    p_fit = p[:n_train]
    
    # Transform with augmented features
    Theta = augmented_lib.transform(X_fit, p_fit)
    
    # Apply integral formulation
    integral = IntegralFormulation(
        window_size=config.window_size,
        dt=config.dt,
        method='trapezoidal',
    )
    
    delta_x, weights = integral.integrate(X_fit)
    Theta_integrated = integral.integrate_library(Theta, weights)
    
    # Fit sparse regression
    coefficients_list = []
    for i in range(n_features):
        monitor.regressor.fit(Theta_integrated, delta_x[:, i])
        coefficients_list.append(monitor.regressor.coef_.copy())
    
    monitor._coefficients = np.column_stack(coefficients_list)
    monitor._is_fitted = True
    monitor._feature_names = augmented_lib.get_feature_names()
    
    # Compute residuals and statistics
    train_residuals = monitor._compute_residuals(X_fit)
    monitor._residual_mean = np.mean(train_residuals, axis=0)
    monitor._residual_std = np.std(train_residuals, axis=0)
    monitor._residual_std = np.maximum(monitor._residual_std, 1e-6)
    
    # Compute metrics
    fit_results = {
        'n_features': len(monitor._feature_names),
        'n_nonzero': np.sum(np.abs(monitor._coefficients) > 1e-10, axis=0),
        'total_nonzero': np.sum(np.abs(monitor._coefficients) > 1e-10),
        'sparsity': 1.0 - np.mean(np.abs(monitor._coefficients) > 1e-10),
        'train_rmse': np.sqrt(np.mean(train_residuals ** 2)),
        'train_mae': np.mean(np.abs(train_residuals)),
    }
    
    # Validation metrics
    val_residuals = monitor._compute_residuals(X_val)
    fit_results['val_rmse'] = np.sqrt(np.mean(val_residuals ** 2))
    fit_results['val_mae'] = np.mean(np.abs(val_residuals))
    
    # Decompose
    decomp_results = monitor.fit_lpv_decomposition(p_fit)
    
    # Combine results
    results = {
        'model_type': 'augmented_lpv',
        'fit_metrics': fit_results,
        'decomposition': {
            'rmse': decomp_results['decomposition_rmse'],
            'r2': decomp_results['explained_variance'],
            'xi_0_norm': float(np.linalg.norm(decomp_results['coefficients_0'])),
            'xi_1_norm': float(np.linalg.norm(decomp_results['coefficients_1'])),
        },
        'scheduling_param_stats': {
            'min': float(p.min()),
            'max': float(p.max()),
            'mean': float(p.mean()),
            'std': float(p.std()),
        },
        'augmented_features': len(augmented_lib.feature_names_),
        'model': monitor,
    }
    
    logger.info(f"✓ Augmented model training complete")
    logger.info(f"  - Augmented features: {len(augmented_lib.feature_names_)}")
    logger.info(f"  - Train RMSE: {fit_results['train_rmse']:.4f}")
    logger.info(f"  - Val RMSE: {fit_results.get('val_rmse', 'N/A')}")
    logger.info(f"  - Non-zero terms: {fit_results['total_nonzero']}")
    logger.info(f"  - Sparsity: {fit_results['sparsity']:.2%}")
    
    return results


def compare_models(standard_results: dict, augmented_results: dict) -> dict:
    """Compare standard vs augmented LPV models.
    
    Args:
        standard_results: Results from standard LPV
        augmented_results: Results from augmented LPV
        
    Returns:
        Comparison metrics
    """
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    
    std_metrics = standard_results['fit_metrics']
    aug_metrics = augmented_results['fit_metrics']
    
    # RMSE comparison
    rmse_improvement = (
        (std_metrics['train_rmse'] - aug_metrics['train_rmse']) /
        std_metrics['train_rmse'] * 100
    )
    
    val_rmse_improvement = (
        (std_metrics['val_rmse'] - aug_metrics['val_rmse']) /
        std_metrics['val_rmse'] * 100
    )
    
    # Sparsity comparison
    sparsity_std = std_metrics['sparsity']
    sparsity_aug = aug_metrics['sparsity']
    
    # Health sensitivity comparison (Ξ₁ norm)
    xi_1_std = standard_results['decomposition']['xi_1_norm']
    xi_1_aug = augmented_results['decomposition']['xi_1_norm']
    
    comparison = {
        'train_rmse': {
            'standard': std_metrics['train_rmse'],
            'augmented': aug_metrics['train_rmse'],
            'improvement_percent': rmse_improvement,
        },
        'val_rmse': {
            'standard': std_metrics['val_rmse'],
            'augmented': aug_metrics['val_rmse'],
            'improvement_percent': val_rmse_improvement,
        },
        'sparsity': {
            'standard': sparsity_std,
            'augmented': sparsity_aug,
        },
        'health_sensitivity': {
            'standard_xi1_norm': xi_1_std,
            'augmented_xi1_norm': xi_1_aug,
            'ratio': xi_1_aug / (xi_1_std + 1e-10),
        },
        'features': {
            'standard': std_metrics['n_features'],
            'augmented': augmented_results['augmented_features'],
        },
    }
    
    logger.info(f"\n✓ RMSE Improvement: {rmse_improvement:.2f}%")
    logger.info(f"  - Standard train RMSE: {std_metrics['train_rmse']:.4f}")
    logger.info(f"  - Augmented train RMSE: {aug_metrics['train_rmse']:.4f}")
    logger.info(f"  - Standard val RMSE:   {std_metrics['val_rmse']:.4f}")
    logger.info(f"  - Augmented val RMSE:  {aug_metrics['val_rmse']:.4f}")
    
    logger.info(f"\n✓ Sparsity:")
    logger.info(f"  - Standard: {sparsity_std:.2%}")
    logger.info(f"  - Augmented: {sparsity_aug:.2%}")
    
    logger.info(f"\n✓ Health Sensitivity (Ξ₁ decomposition):")
    logger.info(f"  - Standard: {xi_1_std:.6f}")
    logger.info(f"  - Augmented: {xi_1_aug:.6f}")
    logger.info(f"  - Ratio: {comparison['health_sensitivity']['ratio']:.2f}x")
    
    return comparison


def main(
    data_dir: Path = Path('CMAPSSData'),
    output_dir: Path = None,
):
    """Main training pipeline.
    
    Args:
        data_dir: Path to C-MAPSS data
        output_dir: Output directory (auto-created if None)
    """
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('outputs') / f'lpv_adaptive_{timestamp}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info("ADAPTIVE LPV-SINDy TRAINING")
    logger.info(f"{'='*60}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load data
    try:
        X_train, X_test, y_train, y_test = load_and_prepare_data(data_dir)
    except Exception as e:
        logger.error(f"Could not load data: {e}")
        logger.info("Using synthetic data for demonstration...")
        # Create synthetic data for demo
        X_train = np.random.randn(5000, 14) * 50 + 100
        X_test = np.random.randn(2000, 14) * 50 + 100
        y_train = np.random.randn(2000) * 20 + 50
        y_test = np.random.randn(100) * 20 + 50
    
    logger.info(f"\nData shapes:")
    logger.info(f"  - Training: {X_train.shape}")
    logger.info(f"  - Test: {X_test.shape}")
    
    # Configuration
    config = LPVSINDyConfig(
        n_features=X_train.shape[1],
        polynomial_degree=2,
        window_size=5,
        threshold=0.1,
        egtm_sensor_idx=9,  # EGT margin sensor
    )
    
    # Train standard LPV
    standard_results = train_standard_lpv(X_train, config, output_dir)
    
    # Train augmented LPV
    augmented_results = train_augmented_lpv(X_train, config, output_dir)
    
    # Compare
    comparison = compare_models(standard_results, augmented_results)
    
    # Save results
    results_file = output_dir / 'comparison_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_comparison = convert_to_serializable(comparison)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_comparison, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {results_file}")
    
    # Save models
    standard_model_file = output_dir / 'standard_lpv_model.pt'
    augmented_model_file = output_dir / 'augmented_lpv_model.pt'
    
    torch.save(standard_results['model'].state_dict(), standard_model_file)
    torch.save(augmented_results['model'].state_dict(), augmented_model_file)
    
    logger.info(f"✓ Models saved")
    logger.info(f"  - Standard: {standard_model_file}")
    logger.info(f"  - Augmented: {augmented_model_file}")
    
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*60}\n")
    
    return {
        'standard': standard_results,
        'augmented': augmented_results,
        'comparison': comparison,
        'output_dir': output_dir,
    }


if __name__ == '__main__':
    main()
