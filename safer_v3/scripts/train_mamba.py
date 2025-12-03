#!/usr/bin/env python3
"""
SAFER v3.0 Mamba Training Script.

This script trains the Mamba-based RUL predictor on C-MAPSS data.

Features:
- Configurable hyperparameters via command line
- Support for all C-MAPSS datasets (FD001-FD004)
- Automatic mixed precision training
- Early stopping with best model checkpoint
- Comprehensive logging and metrics tracking
- ONNX export for deployment
- Ensemble training support

Usage:
    python train_mamba.py --data_dir CMAPSSData --dataset FD001 --epochs 100
    
    # With custom config
    python train_mamba.py --config config.yaml
    
    # Train ensemble
    python train_mamba.py --ensemble 5 --dataset FD001

Author: SAFER v3.0 Development Team
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from safer_v3.utils.config import SAFERConfig, MambaConfig
from safer_v3.utils.logging_config import setup_logging, get_logger
from safer_v3.utils.metrics import RULMetrics, nasa_scoring_function
from safer_v3.core.mamba import MambaRULPredictor, MambaEnsemble
from safer_v3.core.trainer import (
    CMAPSSDataset,
    DataModule,
    Trainer,
    TrainingConfig,
    EarlyStopping,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Mamba RUL Predictor for SAFER v3.0',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir', type=str, default='CMAPSSData',
        help='Directory containing C-MAPSS data files'
    )
    parser.add_argument(
        '--dataset', type=str, default='FD001',
        choices=['FD001', 'FD002', 'FD003', 'FD004'],
        help='C-MAPSS dataset to use'
    )
    parser.add_argument(
        '--max_rul', type=int, default=125,
        help='Maximum RUL clipping value'
    )
    parser.add_argument(
        '--seq_length', type=int, default=50,
        help='Input sequence length'
    )
    parser.add_argument(
        '--val_split', type=float, default=0.2,
        help='Validation split ratio'
    )
    
    # Model arguments
    parser.add_argument(
        '--d_model', type=int, default=64,
        help='Model hidden dimension'
    )
    parser.add_argument(
        '--d_state', type=int, default=16,
        help='SSM state dimension'
    )
    parser.add_argument(
        '--n_layers', type=int, default=4,
        help='Number of Mamba layers'
    )
    parser.add_argument(
        '--expand', type=int, default=2,
        help='Expansion factor for inner dimension'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1,
        help='Dropout rate'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Maximum training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-4,
        help='Weight decay (L2 regularization)'
    )
    parser.add_argument(
        '--patience', type=int, default=15,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--grad_clip', type=float, default=1.0,
        help='Gradient clipping norm'
    )
    parser.add_argument(
        '--use_amp', action='store_true',
        help='Use automatic mixed precision'
    )
    
    # Ensemble arguments
    parser.add_argument(
        '--ensemble', type=int, default=1,
        help='Number of ensemble members (1 for single model)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir', type=str, default='outputs',
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--experiment_name', type=str, default=None,
        help='Experiment name (auto-generated if not provided)'
    )
    parser.add_argument(
        '--export_onnx', action='store_true',
        help='Export model to ONNX format'
    )
    
    # Config file
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML configuration file'
    )
    
    # Misc
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device to use (auto, cpu, cuda, cuda:0, etc.)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    """Get torch device from string."""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def create_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """Create Mamba model from arguments."""
    model = MambaRULPredictor(
        d_input=14,  # 14 prognostic sensors
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        expand=args.expand,
        dropout=args.dropout,
        use_jit=False,  # Disable JIT for training
    )
    
    model = model.to(device)
    
    # Log model info
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Created Mamba model with {n_params:,} trainable parameters")
    
    return model


def train_single_model(
    args: argparse.Namespace,
    data_module: DataModule,
    device: torch.device,
    output_dir: Path,
    model_idx: int = 0,
) -> tuple:
    """Train a single model.
    
    Returns:
        Tuple of (model, metrics)
    """
    logger = logging.getLogger(__name__)
    
    # Set different seed for each ensemble member
    set_seed(args.seed + model_idx)
    
    # Create model
    model = create_model(args, device)
    
    # Create training config
    train_config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
        gradient_clip=args.grad_clip,
        use_amp=args.use_amp,
        device=device.type,
        tensorboard_dir=None,  # Disable TensorBoard due to compatibility issues
    )
    
    # Create trainer
    trainer = Trainer(model, train_config, data_module)
    
    logger.info(f"Training model {model_idx + 1}/{args.ensemble}")
    logger.info(f"  Train samples: {len(data_module.train_dataset)}")
    logger.info(f"  Val samples: {len(data_module.val_dataset)}")
    
    # Train (trainer.fit will get loaders from data_module)
    history = trainer.fit()
    
    # Save checkpoint
    checkpoint_path = output_dir / f'mamba_model_{model_idx}.pt'
    trainer.save_checkpoint(checkpoint_path)
    
    # Evaluate on validation set
    val_loader = data_module.val_dataloader()
    val_metrics = evaluate_model(model, val_loader, device)
    
    return model, val_metrics, history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on data loader."""
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
    
    # Compute metrics
    metrics = RULMetrics.compute_all(y_true, y_pred)
    
    return metrics


def train_ensemble(
    args: argparse.Namespace,
    data_module: DataModule,
    device: torch.device,
    output_dir: Path,
) -> tuple:
    """Train ensemble of models.
    
    Returns:
        Tuple of (ensemble, metrics)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Training ensemble of {args.ensemble} models")
    
    models = []
    all_metrics = []
    
    for i in range(args.ensemble):
        model, metrics, history = train_single_model(
            args, data_module, device, output_dir, model_idx=i
        )
        models.append(model)
        all_metrics.append(metrics)
        
        logger.info(f"Model {i+1} - RMSE: {metrics['rmse']:.2f}, NASA Score: {metrics['nasa_score']:.2f}")
    
    # Create ensemble
    config = MambaConfig(
        d_input=14,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        expand=args.expand,
        dropout=args.dropout,
    )
    
    ensemble = MambaEnsemble(config, n_members=args.ensemble)
    
    # Copy trained models to ensemble
    for i, model in enumerate(models):
        ensemble.members[i].load_state_dict(model.state_dict())
    
    ensemble = ensemble.to(device)
    
    # Evaluate ensemble
    val_loader = data_module.val_dataloader()
    ensemble_metrics = evaluate_ensemble(ensemble, val_loader, device)
    
    # Aggregate individual metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    return ensemble, ensemble_metrics, avg_metrics


def evaluate_ensemble(
    ensemble: MambaEnsemble,
    data_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate ensemble on data loader."""
    ensemble.eval()
    
    all_means = []
    all_stds = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, targets in data_loader:
            sequences = sequences.to(device)
            mean, std = ensemble(sequences)
            
            all_means.append(mean.cpu().numpy())
            all_stds.append(std.cpu().numpy())
            all_targets.append(targets.numpy())
    
    y_pred = np.concatenate(all_means).ravel()
    y_std = np.concatenate(all_stds).ravel()
    y_true = np.concatenate(all_targets).ravel()
    
    # Compute metrics
    metrics = RULMetrics.compute_all(y_true, y_pred)
    metrics['mean_std'] = float(np.mean(y_std))
    
    return metrics


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    
    # Create experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f'mamba_{args.dataset}_{timestamp}'
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(log_dir=output_dir, level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("SAFER v3.0 Mamba Training")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    
    # Save args
    args_file = output_dir / 'args.json'
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    logger.info("Loading C-MAPSS data...")
    
    data_module = DataModule(
        data_dir=args.data_dir,
        dataset=args.dataset,
        window_size=args.seq_length,
        max_rul=args.max_rul,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )
    data_module.setup()
    
    logger.info(f"Train sequences: {len(data_module.train_dataset)}")
    logger.info(f"Val sequences: {len(data_module.val_dataset)}")
    logger.info(f"Test sequences: {len(data_module.test_dataset)}")
    
    # Train
    if args.ensemble > 1:
        model, metrics, avg_metrics = train_ensemble(
            args, data_module, device, output_dir
        )
        logger.info("\nEnsemble Results:")
        logger.info(f"  Ensemble RMSE: {metrics['rmse']:.2f}")
        logger.info(f"  Ensemble NASA Score: {metrics['nasa_score']:.2f}")
        logger.info(f"  Avg Individual RMSE: {avg_metrics['rmse']:.2f}")
        
        # Save ensemble
        ensemble_path = output_dir / 'mamba_ensemble.pt'
        torch.save(model.state_dict(), ensemble_path)
        
    else:
        model, metrics, history = train_single_model(
            args, data_module, device, output_dir
        )
        logger.info("\nValidation Results:")
        logger.info(f"  RMSE: {metrics['rmse']:.2f}")
        logger.info(f"  MAE: {metrics['mae']:.2f}")
        logger.info(f"  NASA Score: {metrics['nasa_score']:.2f}")
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_loader = data_module.test_dataloader()
    
    if args.ensemble > 1:
        test_metrics = evaluate_ensemble(model, test_loader, device)
    else:
        test_metrics = evaluate_model(model, test_loader, device)
    
    logger.info("\nTest Results:")
    logger.info(f"  RMSE: {test_metrics['rmse']:.2f}")
    logger.info(f"  MAE: {test_metrics['mae']:.2f}")
    logger.info(f"  NASA Score: {test_metrics['nasa_score']:.2f}")
    
    # Save metrics
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump({
            'validation': metrics,
            'test': test_metrics,
        }, f, indent=2)
    
    # Export ONNX
    if args.export_onnx and args.ensemble == 1:
        logger.info("\nExporting to ONNX...")
        onnx_path = output_dir / 'mamba_model.onnx'
        
        # Create dummy input
        dummy_input = torch.randn(1, args.seq_length, 14).to(device)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=['sensors'],
            output_names=['rul'],
            dynamic_axes={
                'sensors': {0: 'batch', 1: 'seq_len'},
                'rul': {0: 'batch'},
            },
            opset_version=14,
        )
        logger.info(f"ONNX model saved: {onnx_path}")
    
    logger.info("\nTraining complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
