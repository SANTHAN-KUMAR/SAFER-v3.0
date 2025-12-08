#!/usr/bin/env python3
"""
SAFER v3.0 Baseline Models Training Script.

This script trains baseline models (LSTM, Transformer, CNN-LSTM)
for comparison with the Mamba architecture.

Features:
- Support for multiple baseline architectures
- Consistent training pipeline with Mamba
- Comparative evaluation and reporting
- Model export for deployment

Usage:
    # Train single baseline
    python train_baselines.py --model lstm --dataset FD001
    
    # Train all baselines for comparison
    python train_baselines.py --model all --dataset FD001
    
    # Custom configuration
    python train_baselines.py --model transformer --d_model 128 --epochs 50

Author: SAFER v3.0 Development Team
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from safer_v3.utils.config import SAFERConfig
from safer_v3.utils.logging_config import setup_logging
from safer_v3.utils.metrics import RULMetrics
from safer_v3.core.baselines import (
    LSTMPredictor,
    TransformerPredictor,
    CNNLSTMPredictor,
    BaselineFactory,
)
from safer_v3.core.trainer import (
    CMAPSSDataset,
    DataModule,
    Trainer,
    TrainingConfig,
)


AVAILABLE_MODELS = ['lstm', 'transformer', 'cnn_lstm', 'all']


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Baseline RUL Predictors for SAFER v3.0',
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
    
    # Model selection
    parser.add_argument(
        '--model', type=str, default='lstm',
        choices=AVAILABLE_MODELS,
        help='Model architecture to train'
    )
    
    # Model arguments
    parser.add_argument(
        '--d_model', type=int, default=64,
        help='Model hidden dimension'
    )
    parser.add_argument(
        '--n_layers', type=int, default=2,
        help='Number of layers'
    )
    parser.add_argument(
        '--n_heads', type=int, default=4,
        help='Number of attention heads (Transformer only)'
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
    
    # Output arguments
    parser.add_argument(
        '--output_dir', type=str, default='outputs',
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--experiment_name', type=str, default=None,
        help='Experiment name (auto-generated if not provided)'
    )
    
    # Misc
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device to use (auto, cpu, cuda)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers'
    )
    
    return parser.parse_args()


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


def create_model(
    model_type: str,
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    """Create baseline model from arguments."""
    d_input = 14  # 14 prognostic sensors
    
    if model_type == 'lstm':
        model = LSTMPredictor(
            d_input=d_input,
            d_hidden=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            bidirectional=True,
        )
    elif model_type == 'transformer':
        model = TransformerPredictor(
            d_input=d_input,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            max_seq_len=args.seq_length,
        )
    elif model_type == 'cnn_lstm':
        model = CNNLSTMPredictor(
            d_input=d_input,
            d_hidden=args.d_model,
            n_lstm_layers=args.n_layers,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Log model info
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Created {model_type.upper()} model with {n_params:,} parameters")
    
    return model


def train_model(
    model_type: str,
    args: argparse.Namespace,
    data_module: DataModule,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    """Train a single baseline model.
    
    Returns:
        Dictionary with model, metrics, and path
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'=' * 40}")
    logger.info(f"Training {model_type.upper()}")
    logger.info(f"{'=' * 40}")
    
    # Reset seed
    set_seed(args.seed)
    
    # Create model
    model = create_model(model_type, args, device)
    
    # Create training config
    train_config = TrainingConfig(
        max_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        gradient_clip_val=args.grad_clip,
        use_amp=args.use_amp,
    )
    
    # Create trainer
    trainer = Trainer(train_config, device)
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Train
    history = trainer.fit(model, train_loader, val_loader)
    
    # Save checkpoint
    model_dir = output_dir / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f'{model_type}_model.pt'
    trainer.save_checkpoint(model, checkpoint_path)
    
    # Evaluate on validation set
    val_metrics = evaluate_model(model, val_loader, device)
    
    # Evaluate on test set
    test_loader = data_module.test_dataloader()
    test_metrics = evaluate_model(model, test_loader, device)
    
    # Save metrics
    metrics_file = model_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump({
            'validation': val_metrics,
            'test': test_metrics,
        }, f, indent=2)
    
    # Save training history
    history_file = model_dir / 'history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    return {
        'model': model,
        'model_type': model_type,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'checkpoint_path': str(checkpoint_path),
        'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
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


def create_comparison_report(
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Create comparison report for all models."""
    logger = logging.getLogger(__name__)
    
    # Create comparison table
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 70)
    
    header = f"{'Model':<15} {'Params':>10} {'Val RMSE':>10} {'Test RMSE':>10} {'NASA Score':>12}"
    logger.info(header)
    logger.info("-" * 70)
    
    comparison_data = []
    
    for result in results:
        model_type = result['model_type']
        n_params = result['n_params']
        val_rmse = result['val_metrics']['rmse']
        test_rmse = result['test_metrics']['rmse']
        nasa_score = result['test_metrics']['nasa_score']
        
        row = f"{model_type:<15} {n_params:>10,} {val_rmse:>10.2f} {test_rmse:>10.2f} {nasa_score:>12.2f}"
        logger.info(row)
        
        comparison_data.append({
            'model': model_type,
            'n_params': n_params,
            'val_rmse': val_rmse,
            'test_rmse': test_rmse,
            'nasa_score': nasa_score,
        })
    
    logger.info("-" * 70)
    
    # Find best model
    best_idx = np.argmin([r['test_metrics']['rmse'] for r in results])
    best_model = results[best_idx]['model_type']
    logger.info(f"\nBest model by Test RMSE: {best_model.upper()}")
    
    # Save comparison report
    report_file = output_dir / 'comparison_report.json'
    with open(report_file, 'w') as f:
        json.dump({
            'results': comparison_data,
            'best_model': best_model,
        }, f, indent=2)
    
    logger.info(f"\nComparison report saved: {report_file}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    
    # Determine models to train
    if args.model == 'all':
        models_to_train = ['lstm', 'transformer', 'cnn_lstm']
    else:
        models_to_train = [args.model]
    
    # Create experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_str = args.model if args.model != 'all' else 'baselines'
        args.experiment_name = f'{model_str}_{args.dataset}_{timestamp}'
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(log_dir=output_dir, level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("SAFER v3.0 Baseline Models Training")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Models: {', '.join(models_to_train)}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    
    # Save args
    args_file = output_dir / 'args.json'
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    logger.info("\nLoading C-MAPSS data...")
    
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
    
    # Train models
    results = []
    
    for model_type in models_to_train:
        result = train_model(
            model_type=model_type,
            args=args,
            data_module=data_module,
            device=device,
            output_dir=output_dir,
        )
        results.append(result)
        
        logger.info(f"\n{model_type.upper()} Results:")
        logger.info(f"  Val RMSE: {result['val_metrics']['rmse']:.2f}")
        logger.info(f"  Test RMSE: {result['test_metrics']['rmse']:.2f}")
        logger.info(f"  NASA Score: {result['test_metrics']['nasa_score']:.2f}")
    
    # Generate comparison report if multiple models
    if len(results) > 1:
        create_comparison_report(results, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
