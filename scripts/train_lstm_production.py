#!/usr/bin/env python
"""
Production-Ready LSTM Training Script for SAFER v3.0.

This script trains the LSTM baseline model with optimized hyperparameters
for achieving good performance on C-MAPSS datasets.

Key improvements over default training:
- 100 epochs with early stopping
- Larger model capacity (128 hidden, 3 layers)
- Cosine annealing learning rate schedule
- Gradient clipping for stability
- Comprehensive validation and checkpointing

Usage:
    python scripts/train_lstm_production.py --dataset FD001 --epochs 100
    
Author: SAFER v3.0 Team
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train LSTM Baseline for RUL Prediction'
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='CMAPSSData',
                        help='Path to C-MAPSS data directory')
    parser.add_argument('--dataset', type=str, default='FD001',
                        choices=['FD001', 'FD002', 'FD003', 'FD004'],
                        help='Dataset to use')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=64,
                        help='Hidden dimension (default: 64)')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                        help='Use bidirectional LSTM')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Initial learning rate (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay (default: 1e-3)')
    parser.add_argument('--patience', type=int, default=100,
                        help='Early stopping patience (default: 50)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value (default: 1.0)')
    parser.add_argument('--noise_std', type=float, default=0.1,
                        help='Noise std for data augmentation (default: 0.1)')
    
    # Data processing arguments
    parser.add_argument('--window_size', type=int, default=50,
                        help='Sequence window size (default: 50)')
    parser.add_argument('--max_rul', type=int, default=125,
                        help='Maximum RUL cap (default: 125)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def main():
    """Main training function."""
    print("Starting LSTM Production Training Script...", flush=True)
    args = parse_args()
    print(f"Configuration: dataset={args.dataset}, epochs={args.epochs}", flush=True)
    set_seed(args.seed)
    
    # Delayed imports for faster --help
    print("Importing PyTorch...", flush=True)
    try:
        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        print(f"PyTorch {torch.__version__} imported successfully", flush=True)
    except ImportError as e:
        print(f"ERROR: PyTorch not installed: {e}", flush=True)
        sys.exit(1)
    
    print("Importing SAFER v3.0 modules...", flush=True)
    try:
        from safer_v3.core.baselines import LSTMPredictor
        print("  - LSTMPredictor imported", flush=True)
        from safer_v3.core.trainer import DataModule
        print("  - DataModule imported", flush=True)
        from safer_v3.utils.metrics import calculate_rul_metrics
        print("  - calculate_rul_metrics imported", flush=True)
    except ImportError as e:
        print(f"ERROR: SAFER v3.0 not installed: {e}", flush=True)
        print("Run: pip install -e .", flush=True)
        sys.exit(1)
    
    print("All imports successful!", flush=True)
    
    # Setup device
    print("Setting up device...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    # Create output directory
    print("Creating output directory...", flush=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'lstm_production_{args.dataset}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}", flush=True)
    
    # Save configuration
    config = vars(args)
    config['timestamp'] = timestamp
    config['device'] = str(device)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # ==========================================================================
    # Data Loading
    # ==========================================================================
    print("=" * 60, flush=True)
    print("Loading Data", flush=True)
    print("=" * 60, flush=True)
    
    print(f"Looking for data in: {args.data_dir}", flush=True)
    data_module = DataModule(
        data_dir=args.data_dir,
        dataset=args.dataset,
        window_size=args.window_size,
        stride=1,
        max_rul=args.max_rul,
        batch_size=args.batch_size,
        val_split=0.2,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        seed=args.seed,
    )
    
    print("Calling data_module.setup()...", flush=True)
    data_module.setup()
    print("Data setup complete!", flush=True)
    
    print("Creating data loaders...", flush=True)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    print("Data loaders created!", flush=True)
    
    print(f"Train samples: {len(data_module.train_dataset)}", flush=True)
    print(f"Val samples: {len(data_module.val_dataset)}", flush=True)
    print(f"Test samples: {len(data_module.test_dataset)}", flush=True)
    
    # ==========================================================================
    # Model Creation
    # ==========================================================================
    print("=" * 60, flush=True)
    print("Creating Model", flush=True)
    print("=" * 60, flush=True)
    
    model = LSTMPredictor(
        d_input=14,  # C-MAPSS sensors
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        max_rul=args.max_rul,
    )
    model.to(device)
    print("Model created and moved to device!", flush=True)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable", flush=True)
    
    # ==========================================================================
    # Optimizer and Scheduler
    # ==========================================================================
    print("Setting up optimizer and scheduler...", flush=True)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Use ReduceLROnPlateau - reduces LR when validation stops improving
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=True,
    )
    
    # Use Huber loss for robustness
    criterion = nn.HuberLoss(delta=10.0)
    print("Optimizer, scheduler, and loss function ready!", flush=True)
    
    # ==========================================================================
    # Training Loop
    # ==========================================================================
    print("=" * 60, flush=True)
    print("Training", flush=True)
    print("=" * 60, flush=True)
    
    best_val_rmse = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'learning_rate': [],
    }
    
    start_time = time.time()
    print(f"Starting training for {args.epochs} epochs...", flush=True)
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs} starting...", flush=True)
        # Training
        model.train()
        train_losses = []
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Data augmentation: add Gaussian noise to inputs
            if args.noise_std > 0:
                noise = torch.randn_like(sequences) * args.noise_std
                sequences = sequences + noise
            
            optimizer.zero_grad()
            predictions = model(sequences)
            # Ensure both predictions and targets are 1D for loss calculation
            loss = criterion(predictions.squeeze(-1), targets.squeeze(-1))
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                predictions = model(sequences)
                val_preds.append(predictions.cpu().numpy())
                val_targets.append(targets.numpy())
        
        val_preds = np.concatenate(val_preds).ravel()
        val_targets = np.concatenate(val_targets).ravel()
        
        val_metrics = calculate_rul_metrics(val_targets, val_preds)
        
        # Update history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['val_rmse'].append(val_metrics.rmse)
        history['val_mae'].append(val_metrics.mae)
        history['learning_rate'].append(current_lr)
        
        # Logging
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val RMSE: {val_metrics.rmse:.2f} | "
            f"Val MAE: {val_metrics.mae:.2f} | "
            f"LR: {current_lr:.2e}",
            flush=True
        )
        
        # Checkpointing
        if val_metrics.rmse < best_val_rmse:
            best_val_rmse = val_metrics.rmse
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_rmse': best_val_rmse,
                'config': config,
            }, output_dir / 'best_model.pt')
            
            print(f"  ★ New best model saved (RMSE: {best_val_rmse:.2f})", flush=True)
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}", flush=True)
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs", flush=True)
                break
        
        # Step scheduler with validation metric (ReduceLROnPlateau needs the metric)
        scheduler.step(val_metrics.rmse)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s", flush=True)
    
    # ==========================================================================
    # Test Evaluation
    # ==========================================================================
    print("=" * 60, flush=True)
    print("Test Evaluation", flush=True)
    print("=" * 60, flush=True)
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            predictions = model(sequences)
            test_preds.append(predictions.cpu().numpy())
            test_targets.append(targets.numpy())
    
    test_preds = np.concatenate(test_preds).ravel()
    test_targets = np.concatenate(test_targets).ravel()
    
    test_metrics = calculate_rul_metrics(test_targets, test_preds)
    
    print(f"Test Results:", flush=True)
    print(f"  RMSE: {test_metrics.rmse:.2f}", flush=True)
    print(f"  MAE: {test_metrics.mae:.2f}", flush=True)
    print(f"  R²: {test_metrics.r2:.4f}", flush=True)
    print(f"  NASA Score: {test_metrics.nasa_score:.2f}", flush=True)
    
    # ==========================================================================
    # Save Results
    # ==========================================================================
    results = {
        'model': 'LSTM Baseline (Production)',
        'dataset': args.dataset,
        'config': config,
        'test_metrics': test_metrics.to_dict(),
        'best_val_rmse': best_val_rmse,
        'training_time_seconds': training_time,
        'epochs_trained': epoch + 1,
        'training_history': history,
    }
    
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir}", flush=True)
    print(f"  - best_model.pt", flush=True)
    print(f"  - training_results.json", flush=True)
    print(f"  - config.json", flush=True)
    
    return results


if __name__ == '__main__':
    import traceback
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

