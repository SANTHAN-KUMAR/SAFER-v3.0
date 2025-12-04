#!/usr/bin/env python3
"""
Train LSTM Baseline Model for SAFER v3.0 (DAL C Safety Backup).

This script trains the LSTM baseline predictor that serves as the
safety fallback in the Simplex architecture when Mamba predictions
are flagged as unreliable.

DAL C Classification: This baseline is part of the safety-critical path.
"""

import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from safer_v3.core.baselines import LSTMPredictor
from safer_v3.core.trainer import DataModule
from safer_v3.utils.metrics import RULMetrics, calculate_rul_metrics


def train_baseline(
    data_dir: str = "CMAPSSData",
    dataset: str = "FD001",
    output_dir: str = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    d_model: int = 64,
    n_layers: int = 2,
    dropout: float = 0.2,
    window_size: int = 30,
    max_rul: int = 125,
    device: str = None,
):
    """Train LSTM baseline model.
    
    Args:
        data_dir: Path to C-MAPSS data directory
        dataset: Dataset name (FD001, FD002, etc.)
        output_dir: Output directory (auto-generated if None)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        d_model: Hidden dimension
        n_layers: Number of LSTM layers
        dropout: Dropout probability
        window_size: Sequence window size
        max_rul: Maximum RUL cap
        device: Device to use (auto-detect if None)
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "outputs" / f"lstm_{dataset}_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load data
    print(f"\n{'='*60}")
    print(f"Loading {dataset} dataset...")
    print(f"{'='*60}")
    
    data_module = DataModule(
        data_dir=data_dir,
        dataset=dataset,
        window_size=window_size,
        batch_size=batch_size,
        max_rul=max_rul,
    )
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    print(f"Train samples: {len(data_module.train_dataset)}")
    print(f"Val samples: {len(data_module.val_dataset)}")
    print(f"Test samples: {len(data_module.test_dataset)}")
    
    # Create model
    print(f"\n{'='*60}")
    print("Creating LSTM Baseline Model (DAL C)")
    print(f"{'='*60}")
    
    model = LSTMPredictor(
        d_input=14,  # 14 selected sensors
        d_model=d_model,
        n_layers=n_layers,
        dropout=dropout,
        bidirectional=True,
        max_rul=max_rul,
    )
    model = model.to(device)
    
    # Print model info
    params = model.count_parameters()
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    
    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training for {epochs} epochs...")
    print(f"{'='*60}")
    
    best_val_rmse = float('inf')
    history = {
        'train_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'learning_rate': [],
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            sequences, targets = batch
            x = sequences.to(device)
            y = targets.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred.squeeze(), y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                sequences, targets = batch
                x = sequences.to(device)
                y = targets.to(device)

                y_pred = model(x)
                val_preds.extend(y_pred.squeeze().cpu().numpy())
                val_targets.extend(y.cpu().numpy())
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        # Compute metrics
        val_rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
        val_mae = np.mean(np.abs(val_preds - val_targets))
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Record history
        history['train_loss'].append(np.mean(train_losses))
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['learning_rate'].append(current_lr)
        
        print(f"Epoch {epoch+1}: train_loss={np.mean(train_losses):.4f}, "
              f"val_rmse={val_rmse:.2f}, val_mae={val_mae:.2f}, lr={current_lr:.6f}")
        
        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'config': {
                    'd_input': 14,
                    'd_model': d_model,
                    'n_layers': n_layers,
                    'dropout': dropout,
                    'bidirectional': True,
                    'max_rul': max_rul,
                },
            }, output_dir / "lstm_best.pt")
            print(f"  -> Saved new best model (val_rmse={val_rmse:.2f})")
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'config': {
            'd_input': 14,
            'd_model': d_model,
            'n_layers': n_layers,
            'dropout': dropout,
            'bidirectional': True,
            'max_rul': max_rul,
        },
    }, output_dir / "lstm_final.pt")
    
    # Test evaluation
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}")
    
    # Load best model
    checkpoint = torch.load(output_dir / "lstm_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences, targets = batch
            x = sequences.to(device)
            y = targets.to(device)

            y_pred = model(x)
            test_preds.extend(y_pred.squeeze().cpu().numpy())
            test_targets.extend(y.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    
    # Compute test metrics using utility function
    test_metrics_obj = calculate_rul_metrics(test_targets, test_preds)
    test_metrics = test_metrics_obj.to_dict()
    
    print(f"\nTest Results:")
    print(f"  RMSE: {test_metrics['rmse']:.2f} cycles")
    print(f"  MAE: {test_metrics['mae']:.2f} cycles")
    print(f"  NASA Score: {test_metrics['nasa_score']:.2f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    
    # Save training config and results
    results = {
        'model': 'LSTM Baseline (DAL C)',
        'dataset': dataset,
        'config': {
            'd_input': 14,
            'd_model': d_model,
            'n_layers': n_layers,
            'dropout': dropout,
            'bidirectional': True,
            'max_rul': max_rul,
            'window_size': window_size,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': epochs,
        },
        'test_metrics': test_metrics,
        'best_val_rmse': float(best_val_rmse),
        'training_history': {k: [float(v) for v in vals] for k, vals in history.items()},
    }
    
    with open(output_dir / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*60}")
    
    return output_dir, test_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LSTM Baseline for SAFER v3.0")
    parser.add_argument("--data_dir", type=str, default="CMAPSSData")
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    train_baseline(
        data_dir=args.data_dir,
        dataset=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        window_size=args.window_size,
        device=args.device,
    )
