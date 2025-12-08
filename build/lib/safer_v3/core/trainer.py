"""
Training Infrastructure for SAFER v3.0 RUL Prediction.

This module provides complete training infrastructure:
1. CMAPSSDataset - PyTorch Dataset for C-MAPSS data
2. DataModule - Data loading and preprocessing
3. Trainer - Training loop with validation, checkpointing, and early stopping

Key Features:
- Sliding window sequence generation
- RUL capping (piecewise linear degradation model)
- Sensor normalization (z-score standardization)
- Mixed precision training support
- Learning rate scheduling (cosine annealing with warmup)
- Gradient clipping for stability
- TensorBoard logging integration

Data Format (C-MAPSS FD001):
    - 26 columns: unit_id, time, op1, op2, op3, sensor1-21
    - 14 prognostic sensors selected based on correlation analysis
    - RUL capped at 125 cycles (piecewise linear assumption)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import logging
import time
import json
from collections import defaultdict

from safer_v3.utils.config import MambaConfig, PROGNOSTIC_SENSOR_INDICES
from safer_v3.utils.metrics import (
    calculate_rmse, calculate_mae, nasa_scoring_function,
    MetricsTracker, RULMetrics, calculate_rul_metrics
)


logger = logging.getLogger(__name__)


# C-MAPSS column names
CMAPSS_COLUMNS = (
    ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] +
    [f'sensor_{i}' for i in range(1, 22)]
)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Data
    data_dir: str = "CMAPSSData"
    dataset: str = "FD001"
    window_size: int = 30
    stride: int = 1
    max_rul: int = 125
    
    # Training
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    
    # Scheduler
    scheduler_type: str = "cosine"  # "cosine", "onecycle", "none"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.1
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = True
    
    # Logging
    log_interval: int = 10
    tensorboard_dir: Optional[str] = "runs"


class CMAPSSDataset(Dataset):
    """PyTorch Dataset for C-MAPSS turbofan engine data.
    
    Handles:
    - Loading raw text data
    - Sensor selection (14 prognostic sensors)
    - Z-score normalization
    - Sliding window sequence generation
    - RUL calculation with capping
    
    The dataset uses a piecewise linear degradation model where
    RUL is capped at max_rul (default 125) cycles, assuming
    degradation is not observable in early life.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        rul_path: Optional[Union[str, Path]] = None,
        window_size: int = 30,
        stride: int = 1,
        max_rul: int = 125,
        sensor_indices: Optional[List[int]] = None,
        normalize: bool = True,
        scaler_params: Optional[Dict[str, np.ndarray]] = None,
    ):
        """Initialize C-MAPSS dataset.
        
        Args:
            data_path: Path to train/test data file
            rul_path: Path to RUL file (for test data only)
            window_size: Sequence length for sliding window
            stride: Step size for sliding window
            max_rul: Maximum RUL value (capping threshold)
            sensor_indices: Indices of sensors to use (0-indexed from sensor columns)
            normalize: Whether to apply z-score normalization
            scaler_params: Pre-computed mean/std for normalization
        """
        self.window_size = window_size
        self.stride = stride
        self.max_rul = max_rul
        self.sensor_indices = sensor_indices or PROGNOSTIC_SENSOR_INDICES
        self.normalize = normalize
        
        # Load and process data
        self.data, self.rul_targets = self._load_data(data_path, rul_path)
        
        # Compute or use provided normalization parameters
        if normalize:
            if scaler_params is not None:
                self.mean = scaler_params['mean']
                self.std = scaler_params['std']
            else:
                self.mean, self.std = self._compute_normalization_params()
            
            # Apply normalization
            self.data = self._normalize(self.data)
        else:
            self.mean = None
            self.std = None
        
        # Generate sliding window samples
        self.samples = self._generate_samples()
        
        logger.info(
            f"Created CMAPSSDataset: {len(self.samples)} samples, "
            f"window_size={window_size}, stride={stride}"
        )
    
    def _load_data(
        self,
        data_path: Union[str, Path],
        rul_path: Optional[Union[str, Path]],
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Load and parse C-MAPSS data files.
        
        Args:
            data_path: Path to data file
            rul_path: Optional path to RUL file
            
        Returns:
            Tuple of (data_dict, rul_dict) keyed by unit_id
        """
        data_path = Path(data_path)
        
        # Load main data file
        # Format: space-separated, no header
        df = pd.read_csv(
            data_path,
            sep=r'\s+',
            header=None,
            names=CMAPSS_COLUMNS,
        )
        
        # Load RUL file if provided (test data)
        rul_at_end = {}
        if rul_path is not None:
            rul_path = Path(rul_path)
            rul_values = pd.read_csv(rul_path, header=None).values.flatten()
            for i, rul in enumerate(rul_values, 1):
                rul_at_end[i] = rul
        
        # Process each unit
        data_dict = {}
        rul_dict = {}
        
        for unit_id in df['unit_id'].unique():
            unit_df = df[df['unit_id'] == unit_id].copy()
            unit_df = unit_df.sort_values('time_cycles')
            
            # Extract selected sensor columns
            # sensor_indices already contains 1-indexed sensor numbers (e.g., [2, 3, 4, ...])
            sensor_cols = [f'sensor_{i}' for i in self.sensor_indices]
            sensor_data = unit_df[sensor_cols].values.astype(np.float32)
            
            data_dict[unit_id] = sensor_data
            
            # Calculate RUL for each timestep
            max_cycle = unit_df['time_cycles'].max()
            
            if unit_id in rul_at_end:
                # Test data: RUL at end is provided
                rul_values = np.arange(max_cycle - 1, -1, -1) + rul_at_end[unit_id]
            else:
                # Training data: RUL counts down to 0
                rul_values = np.arange(max_cycle - 1, -1, -1)
            
            # Apply RUL capping (piecewise linear model)
            rul_values = np.minimum(rul_values, self.max_rul).astype(np.float32)
            rul_dict[unit_id] = rul_values
        
        return data_dict, rul_dict
    
    def _compute_normalization_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std for z-score normalization.
        
        Returns:
            Tuple of (mean, std) arrays
        """
        all_data = np.concatenate(list(self.data.values()), axis=0)
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)
        
        # Prevent division by zero
        std = np.where(std < 1e-6, 1.0, std)
        
        return mean, std
    
    def _normalize(self, data: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Apply z-score normalization.
        
        Args:
            data: Dictionary of sensor arrays
            
        Returns:
            Normalized data dictionary
        """
        normalized = {}
        for unit_id, sensor_data in data.items():
            normalized[unit_id] = (sensor_data - self.mean) / self.std
        return normalized
    
    def _generate_samples(self) -> List[Tuple[int, int, int]]:
        """Generate sliding window sample indices.
        
        Returns:
            List of (unit_id, start_idx, end_idx) tuples
        """
        samples = []
        
        for unit_id, sensor_data in self.data.items():
            seq_len = len(sensor_data)
            
            # Generate windows with stride
            for start_idx in range(0, seq_len - self.window_size + 1, self.stride):
                end_idx = start_idx + self.window_size
                samples.append((unit_id, start_idx, end_idx))
        
        return samples
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (sequence, rul_target)
            - sequence: shape (window_size, n_sensors)
            - rul_target: scalar RUL at end of window
        """
        unit_id, start_idx, end_idx = self.samples[idx]
        
        # Extract sequence
        sequence = self.data[unit_id][start_idx:end_idx]
        
        # Get RUL at end of window
        rul = self.rul_targets[unit_id][end_idx - 1]
        
        return (
            torch.from_numpy(sequence),
            torch.tensor([rul], dtype=torch.float32),
        )
    
    def get_scaler_params(self) -> Dict[str, np.ndarray]:
        """Get normalization parameters for transfer to test set.
        
        Returns:
            Dictionary with 'mean' and 'std' arrays
        """
        return {'mean': self.mean, 'std': self.std}
    
    def get_unit_sequence(self, unit_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get full sequence for a unit (for evaluation).
        
        Args:
            unit_id: Unit identifier
            
        Returns:
            Tuple of (full_sequence, full_rul)
        """
        return (
            torch.from_numpy(self.data[unit_id]),
            torch.from_numpy(self.rul_targets[unit_id]),
        )


class DataModule:
    """Data module for managing train/val/test data loading.
    
    Handles:
    - Train/validation split
    - Test data with separate RUL file
    - Consistent normalization across splits
    - DataLoader creation
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        dataset: str = "FD001",
        window_size: int = 30,
        stride: int = 1,
        max_rul: int = 125,
        batch_size: int = 64,
        val_split: float = 0.2,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
    ):
        """Initialize data module.
        
        Args:
            data_dir: Directory containing C-MAPSS data
            dataset: Dataset name (FD001, FD002, FD003, FD004)
            window_size: Sequence length
            stride: Sliding window stride
            max_rul: Maximum RUL for capping
            batch_size: Batch size for data loaders
            val_split: Fraction of training data for validation
            num_workers: DataLoader workers
            pin_memory: Pin memory for GPU transfer
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.window_size = window_size
        self.stride = stride
        self.max_rul = max_rul
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        
        # Will be populated by setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.scaler_params = None
    
    def setup(self) -> None:
        """Set up datasets."""
        train_path = self.data_dir / f"train_{self.dataset}.txt"
        test_path = self.data_dir / f"test_{self.dataset}.txt"
        rul_path = self.data_dir / f"RUL_{self.dataset}.txt"
        
        # Create full training dataset first (for normalization params)
        full_train = CMAPSSDataset(
            data_path=train_path,
            window_size=self.window_size,
            stride=self.stride,
            max_rul=self.max_rul,
            normalize=True,
        )
        
        self.scaler_params = full_train.get_scaler_params()
        
        # Split training data by units
        unit_ids = list(full_train.data.keys())
        np.random.seed(self.seed)
        np.random.shuffle(unit_ids)
        
        n_val = int(len(unit_ids) * self.val_split)
        val_units = set(unit_ids[:n_val])
        train_units = set(unit_ids[n_val:])
        
        # Filter samples by unit
        train_samples = [
            s for s in full_train.samples if s[0] in train_units
        ]
        val_samples = [
            s for s in full_train.samples if s[0] in val_units
        ]
        
        # Create train/val datasets with filtered samples
        self.train_dataset = full_train
        self.train_dataset.samples = train_samples
        
        # Create validation dataset (share data but different samples)
        self.val_dataset = CMAPSSDataset(
            data_path=train_path,
            window_size=self.window_size,
            stride=self.stride,
            max_rul=self.max_rul,
            normalize=True,
            scaler_params=self.scaler_params,
        )
        self.val_dataset.samples = val_samples
        
        # Create test dataset
        self.test_dataset = CMAPSSDataset(
            data_path=test_path,
            rul_path=rul_path,
            window_size=self.window_size,
            stride=1,  # No stride for test evaluation
            max_rul=self.max_rul,
            normalize=True,
            scaler_params=self.scaler_params,
        )
        
        logger.info(
            f"Data setup complete: {len(self.train_dataset)} train, "
            f"{len(self.val_dataset)} val, {len(self.test_dataset)} test samples"
        )
    
    def train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class EarlyStopping:
    """Early stopping to prevent overfitting.
    
    Monitors validation loss and stops training if no improvement
    is seen for `patience` epochs.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for monitoring direction
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation metric
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            score = -score
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class Trainer:
    """Training loop for RUL prediction models.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient clipping
    - Learning rate scheduling (cosine, one-cycle)
    - Checkpointing with best model tracking
    - Early stopping
    - TensorBoard logging
    - Comprehensive metrics tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        data_module: Optional[DataModule] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            data_module: Optional data module (can be set later)
        """
        self.model = model
        self.config = config
        self.data_module = data_module
        
        # Setup device
        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Setup scheduler (will be initialized in fit)
        self.scheduler = None
        
        # Mixed precision
        self.use_amp = config.use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Loss function (Huber loss is more robust to outliers)
        self.criterion = nn.HuberLoss(delta=10.0)
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
        )
        
        # Tracking
        self.metrics_tracker = MetricsTracker()
        self.best_val_rmse = float('inf')
        self.best_model_state = None
        self.current_epoch = 0
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = None
        if config.tensorboard_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(config.tensorboard_dir)
                logger.info(f"TensorBoard logging enabled: {config.tensorboard_dir}")
            except Exception as e:
                logger.warning(f"TensorBoard not available: {e}")
        
        logger.info(f"Trainer initialized on {self.device}")
    
    def _setup_scheduler(self, num_batches: int) -> None:
        """Setup learning rate scheduler.
        
        Args:
            num_batches: Number of batches per epoch
        """
        total_steps = num_batches * self.config.num_epochs
        
        if self.config.scheduler_type == 'cosine':
            # Cosine annealing with warm restarts
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=num_batches * 10,  # Restart every 10 epochs
                T_mult=2,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=self.config.warmup_epochs / self.config.num_epochs,
                anneal_strategy='cos',
                final_div_factor=self.config.learning_rate / self.config.min_lr,
            )
        else:
            self.scheduler = None
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch_idx, (sequences, targets) in enumerate(dataloader):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with optional AMP
            if self.use_amp:
                with autocast():
                    predictions = self.model(sequences)
                    loss = self.criterion(predictions, targets)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(sequences)
                loss = self.criterion(predictions, targets)
                
                loss.backward()
                
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            all_preds.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                logger.debug(
                    f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}"
                )
        
        # Compute epoch metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'rmse': calculate_rmse(all_preds, all_targets),
            'mae': calculate_mae(all_preds, all_targets),
            'score': nasa_scoring_function(all_preds, all_targets),
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for sequences, targets in dataloader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            if self.use_amp:
                with autocast():
                    predictions = self.model(sequences)
                    loss = self.criterion(predictions, targets)
            else:
                predictions = self.model(sequences)
                loss = self.criterion(predictions, targets)
            
            total_loss += loss.item()
            all_preds.append(predictions.cpu())
            all_targets.append(targets.cpu())
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'rmse': calculate_rmse(all_preds, all_targets),
            'mae': calculate_mae(all_preds, all_targets),
            'score': nasa_scoring_function(all_preds, all_targets),
        }
        
        return metrics
    
    def fit(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            train_loader: Training data loader (uses data_module if None)
            val_loader: Validation data loader (uses data_module if None)
            
        Returns:
            Dictionary with training history
        """
        # Setup data loaders
        if train_loader is None:
            if self.data_module is None:
                raise ValueError("No data provided")
            self.data_module.setup()
            train_loader = self.data_module.train_dataloader()
            val_loader = self.data_module.val_dataloader()
        
        # Setup scheduler
        self._setup_scheduler(len(train_loader))
        
        # Training history
        history = defaultdict(list)
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start
            
            # Log metrics
            for key, value in train_metrics.items():
                history[f'train_{key}'].append(value)
                if self.writer:
                    self.writer.add_scalar(f'train/{key}', value, epoch)
            
            for key, value in val_metrics.items():
                history[f'val_{key}'].append(value)
                if self.writer:
                    self.writer.add_scalar(f'val/{key}', value, epoch)
            
            # Track best model
            if val_metrics['rmse'] < self.best_val_rmse:
                self.best_val_rmse = val_metrics['rmse']
                self.best_model_state = self.model.state_dict().copy()
                logger.info(f"New best RMSE: {self.best_val_rmse:.4f}")
            
            # Optional: compute detailed RUL metrics (not persisted here)
            # Detailed metrics available via calculate_rul_metrics if needed.
            
            # Logging
            lr = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} "
                f"({epoch_time:.1f}s) - "
                f"Train Loss: {train_metrics['loss']:.4f}, RMSE: {train_metrics['rmse']:.2f} - "
                f"Val Loss: {val_metrics['loss']:.4f}, RMSE: {val_metrics['rmse']:.2f} - "
                f"LR: {lr:.2e}"
            )
            
            # Checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            # Early stopping
            if self.early_stopping(val_metrics['rmse']):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 60:.1f} minutes")
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model with RMSE: {self.best_val_rmse:.4f}")
        
        # Save final model
        self.save_checkpoint("best_model.pt", is_best=True)
        
        # Close TensorBoard
        if self.writer:
            self.writer.close()
        
        return dict(history)
    
    @torch.no_grad()
    def test(
        self,
        test_loader: Optional[DataLoader] = None,
    ) -> RULMetrics:
        """Evaluate on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Test metrics
        """
        if test_loader is None:
            if self.data_module is None:
                raise ValueError("No test data provided")
            test_loader = self.data_module.test_dataloader()
        
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        for sequences, targets in test_loader:
            sequences = sequences.to(self.device)
            predictions = self.model(sequences)
            
            all_preds.append(predictions.cpu())
            all_targets.append(targets)
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        metrics = RULMetrics(
            rmse=calculate_rmse(all_preds, all_targets),
            mae=calculate_mae(all_preds, all_targets),
            score=nasa_scoring_function(all_preds, all_targets),
        )
        
        logger.info(
            f"Test Results - RMSE: {metrics.rmse:.2f}, "
            f"MAE: {metrics.mae:.2f}, Score: {metrics.score:.2f}"
        )
        
        return metrics
    
    def save_checkpoint(
        self,
        filename: str,
        is_best: bool = False,
    ) -> None:
        """Save training checkpoint.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_rmse': self.best_val_rmse,
            'config': self.config.__dict__,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.checkpoint_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent dirs exist
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_rmse = checkpoint.get('best_val_rmse', float('inf'))
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {path}, epoch {self.current_epoch}")
    
    @torch.no_grad()
    def predict(
        self,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        """Make predictions.
        
        Args:
            sequences: Input sequences, shape (batch, length, n_sensors)
            
        Returns:
            RUL predictions, shape (batch, 1)
        """
        self.model.eval()
        sequences = sequences.to(self.device)
        
        if self.use_amp:
            with autocast():
                predictions = self.model(sequences)
        else:
            predictions = self.model(sequences)
        
        return predictions.cpu()
    
    @torch.no_grad()
    def benchmark_inference(
        self,
        batch_size: int = 1,
        sequence_length: int = 30,
        n_iterations: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """Benchmark inference latency.
        
        Args:
            batch_size: Batch size for inference
            sequence_length: Input sequence length
            n_iterations: Number of iterations to average
            warmup: Warmup iterations
            
        Returns:
            Dictionary with latency statistics
        """
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(
            batch_size, sequence_length, self.model.d_input,
            device=self.device,
        )
        
        # Warmup
        for _ in range(warmup):
            _ = self.model(dummy_input)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = self.model(dummy_input)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            latencies.append((time.perf_counter() - start) * 1000)  # ms
        
        latencies = np.array(latencies)
        
        results = {
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
        }
        
        logger.info(
            f"Inference latency: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms "
            f"(p95: {results['p95_ms']:.2f} ms)"
        )
        
        return results
