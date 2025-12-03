"""
Mamba Architecture for SAFER v3.0 RUL Prediction.

This module implements the complete Mamba-based prognostic model:
1. RMSNorm for stable normalization
2. MambaBlock with residual connections
3. MambaRULPredictor - full model for C-MAPSS RUL prediction

Key Features:
- Selective State Space Model for efficient sequence modeling
- O(1) inference time per step (constant regardless of history)
- JIT compilation support via torch.compile
- ONNX export capability for deployment

Architecture Parameters (from specification):
- Input Dimension: D_in = 14 (C-MAPSS sensor subset)
- Model Dimension: D_model = 64
- State Dimension: D_state = 16
- Number of Layers: N = 4
- Normalization: RMSNorm

References:
    - Gu et al., "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path
import math
import logging

from safer_v3.core.ssm_ops import SelectiveSSM


logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    RMSNorm provides normalization without the mean-centering of LayerNorm,
    which is computationally more efficient and works well with SSMs.
    
    Formula: y = x / RMS(x) * γ
    where RMS(x) = sqrt(mean(x²) + ε)
    
    Attributes:
        d_model: Dimension of the input
        eps: Small constant for numerical stability
        weight: Learnable scale parameter (γ)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """Initialize RMSNorm.
        
        Args:
            d_model: Dimension of input features
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization.
        
        Args:
            x: Input tensor, shape (..., d_model)
            
        Returns:
            Normalized tensor, same shape as input
        """
        # Compute RMS: sqrt(mean(x²))
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.
        
        Args:
            x: Input tensor, shape (..., d_model)
            
        Returns:
            Normalized and scaled tensor
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MambaBlock(nn.Module):
    """Single Mamba block with selective SSM and residual connection.
    
    Architecture:
        x -> RMSNorm -> SelectiveSSM -> + -> output
        |_____________________________|
                (residual)
    
    The block consists of:
    1. Pre-normalization with RMSNorm
    2. Selective SSM processing
    3. Residual connection
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        bias: bool = False,
        dt_rank: Union[int, str] = "auto",
    ):
        """Initialize MambaBlock.
        
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: Expansion factor for inner dimension
            dropout: Dropout probability
            bias: Use bias in linear layers
            dt_rank: Rank of dt projection
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Pre-normalization
        self.norm = RMSNorm(d_model)
        
        # Selective SSM
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            bias=bias,
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        inference_params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Forward pass through Mamba block.
        
        Args:
            x: Input tensor, shape (batch, length, d_model)
            inference_params: Optional dict with inference state
            
        Returns:
            Output tensor, shape (batch, length, d_model)
        """
        # Pre-norm
        residual = x
        x = self.norm(x)
        
        # SSM
        x = self.ssm(x)
        
        # Dropout and residual
        x = self.dropout(x)
        x = x + residual
        
        return x
    
    def step(
        self,
        x_t: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single step for recurrent inference.
        
        Args:
            x_t: Current input, shape (batch, d_model)
            conv_state: Convolution state
            ssm_state: SSM hidden state
            
        Returns:
            Tuple of (output, new_conv_state, new_ssm_state)
        """
        residual = x_t
        x_t = self.norm(x_t)
        
        x_t, conv_state, ssm_state = self.ssm.step(x_t, conv_state, ssm_state)
        
        x_t = x_t + residual
        
        return x_t, conv_state, ssm_state
    
    def init_state(
        self,
        batch_size: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize states for inference.
        
        Args:
            batch_size: Batch size
            device: Device for tensors
            dtype: Data type
            
        Returns:
            Tuple of (conv_state, ssm_state)
        """
        return self.ssm.init_state(batch_size, device, dtype)


class MambaRULPredictor(nn.Module):
    """Complete Mamba-based RUL Prediction Model.
    
    This model predicts Remaining Useful Life (RUL) for turbofan engines
    using the Mamba selective state space architecture.
    
    Architecture:
        Input (14 sensors) -> Input Projection -> [MambaBlock × N] -> 
        Final Norm -> Output Projection -> RUL
    
    Key features:
    - Linear time complexity O(L) for training
    - Constant time O(1) for inference
    - Selective attention mechanism via input-dependent SSM
    - RUL capping for realistic predictions
    
    Safety Documentation (DAL E - Non-Safety-Critical):
    ------------------------------------------------
    This neural network model is classified as DAL E (non-safety-critical)
    per DO-178C guidelines. It serves as a predictive component only.
    Safety-critical decisions are made by the LPV-SINDy Monitor (DAL C)
    and Simplex Switch Logic (DAL C) which provide certification guarantees.
    
    The Mamba model's predictions are:
    1. Validated against physics-based LPV-SINDy model
    2. Bounded by conformal prediction intervals
    3. Subject to override by baseline safety model
    """
    
    def __init__(
        self,
        d_input: int = 14,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 4,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        max_rul: int = 125,
        use_jit: bool = True,
        dt_rank: Union[int, str] = "auto",
    ):
        """Initialize MambaRULPredictor.
        
        Args:
            d_input: Number of input sensors
            d_model: Model dimension
            d_state: SSM state dimension
            n_layers: Number of Mamba blocks
            d_conv: Convolution kernel size
            expand: Expansion factor
            dropout: Dropout probability
            max_rul: Maximum RUL value for capping
            use_jit: Enable JIT compilation for inference
            dt_rank: Rank of dt projection
        """
        super().__init__()
        
        self.d_input = d_input
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.max_rul = max_rul
        self.use_jit = use_jit
        
        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)
        
        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                dt_rank=dt_rank,
            )
            for _ in range(n_layers)
        ])
        
        # Final normalization
        self.final_norm = RMSNorm(d_model)
        
        # Output projection for RUL prediction
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        
        # Initialize weights
        self._init_weights()
        
        # JIT compilation flag
        self._jit_step_fn = None
        
        logger.info(
            f"Initialized MambaRULPredictor: "
            f"d_input={d_input}, d_model={d_model}, d_state={d_state}, "
            f"n_layers={n_layers}, max_rul={max_rul}"
        )
    
    def _init_weights(self) -> None:
        """Initialize model weights using proper initialization schemes."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                # Kaiming initialization for conv layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass for training.
        
        Args:
            x: Input sensor sequence, shape (batch, length, d_input)
            return_sequence: If True, return RUL for all timesteps
            
        Returns:
            RUL predictions:
            - If return_sequence=False: shape (batch, 1) - final RUL only
            - If return_sequence=True: shape (batch, length, 1) - RUL per step
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)  # (batch, length, d_model)
        
        # Process through Mamba blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.final_norm(x)  # (batch, length, d_model)
        
        if return_sequence:
            # Predict RUL for all timesteps
            rul = self.output_proj(x)  # (batch, length, 1)
        else:
            # Only use last timestep for final prediction
            x_last = x[:, -1, :]  # (batch, d_model)
            rul = self.output_proj(x_last)  # (batch, 1)
        
        # Apply RUL capping (ReLU ensures non-negative, clamp ensures max)
        rul = torch.clamp(F.relu(rul), max=self.max_rul)
        
        return rul
    
    @torch.no_grad()
    def predict_step(
        self,
        x_t: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Single step prediction for O(1) inference.
        
        This method enables real-time inference with constant time complexity,
        regardless of the sequence history length.
        
        Args:
            x_t: Current sensor reading, shape (batch, d_input)
            states: List of (conv_state, ssm_state) for each layer
            
        Returns:
            Tuple of (rul_prediction, new_states)
            - rul_prediction: shape (batch, 1)
            - new_states: Updated states for next step
        """
        batch_size = x_t.shape[0]
        device = x_t.device
        dtype = x_t.dtype
        
        # Initialize states if not provided
        if states is None:
            states = [
                layer.init_state(batch_size, device, dtype)
                for layer in self.layers
            ]
        
        # Input projection
        x = self.input_proj(x_t)  # (batch, d_model)
        
        # Process through layers with state updates
        new_states = []
        for layer, (conv_state, ssm_state) in zip(self.layers, states):
            x, new_conv, new_ssm = layer.step(x, conv_state, ssm_state)
            new_states.append((new_conv, new_ssm))
        
        # Final norm and output
        x = self.final_norm(x)
        rul = self.output_proj(x)
        
        # Apply capping
        rul = torch.clamp(F.relu(rul), max=self.max_rul)
        
        return rul, new_states
    
    def get_compiled_step(self) -> callable:
        """Get JIT-compiled step function for maximum inference speed.
        
        Uses torch.compile with reduce-overhead mode for optimal latency.
        
        Returns:
            Compiled step function
        """
        if self._jit_step_fn is None and self.use_jit:
            try:
                self._jit_step_fn = torch.compile(
                    self.predict_step,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                logger.info("Successfully compiled inference step function")
            except Exception as e:
                logger.warning(f"JIT compilation failed, using eager mode: {e}")
                self._jit_step_fn = self.predict_step
        
        return self._jit_step_fn if self._jit_step_fn else self.predict_step
    
    def init_inference_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize inference states for all layers.
        
        Args:
            batch_size: Batch size for inference
            device: Target device
            dtype: Data type
            
        Returns:
            List of (conv_state, ssm_state) tuples
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        
        return [
            layer.init_state(batch_size, device, dtype)
            for layer in self.layers
        ]
    
    def export_onnx(
        self,
        path: Union[str, Path],
        batch_size: int = 1,
        opset_version: int = 17,
    ) -> None:
        """Export model to ONNX format for deployment.
        
        Exports the single-step inference model for real-time deployment
        on ONNX Runtime or other inference engines.
        
        Args:
            path: Output path for ONNX file
            batch_size: Batch size to export with
            opset_version: ONNX opset version
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create wrapper for ONNX export (flattened states)
        class ONNXWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x_t, *flat_states):
                # Reconstruct states from flat list
                states = []
                for i in range(0, len(flat_states), 2):
                    states.append((flat_states[i], flat_states[i + 1]))
                
                rul, new_states = self.model.predict_step(x_t, states)
                
                # Flatten new states
                flat_new = []
                for conv, ssm in new_states:
                    flat_new.extend([conv, ssm])
                
                return (rul, *flat_new)
        
        wrapper = ONNXWrapper(self)
        wrapper.eval()
        
        # Create dummy inputs
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        x_t = torch.randn(batch_size, self.d_input, device=device, dtype=dtype)
        states = self.init_inference_state(batch_size, device, dtype)
        
        flat_states = []
        for conv, ssm in states:
            flat_states.extend([conv, ssm])
        
        # Export
        torch.onnx.export(
            wrapper,
            (x_t, *flat_states),
            str(path),
            opset_version=opset_version,
            input_names=['x_t'] + [f'state_{i}' for i in range(len(flat_states))],
            output_names=['rul'] + [f'new_state_{i}' for i in range(len(flat_states))],
            dynamic_axes={
                'x_t': {0: 'batch'},
                'rul': {0: 'batch'},
            },
        )
        
        logger.info(f"Exported ONNX model to {path}")
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Save model checkpoint.
        
        Args:
            path: Output path for checkpoint
            optimizer: Optimizer state to save
            epoch: Current epoch number
            metrics: Optional metrics dict
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'd_input': self.d_input,
                'd_model': self.d_model,
                'd_state': self.d_state,
                'n_layers': self.n_layers,
                'max_rul': self.max_rul,
            },
            'epoch': epoch,
            'metrics': metrics or {},
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> Tuple['MambaRULPredictor', Dict[str, Any]]:
        """Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file
            device: Device to load model to
            
        Returns:
            Tuple of (model, checkpoint_dict)
        """
        path = Path(path)
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        
        # Create model with saved config
        config = checkpoint['config']
        model = cls(
            d_input=config['d_input'],
            d_model=config['d_model'],
            d_state=config['d_state'],
            n_layers=config['n_layers'],
            max_rul=config['max_rul'],
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"Loaded checkpoint from {path}, epoch {checkpoint.get('epoch', 'N/A')}")
        
        return model, checkpoint
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters.
        
        Returns:
            Dict with total, trainable, and per-component counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Per-component breakdown
        input_proj = sum(p.numel() for p in self.input_proj.parameters())
        layers = sum(p.numel() for p in self.layers.parameters())
        output = sum(p.numel() for p in self.output_proj.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'input_projection': input_proj,
            'mamba_layers': layers,
            'output_projection': output,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        params = self.count_parameters()
        return (
            f"MambaRULPredictor(\n"
            f"  d_input={self.d_input},\n"
            f"  d_model={self.d_model},\n"
            f"  d_state={self.d_state},\n"
            f"  n_layers={self.n_layers},\n"
            f"  max_rul={self.max_rul},\n"
            f"  total_params={params['total']:,},\n"
            f"  trainable_params={params['trainable']:,}\n"
            f")"
        )


class MambaEnsemble(nn.Module):
    """Ensemble of Mamba models for uncertainty estimation.
    
    This class provides uncertainty quantification through model ensembling,
    which can be used alongside conformal prediction for robust intervals.
    """
    
    def __init__(
        self,
        n_models: int = 5,
        **model_kwargs,
    ):
        """Initialize ensemble.
        
        Args:
            n_models: Number of models in ensemble
            **model_kwargs: Arguments passed to each MambaRULPredictor
        """
        super().__init__()
        
        self.n_models = n_models
        self.models = nn.ModuleList([
            MambaRULPredictor(**model_kwargs)
            for _ in range(n_models)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass through ensemble.
        
        Args:
            x: Input tensor, shape (batch, length, d_input)
            return_all: If True, return individual predictions
            
        Returns:
            If return_all=False: Mean prediction, shape (batch, 1)
            If return_all=True: Tuple of (mean, std, all_predictions)
        """
        predictions = torch.stack([model(x) for model in self.models], dim=0)
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        if return_all:
            return mean, std, predictions
        return mean
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty estimation.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean_prediction, uncertainty_std)
        """
        mean, std, _ = self.forward(x, return_all=True)
        return mean, std
