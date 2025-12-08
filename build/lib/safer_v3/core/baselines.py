"""
Baseline Models for SAFER v3.0 RUL Prediction.

This module implements baseline models for benchmarking against the Mamba predictor:
1. LSTMPredictor - Bidirectional LSTM with attention
2. TransformerPredictor - Transformer encoder with positional encoding

These baselines serve two purposes:
1. Performance comparison with state-of-the-art architectures
2. Safety backup in Simplex architecture (TransformerPredictor as baseline controller)

Per the critique modifications, these models provide the comparison baseline
for validating Mamba's performance improvements.

References:
    - Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)
    - Vaswani et al., "Attention is All You Need" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
import math
import logging


logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer.
    
    Adds positional information to input embeddings using sine and cosine
    functions of different frequencies.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor, shape (batch, length, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence aggregation.
    
    Learns to weight different timesteps based on their relevance
    for the final RUL prediction.
    """
    
    def __init__(self, d_model: int):
        """Initialize attention pooling.
        
        Args:
            d_model: Input dimension
        """
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attention pooling.
        
        Args:
            x: Input tensor, shape (batch, length, d_model)
            mask: Optional attention mask, shape (batch, length)
            
        Returns:
            Pooled tensor, shape (batch, d_model)
        """
        # Compute attention weights
        weights = self.attention(x)  # (batch, length, 1)
        
        if mask is not None:
            # Mask out padded positions
            weights = weights.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        
        weights = F.softmax(weights, dim=1)  # (batch, length, 1)
        
        # Weighted sum
        pooled = torch.sum(x * weights, dim=1)  # (batch, d_model)
        
        return pooled


class LSTMPredictor(nn.Module):
    """Bidirectional LSTM with Attention for RUL Prediction.
    
    Architecture:
        Input -> Linear -> BiLSTM -> Attention Pooling -> MLP -> RUL
    
    This model serves as a strong baseline representing traditional
    recurrent architectures for sequence modeling.
    
    Key features:
    - Bidirectional processing for full context
    - Multi-layer stacking for hierarchical features
    - Attention pooling for adaptive aggregation
    - Layer normalization for training stability
    """
    
    def __init__(
        self,
        d_input: int = 14,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        max_rul: int = 125,
    ):
        """Initialize LSTM predictor.
        
        Args:
            d_input: Number of input features
            d_model: Hidden dimension
            n_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
            max_rul: Maximum RUL value for capping
        """
        super().__init__()
        
        self.d_input = d_input
        self.d_model = d_model
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.max_rul = max_rul
        
        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Output dimension after LSTM
        lstm_output_dim = d_model * 2 if bidirectional else d_model
        
        # Layer norm after LSTM
        self.lstm_norm = nn.LayerNorm(lstm_output_dim)
        
        # Attention pooling
        self.attention_pool = AttentionPooling(lstm_output_dim)
        
        # Output MLP
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"Initialized LSTMPredictor: d_input={d_input}, d_model={d_model}, "
            f"n_layers={n_layers}, bidirectional={bidirectional}"
        )
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape (batch, length, d_input)
            lengths: Optional sequence lengths for packed sequences
            return_sequence: If True, return RUL for all timesteps
            
        Returns:
            RUL predictions, shape (batch, 1) or (batch, length, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # LSTM processing
        if lengths is not None:
            # Pack sequence for variable lengths
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(x)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        else:
            lstm_out, _ = self.lstm(x)
        
        # Normalize LSTM output
        lstm_out = self.lstm_norm(lstm_out)
        
        if return_sequence:
            # Predict RUL for each timestep
            rul = self.output_proj(lstm_out)
        else:
            # Attention pooling for final prediction
            pooled = self.attention_pool(lstm_out)
            rul = self.output_proj(pooled)
        
        # Apply RUL capping
        rul = torch.clamp(F.relu(rul), max=self.max_rul)
        
        return rul
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'lstm': sum(p.numel() for p in self.lstm.parameters()),
            'attention': sum(p.numel() for p in self.attention_pool.parameters()),
            'output': sum(p.numel() for p in self.output_proj.parameters()),
        }


class TransformerPredictor(nn.Module):
    """Transformer Encoder for RUL Prediction.
    
    Architecture:
        Input -> Linear -> Positional Encoding -> 
        Transformer Encoder -> Attention Pooling -> MLP -> RUL
    
    This model serves as:
    1. A baseline for comparison with Mamba
    2. The safety backup model in the Simplex architecture
    
    The Transformer provides a well-understood, validated architecture
    that can serve as a trusted baseline when Mamba predictions are
    flagged as unreliable.
    
    Key features:
    - Multi-head self-attention for global context
    - Pre-norm architecture for training stability
    - Learnable positional encoding
    - Attention pooling for sequence aggregation
    """
    
    def __init__(
        self,
        d_input: int = 14,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        max_rul: int = 125,
    ):
        """Initialize Transformer predictor.
        
        Args:
            d_input: Number of input features
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
            max_rul: Maximum RUL value for capping
        """
        super().__init__()
        
        self.d_input = d_input
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_rul = max_rul
        
        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )
        
        # Attention pooling
        self.attention_pool = AttentionPooling(d_model)
        
        # Output MLP
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"Initialized TransformerPredictor: d_input={d_input}, d_model={d_model}, "
            f"n_heads={n_heads}, n_layers={n_layers}"
        )
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape (batch, length, d_input)
            mask: Optional padding mask, shape (batch, length)
            return_sequence: If True, return RUL for all timesteps
            
        Returns:
            RUL predictions, shape (batch, 1) or (batch, length, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask if needed
        attn_mask = None
        if mask is not None:
            # Convert to transformer mask format (True = masked)
            attn_mask = ~mask
        
        # Transformer encoding
        encoded = self.encoder(x, src_key_padding_mask=attn_mask)
        
        if return_sequence:
            # Predict RUL for each timestep
            rul = self.output_proj(encoded)
        else:
            # Attention pooling
            pooled = self.attention_pool(encoded, mask)
            rul = self.output_proj(pooled)
        
        # Apply RUL capping
        rul = torch.clamp(F.relu(rul), max=self.max_rul)
        
        return rul
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'encoder': sum(p.numel() for p in self.encoder.parameters()),
            'attention_pool': sum(p.numel() for p in self.attention_pool.parameters()),
            'output': sum(p.numel() for p in self.output_proj.parameters()),
        }


class CNNLSTMPredictor(nn.Module):
    """CNN-LSTM Hybrid Model for RUL Prediction.
    
    Architecture:
        Input -> 1D CNN (feature extraction) -> LSTM (temporal) -> 
        Attention Pooling -> MLP -> RUL
    
    This architecture combines:
    - CNN for local pattern extraction
    - LSTM for temporal dependencies
    
    Particularly effective for sensor data where local patterns
    (e.g., spikes, trends) are important indicators.
    """
    
    def __init__(
        self,
        d_input: int = 14,
        d_model: int = 64,
        cnn_channels: Tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        max_rul: int = 125,
    ):
        """Initialize CNN-LSTM predictor.
        
        Args:
            d_input: Number of input features
            d_model: Hidden dimension
            cnn_channels: CNN channel progression
            kernel_size: CNN kernel size
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
            max_rul: Maximum RUL value for capping
        """
        super().__init__()
        
        self.d_input = d_input
        self.d_model = d_model
        self.max_rul = max_rul
        
        # CNN layers for feature extraction
        cnn_layers = []
        in_channels = d_input
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(
                    in_channels, out_channels, kernel_size,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Project CNN output to LSTM input
        self.cnn_proj = nn.Linear(cnn_channels[-1], d_model)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )
        
        lstm_out_dim = d_model * 2
        
        # Attention pooling
        self.attention_pool = AttentionPooling(lstm_out_dim)
        
        # Output MLP
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        
        logger.info(
            f"Initialized CNNLSTMPredictor: d_input={d_input}, d_model={d_model}, "
            f"cnn_channels={cnn_channels}"
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape (batch, length, d_input)
            return_sequence: If True, return RUL for all timesteps
            
        Returns:
            RUL predictions
        """
        batch_size, seq_len, _ = x.shape
        
        # CNN expects (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, d_input, length)
        x = self.cnn(x)  # (batch, cnn_out, length)
        x = x.transpose(1, 2)  # (batch, length, cnn_out)
        
        # Project to LSTM input
        x = self.cnn_proj(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        if return_sequence:
            rul = self.output_proj(lstm_out)
        else:
            pooled = self.attention_pool(lstm_out)
            rul = self.output_proj(pooled)
        
        # Apply RUL capping
        rul = torch.clamp(F.relu(rul), max=self.max_rul)
        
        return rul
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
        }


class BaselineFactory:
    """Factory for creating baseline models.
    
    Provides a unified interface for instantiating different baseline
    architectures with consistent configurations.
    """
    
    MODELS = {
        'lstm': LSTMPredictor,
        'transformer': TransformerPredictor,
        'cnn_lstm': CNNLSTMPredictor,
    }
    
    @classmethod
    def create(
        cls,
        model_type: str,
        **kwargs,
    ) -> nn.Module:
        """Create a baseline model.
        
        Args:
            model_type: Type of model ('lstm', 'transformer', 'cnn_lstm')
            **kwargs: Model-specific arguments
            
        Returns:
            Instantiated model
            
        Raises:
            ValueError: If model_type is unknown
        """
        if model_type not in cls.MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(cls.MODELS.keys())}"
            )
        
        return cls.MODELS[model_type](**kwargs)
    
    @classmethod
    def list_models(cls) -> list:
        """List available model types."""
        return list(cls.MODELS.keys())


def compare_model_complexity(
    d_input: int = 14,
    d_model: int = 64,
    n_layers: int = 4,
) -> Dict[str, Dict[str, Any]]:
    """Compare complexity of different model architectures.
    
    Args:
        d_input: Input dimension
        d_model: Model dimension
        n_layers: Number of layers
        
    Returns:
        Dictionary with complexity metrics for each model
    """
    from safer_v3.core.mamba import MambaRULPredictor
    
    models = {
        'mamba': MambaRULPredictor(
            d_input=d_input, d_model=d_model, n_layers=n_layers
        ),
        'lstm': LSTMPredictor(
            d_input=d_input, d_model=d_model, n_layers=n_layers // 2
        ),
        'transformer': TransformerPredictor(
            d_input=d_input, d_model=d_model, n_layers=n_layers // 2
        ),
        'cnn_lstm': CNNLSTMPredictor(
            d_input=d_input, d_model=d_model
        ),
    }
    
    results = {}
    for name, model in models.items():
        params = model.count_parameters()
        results[name] = {
            'total_params': params['total'],
            'trainable_params': params['trainable'],
            'training_complexity': _get_training_complexity(name),
            'inference_complexity': _get_inference_complexity(name),
        }
    
    return results


def _get_training_complexity(model_type: str) -> str:
    """Get theoretical training complexity."""
    complexities = {
        'mamba': 'O(L)',  # Linear in sequence length
        'lstm': 'O(L)',  # Linear in sequence length
        'transformer': 'O(L²)',  # Quadratic due to attention
        'cnn_lstm': 'O(L)',  # Linear in sequence length
    }
    return complexities.get(model_type, 'Unknown')


def _get_inference_complexity(model_type: str) -> str:
    """Get theoretical inference complexity per step."""
    complexities = {
        'mamba': 'O(1)',  # Constant time recurrent
        'lstm': 'O(1)',  # Constant time recurrent
        'transformer': 'O(L)',  # Linear due to KV cache
        'cnn_lstm': 'O(k)',  # Kernel size for CNN
    }
    return complexities.get(model_type, 'Unknown')
