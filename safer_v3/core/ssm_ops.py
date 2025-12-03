"""
State Space Model Operations for SAFER v3.0.

This module implements the core SSM operations required for the Mamba architecture:
1. Zero-Order Hold (ZOH) discretization
2. Parallel associative scan for training
3. O(1) recurrent step for inference
4. Numerically stable implementations

Mathematical Foundation:
-----------------------
Continuous SSM:
    h'(t) = A·h(t) + B·x(t)
    y(t) = C·h(t)

Discretized SSM (ZOH):
    Ā = exp(Δ·A)
    B̄ = (Δ·A)^(-1)·(exp(Δ·A) - I)·(Δ·B)
    
For diagonal A (as in Mamba):
    Ā = exp(Δ·A)
    B̄ = Δ·B  (simplified when A is diagonal)

Recurrence:
    h_t = Ā·h_{t-1} + B̄·x_t
    y_t = C·h_t

References:
    - Gu et al., "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    - Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import math


# Numerical stability constants
EPS = 1e-6
LOG_EPS = 1e-12
MAX_EXP_ARG = 80.0  # Prevent exp overflow


def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable exponential.
    
    Args:
        x: Input tensor
        
    Returns:
        exp(x) with clamping to prevent overflow
    """
    return torch.exp(torch.clamp(x, max=MAX_EXP_ARG))


def _log_safe(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable logarithm.
    
    Args:
        x: Input tensor (should be positive)
        
    Returns:
        log(x) with small epsilon for stability
    """
    return torch.log(torch.clamp(x, min=LOG_EPS))


def discretize_zoh(
    A: torch.Tensor,
    B: torch.Tensor,
    delta: torch.Tensor,
    use_softplus: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Discretize continuous SSM parameters using Zero-Order Hold.
    
    For diagonal A matrices (as used in Mamba), the discretization simplifies to:
        Ā = exp(Δ·A)
        B̄ = Δ·B  (approximation valid for small Δ or diagonal A)
    
    For the exact ZOH discretization with diagonal A:
        Ā = exp(Δ·A)
        B̄ = (exp(Δ·A) - 1) / A · B  (element-wise for diagonal)
    
    Args:
        A: Continuous state matrix, shape (d_model, d_state) or (batch, length, d_model, d_state)
           For Mamba, this is typically negative (representing decay)
        B: Input matrix, shape (batch, length, d_state)
        delta: Step size, shape (batch, length, d_model)
        use_softplus: Apply softplus to delta to ensure positivity
        
    Returns:
        Tuple of (A_bar, B_bar) discretized matrices
        - A_bar: shape (batch, length, d_model, d_state)
        - B_bar: shape (batch, length, d_model, d_state)
    """
    if use_softplus:
        # Ensure delta is positive via softplus
        delta = F.softplus(delta)
    
    # Expand dimensions for broadcasting
    # delta: (batch, length, d_model) -> (batch, length, d_model, 1)
    delta_expanded = delta.unsqueeze(-1)
    
    # A: (d_model, d_state) -> (1, 1, d_model, d_state)
    if A.dim() == 2:
        A_expanded = A.unsqueeze(0).unsqueeze(0)
    else:
        A_expanded = A
    
    # Compute Δ·A
    delta_A = delta_expanded * A_expanded
    
    # A_bar = exp(Δ·A)
    # Use stable exponential
    A_bar = _safe_exp(delta_A)
    
    # B_bar computation
    # For diagonal A: B_bar = (exp(Δ·A) - 1) / A · B
    # But when A is small or delta is small: B_bar ≈ Δ·B
    # 
    # More stable: B_bar = Δ · B · (exp(Δ·A) - 1) / (Δ·A)
    # When Δ·A → 0, this → Δ · B (using L'Hopital)
    
    # Compute (exp(Δ·A) - 1) / (Δ·A) using stable approximation
    # For small x: (exp(x) - 1) / x ≈ 1 + x/2 + x²/6 + ...
    abs_delta_A = torch.abs(delta_A)
    
    # Use Taylor expansion for small values, exact formula for large
    small_mask = abs_delta_A < 1e-4
    
    # Taylor approximation: 1 + x/2 + x²/6
    taylor_approx = 1.0 + delta_A / 2.0 + delta_A.pow(2) / 6.0
    
    # Exact: (exp(x) - 1) / x
    exact = torch.where(
        abs_delta_A > EPS,
        (A_bar - 1.0) / (delta_A + EPS * torch.sign(delta_A)),
        torch.ones_like(delta_A)
    )
    
    # Combine using mask
    scaling = torch.where(small_mask, taylor_approx, exact)
    
    # B: (batch, length, d_state) -> (batch, length, 1, d_state)
    B_expanded = B.unsqueeze(-2)
    
    # B_bar = Δ · scaling · B
    # Shape: (batch, length, d_model, d_state)
    B_bar = delta_expanded * scaling * B_expanded
    
    return A_bar, B_bar


def parallel_selective_scan(
    x: torch.Tensor,
    A_bar: torch.Tensor,
    B_bar: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Parallel associative scan for SSM computation during training.
    
    This implements the parallel scan algorithm for computing:
        h_t = A_bar_{t-1} · h_{t-1} + B_bar_{t-1} · x_t
        y_t = C_t · h_t + D · x_t
    
    The associative scan leverages the semi-group structure:
        (a₁, b₁) ⊕ (a₂, b₂) = (a₁·a₂, a₂·b₁ + b₂)
    
    This allows O(log L) parallel complexity instead of O(L) sequential.
    
    Args:
        x: Input sequence, shape (batch, length, d_model)
        A_bar: Discretized state transition, shape (batch, length, d_model, d_state)
        B_bar: Discretized input matrix, shape (batch, length, d_model, d_state)
        C: Output matrix, shape (batch, length, d_state)
        D: Skip connection (optional), shape (d_model,) or scalar
        
    Returns:
        Output sequence, shape (batch, length, d_model)
    """
    batch_size, seq_len, d_model = x.shape
    d_state = A_bar.shape[-1]
    
    # Compute B_bar · x for each timestep
    # x: (batch, length, d_model) -> (batch, length, d_model, 1)
    # B_bar: (batch, length, d_model, d_state)
    # Result: (batch, length, d_model, d_state)
    x_expanded = x.unsqueeze(-1)
    Bx = B_bar * x_expanded  # Element-wise, broadcasting over d_state
    
    # Initialize hidden states for scan
    # We'll compute the scan using the associative property
    # h_t = A_bar_{t} · h_{t-1} + Bx_t
    
    # For parallel scan, we use a different formulation:
    # Define (a_t, b_t) where composition is: (a₁, b₁) ⊕ (a₂, b₂) = (a₁·a₂, a₂·b₁ + b₂)
    # Then scanning gives us the cumulative products and sums
    
    # Sequential fallback for correctness (parallel scan complex to implement in pure PyTorch)
    # For production, this would use custom CUDA kernels
    # Here we use an efficient sequential implementation that still leverages parallelism
    
    h = torch.zeros(batch_size, d_model, d_state, device=x.device, dtype=x.dtype)
    outputs = []
    
    for t in range(seq_len):
        # h_t = A_bar_t · h_{t-1} + Bx_t
        h = A_bar[:, t] * h + Bx[:, t]
        
        # y_t = C_t · h_t (sum over d_state dimension)
        # C: (batch, length, d_state), h: (batch, d_model, d_state)
        # Need: (batch, d_model)
        C_t = C[:, t]  # (batch, d_state)
        y_t = torch.einsum('bds,bs->bd', h, C_t)
        
        outputs.append(y_t)
    
    # Stack outputs: (length, batch, d_model) -> (batch, length, d_model)
    y = torch.stack(outputs, dim=1)
    
    # Add skip connection if provided
    if D is not None:
        if D.dim() == 0:
            y = y + D * x
        else:
            y = y + D.unsqueeze(0).unsqueeze(0) * x
    
    return y


def parallel_scan_log_space(
    log_A: torch.Tensor,
    Bx: torch.Tensor,
) -> torch.Tensor:
    """Parallel scan in log space for numerical stability.
    
    This is an alternative implementation that works in log space
    to handle very long sequences without numerical issues.
    
    Args:
        log_A: Log of discretized state transition, shape (batch, length, d_model, d_state)
        Bx: B_bar · x, shape (batch, length, d_model, d_state)
        
    Returns:
        Hidden states, shape (batch, length, d_model, d_state)
    """
    batch_size, seq_len, d_model, d_state = log_A.shape
    
    # Cumulative sum of log_A gives us the products
    log_A_cumsum = torch.cumsum(log_A, dim=1)
    
    # For each t, we need: sum_{s=0}^{t} A[t:s] · Bx[s]
    # where A[t:s] = prod_{i=s}^{t-1} A[i]
    # In log space: log(A[t:s]) = sum_{i=s}^{t-1} log(A[i]) = log_A_cumsum[t-1] - log_A_cumsum[s-1]
    
    # This requires O(L²) in naive implementation
    # For truly parallel scan, need custom kernel
    
    # Use stable sequential for now
    h = torch.zeros(batch_size, d_model, d_state, device=log_A.device, dtype=log_A.dtype)
    hidden_states = []
    
    for t in range(seq_len):
        A_t = _safe_exp(log_A[:, t])
        h = A_t * h + Bx[:, t]
        hidden_states.append(h)
    
    return torch.stack(hidden_states, dim=1)


@torch.jit.script
def recurrent_step(
    x_t: torch.Tensor,
    h_prev: torch.Tensor,
    A_bar_t: torch.Tensor,
    B_bar_t: torch.Tensor,
    C_t: torch.Tensor,
    D: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single recurrent step for O(1) inference.
    
    This is the core inference operation, decorated with torch.jit.script
    for JIT compilation and optimal performance.
    
    Computes:
        h_t = A_bar_t · h_{t-1} + B_bar_t · x_t
        y_t = C_t · h_t + D · x_t
    
    Args:
        x_t: Current input, shape (batch, d_model)
        h_prev: Previous hidden state, shape (batch, d_model, d_state)
        A_bar_t: Discretized state transition, shape (batch, d_model, d_state)
        B_bar_t: Discretized input matrix, shape (batch, d_model, d_state)
        C_t: Output matrix, shape (batch, d_state)
        D: Skip connection (optional), shape (d_model,)
        
    Returns:
        Tuple of (output y_t, new hidden state h_t)
        - y_t: shape (batch, d_model)
        - h_t: shape (batch, d_model, d_state)
    """
    # x_t: (batch, d_model) -> (batch, d_model, 1) for broadcasting
    x_expanded = x_t.unsqueeze(-1)
    
    # h_t = A_bar_t · h_prev + B_bar_t · x_t
    # All operations are element-wise due to diagonal structure
    h_t = A_bar_t * h_prev + B_bar_t * x_expanded
    
    # y_t = C_t · h_t (sum over d_state)
    # Using einsum: 'bds,bs->bd'
    y_t = torch.sum(h_t * C_t.unsqueeze(1), dim=-1)
    
    # Add skip connection if provided
    if D is not None:
        y_t = y_t + D * x_t
    
    return y_t, h_t


class SSMKernel(nn.Module):
    """SSM Kernel module encapsulating discretization and computation.
    
    This module manages the SSM parameters and provides both training
    (parallel scan) and inference (recurrent) modes.
    
    Attributes:
        d_model: Model dimension
        d_state: State dimension
        A_log: Learnable log of A matrix (ensures A is negative)
        D: Skip connection parameter
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
    ):
        """Initialize SSM Kernel.
        
        Args:
            d_model: Model dimension
            d_state: State dimension (N in Mamba paper)
            dt_min: Minimum timestep
            dt_max: Maximum timestep
            dt_init: Initialization strategy for dt ("random" or "constant")
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # Initialize A in log space (ensures A is negative after exp(-exp(A_log)))
        # Following S4 initialization: A_n = -1/2 + ni for complex, or just -1/2 for real
        # We use real-valued diagonal A
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log.unsqueeze(0).expand(d_model, -1).clone())
        
        # D skip connection (learnable)
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Store dt bounds for reference
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        # Cache for inference mode
        self._inference_mode = False
        self._h_cache: Optional[torch.Tensor] = None
    
    def get_A(self) -> torch.Tensor:
        """Get the continuous A matrix.
        
        Returns:
            A matrix, shape (d_model, d_state), all negative values
        """
        # A = -exp(A_log) ensures A is always negative (decay)
        return -torch.exp(self.A_log)
    
    def forward(
        self,
        x: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through SSM.
        
        Args:
            x: Input sequence, shape (batch, length, d_model)
            B: Input projection, shape (batch, length, d_state)
            C: Output projection, shape (batch, length, d_state)
            delta: Timestep, shape (batch, length, d_model)
            
        Returns:
            Output sequence, shape (batch, length, d_model)
        """
        # Get continuous A
        A = self.get_A()
        
        # Discretize
        A_bar, B_bar = discretize_zoh(A, B, delta, use_softplus=True)
        
        # Compute SSM output using parallel scan
        y = parallel_selective_scan(x, A_bar, B_bar, C, self.D)
        
        return y
    
    def step(
        self,
        x_t: torch.Tensor,
        B_t: torch.Tensor,
        C_t: torch.Tensor,
        delta_t: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single step for recurrent inference.
        
        Args:
            x_t: Current input, shape (batch, d_model)
            B_t: Current B projection, shape (batch, d_state)
            C_t: Current C projection, shape (batch, d_state)
            delta_t: Current timestep, shape (batch, d_model)
            h_prev: Previous hidden state, shape (batch, d_model, d_state)
            
        Returns:
            Tuple of (output, new_hidden_state)
        """
        batch_size = x_t.shape[0]
        
        # Initialize hidden state if needed
        if h_prev is None:
            h_prev = torch.zeros(
                batch_size, self.d_model, self.d_state,
                device=x_t.device, dtype=x_t.dtype
            )
        
        # Get A and discretize for single step
        A = self.get_A()
        
        # Ensure delta is positive
        delta_t = F.softplus(delta_t)
        
        # Discretize (single timestep)
        delta_expanded = delta_t.unsqueeze(-1)  # (batch, d_model, 1)
        A_expanded = A.unsqueeze(0)  # (1, d_model, d_state)
        
        delta_A = delta_expanded * A_expanded
        A_bar_t = _safe_exp(delta_A)
        
        # B_bar computation (simplified for single step)
        B_expanded = B_t.unsqueeze(1)  # (batch, 1, d_state)
        B_bar_t = delta_expanded * B_expanded.expand(-1, self.d_model, -1)
        
        # Recurrent step
        y_t, h_t = recurrent_step(x_t, h_prev, A_bar_t, B_bar_t, C_t, self.D)
        
        return y_t, h_t
    
    def reset_cache(self) -> None:
        """Reset the hidden state cache for inference."""
        self._h_cache = None
    
    def set_inference_mode(self, mode: bool) -> None:
        """Set inference mode.
        
        Args:
            mode: True for inference (recurrent), False for training (parallel)
        """
        self._inference_mode = mode
        if not mode:
            self.reset_cache()


class SelectiveSSM(nn.Module):
    """Selective State Space Model - the core of Mamba.
    
    This implements the selective scan mechanism where B, C, and Δ
    are computed as functions of the input, making the SSM input-dependent.
    
    The selectivity allows the model to:
    1. Filter out irrelevant information (via input-dependent Δ)
    2. Focus on relevant context (via input-dependent B, C)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
    ):
        """Initialize Selective SSM.
        
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: Expansion factor for inner dimension
            dt_rank: Rank of dt projection ("auto" = ceil(d_model/16))
            dt_min: Minimum timestep value
            dt_max: Maximum timestep value
            dt_init: Timestep initialization strategy
            dt_scale: Scaling factor for dt initialization
            bias: Use bias in linear projections
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        
        # Compute dt_rank
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank
        
        # Input projection (to expanded dimension)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # Convolution (local context)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        
        # Selective projections (input-dependent B, C, Δ)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # Timestep projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt bias for proper range
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize dt bias so that softplus(dt) is in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        # Inverse softplus: x = log(exp(y) - 1)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # SSM kernel
        self.ssm = SSMKernel(self.d_inner, d_state, dt_min, dt_max, dt_init)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape (batch, length, d_model)
            
        Returns:
            Output tensor, shape (batch, length, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)  # (batch, length, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # Each: (batch, length, d_inner)
        
        # Convolution (transpose for conv1d)
        x_conv = x_inner.transpose(1, 2)  # (batch, d_inner, length)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim to original length
        x_conv = x_conv.transpose(1, 2)  # (batch, length, d_inner)
        
        # Apply activation
        x_conv = F.silu(x_conv)
        
        # Selective projections
        x_dbl = self.x_proj(x_conv)  # (batch, length, dt_rank + 2*d_state)
        
        # Split into dt, B, C
        dt, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        
        # Project dt to full dimension
        dt = self.dt_proj(dt)  # (batch, length, d_inner)
        
        # SSM computation
        y = self.ssm(x_conv, B, C, dt)  # (batch, length, d_inner)
        
        # Gating with z
        z = F.silu(z)
        y = y * z
        
        # Output projection
        out = self.out_proj(y)
        
        return out
    
    def step(
        self,
        x_t: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single step for inference.
        
        Args:
            x_t: Current input, shape (batch, d_model)
            conv_state: Convolution state, shape (batch, d_inner, d_conv)
            ssm_state: SSM hidden state, shape (batch, d_inner, d_state)
            
        Returns:
            Tuple of (output, new_conv_state, new_ssm_state)
        """
        # Input projection
        xz = self.in_proj(x_t)  # (batch, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # Each: (batch, d_inner)
        
        # Update conv state (shift and add new)
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = x_inner
        
        # Apply convolution using stored state
        # conv_state: (batch, d_inner, d_conv)
        # conv weight: (d_inner, 1, d_conv) for grouped conv
        x_conv = torch.sum(
            conv_state * self.conv1d.weight.squeeze(1),
            dim=-1
        ) + self.conv1d.bias
        
        x_conv = F.silu(x_conv)
        
        # Selective projections
        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        dt = self.dt_proj(dt)
        
        # SSM step
        y, ssm_state = self.ssm.step(x_conv, B, C, dt, ssm_state)
        
        # Gating
        z = F.silu(z)
        y = y * z
        
        # Output projection
        out = self.out_proj(y)
        
        return out, conv_state, ssm_state
    
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
            dtype: Data type for tensors
            
        Returns:
            Tuple of (conv_state, ssm_state)
        """
        conv_state = torch.zeros(
            batch_size, self.d_inner, self.d_conv,
            device=device, dtype=dtype
        )
        ssm_state = torch.zeros(
            batch_size, self.d_inner, self.d_state,
            device=device, dtype=dtype
        )
        return conv_state, ssm_state
