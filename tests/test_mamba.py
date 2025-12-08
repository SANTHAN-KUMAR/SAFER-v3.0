"""
Unit tests for Mamba RUL Predictor.

Tests cover:
- Model initialization and configuration
- Forward pass correctness
- Output shape and constraints
- Gradient flow
- Parameter counting
- JIT compilation (when enabled)
"""

import pytest
import numpy as np


class TestMambaRULPredictor:
    """Test suite for MambaRULPredictor model."""
    
    @pytest.fixture
    def model(self, torch_available, mamba_config):
        """Create Mamba model for testing."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.core.mamba import MambaRULPredictor
        return MambaRULPredictor(
            d_input=mamba_config.d_input,
            d_model=mamba_config.d_model,
            d_state=mamba_config.d_state,
            n_layers=mamba_config.n_layers,
            dropout=mamba_config.dropout,
            max_rul=mamba_config.max_rul,
            use_jit=False,
        )
    
    def test_initialization(self, model, mamba_config):
        """Test model initializes with correct configuration."""
        assert model.d_input == mamba_config.d_input
        assert model.d_model == mamba_config.d_model
        assert model.n_layers == mamba_config.n_layers
        assert model.max_rul == mamba_config.max_rul
    
    def test_forward_pass_shape(self, model, sample_sequence_batch, batch_size, torch_available):
        """Test forward pass produces correct output shape."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        model = model.cpu()
        tensor_batch = torch.from_numpy(sample_sequence_batch)
        output = model(tensor_batch)
        assert output.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {output.shape}"
    
    def test_output_non_negative(self, model, sample_sequence_batch, torch_available):
        """Test RUL predictions are non-negative."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        model = model.cpu()
        model.eval()
        tensor_batch = torch.from_numpy(sample_sequence_batch)
        with torch.no_grad():
            output = model(tensor_batch)
        assert (output >= 0).all(), "RUL predictions must be non-negative"
    
    def test_output_bounded(self, model, sample_sequence_batch, mamba_config, torch_available):
        """Test RUL predictions are bounded by max_rul."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        model = model.cpu()
        model.eval()
        tensor_batch = torch.from_numpy(sample_sequence_batch)
        with torch.no_grad():
            output = model(tensor_batch)
        assert (output <= mamba_config.max_rul).all(), f"RUL must be <= {mamba_config.max_rul}"
    
    def test_gradient_flow(self, model, sample_sequence_batch, sample_rul_targets, torch_available):
        """Test gradients flow through the model."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        model = model.cpu()
        model.train()
        tensor_batch = torch.from_numpy(sample_sequence_batch)
        targets = torch.from_numpy(sample_rul_targets)
        output = model(tensor_batch)
        loss = torch.nn.functional.mse_loss(output.squeeze(), targets)
        loss.backward()
        
        # Check that gradients are computed for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_parameter_count(self, model):
        """Test parameter counting method."""
        counts = model.count_parameters()
        assert 'total' in counts
        assert 'trainable' in counts
        assert counts['trainable'] > 0
        assert counts['total'] >= counts['trainable']
    
    def test_eval_mode_deterministic(self, model, sample_sequence_batch, torch_available):
        """Test model is deterministic in eval mode."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        model = model.cpu()
        model.eval()
        tensor_batch = torch.from_numpy(sample_sequence_batch)
        with torch.no_grad():
            out1 = model(tensor_batch)
            out2 = model(tensor_batch)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"
    
    def test_single_step_inference(self, model, torch_available, n_sensors):
        """Test inference with single time step (O(1) inference)."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        model = model.cpu()
        model.eval()
        x = torch.randn(1, 1, n_sensors)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 1)
    
    def test_variable_sequence_length(self, model, torch_available, n_sensors):
        """Test model handles variable sequence lengths."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        model = model.cpu()
        model.eval()
        
        for seq_len in [10, 50, 100, 200]:
            x = torch.randn(2, seq_len, n_sensors)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (2, 1), f"Failed for seq_len={seq_len}"


class TestMambaBlock:
    """Test suite for individual Mamba blocks."""
    
    @pytest.fixture
    def mamba_block(self, torch_available):
        """Create MambaBlock for testing."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.core.mamba import MambaBlock
        return MambaBlock(d_model=32, d_state=8, expand=2, dropout=0.1)
    
    def test_block_output_shape(self, mamba_block, torch_available):
        """Test MambaBlock preserves input shape."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        x = torch.randn(4, 50, 32)  # (batch, seq, d_model)
        output = mamba_block(x)
        assert output.shape == x.shape
    
    def test_block_residual_connection(self, mamba_block, torch_available):
        """Test MambaBlock has residual connection."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        x = torch.randn(4, 50, 32)
        output = mamba_block(x)
        # Output should not be exactly the same as input (some processing occurred)
        assert not torch.allclose(x, output)


class TestRMSNorm:
    """Test suite for RMSNorm layer."""
    
    @pytest.fixture
    def rms_norm(self, torch_available):
        """Create RMSNorm for testing."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.core.mamba import RMSNorm
        return RMSNorm(d_model=32)
    
    def test_rms_norm_output_shape(self, rms_norm, torch_available):
        """Test RMSNorm preserves input shape."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        x = torch.randn(4, 50, 32)
        output = rms_norm(x)
        assert output.shape == x.shape
    
    def test_rms_norm_normalizes(self, rms_norm, torch_available):
        """Test RMSNorm produces roughly unit variance."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        x = torch.randn(100, 50, 32) * 100  # Large values
        output = rms_norm(x)
        # RMS of output should be close to 1
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        assert rms.mean() < 10, "RMSNorm should normalize large inputs"
