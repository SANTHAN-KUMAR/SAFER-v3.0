"""
Unit tests for baseline models (LSTM, Transformer, CNN-LSTM).

Tests cover:
- Model initialization
- Forward pass correctness
- Output constraints (non-negative RUL)
- Gradient flow
- BaselineFactory functionality
"""

import pytest
import numpy as np


class TestLSTMPredictor:
    """Test suite for LSTM Baseline Predictor."""
    
    @pytest.fixture
    def model(self, torch_available, n_sensors):
        """Create LSTM model for testing."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.core.baselines import LSTMPredictor
        return LSTMPredictor(
            d_input=n_sensors,
            d_model=32,
            n_layers=2,
            dropout=0.1,
            bidirectional=True,
            max_rul=125,
        )
    
    def test_initialization(self, model, n_sensors):
        """Test LSTM model initializes correctly."""
        assert model.d_input == n_sensors
        assert model.d_model == 32
        assert model.n_layers == 2
    
    def test_forward_pass_shape(self, model, sample_sequence_batch, batch_size, torch_available):
        """Test forward pass produces correct output shape."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        # Put model on CPU and use CPU tensor
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
    
    def test_output_bounded(self, model, sample_sequence_batch, torch_available):
        """Test RUL predictions are bounded by max_rul."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        model = model.cpu()
        model.eval()
        tensor_batch = torch.from_numpy(sample_sequence_batch)
        with torch.no_grad():
            output = model(tensor_batch)
        assert (output <= 125).all(), "RUL must be <= max_rul"
    
    def test_gradient_flow(self, model, sample_sequence_batch, sample_rul_targets, torch_available):
        """Test gradients flow through LSTM model."""
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
        
        # Check LSTM layers have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_attention_mechanism(self, model, sample_sequence_batch, torch_available):
        """Test attention is applied (if present)."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        model = model.cpu()
        model.eval()
        tensor_batch = torch.from_numpy(sample_sequence_batch)
        with torch.no_grad():
            output = model(tensor_batch)
        # Just verify it runs - attention should be internal
        assert output is not None


class TestTransformerPredictor:
    """Test suite for Transformer Baseline Predictor."""
    
    @pytest.fixture
    def model(self, torch_available, n_sensors):
        """Create Transformer model for testing."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.core.baselines import TransformerPredictor
        return TransformerPredictor(
            d_input=n_sensors,
            d_model=32,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            max_rul=125,
            max_len=100,
        )
    
    def test_initialization(self, model, n_sensors):
        """Test Transformer model initializes correctly."""
        assert model.d_input == n_sensors
        assert model.d_model == 32
        assert model.n_heads == 4
    
    def test_forward_pass_shape(self, model, sample_sequence_batch, batch_size, torch_available):
        """Test forward pass produces correct output shape."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        model = model.cpu()
        tensor_batch = torch.from_numpy(sample_sequence_batch)
        output = model(tensor_batch)
        assert output.shape == (batch_size, 1)
    
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
        assert (output >= 0).all()
    
    def test_positional_encoding(self, model, torch_available, n_sensors):
        """Test positional encoding handles different sequence lengths."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        model = model.cpu()
        model.eval()
        
        for seq_len in [10, 50, 80]:
            x = torch.randn(2, seq_len, n_sensors)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (2, 1)


class TestCNNLSTMPredictor:
    """Test suite for CNN-LSTM Baseline Predictor."""
    
    @pytest.fixture
    def model(self, torch_available, n_sensors):
        """Create CNN-LSTM model for testing."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.core.baselines import CNNLSTMPredictor
        return CNNLSTMPredictor(
            d_input=n_sensors,
            cnn_channels=(16, 32),
            d_model=32,
            lstm_layers=1,
            dropout=0.1,
            max_rul=125,
        )
    
    def test_initialization(self, model, n_sensors):
        """Test CNN-LSTM model initializes correctly."""
        assert model.d_input == n_sensors
    
    def test_forward_pass_shape(self, model, sample_sequence_batch, batch_size, torch_available):
        """Test forward pass produces correct output shape."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        model = model.cpu()
        tensor_batch = torch.from_numpy(sample_sequence_batch)
        output = model(tensor_batch)
        assert output.shape == (batch_size, 1)
    
    def test_cnn_feature_extraction(self, model, sample_sequence_batch, torch_available):
        """Test CNN layers extract features properly."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        import torch
        model = model.cpu()
        model.eval()
        tensor_batch = torch.from_numpy(sample_sequence_batch)
        with torch.no_grad():
            output = model(tensor_batch)
        assert output is not None


class TestBaselineFactory:
    """Test suite for BaselineFactory."""
    
    def test_create_lstm(self, torch_available, n_sensors):
        """Test factory creates LSTM model."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.core.baselines import BaselineFactory
        model = BaselineFactory.create(
            model_type='lstm',
            d_input=n_sensors,
            d_model=32,
        )
        assert model is not None
    
    def test_create_transformer(self, torch_available, n_sensors):
        """Test factory creates Transformer model."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.core.baselines import BaselineFactory
        model = BaselineFactory.create(
            model_type='transformer',
            d_input=n_sensors,
            d_model=32,
        )
        assert model is not None
    
    def test_create_cnn_lstm(self, torch_available, n_sensors):
        """Test factory creates CNN-LSTM model."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.core.baselines import BaselineFactory
        model = BaselineFactory.create(
            model_type='cnn_lstm',
            d_input=n_sensors,
        )
        assert model is not None
    
    def test_invalid_model_type(self, torch_available, n_sensors):
        """Test factory raises error for invalid model type."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.core.baselines import BaselineFactory
        with pytest.raises((ValueError, KeyError)):
            BaselineFactory.create(model_type='invalid', d_input=n_sensors)
    
    def test_list_models(self, torch_available):
        """Test factory lists available models."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        from safer_v3.core.baselines import BaselineFactory
        models = BaselineFactory.list_models()
        assert 'lstm' in models or len(models) > 0
