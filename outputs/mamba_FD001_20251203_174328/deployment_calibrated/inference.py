"""
Inference Helper for FD001 Deployment.

This module provides a simple interface for deploying the model
in production environments.

Usage:
    from inference import FD001Predictor
    
    predictor = FD001Predictor('deployment_calibrated/')
    rul, lower, upper = predictor.predict(sensor_data)
"""

import numpy as np
import torch
from pathlib import Path
import json


class FD001Predictor:
    """Production inference wrapper for FD001 model."""
    
    def __init__(self, deployment_dir):
        """Initialize predictor with deployment artifacts.
        
        Args:
            deployment_dir: Path to deployment package directory
        """
        self.deployment_dir = Path(deployment_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        self._load_config()
        
        # Load model
        self._load_model()
        
        # Load calibration parameters
        self._load_calibration()
    
    def _load_config(self):
        """Load model configuration."""
        config_path = self.deployment_dir / 'model_config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def _load_model(self):
        """Load trained model."""
        from safer_v3.core.mamba import MambaRULPredictor
        
        # Extract model architecture parameters
        model_config = {
            'd_input': 14,
            'd_model': self.config.get('d_model', 128),
            'n_layers': self.config.get('n_layers', 6),
            'd_state': self.config.get('d_state', 16),
            'd_conv': 4,
            'expand': self.config.get('expand', 2),
            'dropout': self.config.get('dropout', 0.1),
            'max_rul': self.config.get('max_rul', 125),
        }
        
        # Create model
        self.model = MambaRULPredictor(**model_config)
        
        # Load weights
        checkpoint_path = self.deployment_dir / 'model_checkpoint.pt'
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model = self.model.to(self.device)
    
    def _load_calibration(self):
        """Load calibration parameters."""
        cal_path = self.deployment_dir / 'conformal_params.json'
        with open(cal_path, 'r') as f:
            self.calibration = json.load(f)
        
        self.quantile = self.calibration['quantile']
        self.coverage = self.calibration['coverage']
    
    def predict(self, sensor_data: np.ndarray) -> tuple:
        """Make RUL prediction with confidence bounds.
        
        Args:
            sensor_data: Input sensor data (14 features)
                Expected shape: (seq_len, 14) for single sample
                             or (batch, seq_len, 14) for multiple
        
        Returns:
            Tuple of (rul, rul_lower, rul_upper)
            - rul: Point prediction
            - rul_lower: Lower confidence bound (90%)
            - rul_upper: Upper confidence bound (90%)
        """
        # Convert to tensor
        if isinstance(sensor_data, np.ndarray):
            if sensor_data.ndim == 2:
                # Single sample: (seq_len, 14) -> (1, seq_len, 14)
                sensor_data = np.expand_dims(sensor_data, axis=0)
            
            sensor_tensor = torch.from_numpy(sensor_data).float()
        else:
            sensor_tensor = sensor_data
        
        # Move to device
        sensor_tensor = sensor_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(sensor_tensor)
        
        # Extract predictions
        rul_pred = predictions.cpu().numpy().ravel()
        
        # Apply conformal intervals
        rul_lower = np.maximum(0, rul_pred - self.quantile)
        rul_upper = rul_pred + self.quantile
        
        # Return first if batch size is 1
        if len(rul_pred) == 1:
            return float(rul_pred[0]), float(rul_lower[0]), float(rul_upper[0])
        
        return rul_pred, rul_lower, rul_upper
    
    def get_calibration_info(self) -> dict:
        """Get calibration information.
        
        Returns:
            Dictionary with calibration parameters
        """
        return {
            'coverage': self.coverage,
            'quantile': self.quantile,
            'average_interval_width': self.calibration.get('average_width', 0.0),
        }
    
    def get_model_info(self) -> dict:
        """Get model information.
        
        Returns:
            Dictionary with model architecture and training info
        """
        return {
            'model_type': 'Mamba RUL Predictor',
            'layers': self.config.get('n_layers', 6),
            'd_model': self.config.get('d_model', 128),
            'd_state': self.config.get('d_state', 16),
            'dataset': self.config.get('dataset', 'FD001'),
            'max_rul': self.config.get('max_rul', 125),
        }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <deployment_dir> <sensor_data_file>")
        sys.exit(1)
    
    # Example: load and predict
    deployment_dir = sys.argv[1]
    predictor = FD001Predictor(deployment_dir)
    
    print("Model loaded successfully!")
    print(f"Model info: {predictor.get_model_info()}")
    print(f"Calibration: {predictor.get_calibration_info()}")
    
    # Example prediction (dummy data)
    dummy_data = np.random.randn(30, 14).astype(np.float32)
    rul, lower, upper = predictor.predict(dummy_data)
    
    print(f"\nExample prediction:")
    print(f"  RUL: {rul:.2f} cycles")
    print(f"  90% Interval: [{lower:.2f}, {upper:.2f}]")
    print(f"  Width: {upper - lower:.2f} cycles")
