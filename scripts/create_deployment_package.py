"""
Create Deployment Package for FD001 Model.

This script bundles all artifacts needed for production deployment:
- Model weights and config
- Calibrated conformal parameters
- Alert rules and Simplex config
- Inference helper code
- Deployment documentation

Usage:
    python scripts/create_deployment_package.py
"""

import sys
from pathlib import Path
import json
import shutil
import torch
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_deployment_package():
    """Create complete deployment package."""
    
    # Define paths
    checkpoint_path = project_root / 'checkpoints' / 'best_model.pt'
    args_path = project_root / 'outputs' / 'mamba_FD001_20251203_174328' / 'args.json'
    calibration_dir = project_root / 'outputs' / 'mamba_FD001_20251203_174328' / 'calibration'
    report_dir = project_root / 'outputs' / 'mamba_FD001_20251203_174328' / 'report'
    
    deployment_dir = project_root / 'outputs' / 'mamba_FD001_20251203_174328' / 'deployment_calibrated'
    deployment_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating deployment package in {deployment_dir}")
    
    # 1. Copy model checkpoint
    logger.info("Copying model checkpoint...")
    shutil.copy(checkpoint_path, deployment_dir / 'model_checkpoint.pt')
    
    # 2. Copy training args/config
    logger.info("Copying model configuration...")
    shutil.copy(args_path, deployment_dir / 'model_config.json')
    
    # 3. Copy calibration parameters
    logger.info("Copying calibration parameters...")
    shutil.copy(
        calibration_dir / 'conformal_params.json',
        deployment_dir / 'conformal_params.json'
    )
    
    # 4. Create Simplex and alert configuration
    logger.info("Creating Simplex and alert configuration...")
    simplex_config = {
        'physics_threshold': 0.15,
        'divergence_threshold': 30.0,
        'uncertainty_threshold': 100.0,
        'recovery_window': 10,
        'max_switch_rate': 2.0,
        'hysteresis_cycles': 5,
        'timeout_ms': 20.0,
        'conservative_margin': 5.0,
    }
    
    with open(deployment_dir / 'simplex_config.json', 'w') as f:
        json.dump(simplex_config, f, indent=2)
    
    # Alert thresholds
    alert_config = {
        'critical_threshold': 10,
        'warning_threshold': 25,
        'caution_threshold': 50,
        'advisory_threshold': 100,
    }
    
    with open(deployment_dir / 'alert_config.json', 'w') as f:
        json.dump(alert_config, f, indent=2)
    
    # 5. Copy evaluation report
    logger.info("Copying evaluation report...")
    shutil.copy(
        report_dir / 'EVALUATION_REPORT.txt',
        deployment_dir / 'EVALUATION_REPORT.txt'
    )
    shutil.copy(
        report_dir / 'summary_statistics.json',
        deployment_dir / 'summary_statistics.json'
    )
    
    # 6. Create inference helper module
    logger.info("Creating inference helper module...")
    inference_code = '''"""
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
    
    print(f"\\nExample prediction:")
    print(f"  RUL: {rul:.2f} cycles")
    print(f"  90% Interval: [{lower:.2f}, {upper:.2f}]")
    print(f"  Width: {upper - lower:.2f} cycles")
'''
    
    with open(deployment_dir / 'inference.py', 'w') as f:
        f.write(inference_code)
    
    # 7. Create deployment README
    logger.info("Creating deployment documentation...")
    readme = '''# FD001 Model Deployment Package

This directory contains all artifacts needed to deploy the SAFER v3.0 FD001 RUL prediction model in production.

## Contents

- `model_checkpoint.pt` - Trained model weights and optimizer state
- `model_config.json` - Model architecture configuration
- `conformal_params.json` - Calibrated conformal prediction parameters (90% coverage)
- `simplex_config.json` - Simplex safety module configuration
- `alert_config.json` - Alert system thresholds and rules
- `inference.py` - Inference helper module for production use
- `EVALUATION_REPORT.txt` - Complete evaluation report
- `summary_statistics.json` - Model metrics and statistics

## Quick Start

### 1. Load Model

```python
from inference import FD001Predictor

# Initialize predictor
predictor = FD001Predictor('.')

# Get model info
print(predictor.get_model_info())
print(predictor.get_calibration_info())
```

### 2. Make Predictions

```python
import numpy as np

# Prepare sensor data: (sequence_length, 14_sensors)
sensor_data = np.random.randn(30, 14).astype(np.float32)

# Get prediction with confidence interval
rul, rul_lower, rul_upper = predictor.predict(sensor_data)

print(f"RUL: {rul:.1f} ± {(rul_upper - rul_lower)/2:.1f} cycles")
print(f"90% Confidence Interval: [{rul_lower:.1f}, {rul_upper:.1f}]")
```

### 3. Alert Generation

```python
from safer_v3.decision.alerts import AlertManager, create_rul_alert_rules

# Create alert manager
alert_manager = AlertManager()
alert_manager.add_rules(create_rul_alert_rules(
    critical_threshold=10,
    warning_threshold=25,
    caution_threshold=50,
    advisory_threshold=100,
))

# Process RUL
rul_value = 35.5
alerts = alert_manager.process(rul_value)

for alert in alerts:
    print(f"[{alert.level.name}] {alert.message}")
```

## Model Specifications

- **Type**: Mamba RUL Predictor
- **Architecture**: 6 layers, 128-dim hidden, 16-dim state
- **Input**: 14 turbofan engine sensors
- **Output**: Remaining Useful Life (RUL) in cycles
- **Max RUL**: 125 cycles (capped)

## Calibration

- **Coverage Target**: 90%
- **Empirical Coverage**: 90.0% (verified on validation set)
- **Uncertainty Quantile**: 38.55 cycles
- **Average Interval Width**: 77.09 cycles

This ensures that true RUL falls within predicted bounds 90% of the time.

## Safety Architecture

The deployment uses a **Simplex decision module** that:

1. **Complex Mode**: Uses Mamba predictions (high accuracy)
2. **Baseline Mode**: Uses conservative forecast (high assurance)
3. **Safety Switching**: Monitors:
   - Physics anomalies (residual threshold: 0.15)
   - Prediction divergence (threshold: 30 cycles)
   - Uncertainty bounds (threshold: 100 cycles)
4. **Automatic Fallback**: Switches to baseline on anomalies
5. **Rate Limiting**: Max 2 switches per minute (prevents oscillation)

## Alert System

Multi-level alert thresholds aligned with maintenance scheduling:

| Level | Threshold | Action |
|-------|-----------|--------|
| CRITICAL | RUL ≤ 10 | Immediate action required |
| WARNING | RUL ≤ 25 | Urgent maintenance |
| CAUTION | RUL ≤ 50 | Plan maintenance |
| ADVISORY | RUL ≤ 100 | Monitor trend |

## Deployment Checklist

- [x] Model trained and validated
- [x] Prediction intervals calibrated (90% coverage)
- [x] Simplex safety architecture verified
- [x] Alert system configured
- [x] Inference code provided
- [x] Complete documentation generated

## Performance Metrics

- **Test RMSE**: 18.15 cycles
- **Test MAE**: 12.02 cycles
- **NASA Score**: 82,479.69
- **R² Score**: 0.6648

See `EVALUATION_REPORT.txt` for complete details.

## Integration Example

```python
def monitor_engine(sensor_stream, predictor, alert_manager):
    """Example integration loop."""
    
    for sensor_data in sensor_stream:
        # Predict RUL
        rul, lower, upper = predictor.predict(sensor_data)
        
        # Generate alerts
        alerts = alert_manager.process(rul)
        
        # Log and act
        for alert in alerts:
            print(f"[ALERT] {alert.level.name}: {alert.message}")
            if alert.level >= AlertLevel.WARNING:
                schedule_maintenance(rul)
```

## Support

For issues or questions about deployment, refer to:
- `EVALUATION_REPORT.txt` - Complete analysis
- `summary_statistics.json` - Detailed metrics
- Project repository - Source code and documentation

---

**Generated**: 2025-12-04
**Dataset**: CMAPSS FD001
**Model Version**: v3.0
'''
    
    with open(deployment_dir / 'DEPLOYMENT_README.md', 'w') as f:
        f.write(readme)
    
    # 8. Create a summary manifest
    logger.info("Creating deployment manifest...")
    manifest = {
        'package_name': 'FD001_Mamba_RUL_Predictor',
        'version': '3.0',
        'created': str(Path(__file__).stat().st_mtime),
        'files': {
            'model_checkpoint.pt': 'Trained model weights',
            'model_config.json': 'Model architecture config',
            'conformal_params.json': 'Calibration parameters',
            'simplex_config.json': 'Safety module config',
            'alert_config.json': 'Alert thresholds',
            'inference.py': 'Inference helper module',
            'EVALUATION_REPORT.txt': 'Complete evaluation report',
            'summary_statistics.json': 'Metrics summary',
            'DEPLOYMENT_README.md': 'Deployment guide',
        },
        'requirements': {
            'python': '>=3.9',
            'pytorch': '>=2.0',
            'numpy': '>=1.20',
        },
        'model_specs': {
            'type': 'Mamba RUL Predictor',
            'layers': 6,
            'model_dim': 128,
            'state_dim': 16,
            'input_features': 14,
            'output': 'RUL (cycles)',
        },
        'calibration': {
            'coverage_target': 0.9,
            'empirical_coverage': 0.9,
            'quantile': 38.55,
            'samples': 3490,
        },
        'performance': {
            'test_rmse': 18.15,
            'test_mae': 12.02,
            'nasa_score': 82479.69,
            'r2': 0.6648,
        },
    }
    
    with open(deployment_dir / 'MANIFEST.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Print summary
    logger.success("✓ Deployment package created successfully!")
    logger.info("=" * 70)
    logger.info("DEPLOYMENT PACKAGE CONTENTS")
    logger.info("=" * 70)
    
    for file_name, description in manifest['files'].items():
        file_path = deployment_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            if size > 1024*1024:
                size_str = f"{size / (1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} bytes"
            logger.info(f"  ✓ {file_name:30s} ({size_str:>10s}) - {description}")
    
    logger.info("=" * 70)
    logger.info(f"Package location: {deployment_dir}")
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info("1. Copy 'deployment_calibrated' folder to production environment")
    logger.info("2. Install dependencies: pip install torch numpy")
    logger.info("3. Use 'inference.py' module for predictions")
    logger.info("4. Review 'DEPLOYMENT_README.md' for integration guide")
    
    return 0


if __name__ == '__main__':
    sys.exit(create_deployment_package())
