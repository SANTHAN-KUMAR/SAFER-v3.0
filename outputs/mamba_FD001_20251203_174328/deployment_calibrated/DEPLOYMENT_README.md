# FD001 Model Deployment Package

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
