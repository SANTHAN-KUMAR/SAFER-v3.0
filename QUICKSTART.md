# SAFER v3.0 Quick Start Guide

## Setup Verification

You've already completed:
- ✅ Dependencies installed (`pip install -r requirements.txt`)
- ✅ C-MAPSS FD001 dataset available in `CMAPSSData/`

## Running the Project

### 1. Train Mamba Model (Recommended First Step)

Train the main Mamba RUL predictor on FD001 dataset:

```powershell
# Basic training (recommended for first run)
python -m safer_v3.scripts.train_mamba --data_dir CMAPSSData --dataset FD001 --epochs 50 --batch_size 32

# Full training with ONNX export
python -m safer_v3.scripts.train_mamba --data_dir CMAPSSData --dataset FD001 --epochs 100 --batch_size 64 --export_onnx --use_amp

# Train ensemble (5 models for uncertainty estimation)
python -m safer_v3.scripts.train_mamba --data_dir CMAPSSData --dataset FD001 --ensemble 5 --epochs 50
```

**Output:** Results saved to `outputs/mamba_FD001_<timestamp>/`
- Model checkpoint: `mamba_model_0.pt`
- Metrics: `metrics.json`
- Training log: `train.log`
- ONNX model (if exported): `mamba_model.onnx`

### 2. Train Baseline Models (For Comparison)

Train baseline models to compare with Mamba:

```powershell
# Train all baselines (LSTM, Transformer, CNN-LSTM)
python -m safer_v3.scripts.train_baselines --data_dir CMAPSSData --dataset FD001 --model all --epochs 50

# Train specific baseline
python -m safer_v3.scripts.train_baselines --data_dir CMAPSSData --dataset FD001 --model lstm --epochs 50
python -m safer_v3.scripts.train_baselines --data_dir CMAPSSData --dataset FD001 --model transformer --epochs 50
```

**Output:** Results saved to `outputs/baselines_FD001_<timestamp>/`
- Model checkpoints: `lstm/`, `transformer/`, `cnn_lstm/`
- Comparison report: `comparison_report.json`

### 3. Using Trained Models in Python

After training, use the models for inference:

```python
import torch
import numpy as np
from safer_v3.core.mamba import MambaRULPredictor, MambaConfig
from safer_v3.core.trainer import CMAPSSDataset

# Load trained Mamba model
config = MambaConfig(d_input=14, d_model=64, d_state=16, n_layers=4)
model = MambaRULPredictor(config)
model.load_state_dict(torch.load('outputs/mamba_FD001_<timestamp>/mamba_model_0.pt'))
model.eval()

# Load test data
dataset = CMAPSSDataset(
    data_dir='CMAPSSData',
    dataset='FD001',
    mode='test',
    seq_length=50,
)

# Get a sample
sequence, true_rul = dataset[0]

# Predict
with torch.no_grad():
    sequence = sequence.unsqueeze(0)  # Add batch dimension
    predicted_rul = model(sequence)
    print(f"Predicted RUL: {predicted_rul.item():.1f} cycles")
    print(f"True RUL: {true_rul:.1f} cycles")
```

### 4. Complete SAFER Pipeline Example

Use all SAFER components together:

```python
import torch
from safer_v3.core.mamba import MambaRULPredictor, MambaConfig
from safer_v3.core.baselines import LSTMPredictor
from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
from safer_v3.decision.simplex import SimplexDecisionModule, SimplexConfig
from safer_v3.decision.conformal import SplitConformalPredictor
from safer_v3.decision.alerts import AlertManager, create_rul_alert_rules

# 1. Load models
mamba_config = MambaConfig(d_input=14, d_model=64, d_state=16, n_layers=4)
mamba = MambaRULPredictor(mamba_config)
mamba.load_state_dict(torch.load('outputs/mamba_model.pt'))
mamba.eval()

baseline = LSTMPredictor(d_input=14, d_hidden=64, n_layers=2)
baseline.load_state_dict(torch.load('outputs/lstm_model.pt'))
baseline.eval()

# 2. Create physics monitor
physics = LPVSINDyMonitor(n_states=14, window_size=5)
# Note: Physics monitor needs to be trained on data first
# physics.fit(training_sequences)

# 3. Create decision module
simplex_config = SimplexConfig(
    physics_threshold=0.1,
    divergence_threshold=30.0,
    uncertainty_threshold=50.0,
)
simplex = SimplexDecisionModule(simplex_config)

# 4. Create conformal predictor (requires calibration)
conformal = SplitConformalPredictor(coverage=0.9, symmetric=True)
# Note: Needs calibration on validation set first
# conformal.calibrate(y_cal_true, y_cal_pred)

# 5. Create alert manager
alert_manager = AlertManager()
alert_manager.add_rules(create_rul_alert_rules(
    critical_threshold=10,
    warning_threshold=25,
    caution_threshold=50,
))

# 6. Process sensor data
sensor_sequence = torch.randn(1, 50, 14)  # Example: batch=1, seq=50, sensors=14

with torch.no_grad():
    # Get predictions
    mamba_rul = mamba(sensor_sequence).item()
    baseline_rul = baseline(sensor_sequence).item()
    
    # Physics check (assuming trained)
    # physics_residual = physics.detect_anomaly(sensor_sequence.numpy()[0])
    physics_residual = 0.05  # Example value
    
    # Get confidence interval (assuming calibrated)
    # interval = conformal.predict(mamba_rul)
    # For demo, use fixed interval
    rul_lower = mamba_rul - 15
    rul_upper = mamba_rul + 15
    
    # Simplex decision
    result = simplex.decide(
        complex_rul=mamba_rul,
        baseline_rul=baseline_rul,
        rul_lower=rul_lower,
        rul_upper=rul_upper,
        physics_residual=physics_residual,
    )
    
    # Generate alerts
    alerts = alert_manager.process(
        rul_value=result.rul,
        context={'confidence': (rul_lower, rul_upper)}
    )
    
    # Display results
    print(f"\n{'='*50}")
    print(f"SAFER v3.0 Prediction Results")
    print(f"{'='*50}")
    print(f"Mamba RUL:    {mamba_rul:.1f} cycles")
    print(f"Baseline RUL: {baseline_rul:.1f} cycles")
    print(f"Final RUL:    {result.rul:.1f} cycles [{rul_lower:.1f}, {rul_upper:.1f}]")
    print(f"Decision:     {result.state.name} (using {result.used_source})")
    print(f"Physics:      {physics_residual:.4f}")
    print(f"\nActive Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  [{alert.level.name}] {alert.message}")
```

### 5. Generate Synthetic Data (Optional)

Generate synthetic data for testing:

```python
from safer_v3.simulation.data_generator import CMAPSSGenerator

# Create synthetic dataset
generator = CMAPSSGenerator(
    n_train=100,
    n_test=100,
    seed=42,
)

train_df, test_df, rul_array = generator.generate()
generator.save('synthetic_data', prefix='FD_SYN')

# Train on synthetic data
# python -m safer_v3.scripts.train_mamba --data_dir synthetic_data --dataset FD_SYN
```

### 6. Streaming Data Simulation

Test with streaming data:

```python
from safer_v3.simulation.data_generator import StreamingDataGenerator

# Create streaming generator
with StreamingDataGenerator(n_engines=5, sample_rate_hz=1.0) as generator:
    for _ in range(100):  # Process 100 samples
        packet = generator.get_next(timeout=2.0)
        if packet:
            print(f"Engine {packet['engine_id']}: Cycle {packet['cycle']}, RUL {packet['rul_true']:.0f}")
            # Process with SAFER pipeline...
```

## Expected Performance

Based on C-MAPSS FD001:

| Model | RMSE (cycles) | NASA Score | Training Time |
|-------|---------------|------------|---------------|
| Mamba | 12-15 | 250-300 | ~15 min |
| LSTM | 15-18 | 300-400 | ~10 min |
| Transformer | 14-17 | 280-350 | ~20 min |

*Note: Times on CPU. GPU training is 5-10x faster.*

## Troubleshooting

### Issue: Import errors
```powershell
# Install package in development mode
pip install -e .
```

### Issue: CUDA out of memory
```powershell
# Reduce batch size
python -m safer_v3.scripts.train_mamba --batch_size 16

# Disable AMP
python -m safer_v3.scripts.train_mamba --batch_size 32
```

### Issue: Slow training
```powershell
# Enable AMP for faster training
python -m safer_v3.scripts.train_mamba --use_amp

# Use GPU if available (automatic)
python -m safer_v3.scripts.train_mamba --device cuda
```

## Next Steps

1. **Train Models**: Start with Mamba on FD001
2. **Evaluate**: Check metrics in `outputs/*/metrics.json`
3. **Compare**: Train baselines and compare performance
4. **Integrate**: Use full SAFER pipeline with Simplex + Conformal UQ
5. **Deploy**: Export to ONNX for production

## Quick Test Run

Try this minimal example to verify everything works:

```powershell
# Quick test (5 epochs, small batch)
python -m safer_v3.scripts.train_mamba --data_dir CMAPSSData --dataset FD001 --epochs 5 --batch_size 16
```

This should complete in ~2-3 minutes and create a checkpoint in `outputs/`.
