# SAFER v3.0 Deployment Package

This package contains all artifacts needed to deploy the SAFER v3.0 
Remaining Useful Life (RUL) prediction system.

## Contents

```
deployment/
├── models/
│   ├── mamba_rul.pt              # Trained Mamba model (PyTorch)
│   ├── lstm_baseline.pt          # Trained LSTM baseline
│   ├── lpv_sindy_model.npz       # LPV-SINDy physics monitor
│   └── onnx/
│       └── mamba_rul.onnx        # ONNX export for deployment
├── config/
│   ├── conformal_params.json     # Conformal prediction calibration
│   └── simplex_config.json       # Simplex decision parameters
├── metrics/
│   ├── full_safer_results.json   # Complete pipeline metrics
│   └── full_safer_dashboard.png  # Visualization
├── inference/
│   └── inference_example.py      # Example inference code
└── README.md                     # This file
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Inference

```python
from inference_example import SAFERInference

# Initialize
safer = SAFERInference(
    mamba_checkpoint="models/mamba_rul.pt",
    baseline_checkpoint="models/lstm_baseline.pt",
    physics_model="models/lpv_sindy_model",
    conformal_params="config/conformal_params.json"
)

# Predict RUL
sequence = ...  # Shape: (30, 14) - 30 timesteps, 14 sensors
result = safer.predict(sequence)

print(f"RUL: {result.rul:.1f} cycles")
print(f"Confidence: [{result.rul_lower:.1f}, {result.rul_upper:.1f}]")
print(f"State: {result.state}")
print(f"Alerts: {result.alerts}")
```

## Architecture Components

### 1. Mamba RUL Predictor (DAL E)
- **Type**: Primary high-accuracy predictor
- **Architecture**: Mamba (state-space model)
- **Performance**: RMSE ~20.40 cycles on FD001

### 2. LSTM Baseline (DAL C)
- **Type**: Safety fallback predictor
- **Architecture**: Bidirectional LSTM
- **Performance**: RMSE ~38.24 cycles on FD001

### 3. LPV-SINDy Physics Monitor (DAL C)
- **Type**: Physics-based anomaly detection
- **Method**: Linear Parameter-Varying Sparse Identification
- **Purpose**: Detect physics violations in sensor data

### 4. Conformal Prediction
- **Coverage**: 90% guaranteed
- **Method**: Split conformal prediction
- **Quantile**: 38.55 cycles

### 5. Simplex Decision Module
- **Role**: Safety arbitration between Mamba and baseline
- **States**: COMPLEX (Mamba), BASELINE (LSTM), RECOVERY
- **Thresholds**: 
  - Physics: 3.0σ
  - Divergence: 50 cycles
  - Uncertainty: 100 cycles

### 6. Alert Manager
- **Levels**: CRITICAL (≤10), WARNING (≤25), CAUTION (≤50), ADVISORY (≤100)
- **Purpose**: Multi-level maintenance alerts

## Performance Metrics

Based on C-MAPSS FD001 test set (10,196 samples):

| Metric | Mamba | LSTM Baseline | Simplex Final |
|--------|-------|---------------|---------------|
| RMSE   | 20.40 | 38.24         | ~40.00        |
| MAE    | 12.49 | 35.44         | ~35.00        |
| Coverage | - | -             | 91.2%         |

**Simplex Statistics:**
- Complex mode: ~10%
- Baseline mode: ~90%
- Total alerts: Low (safe operation)

## Deployment Considerations

### Safety Certification
- Mamba (DAL E): Primary predictor, moderate assurance
- LSTM + Physics + Simplex (DAL C): High assurance components
- Full system follows Simplex architecture for runtime safety

### Hardware Requirements
- **Training**: CUDA GPU (8GB+ VRAM)
- **Inference**: CPU sufficient, GPU optional
- **Memory**: ~500MB for all models loaded

### Latency
- Mamba inference: ~5-10ms per sample (GPU)
- Full SAFER pipeline: ~15-20ms per sample
- ONNX runtime: ~3-5ms per sample (optimized)

### Integration Points

#### Option 1: Python API
```python
from safer_v3 import SAFERPredictor
predictor = SAFERPredictor(...)
result = predictor.predict(sensor_data)
```

#### Option 2: ONNX Runtime
```python
import onnxruntime as ort
session = ort.InferenceSession("models/onnx/mamba_rul.onnx")
output = session.run(None, {"input": sensor_data})
```

#### Option 3: REST API
Deploy using FastAPI or Flask with ONNX backend.

## Validation

To validate the deployment:

```bash
python inference/inference_example.py --test
```

## Support

For questions or issues:
- Documentation: See project README.md
- Model cards: See models/README.md
- Training scripts: See scripts/

## License

See LICENSE file in project root.

## Citation

If you use SAFER v3.0 in your research, please cite:

```
@software{safer_v3,
  title={SAFER v3.0: Safety-Aware Framework for Enhanced Reliability},
  year={2024},
  note={Turbofan RUL prediction with Mamba and Simplex architecture}
}
```
