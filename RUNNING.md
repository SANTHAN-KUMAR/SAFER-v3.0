# How to Run SAFER v3.0

✅ **Installation Complete!** All tests passed successfully.

## System Status

```
✅ All imports successful
✅ C-MAPSS data files present (train_FD001.txt, test_FD001.txt, RUL_FD001.txt)
✅ Models working (Mamba: 133,953 params, LSTM: 185,922 params)
✅ Simulation components operational
✅ Decision modules functional
```

## Quick Start

### 1. Verify Installation
```bash
python demo.py
```
Should show: "5/5 tests passed" ✅

### 2. Train the Mamba Model (Quick Test - 5 epochs)
```bash
python -m safer_v3.scripts.train_mamba --data_dir CMAPSSData --dataset FD001 --epochs 5 --batch_size 16
```

### 3. Train Full Mamba Model (Production - 50 epochs)
```bash
python -m safer_v3.scripts.train_mamba --data_dir CMAPSSData --dataset FD001 --epochs 50 --batch_size 32 --lr 0.001
```

Expected output location: `outputs/mamba_fd001_*.pth`

### 4. Train Baseline Models for Comparison
```bash
# Train all baselines (LSTM, Transformer, CNN-LSTM)
python -m safer_v3.scripts.train_baselines --data_dir CMAPSSData --dataset FD001 --model all --epochs 50

# Or train specific baseline
python -m safer_v3.scripts.train_baselines --data_dir CMAPSSData --dataset FD001 --model lstm --epochs 50
```

### 5. Full SAFER Pipeline (Python Script)

Create a file `run_safer.py`:

```python
import torch
import numpy as np
from pathlib import Path

# Import SAFER components
from safer_v3.core.mamba import MambaRULPredictor
from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
from safer_v3.decision.simplex import SimplexDecisionModule
from safer_v3.decision.conformal import SplitConformalPredictor
from safer_v3.decision.alerts import AlertManager, create_rul_alert_rules
from safer_v3.utils.metrics import RULMetrics

# Load trained model
model = MambaRULPredictor(d_input=14, d_model=64, d_state=16, n_layers=4)
model.load_state_dict(torch.load('outputs/mamba_fd001_best.pth'))
model.eval()

# Initialize safety components
physics_monitor = LPVSINDyMonitor(input_dim=14, n_library_terms=20)
simplex = SimplexDecisionModule()
conformal = SplitConformalPredictor(alpha=0.1)
alert_manager = AlertManager()
alert_manager.add_rules(create_rul_alert_rules())

# Make predictions with full SAFER pipeline
sensor_data = torch.randn(1, 50, 14)  # Replace with real sensor data

with torch.no_grad():
    # Complex controller (Mamba)
    complex_rul = model(sensor_data).item()
    
    # Physics monitor
    physics_residual = physics_monitor.compute_residual(sensor_data)
    
    # Conformal prediction intervals
    rul_lower, rul_upper = conformal.predict(sensor_data, coverage=0.9)
    
    # Simplex decision
    decision = simplex.decide(
        complex_rul=complex_rul,
        baseline_rul=complex_rul + 5,  # Replace with actual baseline prediction
        rul_lower=rul_lower,
        rul_upper=rul_upper,
        physics_residual=physics_residual
    )
    
    # Alert generation
    alerts = alert_manager.process(rul_value=decision.rul)
    
    print(f"Final RUL: {decision.rul:.1f} cycles")
    print(f"Controller: {decision.state.name}")
    print(f"Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  - {alert.message}")
```

## Training Parameters

### Mamba Model
- **Input**: 14 sensors (C-MAPSS subset)
- **Model Dim**: 64
- **State Dim**: 16
- **Layers**: 4
- **Parameters**: ~134K

### Expected Performance (FD001)
- **RMSE**: < 15 cycles
- **Score**: < 400
- **Training Time**: ~10 min/epoch on GPU, ~30 min/epoch on CPU

## Project Structure

```
SAFER v3.0 - Initial/
├── safer_v3/                 # Main package
│   ├── core/                 # Mamba & baselines
│   ├── physics/              # LPV-SINDy monitor
│   ├── decision/             # Simplex, conformal, alerts
│   ├── fabric/               # ONNX, JIT, quantization
│   ├── simulation/           # Data generation
│   ├── scripts/              # Training scripts
│   └── utils/                # Config, metrics, logging
├── CMAPSSData/               # Training data
├── outputs/                  # Saved models (created during training)
├── demo.py                   # Installation test
├── QUICKSTART.md             # Detailed documentation
└── RUNNING.md                # This file

```

## Command-Line Tools

After installation, these commands are available:

```bash
# Train Mamba model
safer-train-mamba --data_dir CMAPSSData --dataset FD001 --epochs 50

# Train baseline models
safer-train-baselines --data_dir CMAPSSData --dataset FD001 --model all
```

**Note**: Add the Scripts directory to PATH if these commands don't work:
```
C:\Users\kiran\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts
```

## Troubleshooting

### Import Errors
```bash
# Reinstall in development mode
pip install -e .
```

### CUDA/GPU Issues
```bash
# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Data Loading Errors
Ensure CMAPSSData/ contains:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

## Next Steps

1. ✅ **Verification**: `python demo.py` (Done - 5/5 passed!)
2. 🎯 **Quick Training**: `python -m safer_v3.scripts.train_mamba --data_dir CMAPSSData --dataset FD001 --epochs 5`
3. 🚀 **Production Training**: Train full model with 50 epochs
4. 📊 **Evaluation**: Check `outputs/` for metrics and visualizations
5. 🔧 **Integration**: Use trained model in your application

## Documentation

- **QUICKSTART.md**: Detailed usage guide with code examples
- **README.md**: Project overview and architecture
- **demo.py**: Working examples of all components

---

**Status**: SAFER v3.0 is installed and ready for use! 🚀

For questions or issues, check the docstrings in each module or refer to QUICKSTART.md.
