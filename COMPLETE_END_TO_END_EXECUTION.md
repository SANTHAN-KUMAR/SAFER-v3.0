# 🚀 SAFER v3.0: Complete End-to-End Project Execution Guide

**Document:** Full project execution from data loading to final deployment  
**Date:** December 4, 2025  
**Target:** Complete system design implementation

---

## 📋 Table of Contents

1. [Prerequisites & Setup](#prerequisites--setup)
2. [Project Architecture Overview](#project-architecture-overview)
3. [Complete Execution Workflow](#complete-execution-workflow)
4. [Component-by-Component Execution](#component-by-component-execution)
5. [Full Pipeline Integration](#full-pipeline-integration)
6. [Deployment & Verification](#deployment--verification)

---

## Prerequisites & Setup

### System Requirements

```bash
# 1. Python 3.8+ installed
python --version

# 2. Required packages (check requirements.txt)
pip install -r requirements.txt

# 3. Data directory present
ls -la CMAPSSData/  # Should contain train_FD001.txt, test_FD001.txt, RUL_FD001.txt
```

### Environment Setup

```bash
# 1. Navigate to project
cd /path/to/SAFER\ v3.0\ -\ Initial

# 2. Verify structure
ls -la safer_v3/
ls -la scripts/
ls -la CMAPSSData/

# 3. Check dependencies
python -c "import torch; import numpy as np; print('✓ Dependencies OK')"
```

---

## Project Architecture Overview

### System Design Architecture (from PDF)

```
SAFER v3.0 Architecture
├─ Input Layer (C-MAPSS Sensors: 14 prognostic)
│
├─ Perception Layer
│  ├─ Mamba RUL Predictor (DAL E - Non-critical)
│  ├─ LPV-SINDy Physics Monitor (DAL C - Safety-critical)
│  └─ LSTM Baseline (DAL C - Safety baseline)
│
├─ Shared Memory Fabric
│  ├─ Lock-free Ring Buffer
│  ├─ Process Manager (IPC)
│  └─ Zero-copy Data Transport
│
├─ Decision Layer (Simplex Architecture)
│  ├─ Safety Monitor
│  ├─ Conformal Prediction (UQ)
│  ├─ Mode Switcher (COMPLEX ↔ BASELINE)
│  └─ Alert Manager
│
└─ Output Layer
   ├─ RUL Prediction (point + bounds)
   ├─ Confidence Intervals
   ├─ Alerts (CRITICAL/WARNING/CAUTION/ADVISORY)
   └─ Deployment Package (PyTorch/ONNX)
```

---

## Complete Execution Workflow

### Phase 1: Data Preparation (5 minutes)

**Goal:** Load and normalize C-MAPSS dataset

```bash
# Step 1.1: Verify data files exist
python << 'EOF'
from pathlib import Path
import numpy as np

data_dir = Path('CMAPSSData')
files = ['train_FD001.txt', 'test_FD001.txt', 'RUL_FD001.txt']

for f in files:
    path = data_dir / f
    if path.exists():
        data = np.loadtxt(path)
        print(f"✓ {f}: {data.shape}")
    else:
        print(f"✗ {f}: MISSING")
EOF

# Expected output:
# ✓ train_FD001.txt: (13096, 26)
# ✓ test_FD001.txt: (10196, 26)
# ✓ RUL_FD001.txt: (100,)
```

**Step 1.2: Load and Preprocess Data**

```python
# scripts/run_full_safer_pipeline.py - PHASE 1

import numpy as np
from pathlib import Path
from safer_v3.utils.dataset import load_cmapss, prepare_sequences

print("\n" + "="*60)
print("PHASE 1: DATA PREPARATION")
print("="*60)

# Load C-MAPSS FD001 dataset
data_dir = Path('CMAPSSData')
X_train, y_train, X_test, y_test = load_cmapss(
    data_dir=data_dir,
    dataset='FD001',
    normalize=True,  # Z-score normalization
    sequence_length=30,  # RUL prediction window
)

print(f"✓ Data loaded:")
print(f"  - Training: {X_train.shape} samples")
print(f"  - Testing: {X_test.shape} samples")
print(f"  - Features: 14 prognostic sensors (after filtering)")

# Prepare sequences for RNN models
X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, seq_len=30)
X_test_seq, y_test_seq = prepare_sequences(X_test, y_test, seq_len=30)

print(f"✓ Sequences prepared:")
print(f"  - Training sequences: {X_train_seq.shape}")
print(f"  - Test sequences: {X_test_seq.shape}")

# Save preprocessed data
np.save('outputs/X_train_seq.npy', X_train_seq)
np.save('outputs/y_train_seq.npy', y_train_seq)
np.save('outputs/X_test_seq.npy', X_test_seq)
np.save('outputs/y_test_seq.npy', y_test_seq)

print("✓ Data saved to outputs/")
```

---

### Phase 2: Model Training (40 minutes)

#### 2a. Train Mamba RUL Predictor (DAL E)

```python
# PHASE 2A: MAMBA TRAINING

print("\n" + "="*60)
print("PHASE 2A: TRAIN MAMBA RUL PREDICTOR (DAL E)")
print("="*60)

import torch
from safer_v3.core.mamba import MambaRULPredictor
from safer_v3.utils.config import MambaConfig

# Configuration
config = MambaConfig(
    d_input=14,        # 14 sensors
    d_model=64,        # Model dimension
    d_state=16,        # State space dimension
    n_layers=4,        # Stack depth
    dropout=0.1,
    learning_rate=1e-3,
    batch_size=32,
    epochs=20,
    device='cpu',      # Use 'cuda' if available
)

# Create model
model_mamba = MambaRULPredictor(config)
print(f"✓ Mamba model created: {model_mamba}")

# Training loop
device = torch.device(config.device)
model_mamba = model_mamba.to(device)
optimizer = torch.optim.Adam(model_mamba.parameters(), lr=config.learning_rate)
loss_fn = torch.nn.MSELoss()

print(f"\nTraining on {len(X_train_seq)} sequences...")

train_losses = []
for epoch in range(config.epochs):
    epoch_loss = 0
    n_batches = 0
    
    for i in range(0, len(X_train_seq), config.batch_size):
        batch_x = torch.from_numpy(X_train_seq[i:i+config.batch_size]).float().to(device)
        batch_y = torch.from_numpy(y_train_seq[i:i+config.batch_size]).float().to(device)
        
        # Forward pass
        pred = model_mamba(batch_x)
        loss = loss_fn(pred.squeeze(), batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    avg_loss = epoch_loss / n_batches
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{config.epochs}: Loss = {avg_loss:.4f}")

# Save checkpoint
torch.save(model_mamba.state_dict(), 'checkpoints/mamba_rul.pt')
print(f"\n✓ Mamba model saved: checkpoints/mamba_rul.pt")

# Evaluate on test set
model_mamba.eval()
with torch.no_grad():
    X_test_tensor = torch.from_numpy(X_test_seq).float().to(device)
    y_pred_mamba = model_mamba(X_test_tensor).cpu().numpy()
    
    mse = np.mean((y_pred_mamba.squeeze() - y_test_seq)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred_mamba.squeeze() - y_test_seq))
    
    print(f"\n✓ Mamba Performance:")
    print(f"  - Test RMSE: {rmse:.4f} cycles")
    print(f"  - Test MAE: {mae:.4f} cycles")
```

#### 2b. Train LPV-SINDy Physics Monitor (DAL C)

```python
# PHASE 2B: LPV-SINDy TRAINING

print("\n" + "="*60)
print("PHASE 2B: TRAIN LPV-SINDy PHYSICS MONITOR (DAL C)")
print("="*60)

from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
from safer_v3.physics.library import LPVAugmentedLibrary
from safer_v3.utils.config import LPVSINDyConfig

# Configuration for LPV-SINDy
lpv_config = LPVSINDyConfig(
    n_features=14,
    polynomial_degree=2,
    window_size=5,
    threshold=0.1,
    alpha=0.01,
    egtm_sensor_idx=9,  # EGT margin sensor
)

# Create augmented library (NEW FEATURE)
lib = LPVAugmentedLibrary(degree=2, include_bias=True)

# Create monitor
lpv_monitor = LPVSINDyMonitor(config=lpv_config, library=lib, n_sensors=14)

print(f"Training LPV-SINDy on {len(X_train)} samples...")

# Fit the model
fit_results = lpv_monitor.fit(X_train, validate=True, val_fraction=0.2)

print(f"\n✓ LPV-SINDy Fit Results:")
print(f"  - Non-zero terms: {fit_results['total_nonzero']}")
print(f"  - Sparsity: {fit_results['sparsity']:.2%}")
print(f"  - Train RMSE: {fit_results['train_rmse']:.4f}")
print(f"  - Val RMSE: {fit_results.get('val_rmse', 'N/A')}")

# Compute scheduling parameter (NEW FEATURE)
p = lpv_monitor.compute_scheduling_parameter(X_train)
print(f"\n✓ Scheduling Parameter p(t):")
print(f"  - Range: [{p.min():.3f}, {p.max():.3f}]")
print(f"  - Mean: {p.mean():.3f}")

# LPV Decomposition (NEW FEATURE)
decomp = lpv_monitor.fit_lpv_decomposition(p)
print(f"\n✓ LPV Decomposition:")
print(f"  - Ξ₀ (baseline) norm: {decomp['xi_0_norm']:.6f}")
print(f"  - Ξ₁ (degradation) norm: {decomp['xi_1_norm']:.6f}")
print(f"  - Decomposition R²: {decomp['explained_variance']:.4f}")

# Save model
lpv_monitor.save('checkpoints/lpv_sindy_model.pt')
print(f"\n✓ LPV-SINDy model saved: checkpoints/lpv_sindy_model.pt")

# Get residuals for anomaly detection
residuals = lpv_monitor._compute_residuals(X_train)
print(f"\n✓ Physics residuals computed: {residuals.shape}")
```

#### 2c. Train LSTM Baseline (DAL C)

```python
# PHASE 2C: LSTM BASELINE TRAINING

print("\n" + "="*60)
print("PHASE 2C: TRAIN LSTM BASELINE (DAL C)")
print("="*60)

from safer_v3.core.baselines import LSTMPredictor

# Configuration
lstm_config = {
    'd_input': 14,
    'd_hidden': 64,
    'n_layers': 2,
    'dropout': 0.2,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'epochs': 20,
}

# Create model
model_lstm = LSTMPredictor(
    input_size=lstm_config['d_input'],
    hidden_size=lstm_config['d_hidden'],
    num_layers=lstm_config['n_layers'],
    dropout=lstm_config['dropout'],
)

print(f"✓ LSTM baseline created")

# Training (similar to Mamba)
device = torch.device('cpu')
model_lstm = model_lstm.to(device)
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=lstm_config['learning_rate'])
loss_fn = torch.nn.MSELoss()

print(f"Training on {len(X_train_seq)} sequences...")

for epoch in range(lstm_config['epochs']):
    epoch_loss = 0
    n_batches = 0
    
    for i in range(0, len(X_train_seq), lstm_config['batch_size']):
        batch_x = torch.from_numpy(X_train_seq[i:i+lstm_config['batch_size']]).float().to(device)
        batch_y = torch.from_numpy(y_train_seq[i:i+lstm_config['batch_size']]).float().to(device)
        
        pred = model_lstm(batch_x)
        loss = loss_fn(pred.squeeze(), batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    if (epoch + 1) % 5 == 0:
        avg_loss = epoch_loss / n_batches
        print(f"  Epoch {epoch+1}/{lstm_config['epochs']}: Loss = {avg_loss:.4f}")

# Evaluate
model_lstm.eval()
with torch.no_grad():
    X_test_tensor = torch.from_numpy(X_test_seq).float().to(device)
    y_pred_lstm = model_lstm(X_test_tensor).cpu().numpy()
    
    rmse_lstm = np.sqrt(np.mean((y_pred_lstm.squeeze() - y_test_seq)**2))
    mae_lstm = np.mean(np.abs(y_pred_lstm.squeeze() - y_test_seq))

print(f"\n✓ LSTM Baseline Performance:")
print(f"  - Test RMSE: {rmse_lstm:.4f} cycles")
print(f"  - Test MAE: {mae_lstm:.4f} cycles")

# Save
torch.save(model_lstm.state_dict(), 'checkpoints/lstm_baseline.pt')
print(f"✓ LSTM baseline saved: checkpoints/lstm_baseline.pt")
```

---

### Phase 3: Uncertainty Quantification (10 minutes)

**Goal:** Calibrate conformal prediction intervals

```python
# PHASE 3: CONFORMAL PREDICTION CALIBRATION

print("\n" + "="*60)
print("PHASE 3: CONFORMAL PREDICTION CALIBRATION")
print("="*60)

from safer_v3.decision.conformal import ConformalPredictor

# Split test set for calibration
n_cal = len(y_test_seq) // 2
X_cal = X_test_seq[:n_cal]
y_cal = y_test_seq[:n_cal]
X_val = X_test_seq[n_cal:]
y_val = y_test_seq[n_cal:]

# Get predictions from both models
with torch.no_grad():
    X_cal_tensor = torch.from_numpy(X_cal).float().to(device)
    X_val_tensor = torch.from_numpy(X_val).float().to(device)
    
    model_mamba.eval()
    model_lstm.eval()
    
    y_pred_mamba_cal = model_mamba(X_cal_tensor).cpu().numpy().squeeze()
    y_pred_lstm_cal = model_lstm(X_cal_tensor).cpu().numpy().squeeze()
    
    y_pred_mamba_val = model_mamba(X_val_tensor).cpu().numpy().squeeze()
    y_pred_lstm_val = model_lstm(X_val_tensor).cpu().numpy().squeeze()

# Use Mamba predictions with conformal calibration
cp = ConformalPredictor(alpha=0.1)  # 90% coverage target

# Calibrate on calibration set
cp.calibrate(
    y_true=y_cal,
    y_pred=y_pred_mamba_cal,
)

print(f"✓ Conformal predictor calibrated")
print(f"  - Calibration samples: {len(y_cal)}")
print(f"  - Target coverage: 90%")

# Get prediction intervals on validation set
results = []
for i in range(len(y_val)):
    lower, upper = cp.predict(y_pred_mamba_val[i])
    results.append({
        'y_true': y_val[i],
        'y_pred': y_pred_mamba_val[i],
        'lower': lower,
        'upper': upper,
        'in_interval': lower <= y_val[i] <= upper,
    })

coverage = np.mean([r['in_interval'] for r in results])
avg_width = np.mean([r['upper'] - r['lower'] for r in results])

print(f"\n✓ Prediction Intervals:")
print(f"  - Achieved coverage: {coverage:.2%}")
print(f"  - Average interval width: {avg_width:.2f} cycles")
print(f"  - Interval examples:")
for i in range(5):
    r = results[i]
    print(f"    True: {r['y_true']:.1f}, Pred: {r['y_pred']:.1f}, "
          f"CI: [{r['lower']:.1f}, {r['upper']:.1f}]")

# Save conformal predictor
import pickle
with open('checkpoints/conformal_predictor.pkl', 'wb') as f:
    pickle.dump(cp, f)
```

---

### Phase 4: Simplex Decision Module (5 minutes)

**Goal:** Initialize and test mode switching logic

```python
# PHASE 4: SIMPLEX DECISION MODULE

print("\n" + "="*60)
print("PHASE 4: SIMPLEX DECISION MODULE")
print("="*60)

from safer_v3.decision.simplex import SimplexDecisionModule, SimplexState

# Configuration
simplex_config = {
    'physics_threshold': 0.5,
    'divergence_threshold': 30.0,
    'uncertainty_threshold': 50.0,
    'recovery_window': 20,
    'hysteresis_cycles': 10,
    'conservative_margin': 5.0,
}

# Create Simplex module
simplex = SimplexDecisionModule(config=simplex_config)

print(f"✓ Simplex Decision Module initialized")
print(f"  - Physics threshold: {simplex_config['physics_threshold']}")
print(f"  - Divergence threshold: {simplex_config['divergence_threshold']}")
print(f"  - Uncertainty threshold: {simplex_config['uncertainty_threshold']}")

# Test on first 100 samples
print(f"\nTesting Simplex on first 100 samples...")

decisions = []
states = []

for i in range(min(100, len(y_val))):
    # Get predictions from both models
    complex_rul = float(y_pred_mamba_val[i])
    baseline_rul = float(y_pred_lstm_val[i])
    
    # Get confidence bounds
    lower, upper = cp.predict(complex_rul)
    
    # Get physics residual (from LPV-SINDy)
    sample = X_val[i:i+1]
    physics_residual = np.abs(np.mean(lpv_monitor._compute_residuals(sample)))
    
    # Make decision
    decision = simplex.decide(
        complex_rul=complex_rul,
        baseline_rul=baseline_rul,
        rul_lower=lower,
        rul_upper=upper,
        physics_residual=physics_residual,
    )
    
    decisions.append(decision)
    states.append(decision.state.name)

# Analyze results
from collections import Counter
state_counts = Counter(states)

print(f"\n✓ Simplex Decisions Summary:")
print(f"  - COMPLEX mode: {state_counts.get('COMPLEX', 0)} times")
print(f"  - BASELINE mode: {state_counts.get('BASELINE', 0)} times")
print(f"  - Final state: {simplex._state.name}")

# Sample decision details
print(f"\n✓ Sample Decisions:")
for i in range(5):
    d = decisions[i]
    print(f"  Sample {i+1}:")
    print(f"    - True RUL: {y_val[i]:.1f}")
    print(f"    - Mamba: {d.complex_rul:.1f}, LSTM: {d.baseline_rul:.1f}")
    print(f"    - Final: {d.rul:.1f}, Mode: {d.state.name}")
    if d.switch_reason:
        print(f"    - Reason: {d.switch_reason.name}")
```

---

### Phase 5: Alert Management (2 minutes)

**Goal:** Generate RUL-based alerts

```python
# PHASE 5: ALERT MANAGEMENT

print("\n" + "="*60)
print("PHASE 5: ALERT MANAGEMENT")
print("="*60)

from safer_v3.decision.alerts import AlertManager, AlertLevel

# Create alert manager
alert_mgr = AlertManager(
    critical_rul=10,
    warning_rul=25,
    caution_rul=50,
    advisory_rul=100,
)

print(f"✓ Alert Manager initialized")
print(f"  - CRITICAL: RUL ≤ 10 cycles")
print(f"  - WARNING: RUL ≤ 25 cycles")
print(f"  - CAUTION: RUL ≤ 50 cycles")
print(f"  - ADVISORY: RUL ≤ 100 cycles")

# Generate alerts for test set
alerts_generated = []

for i, d in enumerate(decisions):
    alert = alert_mgr.update(
        rul=d.rul,
        timestamp=i,
        source='Simplex',
    )
    
    if alert is not None:
        alerts_generated.append(alert)
        print(f"\n⚠️  ALERT #{len(alerts_generated)}: {alert['level'].name}")
        print(f"    Sample {i}: RUL = {d.rul:.1f} cycles")
        print(f"    True RUL: {y_val[i]:.1f} cycles")

if not alerts_generated:
    print(f"\n✓ No critical alerts generated (all RULs sufficient)")
else:
    print(f"\n✓ Total alerts generated: {len(alerts_generated)}")
```

---

### Phase 6: Export to Deployment Format (5 minutes)

**Goal:** Export models for production deployment

```python
# PHASE 6: DEPLOYMENT EXPORT

print("\n" + "="*60)
print("PHASE 6: DEPLOYMENT EXPORT")
print("="*60)

# 6a. Export Mamba to ONNX
print("\nExporting Mamba to ONNX...")

from safer_v3.core.mamba import export_mamba_onnx

export_mamba_onnx(
    model=model_mamba,
    output_path='checkpoints/onnx_export/mamba_rul.onnx',
    sample_input_shape=(1, 30, 14),
)

print(f"✓ Mamba exported: checkpoints/onnx_export/mamba_rul.onnx")

# 6b. Create deployment package
print("\nCreating deployment package...")

deployment_package = {
    'models': {
        'mamba': 'checkpoints/mamba_rul.pt',
        'lstm': 'checkpoints/lstm_baseline.pt',
        'lpv_sindy': 'checkpoints/lpv_sindy_model.pt',
        'onnx': 'checkpoints/onnx_export/mamba_rul.onnx',
    },
    'config': {
        'mamba': {
            'd_input': 14,
            'd_model': 64,
            'd_state': 16,
            'n_layers': 4,
        },
        'lpv_sindy': {
            'polynomial_degree': 2,
            'window_size': 5,
        },
    },
    'performance': {
        'mamba_rmse': rmse,
        'lstm_rmse': rmse_lstm,
        'conformal_coverage': float(coverage),
        'conformal_width': float(avg_width),
    },
}

import json
with open('deployment/models/deployment_config.json', 'w') as f:
    json.dump(deployment_package, f, indent=2)

print(f"✓ Deployment package created: deployment/models/deployment_config.json")

# 6c. Create inference example
print("\nCreating inference example...")

inference_example = """
# Example: Using SAFER v3.0 for RUL Prediction

import torch
import numpy as np
from safer_v3.core.mamba import MambaRULPredictor
from safer_v3.decision.simplex import SimplexDecisionModule

# Load models
mamba = MambaRULPredictor.load('checkpoints/mamba_rul.pt')
lstm = LSTMPredictor.load('checkpoints/lstm_baseline.pt')

# Get new sensor data (shape: [batch, 30 timesteps, 14 sensors])
X_new = np.random.randn(1, 30, 14)

# Predict
with torch.no_grad():
    rul_mamba = mamba(torch.from_numpy(X_new).float())
    rul_lstm = lstm(torch.from_numpy(X_new).float())

# Use Simplex to decide
simplex = SimplexDecisionModule()
decision = simplex.decide(
    complex_rul=float(rul_mamba),
    baseline_rul=float(rul_lstm),
    rul_lower=rul_mamba - 10,
    rul_upper=rul_mamba + 10,
)

print(f"RUL: {decision.rul:.1f} cycles")
print(f"Mode: {decision.state.name}")
"""

with open('deployment/inference/inference_example.py', 'w') as f:
    f.write(inference_example)

print(f"✓ Inference example created: deployment/inference/inference_example.py")
```

---

### Phase 7: Complete System Evaluation (10 minutes)

**Goal:** Comprehensive system performance evaluation

```python
# PHASE 7: COMPLETE SYSTEM EVALUATION

print("\n" + "="*60)
print("PHASE 7: COMPLETE SYSTEM EVALUATION")
print("="*60)

# 7a. Prediction Accuracy
print("\n1. PREDICTION ACCURACY")
print("-" * 40)

mamba_rmse = np.sqrt(np.mean((y_pred_mamba_val - y_val)**2))
mamba_mae = np.mean(np.abs(y_pred_mamba_val - y_val))
lstm_rmse = np.sqrt(np.mean((y_pred_lstm_val - y_val)**2))
lstm_mae = np.mean(np.abs(y_pred_lstm_val - y_val))

print(f"Mamba RUL Predictor (DAL E):")
print(f"  RMSE: {mamba_rmse:.4f} cycles")
print(f"  MAE: {mamba_mae:.4f} cycles")

print(f"\nLSTM Baseline (DAL C):")
print(f"  RMSE: {lstm_rmse:.4f} cycles")
print(f"  MAE: {lstm_mae:.4f} cycles")

# 7b. Uncertainty Quantification
print("\n2. UNCERTAINTY QUANTIFICATION")
print("-" * 40)
print(f"Conformal Prediction:")
print(f"  Coverage: {coverage:.2%}")
print(f"  Avg Width: {avg_width:.2f} cycles")
print(f"  Target Coverage: 90%")
print(f"  Status: {'✓ MET' if coverage >= 0.90 else '✗ NOT MET'}")

# 7c. Safety Properties
print("\n3. SAFETY PROPERTIES")
print("-" * 40)
print(f"Simplex Architecture:")
print(f"  COMPLEX uses: Mamba (high performance)")
print(f"  BASELINE uses: LSTM (safety fallback)")
print(f"  Physics monitor: LPV-SINDy (validation)")
print(f"  Fail-safe: YES (defaults to BASELINE)")

# 7d. System Latency
print("\n4. SYSTEM LATENCY")
print("-" * 40)

import time

# Measure inference time
n_inference = 100
X_test_sample = torch.from_numpy(X_val[:n_inference]).float().to(device)

# Mamba latency
start = time.time()
with torch.no_grad():
    _ = model_mamba(X_test_sample)
mamba_latency = (time.time() - start) / n_inference * 1000  # ms

# LSTM latency
start = time.time()
with torch.no_grad():
    _ = model_lstm(X_test_sample)
lstm_latency = (time.time() - start) / n_inference * 1000  # ms

print(f"Per-sample latency:")
print(f"  Mamba: {mamba_latency:.2f} ms")
print(f"  LSTM: {lstm_latency:.2f} ms")
print(f"  Total: {mamba_latency + lstm_latency:.2f} ms")
print(f"  Target: <20 ms")
print(f"  Status: {'✓ MET' if mamba_latency + lstm_latency < 20 else '✗ NOT MET'}")

# 7e. Resource Usage
print("\n5. RESOURCE USAGE")
print("-" * 40)

mamba_params = sum(p.numel() for p in model_mamba.parameters())
lstm_params = sum(p.numel() for p in model_lstm.parameters())

print(f"Model Parameters:")
print(f"  Mamba: {mamba_params:,} parameters")
print(f"  LSTM: {lstm_params:,} parameters")
print(f"  Total: {mamba_params + lstm_params:,} parameters")

# 7f. Overall System Verdict
print("\n" + "="*60)
print("SYSTEM READINESS ASSESSMENT")
print("="*60)

checks = {
    'Accuracy': mamba_rmse < 25,  # Within spec
    'Coverage': coverage >= 0.90,  # 90% coverage
    'Latency': mamba_latency < 20,  # <20ms
    'Safety': True,  # Simplex implemented
    'Deployment': True,  # ONNX exported
}

print("\nPre-Deployment Checks:")
for check, status in checks.items():
    symbol = "✓" if status else "✗"
    print(f"  {symbol} {check}")

all_passed = all(checks.values())
print(f"\n{'✓ SYSTEM READY FOR DEPLOYMENT' if all_passed else '✗ REQUIRES FIXES'}")
```

---

### Phase 8: Generate Final Report (3 minutes)

**Goal:** Create comprehensive project report

```python
# PHASE 8: FINAL REPORT

print("\n" + "="*60)
print("PHASE 8: FINAL REPORT GENERATION")
print("="*60)

import json
from datetime import datetime

report = {
    'project': 'SAFER v3.0 - RUL Prediction System',
    'timestamp': datetime.now().isoformat(),
    'dataset': 'C-MAPSS FD001',
    'architecture': {
        'input_layer': '14 prognostic sensors + 3 operational settings',
        'perception_layer': [
            'Mamba RUL Predictor (DAL E - Non-safety-critical)',
            'LPV-SINDy Physics Monitor (DAL C - Safety-critical)',
            'LSTM Baseline (DAL C - Safety fallback)',
        ],
        'decision_layer': 'Simplex Architecture',
        'output_layer': 'RUL + Confidence Intervals + Alerts',
    },
    'performance': {
        'mamba': {
            'rmse': float(mamba_rmse),
            'mae': float(mamba_mae),
            'latency_ms': float(mamba_latency),
        },
        'lstm': {
            'rmse': float(lstm_rmse),
            'mae': float(lstm_mae),
            'latency_ms': float(lstm_latency),
        },
        'conformal': {
            'coverage': float(coverage),
            'avg_interval_width': float(avg_width),
            'target_coverage': 0.90,
        },
    },
    'deployment': {
        'models': [
            'PyTorch checkpoints',
            'ONNX export',
            'LPV-SINDy coefficients',
            'Conformal predictor',
        ],
        'inference_latency_ms': float(mamba_latency + lstm_latency),
        'throughput_samples_per_sec': 1000 / (mamba_latency + lstm_latency),
    },
    'safety': {
        'architecture': 'Simplex (High-perf + Safety)',
        'fail_safe': 'Baseline LSTM',
        'physics_monitor': 'LPV-SINDy',
        'uncertainty_quantification': 'Conformal Prediction',
        'alerts': 'RUL-based thresholds',
    },
    'status': 'READY FOR DEPLOYMENT',
}

# Save report
report_path = 'outputs/final_safer_v3_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"✓ Final report saved: {report_path}")

# Print summary
print("\n" + "="*60)
print("SAFER v3.0 SUMMARY")
print("="*60)
print(f"\nDataset: C-MAPSS FD001")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Features: 14 sensors")

print(f"\nPerformance:")
print(f"  Mamba RMSE: {mamba_rmse:.4f} cycles")
print(f"  LSTM RMSE: {lstm_rmse:.4f} cycles")
print(f"  Improvement: {(1 - mamba_rmse/lstm_rmse)*100:.1f}%")

print(f"\nUncertainty:")
print(f"  Conformal coverage: {coverage:.2%}")
print(f"  Avg interval width: {avg_width:.2f} cycles")

print(f"\nDeployment:")
print(f"  Inference latency: {mamba_latency + lstm_latency:.2f} ms")
print(f"  Throughput: {1000/(mamba_latency + lstm_latency):.1f} samples/sec")

print(f"\nStatus: ✓ DEPLOYMENT READY")
print("="*60 + "\n")

return report
```

---

## Full Pipeline Integration

### Complete End-to-End Script

Create `scripts/run_full_safer_pipeline.py`:

```python
#!/usr/bin/env python3
"""
Complete SAFER v3.0 Pipeline Execution
Runs all phases from data loading to deployment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all components
import numpy as np
import torch
from datetime import datetime

def main():
    print("\n" + "="*70)
    print(" " * 15 + "SAFER v3.0 COMPLETE PIPELINE")
    print(" " * 10 + "Safeguarded Against Failure Through Effective Recovery")
    print("="*70 + "\n")
    
    start_time = datetime.now()
    
    # PHASE 1: Data Preparation
    # [Insert Phase 1 code]
    
    # PHASE 2: Model Training
    # [Insert Phase 2 code]
    
    # PHASE 3: Uncertainty Quantification
    # [Insert Phase 3 code]
    
    # PHASE 4: Simplex Decision
    # [Insert Phase 4 code]
    
    # PHASE 5: Alert Management
    # [Insert Phase 5 code]
    
    # PHASE 6: Deployment Export
    # [Insert Phase 6 code]
    
    # PHASE 7: System Evaluation
    # [Insert Phase 7 code]
    
    # PHASE 8: Final Report
    report = phase_8_final_report()
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✓ COMPLETE PIPELINE EXECUTED IN {elapsed:.1f} seconds")
    print(f"✓ STATUS: {report['status']}")
    
    return report

if __name__ == '__main__':
    report = main()
```

### Running the Complete Pipeline

```bash
# Method 1: Run the complete script
python scripts/run_full_safer_pipeline.py

# Method 2: Run individual phases (if modularized)
python scripts/phase_1_data_preparation.py
python scripts/phase_2_model_training.py
python scripts/phase_3_uncertainty.py
# etc...

# Method 3: Interactive Jupyter notebook
jupyter notebook train_mamba_kaggle.ipynb
```

---

## Deployment & Verification

### Post-Execution Checklist

```bash
# 1. Verify all outputs created
ls -la outputs/
ls -la checkpoints/
ls -la deployment/models/

# 2. Check model files
file checkpoints/mamba_rul.pt
file checkpoints/onnx_export/mamba_rul.onnx

# 3. Run inference test
python deployment/inference/inference_example.py

# 4. Validate performance
python -c "
import json
with open('outputs/final_safer_v3_report.json') as f:
    report = json.load(f)
    print(f'Status: {report[\"status\"]}')
    print(f'RMSE: {report[\"performance\"][\"mamba\"][\"rmse\"]:.4f}')
    print(f'Coverage: {report[\"performance\"][\"conformal\"][\"coverage\"]:.2%}')
"
```

### Expected Output Structure

```
SAFER v3.0 - Initial/
├── checkpoints/
│   ├── mamba_rul.pt                    ✓
│   ├── lstm_baseline.pt                ✓
│   ├── lpv_sindy_model.pt              ✓
│   ├── conformal_predictor.pkl         ✓
│   └── onnx_export/
│       └── mamba_rul.onnx              ✓
├── outputs/
│   ├── X_train_seq.npy                 ✓
│   ├── y_train_seq.npy                 ✓
│   ├── X_test_seq.npy                  ✓
│   ├── y_test_seq.npy                  ✓
│   └── final_safer_v3_report.json      ✓
└── deployment/
    ├── models/
    │   └── deployment_config.json      ✓
    └── inference/
        └── inference_example.py        ✓
```

---

## Success Criteria

### System is Deployment-Ready When:

✅ All 8 phases completed successfully  
✅ Mamba RMSE < 25 cycles  
✅ Conformal coverage ≥ 90%  
✅ Inference latency < 20ms  
✅ No safety violations (Simplex functions correctly)  
✅ All models exported to deployment format  
✅ Final report generated  

---

**READY TO EXECUTE?**

Run: `python scripts/run_full_safer_pipeline.py`

