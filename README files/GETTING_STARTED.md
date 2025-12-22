# SAFER v3.0 - Complete Getting Started Guide

This guide shows you how to run the entire SAFER v3.0 FD001 pipeline from scratch.

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU (tested on RTX 4060, Kaggle P100/T4)
- **Memory**: 16GB+ RAM, 8GB+ VRAM
- **Storage**: 20GB+ for data and outputs
- **OS**: Linux/Mac/Windows with Docker support

### Software Requirements
- Docker (recommended) or local Python 3.9+
- NVIDIA Docker runtime (for GPU support)

## Option 1: Docker (Recommended)

### Step 1: Build Docker Image

```bash
cd "/run/media/santhankumar/New Volume/SAFER v3.0 - Initial"
docker build -f Dockerfile -t kaggle-image:latest .
```

If using existing image, skip to Step 2.

### Step 2: Start Docker Container

```bash
docker run --rm -it \
  --gpus all \
  -v "/run/media/santhankumar/New Volume/SAFER v3.0 - Initial:/workspace" \
  -w /workspace \
  kaggle-image:latest \
  bash
```

You're now inside the container with all dependencies installed.

### Step 3: Run Complete Pipeline

Inside the container:

```bash
cd /workspace

# Option A: Run entire end-to-end pipeline (recommended)
python scripts/end_to_end_fd001.py

# Option B: Run steps individually (for debugging)
python scripts/calibrate_fd001.py
python scripts/alert_and_simplex_fd001.py
python scripts/generate_report_fd001.py
python scripts/create_deployment_package.py
```

## Option 2: Local Python Environment

### Step 1: Install Dependencies

```bash
cd "/run/media/santhankumar/New Volume/SAFER v3.0 - Initial"

# Install project in development mode
pip install -e .

# Install additional requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib scikit-learn loguru pyyaml
```

### Step 2: Configure GPU (Optional)

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# If GPU not detected, install appropriate CUDA version
# Refer to https://pytorch.org/get-started/locally/
```

### Step 3: Run Pipeline

```bash
# Complete pipeline
python scripts/end_to_end_fd001.py

# Individual steps
python scripts/calibrate_fd001.py
python scripts/alert_and_simplex_fd001.py
python scripts/generate_report_fd001.py
python scripts/create_deployment_package.py
```

## Understanding the Pipeline

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   SAFER v3.0 Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. MODEL TRAINING (Already Done)                          │
│     └─ Mamba RUL Predictor trained on FD001                │
│        Output: checkpoints/best_model.pt                   │
│                                                             │
│  2. CONFORMAL CALIBRATION (Step 1)                         │
│     └─ Run on validation set (80/20 split)                 │
│        ├─ Load model checkpoint                            │
│        ├─ Inference on 3,490 validation samples            │
│        ├─ Calibrate quantile for 90% coverage              │
│        └─ Output: calibration/conformal_params.json        │
│                                                             │
│  3. SIMPLEX + ALERTS (Step 2)                              │
│     └─ Run on test set (10,196 samples)                    │
│        ├─ Load model + calibrated intervals                │
│        ├─ Simplex decision: complex vs baseline            │
│        ├─ Alert generation at thresholds                   │
│        └─ Output: alerts/simplex_and_alerts_results.json   │
│                                                             │
│  4. EVALUATION REPORT (Step 3)                             │
│     └─ Aggregate all results                               │
│        ├─ Compute summary statistics                       │
│        ├─ Generate text report                             │
│        ├─ Create dashboard visualization                   │
│        └─ Output: report/EVALUATION_REPORT.txt             │
│                                                             │
│  5. DEPLOYMENT PACKAGE (Step 4)                            │
│     └─ Bundle for production                               │
│        ├─ Model weights + config                           │
│        ├─ Calibration parameters                           │
│        ├─ Inference helper code                            │
│        └─ Output: deployment_calibrated/                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### What Each Script Does

#### 1. `calibrate_fd001.py` - Conformal Calibration
**Purpose**: Compute prediction intervals with 90% coverage guarantee

**Input**: 
- Model: `checkpoints/best_model.pt`
- Config: `outputs/mamba_FD001_20251203_174328/args.json`
- Data: Training split (validation 20%)

**Process**:
1. Load model architecture from args.json (6 layers, 128-dim)
2. Run inference on 3,490 validation samples
3. Compute residuals as nonconformity scores
4. Compute quantile for 90% coverage
5. Verify empirical coverage matches target

**Output**: 
```
outputs/mamba_FD001_20251203_174328/calibration/
├── conformal_params.json          # Calibration parameters
├── validation_intervals.json       # All intervals
├── calibration_diagnostics.png    # 4-panel plot
└── example_intervals.json         # 100 samples
```

**Time**: ~3-5 seconds
**GPU Memory**: ~2GB

#### 2. `alert_and_simplex_fd001.py` - Safety Integration
**Purpose**: Apply Simplex safety module and generate alerts

**Input**:
- Model: `checkpoints/best_model.pt`
- Calibration: `calibration/conformal_params.json`
- Test data: 10,196 samples

**Process**:
1. Load model + calibration
2. Run Simplex on each test sample:
   - **Complex Mode**: Mamba prediction (high performance)
   - **Baseline Mode**: Mean forecast (high assurance)
   - **Safety Check**: Physics thresholds
   - **Decision**: Use complex or fallback to baseline
3. Generate alerts at thresholds (CRITICAL, WARNING, CAUTION, ADVISORY)
4. Compute alerting metrics

**Output**:
```
outputs/mamba_FD001_20251203_174328/alerts/
├── simplex_and_alerts_results.json  # All decisions + alerts
├── simplex_and_alerts.png           # Timeline visualization
└── alert_statistics.json            # Metrics summary
```

**Time**: ~5-7 seconds
**GPU Memory**: ~2GB

#### 3. `generate_report_fd001.py` - Evaluation Report
**Purpose**: Generate comprehensive summary and visualizations

**Input**:
- All results from steps 1-2
- Test metrics (from earlier evaluation)

**Process**:
1. Load all results from calibration, simplex, alerts
2. Aggregate statistics
3. Generate formatted text report with interpretations
4. Create professional dashboard (4 plots)
5. Save as JSON for machine-readability

**Output**:
```
outputs/mamba_FD001_20251203_174328/report/
├── EVALUATION_REPORT.txt        # Complete analysis
├── summary_statistics.json      # Machine-readable metrics
└── evaluation_dashboard.png     # 4-panel visualization
```

**Time**: ~2-3 seconds

#### 4. `create_deployment_package.py` - Deployment
**Purpose**: Bundle everything for production

**Input**:
- Model, configs, calibration, report

**Process**:
1. Copy model checkpoint
2. Create inference helper module (inference.py)
3. Bundle configs and documentation
4. Generate manifest

**Output**:
```
outputs/mamba_FD001_20251203_174328/deployment_calibrated/
├── model_checkpoint.pt          # 8.2 MB - Model weights
├── model_config.json            # Architecture
├── conformal_params.json        # Calibration
├── simplex_config.json          # Safety config
├── alert_config.json            # Alert thresholds
├── inference.py                 # Production code
├── DEPLOYMENT_README.md         # Integration guide
└── MANIFEST.json                # Package spec
```

**Time**: ~1 second

## Running the Pipeline

### Complete Run (All Steps)

```bash
# Inside Docker or Python environment
python scripts/end_to_end_fd001.py
```

**Expected Output**:
```
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║              SAFER v3.0 - FD001 RUL Prediction Pipeline              ║
║                    End-to-End Automation                              ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

======================================================================
STEP: 1. Conformal Calibration (Validation Set)
======================================================================
...
✓ Conformal calibration complete!

======================================================================
STEP: 2. Simplex Decision & Alert Integration (Test Set)
======================================================================
...
✓ Simplex and alerting integration complete!

======================================================================
STEP: 3. Evaluation Report Generation
======================================================================
...
✓ Evaluation report generation complete!

======================================================================
STEP: 4. Create Deployment Package
======================================================================
...
✓ Deployment package created successfully!

======================================================================
PIPELINE SUMMARY
======================================================================
  ✓ PASSED  1. Conformal Calibration (Validation Set)
  ✓ PASSED  2. Simplex Decision & Alert Integration (Test Set)
  ✓ PASSED  3. Evaluation Report Generation
  ✓ PASSED  4. Create Deployment Package
======================================================================

✓ Pipeline completed successfully!
```

**Total Time**: ~15-20 seconds

### Selective Runs

```bash
# Skip calibration (use existing params)
python scripts/end_to_end_fd001.py --skip-calibration

# Skip alerts
python scripts/end_to_end_fd001.py --skip-alerts

# Skip report
python scripts/end_to_end_fd001.py --skip-report

# Only do deployment
python scripts/end_to_end_fd001.py --skip-calibration --skip-alerts --skip-report
```

### Individual Step Runs

```bash
# Just calibration
python scripts/calibrate_fd001.py
# Output: calibration/ folder with conformal params

# Just alerts
python scripts/alert_and_simplex_fd001.py
# Output: alerts/ folder with decisions

# Just report
python scripts/generate_report_fd001.py
# Output: report/ folder with analysis

# Just deployment
python scripts/create_deployment_package.py
# Output: deployment_calibrated/ folder
```

## Monitoring Progress

All scripts use structured logging with levels:
- **INFO** - General progress
- **SUCCESS** - ✓ Completed steps
- **WARNING** - Non-critical issues
- **ERROR** - Failures

Example log output:
```
2025-12-04 07:06:16.285 | INFO     | Loading checkpoint...
2025-12-04 07:06:16.359 | INFO     | Model loaded: epoch=20, val_rmse=14.2460
2025-12-04 07:06:16.992 | INFO     | Validation samples: 3490
2025-12-04 07:14:29.511 | SUCCESS  | ✓ Calibration complete!
```

## Reviewing Results

### After Calibration
```bash
# View calibration summary
cat outputs/mamba_FD001_20251203_174328/calibration/conformal_params.json

# View diagnostics plot
open outputs/mamba_FD001_20251203_174328/calibration/calibration_diagnostics.png
```

Expected metrics:
- Coverage: 90.0%
- Quantile: 38.55 cycles
- Average interval width: 77.09 cycles

### After Simplex & Alerts
```bash
# View decision statistics
cat outputs/mamba_FD001_20251203_174328/alerts/alert_statistics.json

# View timeline plot
open outputs/mamba_FD001_20251203_174328/alerts/simplex_and_alerts.png
```

Expected results:
- Mode switches: 2
- Complex usage: 1%
- Total alerts: ~1

### After Report
```bash
# View evaluation report
cat outputs/mamba_FD001_20251203_174328/report/EVALUATION_REPORT.txt

# View dashboard
open outputs/mamba_FD001_20251203_174328/report/evaluation_dashboard.png

# View metrics summary
cat outputs/mamba_FD001_20251203_174328/report/summary_statistics.json
```

### Deployment Package
```bash
# Review what's in the package
ls -lh outputs/mamba_FD001_20251203_174328/deployment_calibrated/

# Read deployment guide
cat outputs/mamba_FD001_20251203_174328/deployment_calibrated/DEPLOYMENT_README.md

# Review manifest
cat outputs/mamba_FD001_20251203_174328/deployment_calibrated/MANIFEST.json
```

## Deploying to Production

### Step 1: Copy Deployment Package
```bash
cp -r outputs/mamba_FD001_20251203_174328/deployment_calibrated/ \
      /path/to/production/safer_fd001_model/
```

### Step 2: Use in Your Application
```python
# In your Python code
from safer_fd001_model.inference import FD001Predictor

# Initialize
predictor = FD001Predictor('safer_fd001_model/')

# Make predictions
sensor_data = load_sensor_data()  # Shape: (seq_len, 14)
rul, rul_lower, rul_upper = predictor.predict(sensor_data)

print(f"RUL: {rul:.1f} cycles")
print(f"90% Confidence Interval: [{rul_lower:.1f}, {rul_upper:.1f}]")

# Generate alerts
from safer_v3.decision.alerts import AlertManager, create_rul_alert_rules

alert_manager = AlertManager()
alert_manager.add_rules(create_rul_alert_rules())

alerts = alert_manager.process(rul)
for alert in alerts:
    print(f"[{alert.level.name}] {alert.message}")
    if alert.level >= AlertLevel.WARNING:
        schedule_maintenance(rul)
```

## Troubleshooting

### Issue: GPU Out of Memory
**Solution**: Reduce batch size in data_module initialization
```bash
# Check GPU memory
nvidia-smi

# If needed, run with smaller batch size (modify scripts)
batch_size = 64  # Reduce from 256
```

### Issue: Model Loading Fails
**Error**: `RuntimeError: Unexpected key(s) in state_dict`
**Solution**: Ensure args.json has correct n_layers (should be 6)
```bash
# Verify config
cat outputs/mamba_FD001_20251203_174328/args.json | grep n_layers
# Should show: "n_layers": 6
```

### Issue: Data Not Found
**Error**: `FileNotFoundError: CMAPSSData/`
**Solution**: Ensure data directory exists with required files
```bash
ls -la CMAPSSData/
# Should show: train_FD001.txt, test_FD001.txt, RUL_FD001.txt
```

### Issue: Python Module Not Found
**Error**: `ModuleNotFoundError: No module named 'safer_v3'`
**Solution**: Install project in development mode
```bash
pip install -e .
# Or run scripts from project root
cd /run/media/santhankumar/New\ Volume/SAFER\ v3.0\ -\ Initial
```

## Performance Benchmarks

### Hardware: NVIDIA RTX 4060 / Kaggle P100
| Step | Time | Memory |
|------|------|--------|
| Calibration | 3-5s | ~2GB |
| Simplex+Alerts | 5-7s | ~2GB |
| Report | 2-3s | ~1GB |
| Deployment | 1s | <1GB |
| **Total** | **~15-20s** | **Peak 2GB** |

### Accuracy Metrics (FD001)
| Metric | Value |
|--------|-------|
| Test RMSE | 18.15 cycles |
| Test MAE | 12.02 cycles |
| NASA Score | 82,479.69 |
| R² Score | 0.6648 |

### Calibration Quality
| Metric | Value |
|--------|-------|
| Target Coverage | 90.0% |
| Empirical Coverage | 90.0% |
| Interval Width | 77.09 cycles |
| Quantile | 38.55 cycles |

## Next Steps

1. **Review Documentation**
   - Read `EVALUATION_REPORT.txt` for detailed analysis
   - Review `DEPLOYMENT_README.md` for integration specifics

2. **Deploy to Production**
   - Copy `deployment_calibrated/` folder
   - Use `inference.py` module in your application
   - Monitor interval coverage for domain drift

3. **Monitor and Maintain**
   - Track prediction accuracy over time
   - Detect distribution shifts
   - Schedule recalibration quarterly
   - Update alert thresholds based on operational data

4. **Extend to Other Datasets**
   - Adapt scripts for FD002, FD003, FD004
   - Reuse calibration pipeline architecture
   - Train ensemble models for robustness

## Support & Resources

- **Main README**: `README.md` - Project overview
- **Pipeline Guide**: `PIPELINE_README.md` - Complete workflow
- **Deployment Guide**: `deployment_calibrated/DEPLOYMENT_README.md` - Production setup
- **Code Documentation**: Inline docstrings in all scripts
- **Logs**: Check console output for detailed diagnostics

---

**Summary**: Run `python scripts/end_to_end_fd001.py` to execute the complete SAFER v3.0 FD001 pipeline in ~15-20 seconds. All outputs saved to `outputs/mamba_FD001_20251203_174328/`.
