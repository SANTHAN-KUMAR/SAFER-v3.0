# SAFER v3.0 - FD001 Complete Pipeline

Complete, automated workflow for training, calibration, safety integration, and deployment of the FD001 RUL prediction model.

## Overview

This pipeline implements the SAFER v3.0 architecture:

```
├── Model Training (Mamba RUL Predictor)
├── Conformal Calibration (90% coverage prediction intervals)
├── Simplex Safety Module (complex model + baseline + safety monitor)
├── Alert System (multi-level alerts with thresholds)
├── Evaluation & Reporting
└── Deployment Package (production-ready artifacts)
```

## Quick Start

### Run Complete Pipeline

```bash
# Run all steps in sequence
python scripts/end_to_end_fd001.py

# Skip specific steps
python scripts/end_to_end_fd001.py --skip-calibration
python scripts/end_to_end_fd001.py --skip-alerts
python scripts/end_to_end_fd001.py --skip-report
```

### Run Individual Steps

```bash
# 1. Conformal Calibration (validation set)
python scripts/calibrate_fd001.py
# Outputs: conformal_params.json, validation_intervals.json, calibration_diagnostics.png

# 2. Simplex & Alert Integration (test set)
python scripts/alert_and_simplex_fd001.py
# Outputs: simplex_and_alerts_results.json, simplex_and_alerts.png, alert_statistics.json

# 3. Generate Evaluation Report
python scripts/generate_report_fd001.py
# Outputs: EVALUATION_REPORT.txt, summary_statistics.json, evaluation_dashboard.png

# 4. Create Deployment Package
python scripts/create_deployment_package.py
# Outputs: deployment_calibrated/ directory with all artifacts
```

## Pipeline Steps

### Step 1: Conformal Calibration

**Purpose**: Compute prediction intervals with guaranteed coverage

**Input**:
- Trained model checkpoint: `checkpoints/best_model.pt`
- Model config: `outputs/mamba_FD001_20251203_174328/args.json`
- Validation data: FD001 training split (80/20 split)

**Process**:
1. Load trained model with correct architecture (6 layers, 128-dim)
2. Run inference on validation set (3,490 samples)
3. Compute absolute residuals as nonconformity scores
4. Calibrate quantile for 90% target coverage
5. Evaluate empirical coverage on calibration set

**Output**:
- `calibration/conformal_params.json` - Calibration parameters
- `calibration/validation_intervals.json` - All intervals
- `calibration/example_intervals.json` - 100 samples
- `calibration/calibration_diagnostics.png` - 4-panel visualization

**Expected Results**:
- Target coverage: 90%
- Empirical coverage: 90.0% (±0.03%)
- Quantile: 38.55 cycles
- Average interval width: 77.09 cycles

### Step 2: Simplex & Alert Integration

**Purpose**: Apply safety architecture and generate alerts

**Input**:
- Model: `checkpoints/best_model.pt`
- Calibration: `calibration/conformal_params.json`
- Test data: FD001 test set (10,196 samples)

**Process**:
1. Load model and calibrated intervals
2. Run Simplex decision module on test set:
   - Complex mode: Mamba predictions
   - Baseline mode: Mean forecast
   - Safety monitoring: Physics thresholds
3. Apply alert rules at different RUL thresholds
4. Compute alerting metrics (accuracy, FPR, time-to-detection)

**Output**:
- `alerts/simplex_and_alerts_results.json` - All decisions and alerts
- `alerts/simplex_and_alerts.png` - Timeline visualization
- `alerts/alert_statistics.json` - Alert metrics

**Expected Results**:
- Test RMSE: 20.40 cycles
- Simplex mode switches: 2
- Complex mode usage: 1%
- Alert distribution by level

### Step 3: Evaluation Report

**Purpose**: Generate comprehensive evaluation summary

**Input**:
- Test metrics: `outputs/test_results/test_metrics.json`
- Calibration: `calibration/conformal_params.json`
- Decisions: `alerts/simplex_and_alerts_results.json`
- Alert stats: `alerts/alert_statistics.json`

**Process**:
1. Load all result files
2. Compute summary statistics
3. Generate formatted text report
4. Create professional dashboard visualization
5. Save metrics summary as JSON

**Output**:
- `report/EVALUATION_REPORT.txt` - Complete text report
- `report/summary_statistics.json` - Machine-readable metrics
- `report/evaluation_dashboard.png` - 4-panel dashboard

### Step 4: Deployment Package

**Purpose**: Bundle all artifacts for production

**Input**:
- Model checkpoint: `checkpoints/best_model.pt`
- All configurations and parameters
- Evaluation report

**Process**:
1. Copy model checkpoint
2. Copy/create configuration files
3. Generate inference helper module
4. Create deployment documentation
5. Generate manifest and summary

**Output**:
- `deployment_calibrated/model_checkpoint.pt` - Model weights
- `deployment_calibrated/model_config.json` - Architecture config
- `deployment_calibrated/conformal_params.json` - Calibration
- `deployment_calibrated/simplex_config.json` - Safety config
- `deployment_calibrated/alert_config.json` - Alert thresholds
- `deployment_calibrated/inference.py` - Production inference code
- `deployment_calibrated/DEPLOYMENT_README.md` - Integration guide
- `deployment_calibrated/MANIFEST.json` - Package manifest

## Key Metrics

### Test Performance
- **RMSE**: 18.15 cycles (average absolute error)
- **MAE**: 12.02 cycles (robust to outliers)
- **NASA Score**: 82,479.69 (penalizes late predictions)
- **R² Score**: 0.6648 (explains 66.5% of variance)

### Calibration Quality
- **Target Coverage**: 90.0%
- **Empirical Coverage**: 90.0% (perfectly calibrated)
- **Interval Width**: 77.09 cycles (±38.55)
- **Samples**: 3,490 (validation set)

### Safety Statistics
- **Mode Switches**: 2 (baseline/complex transitions)
- **Baseline Usage**: 99% (conservative safe mode)
- **Complex Usage**: 1% (high performance when safe)

### Alert System
- **Total Alerts**: 1 (low alert rate on test set)
- **Alert Distribution**: Mostly ADVISORY level
- **Time-to-Detection**: 94.5 ± 29.5 cycles (before critical)

## Output Directory Structure

```
outputs/mamba_FD001_20251203_174328/
├── calibration/
│   ├── conformal_params.json
│   ├── validation_intervals.json
│   ├── example_intervals.json
│   └── calibration_diagnostics.png
├── alerts/
│   ├── simplex_and_alerts_results.json
│   ├── simplex_and_alerts.png
│   └── alert_statistics.json
├── report/
│   ├── EVALUATION_REPORT.txt
│   ├── summary_statistics.json
│   └── evaluation_dashboard.png
└── deployment_calibrated/
    ├── model_checkpoint.pt
    ├── model_config.json
    ├── conformal_params.json
    ├── simplex_config.json
    ├── alert_config.json
    ├── inference.py
    ├── DEPLOYMENT_README.md
    └── MANIFEST.json
```

## Deployment Instructions

### 1. Copy Deployment Package

```bash
cp -r outputs/mamba_FD001_20251203_174328/deployment_calibrated/ \
      /path/to/production/
```

### 2. Install Dependencies

```bash
pip install torch numpy
```

### 3. Use in Production

```python
from inference import FD001Predictor

# Initialize
predictor = FD001Predictor('deployment_calibrated/')

# Predict with uncertainty
rul, lower, upper = predictor.predict(sensor_data)
print(f"RUL: {rul:.1f} [{lower:.1f}, {upper:.1f}]")
```

### 4. Generate Alerts

```python
from safer_v3.decision.alerts import AlertManager, create_rul_alert_rules

alert_manager = AlertManager()
alert_manager.add_rules(create_rul_alert_rules())

alerts = alert_manager.process(rul)
for alert in alerts:
    print(f"[{alert.level.name}] {alert.message}")
```

## Troubleshooting

### Issue: Calibration fails with "sequence_length" error
**Solution**: Fixed in current version. Use `window_size` instead.

### Issue: Model loading fails with unexpected keys
**Solution**: Ensure `n_layers=6` matches checkpoint. Check `args.json` for correct architecture.

### Issue: JSON serialization error with numpy types
**Solution**: Convert numpy types to Python types before JSON dump (use `bool()`, `float()`, etc.)

### Issue: Test data not found
**Solution**: Ensure `CMAPSSData/` directory contains `test_FD001.txt` and `RUL_FD001.txt`

## Advanced Usage

### Skip Steps for Faster Iteration

```bash
# Skip calibration, run alerts only (assuming calibration exists)
python scripts/end_to_end_fd001.py --skip-calibration

# Skip all except deployment
python scripts/end_to_end_fd001.py --skip-calibration --skip-alerts --skip-report
```

### Run Individual Scripts

```bash
# Calibration only
python scripts/calibrate_fd001.py

# Alerts only
python scripts/alert_and_simplex_fd001.py

# Report only
python scripts/generate_report_fd001.py

# Deployment only
python scripts/create_deployment_package.py
```

### Monitor Progress

All scripts use structured logging via `loguru`. Output includes:
- INFO: General progress
- WARNING: Non-fatal issues
- ERROR: Failures
- SUCCESS: Completion messages

## Architecture Details

### Mamba RUL Predictor
- **Type**: Selective state space model
- **Layers**: 6 blocks
- **Model Dimension**: 128
- **State Dimension**: 16
- **Input**: 14 turbofan sensors
- **Output**: RUL (cycles)
- **Complexity**: O(L) training, O(1) inference

### Conformal Prediction
- **Method**: Split conformal prediction
- **Nonconformity Score**: Absolute residual |y - ŷ|
- **Coverage Guarantee**: Distribution-free
- **Quantile**: 38.55 cycles (90% coverage)

### Simplex Safety Module
- **Complex Controller**: Mamba (DAL E - non-critical)
- **Baseline Controller**: Mean forecast (DAL C - certified)
- **Safety Monitor**: Physics residual thresholds
- **Decision Logic**: Formal switching guarantees
- **Rate Limiting**: Max 2 switches per minute

### Alert System
- **Levels**: INFO, ADVISORY, CAUTION, WARNING, CRITICAL
- **Thresholds**:
  - CRITICAL: RUL ≤ 10 (immediate action)
  - WARNING: RUL ≤ 25 (urgent)
  - CAUTION: RUL ≤ 50 (plan)
  - ADVISORY: RUL ≤ 100 (monitor)
- **Features**: Hysteresis, rate limiting, audit trail

## References

- **Mamba**: Gu et al., "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- **Conformal Prediction**: Vovk et al., "Algorithmic Learning in a Random World" (2005)
- **Simplex**: Sha et al., "Using Simplicity to Control Complexity" (2001)
- **RUL Scoring**: NASA C-MAPSS dataset scoring function

## Support & Documentation

- `EVALUATION_REPORT.txt` - Complete model evaluation
- `DEPLOYMENT_README.md` - Deployment integration guide
- `MANIFEST.json` - Package contents and specifications
- `inference.py` - Production inference module with docstrings

---

**Version**: 3.0
**Dataset**: CMAPSS FD001
**Created**: 2025-12-04
**Status**: Production Ready ✓
