#!/usr/bin/env python
"""
Test trained Mamba RUL model on test dataset.
Evaluates the model on test data and computes metrics.
"""

import torch
import json
from pathlib import Path
import numpy as np
from safer_v3.core.mamba import MambaRULPredictor
from safer_v3.core.trainer import DataModule
from safer_v3.utils.metrics import calculate_rul_metrics

print(f"\n{'='*80}")
print(f"Testing Trained Mamba RUL Model on Test Data")
print(f"{'='*80}\n")

# Configuration
CHECKPOINT_PATH = "checkpoints/best_model.pt"
DATA_DIR = "CMAPSSData"
DATASET = "FD001"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Dataset: {DATASET}")
print(f"Data directory: {DATA_DIR}\n")

# ============================================================================
# 1. Load Model
# ============================================================================
print(f"{'='*80}")
print(f"Loading Model")
print(f"{'='*80}\n")

if not Path(CHECKPOINT_PATH).exists():
    print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
    exit(1)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
print(f"Checkpoint keys: {list(checkpoint.keys())}")

# Extract config from checkpoint
config_dict = checkpoint.get('config', {})
print(f"\nTraining configuration:")
for key, value in config_dict.items():
    if key not in ['data_dir', 'checkpoint_dir']:
        print(f"  {key}: {value}")

# Create model
model = MambaRULPredictor(
    d_input=14,
    d_model=128,
    d_state=16,
    n_layers=6,
    expand=2,
    dropout=0.1,
    max_rul=125
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

print(f"\n✓ Model loaded successfully")
print(f"  Best training RMSE: {checkpoint.get('best_val_rmse', 'N/A'):.4f}")

# ============================================================================
# 2. Load Test Data
# ============================================================================
print(f"\n{'='*80}")
print(f"Loading Test Data")
print(f"{'='*80}\n")

data_module = DataModule(
    data_dir=DATA_DIR,
    dataset=DATASET,
    window_size=50,
    stride=1,
    max_rul=125,
    batch_size=128,
    val_split=0.2,
    num_workers=4,
    pin_memory=True,
    seed=42,
)

data_module.setup()
test_loader = data_module.test_dataloader()

print(f"✓ Test data loaded")
print(f"  Test samples: {len(data_module.test_dataset)}")
print(f"  Batch size: 128")

# ============================================================================
# 3. Run Inference
# ============================================================================
print(f"\n{'='*80}")
print(f"Running Inference on Test Set")
print(f"{'='*80}\n")

all_preds = []
all_targets = []

with torch.no_grad():
    for batch_idx, (sequences, targets) in enumerate(test_loader):
        sequences = sequences.to(DEVICE)
        predictions = model(sequences)
        
        all_preds.append(predictions.cpu().numpy())
        all_targets.append(targets.numpy())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {(batch_idx + 1) * 128} samples...")

all_preds = np.concatenate(all_preds).ravel()
all_targets = np.concatenate(all_targets).ravel()

print(f"\n✓ Inference complete")
print(f"  Total predictions: {len(all_preds)}")
print(f"  Total targets: {len(all_targets)}")

# ============================================================================
# 4. Compute Metrics
# ============================================================================
print(f"\n{'='*80}")
print(f"Computing Metrics")
print(f"{'='*80}\n")

test_metrics = calculate_rul_metrics(all_targets, all_preds)

print(test_metrics)

# ============================================================================
# 5. Save Results
# ============================================================================
print(f"\n{'='*80}")
print(f"Saving Results")
print(f"{'='*80}\n")

results_dir = Path("outputs/test_results")
results_dir.mkdir(parents=True, exist_ok=True)

# Save metrics
metrics_dict = test_metrics.to_dict()
with open(results_dir / "test_metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)

print(f"✓ Test metrics saved to: {results_dir / 'test_metrics.json'}")

# Save predictions
np.save(results_dir / "predictions.npy", all_preds)
np.save(results_dir / "targets.npy", all_targets)

print(f"✓ Predictions saved to: {results_dir / 'predictions.npy'}")
print(f"✓ Targets saved to: {results_dir / 'targets.npy'}")

# ============================================================================
# 6. Analysis
# ============================================================================
print(f"\n{'='*80}")
print(f"Error Analysis")
print(f"{'='*80}\n")

errors = all_preds - all_targets
abs_errors = np.abs(errors)

print(f"Prediction Statistics:")
print(f"  Mean prediction: {np.mean(all_preds):.2f} cycles")
print(f"  Std prediction: {np.std(all_preds):.2f} cycles")
print(f"  Min prediction: {np.min(all_preds):.2f} cycles")
print(f"  Max prediction: {np.max(all_preds):.2f} cycles")

print(f"\nError Statistics:")
print(f"  Mean error: {np.mean(errors):.2f} cycles")
print(f"  Std error: {np.std(errors):.2f} cycles")
print(f"  Mean abs error: {np.mean(abs_errors):.2f} cycles")
print(f"  Min error: {np.min(errors):.2f} cycles")
print(f"  Max error: {np.max(errors):.2f} cycles")

print(f"\nEarly/Late Predictions:")
early = np.sum(errors < 0)
late = np.sum(errors > 0)
print(f"  Early predictions: {early} ({100*early/len(errors):.1f}%)")
print(f"  Late predictions: {late} ({100*late/len(errors):.1f}%)")

# ============================================================================
# 7. Visualize (optional)
# ============================================================================
print(f"\n{'='*80}")
print(f"Creating Visualizations")
print(f"{'='*80}\n")

try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Scatter plot: predictions vs targets
    axes[0, 0].scatter(all_targets, all_preds, alpha=0.5)
    axes[0, 0].plot([all_targets.min(), all_targets.max()], 
                     [all_targets.min(), all_targets.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True RUL (cycles)')
    axes[0, 0].set_ylabel('Predicted RUL (cycles)')
    axes[0, 0].set_title('Predictions vs Targets')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error distribution
    axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(np.mean(errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    axes[0, 1].set_xlabel('Prediction Error (cycles)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Absolute error over samples
    axes[1, 0].plot(abs_errors, alpha=0.7)
    axes[1, 0].axhline(np.mean(abs_errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(abs_errors):.2f}')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Absolute Error (cycles)')
    axes[1, 0].set_title('Absolute Error by Sample')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot (error normality)
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Error Normality)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "test_analysis.png", dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {results_dir / 'test_analysis.png'}")
    
except Exception as e:
    print(f"⚠️ Could not create visualization: {e}")

print(f"\n{'='*80}")
print(f"✓ Testing Complete!")
print(f"{'='*80}\n")

print(f"Results saved to: {results_dir}/")
print(f"  - test_metrics.json")
print(f"  - predictions.npy")
print(f"  - targets.npy")
print(f"  - test_analysis.png")
