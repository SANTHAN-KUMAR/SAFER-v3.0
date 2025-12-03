#!/usr/bin/env python
"""
Load and verify trained Mamba RUL model.
This script loads the best model checkpoint and shows model info.
"""

import torch
import json
from pathlib import Path
from safer_v3.core.mamba import MambaRULPredictor

# Configuration
EXPERIMENT_DIR = "outputs/mamba_FD001_20251203_174328"  # CHANGE THIS to your experiment folder
MODEL_PATH = Path(EXPERIMENT_DIR) / "best_model.pt"
ARGS_PATH = Path(EXPERIMENT_DIR) / "args.json"

print(f"\n{'='*80}")
print(f"Loading Trained Mamba RUL Model")
print(f"{'='*80}\n")

# Check if paths exist
if not EXPERIMENT_DIR or not Path(EXPERIMENT_DIR).exists():
    print(f"❌ Experiment directory not found: {EXPERIMENT_DIR}")
    print("\nAvailable experiments:")
    for exp_dir in sorted(Path("outputs").glob("mamba_*")):
        print(f"  - {exp_dir.name}")
    exit(1)

print(f"Experiment directory: {EXPERIMENT_DIR}")
print(f"  Exists: {Path(EXPERIMENT_DIR).exists()}")
print(f"  Contents: {list(Path(EXPERIMENT_DIR).glob('*'))}\n")

# Load args
if ARGS_PATH.exists():
    with open(ARGS_PATH, 'r') as f:
        args = json.load(f)
    print(f"✓ Loaded training configuration from {ARGS_PATH}")
    print(f"\nTraining Configuration:")
    for key, value in args.items():
        if key not in ['checkpoint_dir', 'data_dir']:  # Skip long paths
            print(f"  {key}: {value}")
else:
    print(f"⚠️ args.json not found at {ARGS_PATH}")
    args = {
        'd_input': 14,
        'd_model': 128,
        'd_state': 16,
        'n_layers': 6,
        'expand': 2,
        'dropout': 0.1,
        'max_rul': 125,
    }

# Create model with saved configuration
model_config = {
    'd_input': args.get('d_input', 14),
    'd_model': args.get('d_model', 128),
    'd_state': args.get('d_state', 16),
    'n_layers': args.get('n_layers', 6),
    'expand': args.get('expand', 2),
    'dropout': args.get('dropout', 0.1),
    'max_rul': args.get('max_rul', 125),
}

print(f"\n{'='*80}")
print(f"Creating Model with Configuration:")
print(f"{'='*80}")
for key, value in model_config.items():
    print(f"  {key}: {value}")

model = MambaRULPredictor(**model_config)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel Statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# Load checkpoint
if MODEL_PATH.exists():
    print(f"\n{'='*80}")
    print(f"Loading Checkpoint: {MODEL_PATH}")
    print(f"{'='*80}")
    
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model weights loaded successfully")
        
        if 'epoch' in checkpoint:
            print(f"  Trained for {checkpoint['epoch']} epochs")
        if 'best_val_rmse' in checkpoint:
            print(f"  Best validation RMSE: {checkpoint['best_val_rmse']:.4f}")
    else:
        print(f"❌ 'model_state_dict' not found in checkpoint")
        print(f"   Available keys: {list(checkpoint.keys())}")
        exit(1)
else:
    print(f"⚠️ Model checkpoint not found at {MODEL_PATH}")
    print(f"\nAvailable files in {EXPERIMENT_DIR}:")
    for file in Path(EXPERIMENT_DIR).glob('*'):
        print(f"  - {file.name}")

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

print(f"\n{'='*80}")
print(f"✓ Model loaded successfully!")
print(f"  Device: {device}")
print(f"  Mode: Evaluation")
print(f"{'='*80}\n")

# Test inference with dummy data
print(f"Testing inference with dummy data...")
with torch.no_grad():
    dummy_input = torch.randn(1, 50, 14).to(device)  # batch=1, seq_len=50, n_sensors=14
    output = model(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Predicted RUL: {output.item():.2f} cycles")

print(f"\n{'='*80}")
print(f"✓ Model is ready for use!")
print(f"{'='*80}\n")

# Save as a clean checkpoint for deployment
deployment_dir = Path(EXPERIMENT_DIR) / "deployment"
deployment_dir.mkdir(exist_ok=True)

torch.save(model.state_dict(), deployment_dir / "model_weights.pt")
print(f"✓ Model weights saved to: {deployment_dir / 'model_weights.pt'}")

# Save model configuration for easy reloading
with open(deployment_dir / "model_config.json", 'w') as f:
    json.dump(model_config, f, indent=2)
print(f"✓ Model configuration saved to: {deployment_dir / 'model_config.json'}")

print(f"\nTo use the model in the future:")
print(f"  1. Copy the deployment folder: {deployment_dir}")
print(f"  2. Load weights: model.load_state_dict(torch.load('model_weights.pt'))")
print(f"  3. Load config: config = json.load(open('model_config.json'))")
print(f"\n")
