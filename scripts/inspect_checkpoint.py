"""Quick script to inspect checkpoint contents."""
import torch
from pathlib import Path

checkpoint_path = Path('/workspace/checkpoints/best_model.pt')
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("Checkpoint keys:", checkpoint.keys())
print("\nModel config keys:")
for key in checkpoint.keys():
    if key not in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']:
        print(f"  {key}: {checkpoint[key]}")

print("\nModel state_dict keys (first 10):")
for i, key in enumerate(list(checkpoint['model_state_dict'].keys())[:10]):
    print(f"  {key}")

# Count layers
layer_keys = [k for k in checkpoint['model_state_dict'].keys() if k.startswith('layers.')]
max_layer = max([int(k.split('.')[1]) for k in layer_keys if k.split('.')[1].isdigit()])
print(f"\nNumber of layers in checkpoint: {max_layer + 1}")
