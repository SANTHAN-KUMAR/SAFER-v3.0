#!/usr/bin/env python3
"""
Export Mamba Model to ONNX Format.

This script exports the trained Mamba RUL predictor to ONNX format
for production deployment in environments without PyTorch.
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from safer_v3.core.mamba import MambaRULPredictor


def export_to_onnx(
    checkpoint_path: str,
    output_path: str = None,
    opset_version: int = 14,
    validate: bool = True,
):
    """Export Mamba model to ONNX format.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Output ONNX file path (auto-generated if None)
        opset_version: ONNX opset version
        validate: Whether to validate the exported model
    """
    checkpoint_path = Path(checkpoint_path)
    
    if output_path is None:
        output_path = checkpoint_path.parent / "mamba_model.onnx"
    output_path = Path(output_path)
    
    print(f"\n{'='*60}")
    print("ONNX Export for SAFER v3.0 Mamba Model")
    print(f"{'='*60}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Try loading from args.json
        args_path = checkpoint_path.parent / "args.json"
        if args_path.exists():
            with open(args_path) as f:
                args = json.load(f)
                config = {
                    'd_input': 14,
                    'd_model': args.get('d_model', 128),
                    'd_state': args.get('d_state', 16),
                    'n_layers': args.get('n_layers', 6),
                    'expand': args.get('expand', 2),
                    'dropout': args.get('dropout', 0.1),
                    'max_rul': args.get('max_rul', 125),
                }
        else:
            raise ValueError("Cannot find model configuration")
    
    print(f"\nModel configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Ensure required model config keys with defaults
    model_config = {
        'd_input': config.get('d_input', 14),
        'd_model': config.get('d_model', 128),
        'd_state': config.get('d_state', 16),
        'n_layers': config.get('n_layers', 6),
        'expand': config.get('expand', 2),
        'dropout': config.get('dropout', 0.1),
        'max_rul': config.get('max_rul', 125),
    }
    
    # Create model
    model = MambaRULPredictor(
        d_input=model_config['d_input'],
        d_model=model_config['d_model'],
        d_state=model_config['d_state'],
        n_layers=model_config['n_layers'],
        expand=model_config['expand'],
        dropout=model_config['dropout'],
        max_rul=model_config['max_rul'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    batch_size = 1
    seq_length = 30
    n_sensors = 14
    dummy_input = torch.randn(batch_size, seq_length, n_sensors)
    
    print(f"\nDummy input shape: {dummy_input.shape}")
    
    # Test forward pass
    with torch.no_grad():
        dummy_output = model(dummy_input)
    print(f"Dummy output shape: {dummy_output.shape}")
    
    # Export to ONNX
    print(f"\nExporting to ONNX (opset {opset_version})...")
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['sensor_sequence'],
        output_names=['rul_prediction'],
        dynamic_axes={
            'sensor_sequence': {0: 'batch_size', 1: 'sequence_length'},
            'rul_prediction': {0: 'batch_size'},
        },
    )
    
    print(f"✓ Exported to: {output_path}")
    
    # Validate if requested
    if validate:
        print("\nValidating ONNX model...")
        try:
            import onnx
            import onnxruntime as ort
            
            # Load and check ONNX model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model structure validated")
            
            # Test with ONNX Runtime
            ort_session = ort.InferenceSession(str(output_path))
            
            # Run inference
            ort_input = {
                'sensor_sequence': dummy_input.numpy()
            }
            ort_output = ort_session.run(None, ort_input)
            
            # Compare outputs
            pytorch_output = dummy_output.numpy()
            onnx_output = ort_output[0]
            
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            print(f"✓ ONNX Runtime inference successful")
            print(f"  Max difference from PyTorch: {max_diff:.6f}")
            
            if max_diff < 1e-4:
                print("✓ Output matches PyTorch (within tolerance)")
            else:
                print("⚠ Output differs from PyTorch (may be due to numerical precision)")
            
            # Print model info
            print(f"\nONNX Model Info:")
            print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            print(f"  Inputs: {[inp.name for inp in onnx_model.graph.input]}")
            print(f"  Outputs: {[out.name for out in onnx_model.graph.output]}")
            
        except ImportError:
            print("⚠ onnx/onnxruntime not installed, skipping validation")
    
    # Save export metadata
    metadata = {
        'export_date': datetime.now().isoformat(),
        'source_checkpoint': str(checkpoint_path),
        'onnx_file': str(output_path),
        'opset_version': opset_version,
        'config': config,
        'input_shape': list(dummy_input.shape),
        'output_shape': list(dummy_output.shape),
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Export metadata saved to: {metadata_path}")
    
    print(f"\n{'='*60}")
    print("ONNX Export Complete!")
    print(f"{'='*60}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export Mamba Model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=14,
                        help="ONNX opset version")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validation")
    
    args = parser.parse_args()
    
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        validate=not args.no_validate,
    )
