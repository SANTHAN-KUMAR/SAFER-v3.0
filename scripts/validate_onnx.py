#!/usr/bin/env python3
"""
Simple ONNX model validation script.
Validates exported ONNX models can be loaded and run inference.
"""

import sys
import json
import numpy as np
from pathlib import Path

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("⚠ onnx/onnxruntime not installed")
    print("  Install with: pip install onnx onnxruntime")
    sys.exit(1)


def validate_onnx_model(model_path: str, metadata_path: str = None):
    """Validate ONNX model.
    
    Args:
        model_path: Path to .onnx file
        metadata_path: Path to .json metadata file (optional)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        return False
    
    print(f"\nValidating: {model_path}")
    print("=" * 60)
    
    # Load and check ONNX model
    try:
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        print("✓ ONNX model structure is valid")
    except Exception as e:
        print(f"✗ ONNX validation failed: {e}")
        return False
    
    # Load metadata if available
    if metadata_path is None:
        metadata_path = model_path.with_suffix('.json')
    
    if Path(metadata_path).exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"\nModel metadata:")
        print(f"  Input shape: {metadata.get('input_shape')}")
        print(f"  Output shape: {metadata.get('output_shape')}")
        print(f"  Opset version: {metadata.get('opset_version')}")
    
    # Create inference session
    try:
        session = ort.InferenceSession(str(model_path))
        print(f"\n✓ ONNX Runtime session created")
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_name = session.get_outputs()[0].name
        
        print(f"  Input: {input_name}, shape: {input_shape}")
        print(f"  Output: {output_name}")
        
    except Exception as e:
        print(f"✗ Failed to create inference session: {e}")
        return False
    
    # Test inference with dummy data
    try:
        # Replace dynamic dimensions with concrete values
        concrete_shape = []
        for dim in input_shape:
            if isinstance(dim, str) or dim is None:
                concrete_shape.append(1)
            else:
                concrete_shape.append(dim)
        
        dummy_input = np.random.randn(*concrete_shape).astype(np.float32)
        outputs = session.run([output_name], {input_name: dummy_input})
        
        print(f"\n✓ Test inference successful")
        print(f"  Input: {dummy_input.shape}")
        print(f"  Output: {outputs[0].shape}")
        print(f"  Sample prediction: {outputs[0][0]}")
        
    except Exception as e:
        print(f"✗ Test inference failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ ONNX model validation PASSED")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate ONNX model")
    parser.add_argument("model", type=str, help="Path to ONNX model file")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Path to metadata JSON file")
    
    args = parser.parse_args()
    
    success = validate_onnx_model(args.model, args.metadata)
    sys.exit(0 if success else 1)
