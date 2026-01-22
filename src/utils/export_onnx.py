"""
Model Export Utilities
Convert PyTorch models to ONNX format for optimized inference.
"""

import torch
import torch.nn as nn
import os
from typing import Tuple


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
    opset_version: int = 11,
    dynamic_axes: dict = None,
    verbose: bool = True
):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_shape: Input tensor shape (B, C, H, W)
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes specification
        verbose: Print export information
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Default dynamic axes for batch size
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=verbose
    )
    
    if verbose:
        print(f"Model exported to: {output_path}")
        print(f"Input shape: {input_shape}")
        
        # Check file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Model size: {file_size:.2f} MB")


def verify_onnx_model(model_path: str, input_shape: Tuple[int, ...]):
    """
    Verify ONNX model can be loaded and run
    
    Args:
        model_path: Path to ONNX model
        input_shape: Input tensor shape
    """
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Load model
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        output = session.run([output_name], {input_name: dummy_input})
        
        print(f"✓ ONNX model verification successful")
        print(f"  Input: {input_name} {dummy_input.shape}")
        print(f"  Output: {output_name} {output[0].shape}")
        
        return True
    
    except Exception as e:
        print(f"✗ ONNX model verification failed: {e}")
        return False


def export_recognition_model(
    model: nn.Module,
    output_path: str,
    input_size: int = 112
):
    """
    Export face recognition backbone to ONNX
    
    Args:
        model: Recognition model (MobileFaceNet, ResNet, etc.)
        output_path: Output ONNX path
        input_size: Face input size (default: 112x112)
    """
    print(f"Exporting recognition model to ONNX...")
    
    export_to_onnx(
        model=model,
        output_path=output_path,
        input_shape=(1, 3, input_size, input_size),
        opset_version=11,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify
    verify_onnx_model(output_path, (1, 3, input_size, input_size))


def export_liveness_model(
    model: nn.Module,
    output_path: str,
    input_size: int = 128
):
    """
    Export liveness detection model to ONNX
    
    Args:
        model: Liveness model
        output_path: Output ONNX path
        input_size: Input size (default: 128x128)
    """
    print(f"Exporting liveness model to ONNX...")
    
    # Set model to inference mode (disable auxiliary heads)
    if hasattr(model, 'use_auxiliary'):
        model.use_auxiliary = False
    
    export_to_onnx(
        model=model,
        output_path=output_path,
        input_shape=(1, 3, input_size, input_size),
        opset_version=11,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify
    verify_onnx_model(output_path, (1, 3, input_size, input_size))


if __name__ == "__main__":
    """Test model export"""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models import get_mobilefacenet, LivenessDetectionNet
    
    print("=" * 80)
    print("Model Export Test")
    print("=" * 80)
    
    # Create output directory
    os.makedirs("../../weights", exist_ok=True)
    
    # Export MobileFaceNet
    print("\n1. Exporting MobileFaceNet...")
    model_recognition = get_mobilefacenet(embedding_size=512)
    export_recognition_model(
        model=model_recognition,
        output_path="../../weights/mobilefacenet.onnx",
        input_size=112
    )
    
    # Export Liveness model
    print("\n2. Exporting LivenessNet...")
    model_liveness = LivenessDetectionNet(input_size=128, num_classes=2, use_auxiliary=False)
    export_liveness_model(
        model=model_liveness,
        output_path="../../weights/liveness_net.onnx",
        input_size=128
    )
    
    print("\n" + "=" * 80)
    print("Export complete!")
    print("=" * 80)
    print("""
    Exported models:
    - weights/mobilefacenet.onnx (Face recognition)
    - weights/liveness_net.onnx (Anti-spoofing)
    
    Usage:
    1. Place pretrained weights in PyTorch format
    2. Export to ONNX using this script
    3. Configure pipeline to use ONNX models
    4. Enjoy 2-3x faster inference on CPU!
    """)
