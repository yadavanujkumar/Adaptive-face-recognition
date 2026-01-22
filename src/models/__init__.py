"""
Face Recognition Models Package

Exports:
- AdaFace loss (quality-adaptive margin)
- Liveness detection network (anti-spoofing)
- Recognition backbones (MobileFaceNet, ResNet)
"""

from .adaface_loss import AdaFace, AdaFaceLoss
from .liveness_net import LivenessDetectionNet, LivenessLoss
from .backbones import MobileFaceNet, ResNet, get_mobilefacenet, get_resnet50, get_resnet18

__all__ = [
    'AdaFace',
    'AdaFaceLoss',
    'LivenessDetectionNet',
    'LivenessLoss',
    'MobileFaceNet',
    'ResNet',
    'get_mobilefacenet',
    'get_resnet50',
    'get_resnet18',
]
