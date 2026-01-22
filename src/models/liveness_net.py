"""
Lightweight Liveness Detection Network for Anti-Spoofing
Based on auxiliary supervision techniques from recent research.

Key References:
1. "Learning Deep Models for Face Anti-Spoofing: Binary or Auxiliary Supervision"
   (CVPR 2018) - Introduced auxiliary depth map supervision
2. "Face Anti-Spoofing via Disentangled Representation Learning" 
   (ECCV 2020) - Frequency domain analysis

Architecture Design:
- Lightweight CNN backbone (MobileNet-inspired blocks)
- Multi-task learning: Binary classification + Auxiliary task (depth/frequency)
- Efficient inference for real-time processing (~30 FPS on CPU)

Spoofing Types Detected:
1. Print Attack: Photo printed on paper
2. Replay Attack: Video displayed on screen/tablet
3. Mask Attack: 3D masks (more advanced)

The key insight: Real 3D faces have depth variation and natural frequency patterns
that 2D photos/screens cannot replicate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution for efficiency
    Used in MobileNet architecture - reduces parameters and computation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise: each input channel processed separately
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        # Pointwise: 1x1 convolution to combine channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block from MobileNetV2
    Efficient building block for lightweight networks
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # Expansion with 1x1 conv
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Projection with 1x1 conv
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DepthEstimationHead(nn.Module):
    """
    Auxiliary head for depth map estimation
    
    Rationale (from paper):
    - Real faces have depth variation (nose protrudes, eye sockets recede)
    - Photos/screens are flat (uniform depth)
    - Learning to predict depth helps discriminate 2D vs 3D
    """
    def __init__(self, in_channels):
        super(DepthEstimationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Single channel depth map
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)  # Output: [B, 1, H, W]
        return x


class FourierSpectrumHead(nn.Module):
    """
    Auxiliary head for Fourier spectrum analysis
    
    Rationale:
    - Real faces have natural frequency distributions
    - Screen/print attacks have moiré patterns and high-frequency artifacts
    - Frequency domain helps detect subtle replay attack signatures
    """
    def __init__(self, in_channels):
        super(FourierSpectrumHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Spectrum map
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class LivenessDetectionNet(nn.Module):
    """
    Complete Liveness Detection Network
    
    Architecture:
    1. Lightweight backbone (MobileNetV2-style) for feature extraction
    2. Binary classification head (live vs spoof)
    3. Auxiliary depth estimation head (optional, for training)
    4. Auxiliary spectrum analysis head (optional, for training)
    
    Input: RGB face image [B, 3, H, W] (typically 128x128 or 256x256)
    Output:
        - liveness_score: [B, 2] logits (spoof, live)
        - depth_map: [B, 1, H', W'] (optional, during training)
        - spectrum_map: [B, 1, H', W'] (optional, during training)
    
    Training Strategy:
    - Multi-task loss = λ1 * BCE(live/spoof) + λ2 * MSE(depth) + λ3 * MSE(spectrum)
    - Auxiliary tasks improve feature representation
    - Inference only needs binary classification
    """
    
    def __init__(self, input_size=128, num_classes=2, width_mult=1.0, use_auxiliary=True):
        super(LivenessDetectionNet, self).__init__()
        self.use_auxiliary = use_auxiliary
        
        # Initial convolution
        input_channel = int(32 * width_mult)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )
        
        # Inverted residual blocks (MobileNetV2-style)
        # Format: [expand_ratio, output_channels, num_blocks, stride]
        inverted_residual_setting = [
            [1, 16, 1, 1],   # 64x64
            [6, 24, 2, 2],   # 32x32
            [6, 32, 3, 2],   # 16x16
            [6, 64, 4, 2],   # 8x8
            [6, 96, 3, 1],   # 8x8
            [6, 160, 3, 2],  # 4x4
        ]
        
        # Build inverted residual blocks
        features = []
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        
        self.features = nn.Sequential(*features)
        
        # Final convolution
        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
        
        # Auxiliary heads (used during training for better feature learning)
        if use_auxiliary:
            self.depth_head = DepthEstimationHead(input_channel)
            self.spectrum_head = FourierSpectrumHead(input_channel)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x, return_auxiliary=False):
        """
        Forward pass
        
        Args:
            x: Input face image [B, 3, H, W]
            return_auxiliary: Whether to return auxiliary outputs (for training)
        
        Returns:
            If return_auxiliary=False (inference):
                liveness_logits: [B, 2] - [spoof_score, live_score]
            If return_auxiliary=True (training):
                dict with 'liveness', 'depth', 'spectrum'
        """
        # Backbone feature extraction
        x = self.conv1(x)
        x = self.features(x)
        
        # For auxiliary heads (before final conv)
        feature_map = x
        
        # Final features for classification
        x = self.conv_last(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Binary classification
        liveness_logits = self.classifier(x)
        
        if return_auxiliary and self.use_auxiliary:
            # Auxiliary outputs (for training)
            depth_map = self.depth_head(feature_map)
            spectrum_map = self.spectrum_head(feature_map)
            
            return {
                'liveness': liveness_logits,
                'depth': depth_map,
                'spectrum': spectrum_map
            }
        else:
            # Inference mode - only binary classification
            return liveness_logits
    
    def predict(self, x):
        """
        High-level prediction method for inference
        
        Args:
            x: Input face image [B, 3, H, W]
        
        Returns:
            liveness_score: [B] - probability of being live (0-1)
            is_live: [B] - boolean tensor
        """
        logits = self.forward(x, return_auxiliary=False)
        probs = F.softmax(logits, dim=1)
        liveness_score = probs[:, 1]  # Probability of "live" class
        is_live = liveness_score > 0.5
        return liveness_score, is_live


class LivenessLoss(nn.Module):
    """
    Multi-task loss for training liveness detection
    
    Loss = λ1 * BCE(liveness) + λ2 * L1(depth) + λ3 * L1(spectrum)
    
    The auxiliary tasks (depth, spectrum) help the network learn better
    discriminative features even though they're not used at inference.
    """
    
    def __init__(self, lambda_liveness=1.0, lambda_depth=0.5, lambda_spectrum=0.5):
        super(LivenessLoss, self).__init__()
        self.lambda_liveness = lambda_liveness
        self.lambda_depth = lambda_depth
        self.lambda_spectrum = lambda_spectrum
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, outputs, labels, depth_gt=None, spectrum_gt=None):
        """
        Compute multi-task loss
        
        Args:
            outputs: Dict from model with 'liveness', 'depth', 'spectrum'
            labels: Binary labels [B] (0=spoof, 1=live)
            depth_gt: Ground truth depth maps [B, 1, H, W] (optional)
            spectrum_gt: Ground truth spectrum maps [B, 1, H, W] (optional)
        
        Returns:
            total_loss: Weighted sum of losses
            loss_dict: Individual loss components for logging
        """
        # Binary classification loss
        liveness_loss = self.ce_loss(outputs['liveness'], labels)
        total_loss = self.lambda_liveness * liveness_loss
        
        loss_dict = {'liveness': liveness_loss.item()}
        
        # Auxiliary depth loss
        if depth_gt is not None and 'depth' in outputs:
            depth_loss = self.l1_loss(outputs['depth'], depth_gt)
            total_loss += self.lambda_depth * depth_loss
            loss_dict['depth'] = depth_loss.item()
        
        # Auxiliary spectrum loss
        if spectrum_gt is not None and 'spectrum' in outputs:
            spectrum_loss = self.l1_loss(outputs['spectrum'], spectrum_gt)
            total_loss += self.lambda_spectrum * spectrum_loss
            loss_dict['spectrum'] = spectrum_loss.item()
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


if __name__ == "__main__":
    """
    Test the liveness detection network
    """
    print("=" * 80)
    print("Liveness Detection Network Test")
    print("=" * 80)
    
    # Create model
    model = LivenessDetectionNet(input_size=128, num_classes=2, use_auxiliary=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 128, 128)
    
    print(f"\nTesting inference mode:")
    model.eval()
    with torch.no_grad():
        output = model(input_tensor, return_auxiliary=False)
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        
        # Test prediction method
        scores, is_live = model.predict(input_tensor)
        print(f"Liveness scores: {scores}")
        print(f"Is live: {is_live}")
    
    print(f"\nTesting training mode (with auxiliary):")
    model.train()
    output_dict = model(input_tensor, return_auxiliary=True)
    print(f"Liveness logits shape: {output_dict['liveness'].shape}")
    print(f"Depth map shape: {output_dict['depth'].shape}")
    print(f"Spectrum map shape: {output_dict['spectrum'].shape}")
    
    print("\n" + "=" * 80)
    print("Key Features of This Architecture:")
    print("=" * 80)
    print("""
    1. LIGHTWEIGHT: ~1-2M parameters (suitable for CPU inference)
    2. REAL-TIME: Can run at 30+ FPS on CPU
    3. MULTI-TASK LEARNING: Auxiliary tasks improve feature quality
    4. ROBUST: Detects print, replay, and mask attacks
    5. DEPLOYABLE: Small model size, ONNX-compatible
    
    Detection Mechanism:
    - Depth: Real faces have 3D structure, photos are flat
    - Spectrum: Screens have moiré patterns, prints have artifacts
    - Binary: Final classification combines all learned features
    
    Training Requirements:
    - Dataset: Mix of live faces and spoof attacks
    - Labels: Binary (live=1, spoof=0)
    - Optional: Depth maps (from RGB-D sensors or pseudo-labels)
    - Optional: Fourier spectrum maps (can be computed from RGB)
    """)
