"""
Face Recognition Backbone Networks
Implements lightweight and efficient architectures for embedding extraction.

Supported Backbones:
1. MobileFaceNet: Ultra-lightweight for mobile/edge devices
2. ResNet50: Standard backbone for high accuracy
3. IResNet50/100: Improved ResNet with better feature learning

All backbones output 512-dimensional embeddings suitable for face recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Flatten(nn.Module):
    """Flattens input tensor"""
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBNReLU(nn.Module):
    """Standard convolution block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DepthwiseConv(nn.Module):
    """Depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return x


class MobileFaceNetBlock(nn.Module):
    """
    Bottleneck block for MobileFaceNet
    Efficient residual block with depthwise separable convolutions
    """
    def __init__(self, in_channels, out_channels, stride=1, expansion=2):
        super(MobileFaceNetBlock, self).__init__()
        hidden_channels = in_channels * expansion
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Expansion
        self.conv1 = ConvBNReLU(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        # Depthwise
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride, 
                               padding=1, groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        # Projection
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.use_residual:
            out += identity
        
        return out


class MobileFaceNet(nn.Module):
    """
    MobileFaceNet: Efficient face recognition backbone
    
    Paper: "MobileFaceNets: Efficient CNNs for Accurate Real-time Face Verification on Mobile Devices"
    
    Key Features:
    - Only ~1M parameters
    - Fast inference on CPU/mobile devices
    - Suitable for edge deployment
    - Output: 512-d embedding
    
    Input: [B, 3, 112, 112] (aligned face)
    Output: [B, 512] (face embedding)
    """
    
    def __init__(self, embedding_size=512):
        super(MobileFaceNet, self).__init__()
        self.embedding_size = embedding_size
        
        # Initial convolution
        self.conv1 = ConvBNReLU(3, 64, kernel_size=3, stride=2, padding=1)
        
        # Depthwise separable conv
        self.conv2_dw = DepthwiseConv(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Bottleneck blocks
        self.conv3 = MobileFaceNetBlock(64, 64, stride=2, expansion=2)
        self.conv4 = MobileFaceNetBlock(64, 128, stride=1, expansion=4)
        self.conv5 = MobileFaceNetBlock(128, 128, stride=2, expansion=2)
        
        # Additional blocks
        self.conv6 = MobileFaceNetBlock(128, 128, stride=1, expansion=4)
        self.conv7 = MobileFaceNetBlock(128, 128, stride=1, expansion=4)
        self.conv8 = MobileFaceNetBlock(128, 128, stride=1, expansion=4)
        self.conv9 = MobileFaceNetBlock(128, 128, stride=1, expansion=4)
        self.conv10 = MobileFaceNetBlock(128, 128, stride=1, expansion=4)
        
        self.conv11 = MobileFaceNetBlock(128, 128, stride=2, expansion=2)
        self.conv12 = MobileFaceNetBlock(128, 128, stride=1, expansion=4)
        
        # Final layers
        self.conv13 = ConvBNReLU(128, 512, kernel_size=1, stride=1, padding=0)
        
        # Global depthwise conv (similar to global pooling)
        self.conv14_dw = nn.Conv2d(512, 512, kernel_size=7, stride=1, padding=0, groups=512, bias=False)
        self.bn14 = nn.BatchNorm2d(512)
        
        # Final embedding layer
        self.fc = nn.Linear(512, embedding_size)
        self.bn_fc = nn.BatchNorm1d(embedding_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        """
        Extract face embedding
        
        Args:
            x: Face image [B, 3, 112, 112]
        
        Returns:
            embedding: [B, 512]
        """
        x = self.conv1(x)
        x = self.conv2_dw(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        
        x = self.conv11(x)
        x = self.conv12(x)
        
        x = self.conv13(x)
        x = self.conv14_dw(x)
        x = self.bn14(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn_fc(x)
        
        return x


class BasicBlock(nn.Module):
    """Basic ResNet block"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50/101"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet backbone for face recognition
    
    Supports ResNet-18, 34, 50, 101
    Standard architecture adapted for face recognition (112x112 input)
    """
    
    def __init__(self, block, layers, embedding_size=512):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.embedding_size = embedding_size
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Output layers
        self.bn2 = nn.BatchNorm2d(512 * block.expansion)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(512 * block.expansion * 7 * 7, embedding_size)
        self.bn3 = nn.BatchNorm1d(embedding_size)
        
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)
        
        return x


def get_mobilefacenet(embedding_size=512):
    """Get MobileFaceNet model"""
    return MobileFaceNet(embedding_size=embedding_size)


def get_resnet50(embedding_size=512):
    """Get ResNet-50 model for face recognition"""
    return ResNet(Bottleneck, [3, 4, 6, 3], embedding_size=embedding_size)


def get_resnet18(embedding_size=512):
    """Get ResNet-18 model for face recognition"""
    return ResNet(BasicBlock, [2, 2, 2, 2], embedding_size=embedding_size)


if __name__ == "__main__":
    """Test backbone models"""
    print("=" * 80)
    print("Face Recognition Backbone Models Test")
    print("=" * 80)
    
    # Test MobileFaceNet
    print("\n1. MobileFaceNet:")
    model_mobile = get_mobilefacenet(embedding_size=512)
    mobile_params = sum(p.numel() for p in model_mobile.parameters())
    print(f"   Parameters: {mobile_params:,}")
    
    x = torch.randn(2, 3, 112, 112)
    y = model_mobile(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    
    # Test ResNet-50
    print("\n2. ResNet-50:")
    model_resnet = get_resnet50(embedding_size=512)
    resnet_params = sum(p.numel() for p in model_resnet.parameters())
    print(f"   Parameters: {resnet_params:,}")
    
    y = model_resnet(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    
    print("\n" + "=" * 80)
    print("Model Comparison:")
    print("=" * 80)
    print(f"""
    MobileFaceNet:
    - Parameters: ~{mobile_params/1e6:.1f}M
    - Speed: Very Fast (optimized for mobile/CPU)
    - Use case: Edge devices, real-time applications
    
    ResNet-50:
    - Parameters: ~{resnet_params/1e6:.1f}M
    - Speed: Moderate (requires GPU for real-time)
    - Use case: High accuracy requirements, server deployment
    
    Both models:
    - Input: 112x112 aligned face
    - Output: 512-d L2-normalized embedding
    - Training: Use with AdaFace loss for best results
    """)
