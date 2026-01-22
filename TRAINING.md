# Training Guide for Adaptive Face Recognition

This guide provides professional instructions for training all components of the face recognition system.

## Table of Contents
1. [Dataset Preparation](#dataset-preparation)
2. [Training AdaFace Recognition Model](#training-adaface-recognition-model)
3. [Training Liveness Detection Model](#training-liveness-detection-model)
4. [Model Export to ONNX](#model-export-to-onnx)
5. [Evaluation and Testing](#evaluation-and-testing)

---

## 1. Dataset Preparation

### Face Recognition Dataset

**Recommended Dataset: MS1MV3 (MS-Celeb-1M refined version)**

- **Size**: ~5.8M images, 93K identities
- **Download**: [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
- **Format**: Aligned faces (112x112) with identity labels

**Directory Structure:**
```
data/
├── ms1m-retinaface/
│   ├── train.rec
│   ├── train.idx
│   └── property
```

**Alternative Datasets:**
- CASIA-WebFace: 500K images, 10K identities (smaller, faster training)
- VGGFace2: 3.3M images, 9K identities (high quality)
- Glint360K: 17M images, 360K identities (very large scale)

### Liveness Detection Dataset

**Recommended Datasets:**

1. **CASIA-FASD (Face Anti-Spoofing Database)**
   - Live faces + Print/Replay attacks
   - Download: Available upon request from CASIA

2. **OULU-NPU**
   - Multiple attack types and protocols
   - Public benchmark dataset

3. **CelebA-Spoof**
   - Large-scale (625K images)
   - Rich spoof types
   - Download: [CelebA-Spoof](https://github.com/Davidzhangyuanhan/CelebA-Spoof)

**Directory Structure:**
```
data/
├── liveness/
│   ├── train/
│   │   ├── live/
│   │   └── spoof/
│   └── val/
│       ├── live/
│       └── spoof/
```

---

## 2. Training AdaFace Recognition Model

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Training Script

Create `train_recognition.py`:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models import get_mobilefacenet, AdaFaceLoss
import sys

# Configuration
config = {
    'backbone': 'mobilefacenet',  # or 'resnet50'
    'embedding_size': 512,
    'num_classes': 93431,  # MS1MV3 identities
    'batch_size': 512,
    'lr': 0.1,
    'epochs': 30,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    # AdaFace parameters
    'adaface_m': 0.4,
    'adaface_h': 0.333,
    'adaface_s': 64.0,
}

def main():
    # Initialize model
    if config['backbone'] == 'mobilefacenet':
        from src.models import get_mobilefacenet
        backbone = get_mobilefacenet(embedding_size=config['embedding_size'])
    else:
        from src.models import get_resnet50
        backbone = get_resnet50(embedding_size=config['embedding_size'])
    
    # Initialize AdaFace loss
    criterion = AdaFaceLoss(
        embedding_size=config['embedding_size'],
        num_classes=config['num_classes'],
        m=config['adaface_m'],
        h=config['adaface_h'],
        s=config['adaface_s']
    )
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = backbone.to(device)
    criterion = criterion.to(device)
    
    # Optimizer (separate for backbone and head)
    params = [
        {'params': backbone.parameters()},
        {'params': criterion.parameters()},
    ]
    optimizer = torch.optim.SGD(
        params,
        lr=config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[10, 18, 22], 
        gamma=0.1
    )
    
    # Load dataset (use MXNet RecordIO or standard PyTorch dataset)
    # Example placeholder:
    # train_loader = get_dataloader(config)
    
    # Training loop
    for epoch in range(config['epochs']):
        backbone.train()
        criterion.train()
        
        # for batch_idx, (images, labels) in enumerate(train_loader):
        #     images, labels = images.to(device), labels.to(device)
        #     
        #     # Forward pass
        #     embeddings = backbone(images)
        #     loss = criterion(embeddings, labels)
        #     
        #     # Backward pass
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': backbone.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoints/adaface_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(backbone.state_dict(), 'weights/mobilefacenet_adaface.pth')

if __name__ == '__main__':
    main()
```

### Step 3: Training Commands

```bash
# Single GPU
python train_recognition.py

# Multi-GPU (distributed)
python -m torch.distributed.launch --nproc_per_node=4 train_recognition.py
```

### Step 4: Hyperparameters

**Key AdaFace Parameters:**
- `m` (margin): 0.4 (base margin for hard samples)
- `h` (scale): 0.333 (adaptive margin scaling)
- `s` (feature scale): 64.0 (logits scaling)

**Training Schedule:**
- Batch size: 512 (adjust based on GPU memory)
- Initial LR: 0.1
- LR decay: 0.1x at epochs [10, 18, 22]
- Total epochs: 24-30
- Optimizer: SGD with momentum 0.9

**Expected Results (MS1MV3):**
- LFW: 99.7%+
- CFP-FP: 98.5%+
- AgeDB-30: 98.0%+

---

## 3. Training Liveness Detection Model

### Step 1: Prepare Training Script

Create `train_liveness.py`:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models import LivenessDetectionNet, LivenessLoss

# Configuration
config = {
    'input_size': 128,
    'batch_size': 64,
    'lr': 0.001,
    'epochs': 50,
    'use_auxiliary': True,
    'lambda_liveness': 1.0,
    'lambda_depth': 0.5,
    'lambda_spectrum': 0.5,
}

def main():
    # Initialize model
    model = LivenessDetectionNet(
        input_size=config['input_size'],
        num_classes=2,
        use_auxiliary=config['use_auxiliary']
    )
    
    # Loss function
    criterion = LivenessLoss(
        lambda_liveness=config['lambda_liveness'],
        lambda_depth=config['lambda_depth'],
        lambda_spectrum=config['lambda_spectrum']
    )
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Load datasets
    # train_loader = get_liveness_dataloader('train', config)
    # val_loader = get_liveness_dataloader('val', config)
    
    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        
        # for images, labels, depth_maps, spectrum_maps in train_loader:
        #     images = images.to(device)
        #     labels = labels.to(device)
        #     
        #     # Forward pass
        #     outputs = model(images, return_auxiliary=True)
        #     loss, loss_dict = criterion(outputs, labels, depth_maps, spectrum_maps)
        #     
        #     # Backward pass
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/liveness_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), 'weights/liveness_net.pth')

if __name__ == '__main__':
    main()
```

### Step 2: Training Commands

```bash
python train_liveness.py
```

### Step 3: Hyperparameters

**Model Parameters:**
- Input size: 128x128 or 256x256
- Architecture: MobileNetV2-based (lightweight)
- Use auxiliary heads during training only

**Training Schedule:**
- Batch size: 64
- Initial LR: 0.001
- LR decay: 0.1x every 20 epochs
- Total epochs: 50-100
- Optimizer: Adam

**Loss Weights:**
- Binary classification: 1.0
- Depth auxiliary: 0.5
- Spectrum auxiliary: 0.5

**Expected Results:**
- HTER (Half Total Error Rate): < 5%
- EER (Equal Error Rate): < 3%
- APCER @ BPCER=1%: < 2%

---

## 4. Model Export to ONNX

### Export Recognition Model

```python
from src.models import get_mobilefacenet
from src.utils.export_onnx import export_recognition_model
import torch

# Load trained model
model = get_mobilefacenet(embedding_size=512)
model.load_state_dict(torch.load('weights/mobilefacenet_adaface.pth'))
model.eval()

# Export to ONNX
export_recognition_model(
    model=model,
    output_path='weights/mobilefacenet.onnx',
    input_size=112
)
```

### Export Liveness Model

```python
from src.models import LivenessDetectionNet
from src.utils.export_onnx import export_liveness_model
import torch

# Load trained model
model = LivenessDetectionNet(input_size=128, num_classes=2, use_auxiliary=False)
model.load_state_dict(torch.load('weights/liveness_net.pth'))
model.eval()

# Export to ONNX
export_liveness_model(
    model=model,
    output_path='weights/liveness_net.onnx',
    input_size=128
)
```

---

## 5. Evaluation and Testing

### Face Recognition Evaluation

Standard benchmarks:
- **LFW** (Labeled Faces in the Wild)
- **CFP-FP** (Celebrities in Frontal-Profile)
- **AgeDB-30** (Age variation)
- **IJB-B/C** (Challenging scenarios)

### Liveness Detection Evaluation

Metrics:
- **APCER**: Attack Presentation Classification Error Rate
- **BPCER**: Bona Fide Presentation Classification Error Rate
- **ACER**: Average Classification Error Rate
- **EER**: Equal Error Rate

---

## Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
# MS1MV3: Follow InsightFace instructions
# Liveness: Download CASIA-FASD or CelebA-Spoof

# 3. Train recognition model
python train_recognition.py

# 4. Train liveness model
python train_liveness.py

# 5. Export to ONNX
python src/utils/export_onnx.py

# 6. Run inference pipeline
python src/pipeline/pipeline.py
```

---

## Pre-trained Models

For quick deployment, consider using pre-trained models:

1. **InsightFace Model Zoo**: Pre-trained face recognition models
2. **FaceX-Zoo**: Collection of face recognition models
3. **Silent-Face-Anti-Spoofing**: Pre-trained liveness models

---

## Notes

- **GPU Requirements**: Training recognition models requires at least 1x RTX 3090 or V100
- **Training Time**: Recognition ~2-3 days on 4x V100; Liveness ~1 day on 1x V100
- **Mixed Precision**: Use `torch.cuda.amp` for faster training
- **Distributed Training**: Use DDP for multi-GPU training

---

For more details, refer to:
- [AdaFace Paper](https://arxiv.org/abs/2204.00964)
- [Face Anti-Spoofing Survey](https://arxiv.org/abs/2101.04558)
