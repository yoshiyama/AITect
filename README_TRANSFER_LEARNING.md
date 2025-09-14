# Transfer Learning System for Lightweight Object Detection

This system provides a comprehensive pipeline for pretraining object detection models on multiple classes and fine-tuning them for specific single-class detection tasks.

## Overview

The transfer learning pipeline consists of two main stages:

1. **Pretraining**: Train on multiple COCO classes to learn robust feature representations
2. **Fine-tuning**: Adapt the pretrained model for specific single-class detection tasks

## System Components

### 1. Pretraining Script (`pretrain_multi_class.py`)

Trains the model on multiple COCO classes to create a robust pretrained backbone.

**Key Features:**
- Trains on 20-30 common COCO classes
- Saves backbone weights separately for transfer learning
- Implements different learning rates for backbone, FPN, and detection heads
- Uses cosine annealing with warm restarts for learning rate scheduling
- Supports data augmentation and focal loss

**Usage:**
```bash
python pretrain_multi_class.py \
    --coco-root /path/to/coco/dataset \
    --epochs 100 \
    --batch-size 32 \
    --model-size tiny
```

### 2. Fine-tuning Script (`finetune_single_class.py`)

Fine-tunes the pretrained model for single-class detection using a multi-stage approach.

**Key Features:**
- Three-stage fine-tuning strategy:
  - Stage 1: Freeze backbone, train detection heads only
  - Stage 2: Unfreeze FPN layers, reduce learning rates
  - Stage 3: Unfreeze all layers with very low learning rates
- Differential learning rates for different layer groups
- Gradual unfreezing to prevent catastrophic forgetting
- Comprehensive evaluation metrics (mAP, precision, recall, F1)

**Usage:**
```bash
python finetune_single_class.py \
    --pretrained-backbone path/to/backbone_weights.pth \
    --class-name whiteline \
    --train-root datasets/inaoka/train/JPEGImages \
    --train-ann datasets/inaoka/train/annotations.json \
    --val-root datasets/inaoka/val \
    --val-ann datasets/inaoka/val/annotations.json \
    --batch-size 16
```

### 3. Enhanced Model (`model_lightweight.py`)

The lightweight model has been enhanced with transfer learning capabilities:

**New Methods:**
- `load_backbone_weights()`: Load only backbone weights from checkpoint
- `get_param_groups()`: Get parameter groups with different learning rates
- `freeze_backbone()`: Freeze/unfreeze backbone parameters
- `freeze_fpn()`: Freeze/unfreeze FPN parameters  
- `replace_detection_heads()`: Replace heads for different number of classes

**Example:**
```python
# Create model with pretrained backbone
model = create_lightweight_model(
    num_classes=1,
    model_size='tiny',
    backbone_weights_path='path/to/pretrained_backbone.pth'
)

# Setup optimizer with different learning rates
param_groups = model.get_param_groups(
    backbone_lr=1e-5,
    fpn_lr=1e-4,
    head_lr=1e-3
)
optimizer = torch.optim.AdamW(param_groups)

# Freeze backbone for initial training
model.freeze_backbone(True)
```

### 4. Configuration File (`config_transfer_learning.json`)

Comprehensive configuration defining best practices for transfer learning:

- Pretraining settings and strategies
- Multi-stage fine-tuning configurations
- Layer group definitions
- Optimization tricks and tips
- Best practices and recommendations

## Transfer Learning Pipeline

### Step 1: Prepare Data

For pretraining, you need COCO dataset:
```bash
datasets/
├── coco/
│   ├── train2017/
│   ├── val2017/
│   └── annotations/
│       ├── instances_train2017.json
│       └── instances_val2017.json
```

For fine-tuning, prepare your single-class dataset in COCO format.

### Step 2: Pretrain on Multiple Classes

```bash
# Pretrain on 30 common COCO classes
python pretrain_multi_class.py \
    --coco-root datasets/coco \
    --epochs 100 \
    --batch-size 32 \
    --model-size tiny
```

This will save:
- Full model checkpoints: `pretrain_results/multi_class_*/checkpoints/`
- Backbone weights only: `pretrain_results/multi_class_*/backbone_weights/`

### Step 3: Fine-tune for Single Class

```bash
# Fine-tune for whiteline detection
python finetune_single_class.py \
    --pretrained-backbone pretrain_results/multi_class_*/backbone_weights/best_backbone.pth \
    --class-name whiteline \
    --train-root datasets/inaoka/train/JPEGImages \
    --train-ann datasets/inaoka/train/annotations.json \
    --val-root datasets/inaoka/val \
    --val-ann datasets/inaoka/val/annotations.json
```

### Step 4: Use the Fine-tuned Model

```python
import torch
from model_lightweight import create_lightweight_model

# Load model
model = create_lightweight_model(num_classes=1, model_size='tiny')
checkpoint = torch.load('finetune_results/whiteline_*/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    predictions = model.predict(image_tensor, conf_threshold=0.5)
```

## Best Practices

### 1. Learning Rate Strategy
- Backbone: 1e-5 (very low to preserve learned features)
- FPN: 1e-4 (moderate for adaptation)
- Detection heads: 1e-3 (higher for quick adaptation)

### 2. Freezing Strategy
- Start with frozen backbone to stabilize training
- Gradually unfreeze layers from top to bottom
- Monitor validation metrics when unfreezing

### 3. Data Augmentation
- Use moderate augmentation during fine-tuning
- Too aggressive augmentation can hurt transfer learning
- Consider domain-specific augmentations

### 4. Monitoring
- Track gradient flow to ensure all layers are learning properly
- Monitor for overfitting, especially with small datasets
- Save checkpoints frequently

### 5. Optimization Tips
- Use gradient clipping (max norm ~5.0)
- Consider exponential moving average (EMA) for stability
- Fine-tune batch norm statistics in later stages

## Common Issues and Solutions

### Issue 1: Catastrophic Forgetting
**Solution**: Use gradual unfreezing and very low learning rates for pretrained layers

### Issue 2: Poor Initial Performance
**Solution**: Ensure proper initialization of detection heads and use warmup

### Issue 3: Overfitting on Small Dataset
**Solution**: Keep backbone frozen longer, use stronger regularization

### Issue 4: Training Instability
**Solution**: Reduce learning rates, use gradient clipping, check data quality

## Example Results

Expected improvements with transfer learning:
- **Convergence Speed**: 2-3x faster than training from scratch
- **Final Performance**: 10-20% higher mAP on small datasets
- **Stability**: More stable training with pretrained features
- **Generalization**: Better performance on edge cases

## Advanced Usage

### Custom Configuration

Create a custom config file:
```json
{
  "num_classes": 1,
  "pretrained_backbone_path": "path/to/backbone.pth",
  "stage1_epochs": 15,
  "stage1_head_lr": 2e-3,
  "stage2_epochs": 15,
  "stage2_head_lr": 1e-3,
  "stage2_fpn_lr": 2e-4,
  "stage3_epochs": 30,
  "stage3_head_lr": 2e-4,
  "stage3_fpn_lr": 1e-4,
  "stage3_backbone_lr": 2e-5
}
```

Use with:
```bash
python finetune_single_class.py --config custom_config.json
```

### Multi-GPU Training

Both scripts support multi-GPU training:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python pretrain_multi_class.py ...
```

### Mixed Precision Training

Enable mixed precision for faster training:
```python
# In the training scripts, add:
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(images)
    loss = criterion(outputs, targets)
```

## Conclusion

This transfer learning system provides a robust pipeline for adapting object detection models to new tasks with limited data. By leveraging pretrained features and careful fine-tuning strategies, you can achieve better performance with less training time and data.