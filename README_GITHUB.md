# AITect - Lightweight Object Detection Framework

A PyTorch-based lightweight object detection framework designed for efficient multi-class and single-class detection tasks. Features transfer learning capabilities, COCO dataset integration, and optimized for edge deployment.

## 🌟 Key Features

- **Lightweight Models**: MobileNetV2, ResNet18, ShuffleNet backbones (~2M-13M parameters)
- **Multi-Scale Detection**: Feature Pyramid Network (FPN) for detecting objects at different scales
- **Advanced Loss Functions**: Focal Loss for class imbalance, IoU Loss for accurate bounding boxes
- **Transfer Learning**: Pre-train on multiple classes, fine-tune for specific tasks
- **COCO Integration**: Easy setup for training with COCO dataset
- **Edge-Ready**: Optimized for deployment on resource-constrained devices

## 📁 Project Structure

```
AITect/
├── models/
│   ├── model_lightweight.py          # Lightweight detection architectures
│   ├── model_improved_v2.py          # Improved detection model with FPN
│   └── detection_loss_focal.py       # Focal detection loss implementation
├── training/
│   ├── pretrain_multi_class_focal.py # Multi-class pretraining script
│   ├── finetune_single_class.py      # Single-class fine-tuning
│   └── train_lightweight_coco.py     # COCO training pipeline
├── utils/
│   ├── postprocess.py                # NMS and post-processing utilities
│   ├── bbox.py                       # Bounding box utilities
│   └── metrics.py                    # Evaluation metrics
├── data/
│   ├── setup_coco_training.py        # COCO dataset setup
│   └── dataset.py                    # Custom dataset classes
└── configs/
    └── config_transfer_learning.json  # Transfer learning configurations
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AITect.git
cd AITect

# Create conda environment
conda create -n aitect python=3.10
conda activate aitect

# Install dependencies
pip install torch torchvision
pip install pycocotools tqdm matplotlib
```

### 2. Prepare COCO Dataset

```python
from setup_coco_training import setup_coco_dataset

# Download specific classes (e.g., person detection)
setup_coco_dataset(
    selected_classes=['person'],
    dataset_type='val',
    max_images=500
)
```

### 3. Train a Model

#### Option A: Direct Training
```bash
# Train lightweight model on single class
python train_lightweight_coco.py \
    --classes person \
    --model_size tiny \
    --epochs 30
```

#### Option B: Transfer Learning (Recommended)
```bash
# Step 1: Pre-train on multiple classes
python pretrain_multi_class_focal.py \
    --model_size small \
    --epochs 30

# Step 2: Fine-tune for specific class
python finetune_single_class.py \
    --pretrained_path ./pretrain_focal_results/*/checkpoints/pretrained_focal_small_best.pth \
    --target_class person \
    --epochs 20
```

## 🏗️ Model Architecture

### Lightweight Backbone Options

| Model Size | Backbone | Parameters | Speed (FPS) | Use Case |
|------------|----------|------------|-------------|----------|
| Tiny | MobileNetV2 | ~2M | 60+ | Edge devices, real-time |
| Small | ResNet18 | ~11M | 30 | Balanced performance |
| Medium | ShuffleNet | ~5M | 45 | Mobile applications |

### Detection Head Architecture
- Multi-scale feature maps (52×52, 26×26, 13×13)
- Depthwise separable convolutions for efficiency
- Anchor-based detection with IoU assignment

## 📊 Loss Functions

### Focal Detection Loss
```python
FocalDetectionLoss(
    focal_alpha=0.25,    # Balance positive/negative samples
    focal_gamma=2.0,     # Focus on hard examples
    bbox_weight=5.0,     # Bounding box regression weight
    obj_weight=1.0,      # Objectness weight
    cls_weight=1.0       # Classification weight
)
```

Components:
- **Focal Loss**: Addresses class imbalance in object detection
- **IoU Loss**: Better bounding box regression than L2 loss
- **Multi-scale supervision**: Different anchors for different object sizes

## 🔧 Advanced Usage

### Custom Dataset Training
```python
from model_lightweight import create_lightweight_model
from detection_loss_focal import FocalDetectionLoss

# Create model
model = create_lightweight_model(
    num_classes=5,
    model_size="small",
    pretrained=True
)

# Define loss
criterion = FocalDetectionLoss(
    num_classes=5,
    focal_alpha=0.25,
    focal_gamma=2.0
)

# Training loop
for epoch in range(num_epochs):
    for images, targets in dataloader:
        outputs = model(images)
        loss, loss_dict = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

### Inference
```python
# Load trained model
checkpoint = torch.load('path/to/model.pth')
model = create_lightweight_model(
    num_classes=checkpoint['num_classes'],
    model_size=checkpoint['model_size']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    predictions = model.predict(image_tensor, conf_threshold=0.5)
```

## 📈 Performance

Typical performance on COCO validation set:

| Model | Classes | mAP@0.5 | Inference Time |
|-------|---------|---------|----------------|
| Tiny-1 | Person | ~0.45 | 16ms |
| Small-5 | 5 classes | ~0.40 | 33ms |
| Small-20 | 20 classes | ~0.35 | 35ms |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{aitect2024,
  title = {AITect: Lightweight Object Detection Framework},
  year = {2024},
  url = {https://github.com/yourusername/AITect}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- COCO dataset team for the comprehensive object detection dataset
- PyTorch team for the excellent deep learning framework
- Focal Loss paper (Lin et al., 2017) for addressing class imbalance