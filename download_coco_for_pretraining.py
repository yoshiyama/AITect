#!/usr/bin/env python3
"""
Download COCO dataset for multi-class pretraining
"""

from setup_coco_training import setup_coco_dataset

def main():
    print("=== COCO Dataset Download for Multi-class Pretraining ===\n")
    
    # Get common classes for pretraining
    common_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'backpack', 'umbrella', 'handbag', 'bottle'
    ]
    
    print(f"Downloading COCO dataset for {len(common_classes)} classes:")
    print(f"Classes: {', '.join(common_classes)}\n")
    
    # Download training data (limited to 5000 images for faster download)
    print("1. Downloading training data...")
    setup_coco_dataset(
        data_dir="./datasets/coco2017",
        download_images=True,
        dataset_type="train",
        selected_classes=common_classes,
        max_images=5000  # Limit for faster download
    )
    
    # Download validation data
    print("\n2. Downloading validation data...")
    setup_coco_dataset(
        data_dir="./datasets/coco2017",
        download_images=True,
        dataset_type="val",
        selected_classes=common_classes,
        max_images=1000  # Smaller for validation
    )
    
    print("\n=== Download Complete! ===")
    print("\nNext steps:")
    print("1. Run pretraining:")
    print("   python pretrain_multi_class_simple.py --model_size small --epochs 50")
    print("\n2. Then fine-tune for single class:")
    print("   python finetune_single_class.py --pretrained_path pretrain_results/multi_class_*/checkpoints/pretrained_multi_class_small_best.pth --target_class person")

if __name__ == "__main__":
    main()