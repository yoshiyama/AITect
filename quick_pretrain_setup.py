#!/usr/bin/env python3
"""
Quick setup for pretraining with fewer images to start faster
"""

from setup_coco_training import setup_coco_dataset

def main():
    print("=== Quick COCO Setup for Testing ===\n")
    
    # Common classes for pretraining (fewer for quick start)
    common_classes = [
        'person', 'car', 'dog', 'cat', 'bicycle'
    ]
    
    print(f"Setting up COCO with {len(common_classes)} classes for quick testing")
    print(f"Classes: {', '.join(common_classes)}\n")
    
    # Download smaller dataset for quick testing
    print("Downloading validation data (smaller, faster)...")
    setup_coco_dataset(
        data_dir="./datasets/coco2017",
        download_images=True,
        dataset_type="val",  # Use validation set (smaller)
        selected_classes=common_classes,
        max_images=500  # Only 500 images for quick start
    )
    
    print("\n=== Quick Setup Complete! ===")
    print("\nYou can now run pretraining with:")
    print("python pretrain_multi_class_simple.py --model_size tiny --epochs 10 --batch_size 8")
    print("\nNote: This is a smaller dataset for testing. For full pretraining, use the full dataset.")

if __name__ == "__main__":
    main()