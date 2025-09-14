#!/usr/bin/env python3
"""
Start pretraining with existing downloaded data
"""

import os
import torch
from train_lightweight_coco import COCODatasetLightweight
from model_lightweight import create_lightweight_model

def check_existing_data():
    """Check how many images are already downloaded"""
    train_dir = "./datasets/coco2017/train2017"
    val_dir = "./datasets/coco2017/val2017"
    
    train_images = 0
    val_images = 0
    
    if os.path.exists(train_dir):
        train_images = len(os.listdir(train_dir))
    
    if os.path.exists(val_dir):
        val_images = len(os.listdir(val_dir))
    
    print(f"Found {train_images} training images")
    print(f"Found {val_images} validation images")
    
    return train_images, val_images

def main():
    print("=== Checking existing COCO data ===\n")
    
    train_images, val_images = check_existing_data()
    
    if train_images < 100 and val_images < 100:
        print("\nNot enough images downloaded yet.")
        print("Options:")
        print("1. Wait for download to complete")
        print("2. Run quick setup: python quick_pretrain_setup.py")
        print("3. Use validation set for both training and validation (if available)")
        return
    
    print("\nYou can start pretraining with existing data:")
    print(f"python pretrain_multi_class_simple.py --model_size tiny --epochs 10 --batch_size 4")
    
    # Check if we can use val set
    if val_images > 100:
        print("\nAlternatively, use validation set for quick testing:")
        print("Edit pretrain_multi_class_simple.py to use 'val2017' for both train and val")

if __name__ == "__main__":
    main()