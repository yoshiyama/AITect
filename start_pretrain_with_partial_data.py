#!/usr/bin/env python3
"""
Start pretraining with partially downloaded data
"""

import os
import subprocess

def check_data_status():
    """Check current download status"""
    train_dir = "./datasets/coco2017/train2017"
    val_dir = "./datasets/coco2017/val2017"
    
    train_count = 0
    val_count = 0
    
    if os.path.exists(train_dir):
        train_count = len([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
    
    if os.path.exists(val_dir):
        val_count = len([f for f in os.listdir(val_dir) if f.endswith('.jpg')])
    
    return train_count, val_count

def main():
    print("=== Starting Pretraining with Partial Data ===\n")
    
    train_count, val_count = check_data_status()
    print(f"Current status:")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    
    if train_count >= 500:
        print(f"\n✅ Enough training data ({train_count} images) to start pretraining!")
        print("\nStarting pretraining with current data...")
        print("Note: This will use whatever data is currently downloaded.")
        print("\nCommand to run:")
        print("python pretrain_multi_class_simple.py --model_size small --epochs 30 --batch_size 16")
        
        # Start pretraining
        subprocess.run([
            "python", "pretrain_multi_class_simple.py",
            "--model_size", "small",
            "--epochs", "30",
            "--batch_size", "16"
        ])
    else:
        print(f"\n⏳ Not enough data yet ({train_count} images). Need at least 500.")
        print("Options:")
        print("1. Wait for more downloads")
        print("2. Use validation data for both train/val (if available)")
        print("3. Start with tiny model and fewer epochs")
        
        if train_count >= 100:
            print(f"\nYou can try with current {train_count} images:")
            print("python pretrain_multi_class_simple.py --model_size tiny --epochs 10 --batch_size 8")

if __name__ == "__main__":
    main()