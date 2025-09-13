#!/usr/bin/env python3
"""GTボックスのサイズ分析"""

import json
import numpy as np
from dataset import CocoDataset
from torchvision import transforms

def analyze_gt_boxes():
    print("=== GTボックス（白線）のサイズ分析 ===\n")
    
    # 設定読み込み
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # データセット準備
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    image_dir = config['paths']['train_images']
    annotation_path = config['paths']['train_annotations']
    dataset = CocoDataset(image_dir, annotation_path, transform=transform)
    
    all_widths = []
    all_heights = []
    all_aspects = []
    
    # 最初の20サンプルを分析
    for i in range(min(20, len(dataset))):
        _, target = dataset[i]
        boxes = target['boxes']
        
        for box in boxes:
            width = box[2] - box[0]
            height = box[3] - box[1]
            aspect = width / height if height > 0 else 0
            
            all_widths.append(width.item())
            all_heights.append(height.item())
            all_aspects.append(aspect)
    
    print(f"分析したボックス数: {len(all_widths)}")
    
    print(f"\n幅の統計（ピクセル）:")
    print(f"  最小: {np.min(all_widths):.1f}")
    print(f"  最大: {np.max(all_widths):.1f}")
    print(f"  平均: {np.mean(all_widths):.1f}")
    print(f"  中央値: {np.median(all_widths):.1f}")
    
    print(f"\n高さの統計（ピクセル）:")
    print(f"  最小: {np.min(all_heights):.1f}")
    print(f"  最大: {np.max(all_heights):.1f}")
    print(f"  平均: {np.mean(all_heights):.1f}")
    print(f"  中央値: {np.median(all_heights):.1f}")
    
    print(f"\nアスペクト比（幅/高さ）:")
    print(f"  最小: {np.min(all_aspects):.2f}")
    print(f"  最大: {np.max(all_aspects):.2f}")
    print(f"  平均: {np.mean(all_aspects):.2f}")
    print(f"  中央値: {np.median(all_aspects):.2f}")
    
    # アンカーとの比較
    print(f"\n現在のアンカーサイズ:")
    print(f"  幅: 102.4px")
    print(f"  高さ: 25.6px")
    print(f"  アスペクト比: 4.0")
    
    print(f"\n推奨アンカーサイズ:")
    print(f"  幅: {np.median(all_widths):.1f}px")
    print(f"  高さ: {np.median(all_heights):.1f}px")
    print(f"  アスペクト比: {np.median(all_aspects):.1f}")

if __name__ == "__main__":
    analyze_gt_boxes()