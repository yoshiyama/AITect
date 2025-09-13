#!/usr/bin/env python3
"""学習データを分析してモデルパラメータを自動設定"""

import json
import numpy as np
import torch
from dataset import CocoDataset
from torchvision import transforms

def analyze_dataset(image_dir, annotation_path, sample_size=100):
    """データセットのGTボックスを分析"""
    print("=== データセット分析中 ===")
    
    # データセット準備
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    dataset = CocoDataset(image_dir, annotation_path, transform=transform)
    
    all_widths = []
    all_heights = []
    all_aspects = []
    all_areas = []
    
    # サンプルを分析
    num_samples = min(sample_size, len(dataset))
    for i in range(num_samples):
        _, target = dataset[i]
        boxes = target['boxes']
        
        for box in boxes:
            width = box[2] - box[0]
            height = box[3] - box[1]
            aspect = width / height if height > 0 else 0
            area = width * height
            
            all_widths.append(width.item())
            all_heights.append(height.item())
            all_aspects.append(aspect)
            all_areas.append(area.item())
    
    # 統計計算
    stats = {
        'num_boxes': len(all_widths),
        'width': {
            'mean': np.mean(all_widths),
            'median': np.median(all_widths),
            'std': np.std(all_widths),
            'min': np.min(all_widths),
            'max': np.max(all_widths)
        },
        'height': {
            'mean': np.mean(all_heights),
            'median': np.median(all_heights),
            'std': np.std(all_heights),
            'min': np.min(all_heights),
            'max': np.max(all_heights)
        },
        'aspect_ratio': {
            'mean': np.mean(all_aspects),
            'median': np.median(all_aspects),
            'std': np.std(all_aspects),
            'min': np.min(all_aspects),
            'max': np.max(all_aspects)
        },
        'area': {
            'mean': np.mean(all_areas),
            'median': np.median(all_areas),
            'percentile_25': np.percentile(all_areas, 25),
            'percentile_75': np.percentile(all_areas, 75)
        }
    }
    
    return stats

def determine_grid_size(stats, image_size=512):
    """GTボックスのサイズに基づいて最適なグリッドサイズを決定"""
    median_area = stats['area']['median']
    image_area = image_size * image_size
    
    # 各グリッドセルが中央値のボックスの2-3倍の面積を持つように設定
    target_cell_area = median_area * 2.5
    grid_size = int(np.sqrt(image_area / target_cell_area))
    
    # グリッドサイズを妥当な範囲に制限
    grid_size = max(8, min(16, grid_size))
    
    return grid_size

def determine_anchors(stats, grid_size, image_size=512):
    """GTボックスの統計に基づいてアンカーサイズを決定"""
    cell_size = image_size / grid_size
    
    # 中央値を基準にアンカーサイズを設定
    anchor_w = stats['width']['median'] / cell_size
    anchor_h = stats['height']['median'] / cell_size
    
    # アスペクト比の変動が大きい場合は複数アンカーを使用
    aspect_std = stats['aspect_ratio']['std']
    if aspect_std > 0.5:
        # 3つのアンカー：小さい、中央値、大きい
        anchors = [
            [anchor_w * 0.7, anchor_h * 1.3],  # 縦長
            [anchor_w, anchor_h],               # 中央値
            [anchor_w * 1.3, anchor_h * 0.7]   # 横長
        ]
        num_anchors = 3
    else:
        # 1つのアンカーで十分
        anchors = [[anchor_w, anchor_h]]
        num_anchors = 1
    
    return anchors, num_anchors

def determine_loss_weights(stats):
    """GTボックスの特性に基づいて損失関数の重みを決定"""
    # サイズのばらつきが大きい場合はIoU重視
    size_variance = stats['width']['std'] + stats['height']['std']
    avg_size = stats['width']['mean'] + stats['height']['mean']
    relative_variance = size_variance / avg_size
    
    if relative_variance > 0.3:
        iou_weight = 3.0  # サイズのばらつきが大きい
        l1_weight = 1.0
    else:
        iou_weight = 2.0  # サイズが比較的一定
        l1_weight = 1.0
    
    return iou_weight, l1_weight

def auto_configure(config_path='config.json'):
    """設定ファイルを読み込み、データセット分析に基づいて自動調整"""
    # 既存の設定を読み込み
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # データセットパス取得
    image_dir = config['paths']['train_images']
    annotation_path = config['paths']['train_annotations']
    
    # データセット分析
    stats = analyze_dataset(image_dir, annotation_path)
    
    print(f"\n分析したボックス数: {stats['num_boxes']}")
    print(f"幅の中央値: {stats['width']['median']:.1f}px")
    print(f"高さの中央値: {stats['height']['median']:.1f}px")
    print(f"アスペクト比の中央値: {stats['aspect_ratio']['median']:.2f}")
    
    # パラメータ決定
    grid_size = determine_grid_size(stats)
    anchors, num_anchors = determine_anchors(stats, grid_size)
    iou_weight, l1_weight = determine_loss_weights(stats)
    
    print(f"\n=== 自動設定されたパラメータ ===")
    print(f"グリッドサイズ: {grid_size}x{grid_size}")
    print(f"アンカー数: {num_anchors}")
    print(f"アンカーサイズ:")
    for i, (w, h) in enumerate(anchors):
        print(f"  アンカー{i+1}: 幅{w:.2f}, 高さ{h:.2f} (グリッドセル単位)")
    print(f"IoU重み: {iou_weight}")
    print(f"L1重み: {l1_weight}")
    
    # 設定を更新
    config['model']['grid_size'] = grid_size
    config['model']['num_anchors'] = num_anchors
    config['model']['auto_anchors'] = anchors
    config['training']['iou_weight'] = iou_weight
    config['training']['l1_weight'] = l1_weight
    
    # 自動設定フラグを追加
    config['model']['auto_configured'] = True
    config['model']['dataset_stats'] = {
        'width_median': stats['width']['median'],
        'height_median': stats['height']['median'],
        'aspect_median': stats['aspect_ratio']['median']
    }
    
    return config, stats

if __name__ == "__main__":
    config, stats = auto_configure()
    print("\n自動設定が完了しました。")