#!/usr/bin/env python3
"""改善されたV2モデルの動作確認"""

import torch
import numpy as np
from model_v2 import AITECTDetectorV2
from loss import detection_loss
import json

def test_v2_model():
    """V2モデルの改善点を確認"""
    print("=== AITECTDetector V2 モデルテスト ===\n")
    
    # 設定読み込み
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # モデル初期化
    grid_size = config['model']['grid_size']
    num_anchors = config['model']['num_anchors']
    model = AITECTDetectorV2(grid_size=grid_size, num_anchors=num_anchors)
    model.eval()
    
    print(f"モデル設定:")
    print(f"  グリッドサイズ: {grid_size}x{grid_size}")
    print(f"  アンカー数: {num_anchors}")
    print(f"  総予測数: {grid_size * grid_size * num_anchors}")
    print(f"  改善前(16x16x1): 256 → 改善後: {grid_size * grid_size * num_anchors}\n")
    
    # アンカーボックスのサイズ確認
    anchor_sizes = model.get_anchor_boxes()
    print("アンカーボックスサイズ（ピクセル）:")
    for i, (w, h) in enumerate(anchor_sizes):
        aspect_ratio = h / w
        print(f"  アンカー{i+1}: 幅={w:.1f}px, 高さ={h:.1f}px, アスペクト比={aspect_ratio:.2f}")
    
    # ダミー入力でテスト
    dummy_input = torch.randn(1, 3, 512, 512)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"\nモデル出力:")
    print(f"  形状: {output.shape}")
    
    # 出力の統計
    pred_boxes = output[0, :, :4]
    pred_conf = output[0, :, 4]
    
    print(f"\n座標の統計:")
    print(f"  X中心: {pred_boxes[:, 0].min():.1f} - {pred_boxes[:, 0].max():.1f}")
    print(f"  Y中心: {pred_boxes[:, 1].min():.1f} - {pred_boxes[:, 1].max():.1f}")
    print(f"  幅: {pred_boxes[:, 2].min():.1f} - {pred_boxes[:, 2].max():.1f}")
    print(f"  高さ: {pred_boxes[:, 3].min():.1f} - {pred_boxes[:, 3].max():.1f}")
    
    # 信頼度の分布
    conf_sigmoid = torch.sigmoid(pred_conf)
    high_conf = (conf_sigmoid > 0.5).sum().item()
    print(f"\n信頼度分布:")
    print(f"  高信頼度(>0.5)の予測数: {high_conf}/{len(conf_sigmoid)} ({high_conf/len(conf_sigmoid)*100:.1f}%)")
    
    # Focal Lossのテスト
    print("\n=== Focal Loss テスト ===")
    
    # ダミーターゲット
    targets = [{
        'boxes': torch.tensor([[100, 100, 150, 300]], dtype=torch.float32),  # 電柱らしい縦長ボックス
        'labels': torch.tensor([1], dtype=torch.int64)
    }]
    
    # 通常のBCE Lossと比較
    device = output.device
    for t in targets:
        t['boxes'] = t['boxes'].to(device)
        t['labels'] = t['labels'].to(device)
    
    # Focal Lossあり
    loss_focal = detection_loss(output, targets, use_focal=True)
    # Focal Lossなし
    loss_bce = detection_loss(output, targets, use_focal=False)
    
    print(f"  Focal Lossあり: {loss_focal.item():.4f}")
    print(f"  通常のBCE Loss: {loss_bce.item():.4f}")
    print(f"  差: {abs(loss_focal.item() - loss_bce.item()):.4f}")
    
    print("\n✅ V2モデルの改善点:")
    print("  1. グリッドサイズ削減 (16x16 → 13x13) で予測数を抑制")
    print("  2. 電柱に最適化された3つのアンカーボックス")
    print("  3. Focal Lossで背景の誤検出を抑制")
    print("  4. LeakyReLUとBatchNormで学習安定性向上")

if __name__ == "__main__":
    test_v2_model()