#!/usr/bin/env python3
"""グリッドベース予測の問題を確認するデバッグスクリプト"""

import torch
import numpy as np
from model import AITECTDetector

def debug_model_output():
    """モデルの出力形状と値の範囲を確認"""
    print("=== モデル出力のデバッグ ===\n")
    
    # モデルとダミー入力
    model = AITECTDetector()
    model.eval()
    
    # ダミー画像 (1, 3, 512, 512)
    dummy_input = torch.randn(1, 3, 512, 512)
    
    with torch.no_grad():
        # モデルの中間出力を確認
        feat = model.backbone(dummy_input)
        print(f"バックボーン出力形状: {feat.shape}")
        
        # Head出力
        head_out = model.head(feat)
        print(f"Head出力形状: {head_out.shape}")
        
        # 最終出力
        output = model(dummy_input)
        print(f"最終出力形状: {output.shape}")
        
        # 出力の統計情報
        pred_boxes = output[0, :, :4]  # [N, 4]
        pred_conf = output[0, :, 4]     # [N]
        
        print(f"\n予測ボックス座標の統計:")
        print(f"  X座標 - min: {pred_boxes[:, 0].min():.2f}, max: {pred_boxes[:, 0].max():.2f}, mean: {pred_boxes[:, 0].mean():.2f}")
        print(f"  Y座標 - min: {pred_boxes[:, 1].min():.2f}, max: {pred_boxes[:, 1].max():.2f}, mean: {pred_boxes[:, 1].mean():.2f}")
        print(f"  幅    - min: {pred_boxes[:, 2].min():.2f}, max: {pred_boxes[:, 2].max():.2f}, mean: {pred_boxes[:, 2].mean():.2f}")
        print(f"  高さ  - min: {pred_boxes[:, 3].min():.2f}, max: {pred_boxes[:, 3].max():.2f}, mean: {pred_boxes[:, 3].mean():.2f}")
        print(f"  信頼度 - min: {pred_conf.min():.2f}, max: {pred_conf.max():.2f}, mean: {pred_conf.mean():.2f}")
        
        # グリッド情報
        B, C, H, W = head_out.shape
        print(f"\nグリッド情報:")
        print(f"  グリッドサイズ: {H}x{W}")
        print(f"  総予測数: {H*W}")
        print(f"  1グリッドあたりの画像サイズ: {512/H:.1f}x{512/W:.1f} pixels")
        
        # 予測を2Dグリッドに戻して各位置の予測を確認
        pred_grid = output[0].view(H, W, 5)
        
        print(f"\nグリッド位置ごとの予測例（信頼度 > 0）:")
        for i in range(min(5, H)):
            for j in range(min(5, W)):
                conf = torch.sigmoid(pred_grid[i, j, 4])
                if conf > 0.1:
                    x, y, w, h = pred_grid[i, j, :4]
                    print(f"  Grid[{i},{j}]: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}, conf={conf:.3f}")

if __name__ == "__main__":
    debug_model_output()