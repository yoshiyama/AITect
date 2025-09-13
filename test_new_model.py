#!/usr/bin/env python3
"""新しいグリッドベースモデルの動作確認"""

import torch
import numpy as np
from model import AITECTDetector

def test_new_model():
    """新しいモデルの出力を確認"""
    print("=== 新しいグリッドベースモデルのテスト ===\n")
    
    # モデルとダミー入力
    model = AITECTDetector()
    model.eval()
    
    # ダミー画像 (1, 3, 512, 512)
    dummy_input = torch.randn(1, 3, 512, 512)
    
    with torch.no_grad():
        # モデル出力
        output = model(dummy_input)
        print(f"モデル出力形状: {output.shape}")
        
        # 出力の統計情報
        pred_boxes = output[0, :, :4]  # [N, 4]
        pred_conf = output[0, :, 4]     # [N]
        
        print(f"\n予測ボックス座標の統計（画像座標系）:")
        print(f"  X中心 - min: {pred_boxes[:, 0].min():.2f}, max: {pred_boxes[:, 0].max():.2f}, mean: {pred_boxes[:, 0].mean():.2f}")
        print(f"  Y中心 - min: {pred_boxes[:, 1].min():.2f}, max: {pred_boxes[:, 1].max():.2f}, mean: {pred_boxes[:, 1].mean():.2f}")
        print(f"  幅    - min: {pred_boxes[:, 2].min():.2f}, max: {pred_boxes[:, 2].max():.2f}, mean: {pred_boxes[:, 2].mean():.2f}")
        print(f"  高さ  - min: {pred_boxes[:, 3].min():.2f}, max: {pred_boxes[:, 3].max():.2f}, mean: {pred_boxes[:, 3].mean():.2f}")
        print(f"  信頼度 - min: {pred_conf.min():.2f}, max: {pred_conf.max():.2f}, mean: {pred_conf.mean():.2f}")
        
        # グリッド位置の分布を確認
        pred_grid = output[0].view(16, 16, 5)
        
        print(f"\nグリッド位置ごとの予測分布:")
        print("(高信頼度の予測がある位置を表示)")
        
        grid_map = np.zeros((16, 16))
        for i in range(16):
            for j in range(16):
                conf = torch.sigmoid(pred_grid[i, j, 4])
                grid_map[i, j] = conf.item()
                if conf > 0.5:
                    x, y, w, h = pred_grid[i, j, :4]
                    print(f"  Grid[{i:2d},{j:2d}]: x={x:6.1f}, y={y:6.1f}, w={w:5.1f}, h={h:5.1f}, conf={conf:.3f}")
        
        # グリッドマップの可視化（簡易版）
        print("\n信頼度マップ（0-9スケール）:")
        for i in range(16):
            row = ""
            for j in range(16):
                val = int(grid_map[i, j] * 9)
                row += str(val) if val > 0 else "."
            print(f"  {row}")
        
        # 座標の妥当性チェック
        print("\n座標の妥当性チェック:")
        x_valid = (pred_boxes[:, 0] >= 0) & (pred_boxes[:, 0] <= 512)
        y_valid = (pred_boxes[:, 1] >= 0) & (pred_boxes[:, 1] <= 512)
        w_valid = pred_boxes[:, 2] > 0
        h_valid = pred_boxes[:, 3] > 0
        
        print(f"  X座標が画像内: {x_valid.sum()}/{len(x_valid)} ({x_valid.float().mean()*100:.1f}%)")
        print(f"  Y座標が画像内: {y_valid.sum()}/{len(y_valid)} ({y_valid.float().mean()*100:.1f}%)")
        print(f"  幅が正の値: {w_valid.sum()}/{len(w_valid)} ({w_valid.float().mean()*100:.1f}%)")
        print(f"  高さが正の値: {h_valid.sum()}/{len(h_valid)} ({h_valid.float().mean()*100:.1f}%)")

if __name__ == "__main__":
    test_new_model()