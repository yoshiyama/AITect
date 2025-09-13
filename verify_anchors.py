#!/usr/bin/env python3
"""アンカーサイズの確認"""

from model_whiteline import WhiteLineDetector

# モデル初期化
model = WhiteLineDetector(grid_size=10, num_anchors=1)

# アンカー情報を取得
info = model.get_anchor_info()

print("=== 更新されたアンカーサイズ ===")
print(f"幅: {info['anchor_width']:.1f}px")
print(f"高さ: {info['anchor_height']:.1f}px") 
print(f"アスペクト比: {info['aspect_ratio']:.2f}")
print(f"\nGT中央値との比較:")
print(f"GT幅: 60.0px → アンカー幅: {info['anchor_width']:.1f}px")
print(f"GT高さ: 39.3px → アンカー高さ: {info['anchor_height']:.1f}px")
print(f"GTアスペクト比: 1.9 → アンカー: {info['aspect_ratio']:.2f}")