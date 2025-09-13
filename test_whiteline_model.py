#!/usr/bin/env python3
"""白線検出モデルの動作確認"""

import torch
from model_whiteline import WhiteLineDetector

def test_whiteline_model():
    print("=== 白線検出モデルのテスト ===\n")
    
    # モデル初期化
    model = WhiteLineDetector(grid_size=10, num_anchors=1)
    model.eval()
    
    # モデル情報を表示
    anchor_info = model.get_anchor_info()
    print(f"モデル設定:")
    print(f"  グリッドサイズ: 10x10")
    print(f"  アンカー数: 1")
    print(f"  総予測数: {anchor_info['total_predictions']}")
    print(f"\nアンカーボックス:")
    print(f"  幅: {anchor_info['anchor_width']:.1f}px")
    print(f"  高さ: {anchor_info['anchor_height']:.1f}px")
    print(f"  アスペクト比: {anchor_info['aspect_ratio']:.1f} (横長)")
    
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
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"\n信頼度分布:")
    for thresh in thresholds:
        count = (conf_sigmoid > thresh).sum().item()
        print(f"  閾値 {thresh}: {count}/{len(conf_sigmoid)} ({count/len(conf_sigmoid)*100:.1f}%)")
    
    print("\n✅ 白線検出モデルの特徴:")
    print("  - シンプルな10x10グリッド（総予測数100）")
    print("  - 横長のアンカーボックス（白線に最適）")
    print("  - 軽量で高速な推論")
    
    # 以前のモデルとの比較
    print("\n📊 モデル比較:")
    print("  V1モデル: 16x16x1 = 256予測")
    print("  V2モデル: 13x13x3 = 507予測")
    print("  白線モデル: 10x10x1 = 100予測 ← シンプル！")

if __name__ == "__main__":
    test_whiteline_model()