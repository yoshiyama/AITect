#!/usr/bin/env python3
"""新しいモデルと損失関数の互換性確認"""

import torch
from model import AITECTDetector
from loss import detection_loss
import json

def test_training_compatibility():
    """学習の互換性を確認"""
    print("=== 新モデルと損失関数の互換性テスト ===\n")
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}\n")
    
    # モデル初期化
    model = AITECTDetector().to(device)
    model.train()
    
    # ダミーデータ作成
    batch_size = 2
    images = torch.randn(batch_size, 3, 512, 512).to(device)
    
    # ダミーターゲット（COCO形式）
    targets = []
    for i in range(batch_size):
        target = {
            'boxes': torch.tensor([
                [100, 100, 200, 200],  # [x1, y1, x2, y2]
                [300, 300, 400, 400]
            ], dtype=torch.float32).to(device),
            'labels': torch.tensor([1, 1], dtype=torch.int64).to(device),
            'image_id': i
        }
        targets.append(target)
    
    # 順伝播
    print("1. 順伝播テスト...")
    try:
        predictions = model(images)
        print(f"   ✓ モデル出力形状: {predictions.shape}")
        print(f"   ✓ 座標範囲:")
        print(f"     X: {predictions[:, :, 0].min():.1f} - {predictions[:, :, 0].max():.1f}")
        print(f"     Y: {predictions[:, :, 1].min():.1f} - {predictions[:, :, 1].max():.1f}")
        print(f"     W: {predictions[:, :, 2].min():.1f} - {predictions[:, :, 2].max():.1f}")
        print(f"     H: {predictions[:, :, 3].min():.1f} - {predictions[:, :, 3].max():.1f}")
    except Exception as e:
        print(f"   ✗ エラー: {e}")
        return
    
    # 損失計算
    print("\n2. 損失計算テスト...")
    try:
        # config.jsonから設定を読み込み
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        loss_type = config['training'].get('loss_type', 'mixed')
        iou_weight = config['training'].get('iou_weight', 2.0)
        l1_weight = config['training'].get('l1_weight', 0.5)
        
        loss = detection_loss(
            predictions, targets,
            loss_type=loss_type,
            iou_weight=iou_weight,
            l1_weight=l1_weight
        )
        print(f"   ✓ 損失値: {loss.item():.4f}")
        print(f"   ✓ 損失タイプ: {loss_type}")
    except Exception as e:
        print(f"   ✗ エラー: {e}")
        return
    
    # 逆伝播
    print("\n3. 逆伝播テスト...")
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        optimizer.zero_grad()
        loss.backward()
        
        # 勾配が計算されているか確認
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        print(f"   ✓ 勾配が計算されたパラメータ数: {len(grad_norms)}")
        print(f"   ✓ 平均勾配ノルム: {sum(grad_norms)/len(grad_norms):.6f}")
        
        # 最適化ステップ
        optimizer.step()
        print("   ✓ 最適化ステップ完了")
    except Exception as e:
        print(f"   ✗ エラー: {e}")
        return
    
    print("\n4. メモリ使用量...")
    if device.type == 'cuda':
        print(f"   GPU メモリ使用量: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    print("\n✅ すべてのテストが成功しました！")
    print("新しいモデルは学習可能です。")

if __name__ == "__main__":
    test_training_compatibility()