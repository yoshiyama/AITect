#!/usr/bin/env python3
"""IoU損失関数の動作確認スクリプト"""

import torch
import sys
sys.path.append('.')

from loss import iou_loss, detection_loss
from utils.bbox import box_iou

def test_iou_loss():
    """IoU損失関数の基本的な動作をテスト"""
    print("=== IoU損失関数のテスト ===\n")
    
    # テストケース1: 完全一致のボックス（IoU = 1.0）
    pred_boxes1 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    gt_boxes1 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    loss1 = iou_loss(pred_boxes1, gt_boxes1)
    iou1 = box_iou(pred_boxes1, gt_boxes1)[0, 0]
    print(f"テスト1 - 完全一致:")
    print(f"  予測: {pred_boxes1[0].tolist()}")
    print(f"  正解: {gt_boxes1[0].tolist()}")
    print(f"  IoU: {iou1:.4f}")
    print(f"  IoU損失: {loss1:.4f} (期待値: 0.0)\n")
    
    # テストケース2: 50%重なり
    pred_boxes2 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    gt_boxes2 = torch.tensor([[30, 10, 70, 50]], dtype=torch.float32)
    loss2 = iou_loss(pred_boxes2, gt_boxes2)
    iou2 = box_iou(pred_boxes2, gt_boxes2)[0, 0]
    print(f"テスト2 - 部分的な重なり:")
    print(f"  予測: {pred_boxes2[0].tolist()}")
    print(f"  正解: {gt_boxes2[0].tolist()}")
    print(f"  IoU: {iou2:.4f}")
    print(f"  IoU損失: {loss2:.4f}\n")
    
    # テストケース3: 重なりなし（IoU = 0.0）
    pred_boxes3 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    gt_boxes3 = torch.tensor([[60, 60, 100, 100]], dtype=torch.float32)
    loss3 = iou_loss(pred_boxes3, gt_boxes3)
    iou3 = box_iou(pred_boxes3, gt_boxes3)[0, 0]
    print(f"テスト3 - 重なりなし:")
    print(f"  予測: {pred_boxes3[0].tolist()}")
    print(f"  正解: {gt_boxes3[0].tolist()}")
    print(f"  IoU: {iou3:.4f}")
    print(f"  IoU損失: {loss3:.4f} (期待値: 1.0)\n")
    
    # テストケース4: 複数ボックス
    pred_boxes4 = torch.tensor([
        [10, 10, 50, 50],
        [60, 60, 100, 100]
    ], dtype=torch.float32)
    gt_boxes4 = torch.tensor([
        [15, 15, 55, 55],
        [65, 65, 105, 105]
    ], dtype=torch.float32)
    loss4 = iou_loss(pred_boxes4, gt_boxes4)
    ious4 = box_iou(pred_boxes4, gt_boxes4).diag()
    print(f"テスト4 - 複数ボックス:")
    print(f"  IoU (Box1): {ious4[0]:.4f}")
    print(f"  IoU (Box2): {ious4[1]:.4f}")
    print(f"  平均IoU: {ious4.mean():.4f}")
    print(f"  IoU損失: {loss4:.4f}\n")
    
    print("=== detection_loss関数の損失タイプテスト ===\n")
    
    # ダミーの予測とターゲットを作成
    batch_size = 2
    num_preds = 16  # 4x4グリッド
    preds = torch.randn(batch_size, num_preds, 5)  # [B, N, 5]
    
    # ダミーターゲット
    targets = []
    for b in range(batch_size):
        target = {
            'boxes': torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32),
            'labels': torch.tensor([1, 1], dtype=torch.int64)
        }
        targets.append(target)
    
    # 各損失タイプでテスト
    for loss_type in ['iou_only', 'l1_only', 'mixed']:
        loss = detection_loss(preds, targets, loss_type=loss_type)
        print(f"損失タイプ '{loss_type}': {loss:.4f}")
    
    print("\n✅ すべてのテストが完了しました！")

if __name__ == "__main__":
    test_iou_loss()