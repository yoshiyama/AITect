#!/usr/bin/env python3
"""損失関数のデバッグ"""

import torch
import json
from model_whiteline import WhiteLineDetector
from dataset import CocoDataset
from torchvision import transforms
from loss_improved import detection_loss_improved
from loss import detection_loss

def debug_loss():
    print("=== 損失関数のデバッグ ===\n")
    
    # 設定読み込み
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print(f"改善された損失関数を使用: {config['model'].get('use_improved_loss', False)}")
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデル初期化
    model = WhiteLineDetector(grid_size=10, num_anchors=1).to(device)
    
    # データセット準備（1サンプル）
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    image_dir = config['paths']['train_images']
    annotation_path = config['paths']['train_annotations']
    dataset = CocoDataset(image_dir, annotation_path, transform=transform)
    
    # 1つのサンプルで確認
    image, target = dataset[0]
    image = image.unsqueeze(0).to(device)
    target['boxes'] = target['boxes'].to(device)
    target['labels'] = target['labels'].to(device)
    
    # 順伝播
    model.eval()
    with torch.no_grad():
        output = model(image)
    
    # 予測とGTの情報
    pred_boxes = output[0, :, :4]
    gt_boxes = target['boxes']
    
    print(f"\nGTボックス数: {len(gt_boxes)}")
    print(f"予測数: {len(pred_boxes)}")
    
    # 両方の損失関数で計算
    targets = [target]
    
    # 通常の損失関数
    loss_normal = detection_loss(output, targets, use_focal=True)
    print(f"\n通常の損失関数: {loss_normal.item():.4f}")
    
    # 改善された損失関数
    loss_improved = detection_loss_improved(output, targets, use_focal=True)
    print(f"改善された損失関数: {loss_improved.item():.4f}")
    
    # IoUを計算して確認
    from utils.bbox import box_iou
    
    # xyxy形式に変換
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    pred_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
    
    # すべての予測とGTのIoU
    ious = box_iou(pred_boxes_xyxy, gt_boxes)
    max_ious, _ = ious.max(dim=1)
    
    print(f"\n最大IoU統計:")
    print(f"  最小: {max_ious.min():.4f}")
    print(f"  最大: {max_ious.max():.4f}")
    print(f"  平均: {max_ious.mean():.4f}")
    print(f"  IoU > 0.1の予測数: {(max_ious > 0.1).sum().item()}")
    print(f"  IoU > 0.5の予測数: {(max_ious > 0.5).sum().item()}")

if __name__ == "__main__":
    debug_loss()