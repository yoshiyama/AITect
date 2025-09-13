"""
シンプルな評価スクリプト（基本的なメトリクスのみ）
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CocoDataset
from model_whiteline import WhiteLineDetector
from utils.bbox import box_iou, nms
import json
import numpy as np
from datetime import datetime

def evaluate_simple(model_path='result/aitect_model_improved.pth', 
                   conf_threshold=0.5, 
                   iou_threshold=0.5,
                   max_samples=None):
    """シンプルな評価を実行"""
    
    # 設定読み込み
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデル読み込み
    model = WhiteLineDetector(grid_size=8, num_anchors=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # データセット
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    val_dataset = CocoDataset(
        config['paths']['val_images'],
        config['paths']['val_annotations'],
        transform=transform
    )
    
    # 評価
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_pred = 0
    all_ious = []  # IoU値を保存
    
    num_samples = min(max_samples, len(val_dataset)) if max_samples else len(val_dataset)
    
    print(f"評価開始: {num_samples}枚の画像")
    print(f"設定: Conf={conf_threshold}, IoU={iou_threshold}")
    print("-" * 50)
    
    for i in range(num_samples):
        image, target = val_dataset[i]
        
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))[0]
        
        # 後処理
        scores = torch.sigmoid(output[:, 4])
        mask = scores > conf_threshold
        
        if mask.sum() > 0:
            boxes = output[mask, :4]
            scores = scores[mask]
            
            # xyxy変換
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2
            pred_boxes = torch.stack([x1, y1, x2, y2], dim=1)
            
            # NMS
            keep = nms(pred_boxes, scores, 0.5)
            pred_boxes = pred_boxes[keep]
            
            total_pred += len(pred_boxes)
        else:
            pred_boxes = torch.zeros((0, 4))
        
        gt_boxes = target['boxes'].to(device)
        total_gt += len(gt_boxes)
        
        # TP/FP/FN計算
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)
            max_ious, _ = ious.max(dim=1)
            
            # IoU値を保存（TP予測のみ）
            tp_mask = max_ious >= iou_threshold
            if tp_mask.any():
                all_ious.extend(max_ious[tp_mask].cpu().numpy().tolist())
            
            tp = tp_mask.sum().item()
            fp = len(pred_boxes) - tp
            
            # FN計算
            matched_gt = (ious >= iou_threshold).any(dim=0).sum().item()
            fn = len(gt_boxes) - matched_gt
        elif len(pred_boxes) == 0:
            tp = 0
            fp = 0
            fn = len(gt_boxes)
        else:  # len(gt_boxes) == 0
            tp = 0
            fp = len(pred_boxes)
            fn = 0
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        if (i + 1) % 10 == 0:
            print(f"進捗: {i+1}/{num_samples}")
    
    # メトリクス計算
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # IoU統計
    if all_ious:
        mean_iou = np.mean(all_ious)
        std_iou = np.std(all_ious)
        min_iou = np.min(all_ious)
        max_iou = np.max(all_ious)
        median_iou = np.median(all_ious)
    else:
        mean_iou = std_iou = min_iou = max_iou = median_iou = 0.0
    
    # 結果表示
    print("\n" + "="*50)
    print("評価結果")
    print("="*50)
    print(f"評価時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"モデル: {model_path}")
    print(f"評価画像数: {num_samples}")
    print(f"総GT数: {total_gt}")
    print(f"総予測数: {total_pred}")
    print("\nメトリクス:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\nIoU統計 (TP予測のみ):")
    print(f"  平均IoU: {mean_iou:.4f}")
    print(f"  標準偏差: {std_iou:.4f}")
    print(f"  最小IoU: {min_iou:.4f}")
    print(f"  最大IoU: {max_iou:.4f}")
    print(f"  中央値IoU: {median_iou:.4f}")
    print(f"  TP予測数: {len(all_ious)}")
    print(f"\n詳細:")
    print(f"  True Positives (TP): {total_tp}")
    print(f"  False Positives (FP): {total_fp}")
    print(f"  False Negatives (FN): {total_fn}")
    print("="*50)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'mean_iou': mean_iou,
        'std_iou': std_iou,
        'min_iou': min_iou,
        'max_iou': max_iou,
        'median_iou': median_iou,
        'iou_samples': len(all_ious)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='シンプルな評価スクリプト')
    parser.add_argument('--model', type=str, default='result/aitect_model_improved.pth',
                       help='モデルパス')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='信頼度閾値')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU閾値')
    parser.add_argument('--samples', type=int, default=None,
                       help='評価サンプル数（None=全データ）')
    
    args = parser.parse_args()
    
    evaluate_simple(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_samples=args.samples
    )