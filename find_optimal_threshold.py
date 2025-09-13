import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CocoDataset
from model_whiteline import WhiteLineDetector
from utils.bbox import box_iou, nms
import json
import numpy as np
import matplotlib.pyplot as plt

def evaluate_with_threshold(model, dataset, device, conf_threshold, nms_threshold=0.5):
    """特定の閾値での評価"""
    model.eval()
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    with torch.no_grad():
        for i in range(min(50, len(dataset))):  # 最大50枚で評価
            image, target = dataset[i]
            image_tensor = image.unsqueeze(0).to(device)
            
            # 予測
            output = model(image_tensor)[0]
            scores = torch.sigmoid(output[:, 4])
            
            # 閾値でフィルタリング
            mask = scores > conf_threshold
            
            if mask.sum() == 0:
                # 予測なし
                total_fn += len(target['boxes'])
                continue
            
            filtered_boxes = output[mask, :4]
            filtered_scores = scores[mask]
            
            # xyxy形式に変換
            x1 = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2
            y1 = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2
            x2 = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2
            y2 = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
            
            # NMS
            keep = nms(boxes_xyxy, filtered_scores, nms_threshold)
            final_boxes = boxes_xyxy[keep]
            
            gt_boxes = target['boxes'].to(device)
            
            if len(gt_boxes) == 0:
                # GTなし
                total_fp += len(final_boxes)
                continue
            
            # IoU計算
            ious = box_iou(final_boxes, gt_boxes)
            
            if len(final_boxes) > 0:
                # 各予測に対して最もIoUが高いGTを見つける
                max_ious, matched_gt = ious.max(dim=1)
                
                # TP, FP
                tp_mask = max_ious > 0.5
                total_tp += tp_mask.sum().item()
                total_fp += (~tp_mask).sum().item()
                
                # FN（マッチしなかったGT）
                matched_gt_set = set(matched_gt[tp_mask].tolist())
                total_fn += len(gt_boxes) - len(matched_gt_set)
            else:
                total_fn += len(gt_boxes)
    
    # メトリクス計算
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'threshold': conf_threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }

def find_optimal_threshold():
    """最適な信頼度閾値を探索"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデル読み込み
    model = WhiteLineDetector(grid_size=8, num_anchors=3).to(device)
    model.load_state_dict(torch.load("result/aitect_model_improved.pth", map_location=device))
    
    # データセット
    with open('config.json') as f:
        config = json.load(f)
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    val_dataset = CocoDataset(
        config['paths']['val_images'],
        config['paths']['val_annotations'],
        transform=transform
    )
    
    print(f"検証データ数: {len(val_dataset)}")
    print("信頼度閾値を探索中...\n")
    
    # 異なる閾値で評価
    thresholds = np.arange(0.1, 0.95, 0.05)
    results = []
    
    for threshold in thresholds:
        result = evaluate_with_threshold(model, val_dataset, device, threshold)
        results.append(result)
        print(f"閾値 {threshold:.2f}: "
              f"Precision={result['precision']:.3f}, "
              f"Recall={result['recall']:.3f}, "
              f"F1={result['f1']:.3f} "
              f"(TP={result['tp']}, FP={result['fp']}, FN={result['fn']})")
    
    # 結果の可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Precision-Recall曲線
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    
    ax1.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    ax1.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
    ax1.plot(thresholds, f1s, 'g-', label='F1 Score', linewidth=2)
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Metrics vs Confidence Threshold')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(0.1, 0.9)
    ax1.set_ylim(0, 1)
    
    # 最適なF1スコアの位置にマーカー
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    ax1.axvline(best_threshold, color='black', linestyle='--', alpha=0.5)
    ax1.text(best_threshold + 0.01, 0.5, f'Best F1\n@ {best_threshold:.2f}', 
             fontsize=10, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
    
    # Precision-Recall曲線（別の見方）
    ax2.plot(recalls, precisions, 'b-', linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.grid(True)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # 各点の閾値を表示
    for i in range(0, len(thresholds), 2):
        ax2.annotate(f'{thresholds[i]:.2f}', 
                    (recalls[i], precisions[i]), 
                    fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('threshold_optimization.png', dpi=150)
    print(f"\n結果を threshold_optimization.png に保存しました")
    
    # 最適な閾値の結果を表示
    best_result = results[best_idx]
    print(f"\n=== 最適な閾値 ===")
    print(f"信頼度閾値: {best_threshold:.2f}")
    print(f"Precision: {best_result['precision']:.3f}")
    print(f"Recall: {best_result['recall']:.3f}")
    print(f"F1 Score: {best_result['f1']:.3f}")
    print(f"TP: {best_result['tp']}, FP: {best_result['fp']}, FN: {best_result['fn']}")
    
    # 実用的な閾値の提案
    print(f"\n=== 用途別の推奨閾値 ===")
    
    # 高精度重視（誤検出を最小化）
    high_precision_idx = np.where(np.array(precisions) > 0.8)[0]
    if len(high_precision_idx) > 0:
        hp_threshold = thresholds[high_precision_idx[0]]
        hp_result = results[high_precision_idx[0]]
        print(f"高精度重視 (Precision > 0.8): 閾値={hp_threshold:.2f}, "
              f"P={hp_result['precision']:.3f}, R={hp_result['recall']:.3f}")
    
    # バランス重視（F1最大）
    print(f"バランス重視 (F1最大): 閾値={best_threshold:.2f}, "
          f"P={best_result['precision']:.3f}, R={best_result['recall']:.3f}")
    
    # 高再現率重視（見逃しを最小化）
    high_recall_idx = np.where(np.array(recalls) > 0.8)[0]
    if len(high_recall_idx) > 0:
        hr_threshold = thresholds[high_recall_idx[-1]]
        hr_result = results[high_recall_idx[-1]]
        print(f"高再現率重視 (Recall > 0.8): 閾値={hr_threshold:.2f}, "
              f"P={hr_result['precision']:.3f}, R={hr_result['recall']:.3f}")
    
    return best_threshold, results

if __name__ == "__main__":
    find_optimal_threshold()