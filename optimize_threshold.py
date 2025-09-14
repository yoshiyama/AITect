import torch
from model import AITECTDetector
from dataset import CocoDataset
from torchvision import transforms
from utils.postprocess import postprocess_predictions
from torchvision.ops import box_iou
import numpy as np
import matplotlib.pyplot as plt

def evaluate_at_threshold(model, dataset, device, conf_threshold):
    """特定の閾値での精度を評価"""
    tp, fp, fn = 0, 0, 0
    
    for idx in range(min(20, len(dataset))):  # 20枚で評価
        image, target = dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(image_batch)
        
        processed = postprocess_predictions(
            predictions,
            conf_threshold=conf_threshold,
            nms_threshold=0.5
        )[0]
        
        pred_boxes = processed['boxes'].cpu()
        gt_boxes = target['boxes']
        
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)
            max_ious, _ = ious.max(dim=1)
            tp += (max_ious > 0.5).sum().item()
            fp += (max_ious <= 0.5).sum().item()
            gt_detected = (ious.max(dim=0)[0] > 0.5)
            fn += (~gt_detected).sum().item()
        else:
            fp += len(pred_boxes)
            fn += len(gt_boxes)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, tp, fp, fn

def find_optimal_threshold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルを読み込む
    model = AITECTDetector(num_classes=1, grid_size=16, num_anchors=3).to(device)
    model.load_state_dict(torch.load("result/aitect_model_simple.pth"))
    model.eval()
    
    # 検証データセット
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    val_dataset = CocoDataset(
        "datasets/inaoka/val/JPEGImages",
        "datasets/inaoka/val/annotations.json",
        transform=transform
    )
    
    # 異なる閾値で評価
    thresholds = np.arange(0.3, 0.5, 0.01)
    precisions = []
    recalls = []
    f1_scores = []
    
    print("Evaluating different confidence thresholds...")
    for thresh in thresholds:
        precision, recall, f1, tp, fp, fn = evaluate_at_threshold(model, val_dataset, device, thresh)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        if thresh in [0.30, 0.35, 0.40, 0.45]:
            print(f"\nThreshold {thresh:.2f}:")
            print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")
    
    # 最適な閾値を見つける
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    
    print(f"\n=== Best Threshold ===")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Precision: {precisions[best_f1_idx]:.4f}")
    print(f"Recall: {recalls[best_f1_idx]:.4f}")
    print(f"F1 Score: {f1_scores[best_f1_idx]:.4f}")
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Precision-Recall曲線
    ax1.plot(recalls, precisions, 'b-', linewidth=2)
    ax1.scatter(recalls[best_f1_idx], precisions[best_f1_idx], 
                color='red', s=100, zorder=5, label=f'Best F1 @ {best_threshold:.3f}')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 閾値 vs メトリクス
    ax2.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    ax2.plot(thresholds, recalls, 'g-', label='Recall', linewidth=2)
    ax2.plot(thresholds, f1_scores, 'r-', label='F1 Score', linewidth=2)
    ax2.axvline(best_threshold, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('Score')
    ax2.set_title('Metrics vs Confidence Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_optimization.png', dpi=150)
    print(f"\nPlot saved to: threshold_optimization.png")

if __name__ == "__main__":
    find_optimal_threshold()