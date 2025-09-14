import torch
from model import AITECTDetector
from dataset import CocoDataset
from torchvision import transforms
from utils.postprocess import postprocess_predictions
from torchvision.ops import box_iou
import numpy as np
import matplotlib.pyplot as plt
import json

def evaluate_at_threshold(model, dataset, device, conf_threshold):
    """特定の閾値での精度を評価"""
    tp, fp, fn = 0, 0, 0
    
    for idx in range(len(dataset)):
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

def optimize_threshold_for_model(model_path, config_path):
    """特定のモデルの最適閾値を見つける"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 設定読み込み
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # モデルを読み込む
    model = AITECTDetector(
        num_classes=config['model']['num_classes'],
        grid_size=config['model']['grid_size'],
        num_anchors=config['model']['num_anchors']
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 検証データセット
    transform = transforms.Compose([
        transforms.Resize((config['training']['image_size'], config['training']['image_size'])),
        transforms.ToTensor(),
    ])
    
    val_dataset = CocoDataset(
        config['paths']['val_images'],
        config['paths']['val_annotations'],
        transform=transform
    )
    
    # 異なる閾値で評価
    thresholds = np.arange(0.1, 0.6, 0.01)
    results = []
    
    print(f"Optimizing threshold for: {model_path}")
    print("Testing thresholds from 0.10 to 0.60...")
    
    for thresh in thresholds:
        precision, recall, f1, tp, fp, fn = evaluate_at_threshold(model, val_dataset, device, thresh)
        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
        
        if thresh in [0.20, 0.30, 0.39, 0.40, 0.50]:
            print(f"  Threshold {thresh:.2f}: F1={f1:.4f}, Prec={precision:.4f}, Rec={recall:.4f}")
    
    # 最適な閾値を見つける
    f1_scores = [r['f1'] for r in results]
    best_idx = np.argmax(f1_scores)
    best_result = results[best_idx]
    
    return best_result, results

def main():
    # 両方のモデルで最適閾値を探索
    models = [
        ('result/aitect_model_improved_training_best.pth', 'Best Model'),
        ('result/aitect_model_improved_training.pth', 'Final Model')
    ]
    
    all_results = {}
    
    for model_path, model_name in models:
        print(f"\n{'='*60}")
        print(f"Optimizing: {model_name}")
        print(f"{'='*60}")
        
        best_result, all_thresholds = optimize_threshold_for_model(
            model_path, 
            'config_improved_training.json'
        )
        
        all_results[model_name] = {
            'best': best_result,
            'all': all_thresholds
        }
        
        print(f"\nBest threshold for {model_name}: {best_result['threshold']:.3f}")
        print(f"Best F1 Score: {best_result['f1']:.4f}")
        print(f"Precision: {best_result['precision']:.4f}")
        print(f"Recall: {best_result['recall']:.4f}")
        print(f"TP: {best_result['tp']}, FP: {best_result['fp']}, FN: {best_result['fn']}")
    
    # プロット作成
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        thresholds = [r['threshold'] for r in results['all']]
        precisions = [r['precision'] for r in results['all']]
        recalls = [r['recall'] for r in results['all']]
        f1_scores = [r['f1'] for r in results['all']]
        
        ax.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
        ax.plot(thresholds, recalls, 'g-', label='Recall', linewidth=2)
        ax.plot(thresholds, f1_scores, 'r-', label='F1 Score', linewidth=2)
        
        # 最適点をマーク
        best = results['best']
        ax.axvline(best['threshold'], color='black', linestyle='--', alpha=0.5)
        ax.scatter([best['threshold']], [best['f1']], color='red', s=100, zorder=5)
        
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'{model_name} - Optimal: {best["threshold"]:.3f} (F1: {best["f1"]:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.1, 0.6)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('final_threshold_optimization.png', dpi=150)
    print(f"\nOptimization plot saved to: final_threshold_optimization.png")
    
    # 推奨設定を保存
    recommendations = {
        'best_model': {
            'path': 'result/aitect_model_improved_training_best.pth',
            'optimal_threshold': all_results['Best Model']['best']['threshold'],
            'expected_f1': all_results['Best Model']['best']['f1'],
            'expected_precision': all_results['Best Model']['best']['precision'],
            'expected_recall': all_results['Best Model']['best']['recall']
        },
        'final_model': {
            'path': 'result/aitect_model_improved_training.pth',
            'optimal_threshold': all_results['Final Model']['best']['threshold'],
            'expected_f1': all_results['Final Model']['best']['f1'],
            'expected_precision': all_results['Final Model']['best']['precision'],
            'expected_recall': all_results['Final Model']['best']['recall']
        }
    }
    
    with open('optimal_thresholds.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print("\nRecommendations saved to: optimal_thresholds.json")

if __name__ == "__main__":
    main()