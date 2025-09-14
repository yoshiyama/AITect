import torch
from model import AITECTDetector
from dataset import CocoDataset
from torchvision import transforms
from utils.postprocess import postprocess_predictions
from torchvision.ops import box_iou
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def evaluate_model(model_path, config_path, visualize=True):
    """改善されたモデルの評価"""
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
    
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError as e:
            # チェックポイント形式の場合
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise e
    else:
        print(f"Model file not found: {model_path}")
        return
    
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
    
    # 評価用の閾値
    conf_threshold = config['evaluation']['conf_threshold']
    
    # 全体の統計
    all_predictions = []
    all_targets = []
    
    print(f"\nEvaluating on {len(val_dataset)} validation images...")
    print(f"Using confidence threshold: {conf_threshold}")
    
    for idx in range(len(val_dataset)):
        image, target = val_dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(image_batch)
        
        processed = postprocess_predictions(
            predictions,
            conf_threshold=conf_threshold,
            nms_threshold=config['evaluation']['iou_threshold']
        )[0]
        
        all_predictions.append(processed)
        all_targets.append(target)
    
    # 精度計算
    tp, fp, fn = 0, 0, 0
    detection_results = []
    
    for pred, target in zip(all_predictions, all_targets):
        pred_boxes = pred['boxes'].cpu()
        gt_boxes = target['boxes']
        
        sample_tp, sample_fp, sample_fn = 0, 0, len(gt_boxes)
        
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)
            max_ious, _ = ious.max(dim=1)
            sample_tp = (max_ious > 0.5).sum().item()
            sample_fp = (max_ious <= 0.5).sum().item()
            gt_detected = (ious.max(dim=0)[0] > 0.5)
            sample_fn = (~gt_detected).sum().item()
        elif len(pred_boxes) > 0:
            sample_fp = len(pred_boxes)
        
        tp += sample_tp
        fp += sample_fp
        fn += sample_fn
        
        detection_results.append({
            'tp': sample_tp,
            'fp': sample_fp,
            'fn': sample_fn,
            'num_pred': len(pred_boxes),
            'num_gt': len(gt_boxes)
        })
    
    # メトリクス計算
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 結果表示
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {model_path}")
    print(f"Total Ground Truth: {tp + fn}")
    print(f"Total Predictions: {tp + fp}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"mAP@0.5: {precision:.4f}")  # 簡易版
    
    # 統計情報
    avg_pred_per_image = np.mean([r['num_pred'] for r in detection_results])
    avg_gt_per_image = np.mean([r['num_gt'] for r in detection_results])
    
    print(f"\nAverage predictions per image: {avg_pred_per_image:.2f}")
    print(f"Average ground truth per image: {avg_gt_per_image:.2f}")
    
    # 可視化
    if visualize:
        visualize_predictions(val_dataset, all_predictions, all_targets, detection_results, model_path)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def visualize_predictions(dataset, predictions, targets, detection_results, model_name):
    """予測結果の可視化"""
    # 最も良い結果と悪い結果を選択
    f1_scores = []
    for result in detection_results:
        tp = result['tp']
        fp = result['fp']
        fn = result['fn']
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)
    
    f1_scores = np.array(f1_scores)
    best_indices = np.argsort(f1_scores)[-3:]  # 上位3つ
    worst_indices = np.argsort(f1_scores)[:3]  # 下位3つ
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    sample_indices = list(worst_indices) + list(best_indices)
    titles = ['Worst 1', 'Worst 2', 'Worst 3', 'Best 1', 'Best 2', 'Best 3']
    
    for ax_idx, (sample_idx, title) in enumerate(zip(sample_indices, titles)):
        ax = axes[ax_idx]
        
        # 元画像を読み込み
        image_info = dataset.image_info[sample_idx]
        image_path = f"{dataset.image_dir}/{image_info['file_name'].split('/')[-1]}"
        orig_image = Image.open(image_path)
        
        ax.imshow(orig_image)
        
        # スケーリング係数
        scale_x = orig_image.width / 512
        scale_y = orig_image.height / 512
        
        # GT（緑）
        target = targets[sample_idx]
        for box in target['boxes']:
            x1, y1, x2, y2 = box.cpu().numpy() if torch.is_tensor(box) else box
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
        
        # 予測（赤）
        pred = predictions[sample_idx]
        for box, score in zip(pred['boxes'], pred['scores']):
            x1, y1, x2, y2 = box.cpu().numpy()
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=8)
        
        result = detection_results[sample_idx]
        ax.set_title(f'{title} - TP:{result["tp"]}, FP:{result["fp"]}, FN:{result["fn"]}')
        ax.axis('off')
    
    plt.suptitle(f'Model: {os.path.basename(model_name)}', fontsize=16)
    plt.tight_layout()
    
    output_name = f'evaluation_{os.path.basename(model_name).replace(".pth", "")}.png'
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_name}")

def compare_models():
    """複数のモデルを比較"""
    models_to_evaluate = [
        {
            'path': 'result/aitect_model_simple.pth',
            'config': 'config.json',
            'name': 'Original Model'
        },
        {
            'path': 'result/aitect_model_improved_training_best.pth',
            'config': 'config_improved_training.json',
            'name': 'Improved Model (Best)'
        },
        {
            'path': 'result/aitect_model_improved_training.pth',
            'config': 'config_improved_training.json',
            'name': 'Improved Model (Latest)'
        }
    ]
    
    results = []
    
    for model_info in models_to_evaluate:
        if os.path.exists(model_info['path']):
            print(f"\n{'='*60}")
            print(f"Evaluating: {model_info['name']}")
            print(f"{'='*60}")
            
            result = evaluate_model(
                model_info['path'], 
                model_info['config'],
                visualize=(model_info['name'] == 'Improved Model (Best)')
            )
            
            if result:
                result['name'] = model_info['name']
                results.append(result)
        else:
            print(f"\nSkipping {model_info['name']} - file not found: {model_info['path']}")
    
    # 結果の比較表示
    if results:
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(f"{'Model':<30} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'TP':<8} {'FP':<8} {'FN':<8}")
        print("-"*80)
        
        for r in results:
            print(f"{r['name']:<30} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1']:<10.4f} "
                  f"{r['tp']:<8} {r['fp']:<8} {r['fn']:<8}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        compare_models()
    else:
        # デフォルトは最新の改善モデルを評価
        if os.path.exists('result/aitect_model_improved_training_best.pth'):
            evaluate_model('result/aitect_model_improved_training_best.pth', 'config_improved_training.json')
        else:
            print("Improved model not found yet. Evaluating original model...")
            evaluate_model('result/aitect_model_simple.pth', 'config.json')