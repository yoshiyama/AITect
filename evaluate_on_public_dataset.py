import torch
from model import AITECTDetector
from dataset import CocoDataset
from torchvision import transforms
from utils.postprocess import postprocess_predictions
from torchvision.ops import box_iou
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os

def evaluate_on_dataset(dataset_path, dataset_name="Public Dataset"):
    """公開データセットでの評価"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルと最適閾値を読み込み
    with open('optimal_thresholds.json', 'r') as f:
        optimal = json.load(f)
    
    model_path = optimal['best_model']['path']
    threshold = optimal['best_model']['optimal_threshold']
    
    print(f"=== {dataset_name} での評価 ===")
    print(f"モデル: {model_path}")
    print(f"最適閾値: {threshold}")
    
    # モデル読み込み
    model = AITECTDetector(num_classes=1, grid_size=16, num_anchors=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # データセット読み込み
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    test_dataset = CocoDataset(
        f"{dataset_path}/images",
        f"{dataset_path}/annotations.json",
        transform=transform
    )
    
    print(f"テスト画像数: {len(test_dataset)}")
    
    # 評価
    results = []
    total_tp, total_fp, total_fn = 0, 0, 0
    
    # 様々な閾値でテスト
    thresholds_to_test = [0.3, 0.4, threshold, 0.6, 0.7]
    threshold_results = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in thresholds_to_test}
    
    for idx in range(len(test_dataset)):
        image, target = test_dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(image_batch)
        
        # 各閾値で評価
        for test_thresh in thresholds_to_test:
            processed = postprocess_predictions(
                predictions,
                conf_threshold=test_thresh,
                nms_threshold=0.5
            )[0]
            
            pred_boxes = processed['boxes'].cpu()
            gt_boxes = target['boxes']
            
            tp, fp, fn = 0, 0, len(gt_boxes)
            
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                ious = box_iou(pred_boxes, gt_boxes)
                max_ious, _ = ious.max(dim=1)
                tp = (max_ious > 0.5).sum().item()
                fp = (max_ious <= 0.5).sum().item()
                gt_detected = (ious.max(dim=0)[0] > 0.5)
                fn = (~gt_detected).sum().item()
            elif len(pred_boxes) > 0:
                fp = len(pred_boxes)
            
            threshold_results[test_thresh]['tp'] += tp
            threshold_results[test_thresh]['fp'] += fp
            threshold_results[test_thresh]['fn'] += fn
            
            # 最適閾値での詳細結果を保存
            if test_thresh == threshold:
                results.append({
                    'idx': idx,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'num_pred': len(pred_boxes),
                    'num_gt': len(gt_boxes)
                })
                total_tp += tp
                total_fp += fp
                total_fn += fn
    
    # 結果表示
    print("\n=== 閾値別の性能 ===")
    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 40)
    
    best_f1 = 0
    best_thresh = 0
    
    for thresh in thresholds_to_test:
        tp = threshold_results[thresh]['tp']
        fp = threshold_results[thresh]['fp']
        fn = threshold_results[thresh]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{thresh:<10.2f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    print(f"\n最適閾値: {best_thresh} (F1: {best_f1:.4f})")
    
    # 最適閾値での全体性能
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n=== 白線検出モデルの{dataset_name}での性能 ===")
    print(f"Total TP: {total_tp}")
    print(f"Total FP: {total_fp}")
    print(f"Total FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # 可視化
    visualize_results(test_dataset, results, model, threshold, device, dataset_name)
    
    return results

def visualize_results(dataset, results, model, threshold, device, dataset_name):
    """結果の可視化"""
    # F1スコアでソート
    for r in results:
        tp, fp, fn = r['tp'], r['fp'], r['fn']
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        r['f1'] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)
    
    # 上位と下位の例
    examples = sorted_results[:3] + sorted_results[-3:]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{dataset_name} - Detection Results (Threshold: {threshold:.2f})', fontsize=16)
    axes = axes.ravel()
    
    for i, result in enumerate(examples[:6]):
        idx = result['idx']
        image, target = dataset[idx]
        
        # 元画像
        image_info = dataset.image_info[idx]
        image_path = f"{dataset.image_dir}/{image_info['file_name']}"
        orig_image = Image.open(image_path)
        
        # 予測
        image_batch = image.unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = model(image_batch)
        
        processed = postprocess_predictions(
            predictions,
            conf_threshold=threshold,
            nms_threshold=0.5
        )[0]
        
        ax = axes[i]
        ax.imshow(orig_image)
        
        # スケーリング
        scale_x = orig_image.width / 512
        scale_y = orig_image.height / 512
        
        # GT (緑)
        for box in target['boxes']:
            x1, y1, x2, y2 = box.cpu().numpy() if torch.is_tensor(box) else box
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
        
        # 予測 (赤)
        for box, score in zip(processed['boxes'], processed['scores']):
            x1, y1, x2, y2 = box.cpu().numpy()
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=10)
        
        quality = "Best" if i < 3 else "Worst"
        ax.set_title(f'{quality} {i%3 + 1} - F1:{result["f1"]:.3f} (TP:{result["tp"]} FP:{result["fp"]} FN:{result["fn"]})')
        ax.axis('off')
    
    plt.tight_layout()
    output_name = f'{dataset_name.lower().replace(" ", "_")}_evaluation.png'
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"\n可視化を保存: {output_name}")

def compare_datasets():
    """複数のデータセットで比較評価"""
    datasets = [
        ("./datasets/simple_shapes", "Simple Shapes")
    ]
    
    # 元のデータセットがあれば追加
    if os.path.exists("./datasets/inaoka/val/JPEGImages"):
        datasets.insert(0, ("./datasets/inaoka/val", "White Line (Original)"))
    
    # Pascal VOCがあれば追加
    if os.path.exists("./datasets/voc2007"):
        datasets.append(("./datasets/voc2007/VOCdevkit/VOC2007", "Pascal VOC 2007"))
    
    all_results = {}
    
    for dataset_path, dataset_name in datasets:
        if os.path.exists(dataset_path):
            print(f"\n{'='*60}")
            results = evaluate_on_dataset(dataset_path, dataset_name)
            all_results[dataset_name] = results
        else:
            print(f"\n{dataset_name} not found at {dataset_path}")
    
    # 比較サマリー
    print("\n" + "="*60)
    print("データセット間の比較")
    print("="*60)
    
    for name, results in all_results.items():
        total_tp = sum(r['tp'] for r in results)
        total_fp = sum(r['fp'] for r in results)
        total_fn = sum(r['fn'] for r in results)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{name}:")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")

if __name__ == "__main__":
    # 単一データセット評価
    if os.path.exists("./datasets/simple_shapes"):
        evaluate_on_dataset("./datasets/simple_shapes", "Simple Shapes")
    
    # 複数データセット比較
    print("\n\n" + "="*60)
    print("複数データセットでの比較評価")
    print("="*60)
    compare_datasets()