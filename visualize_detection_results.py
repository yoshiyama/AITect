import torch
from model import AITECTDetector
from dataset import CocoDataset
from torchvision import transforms
from utils.postprocess import postprocess_predictions
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import json
import os

def visualize_detections_with_optimal_threshold():
    """最適化された閾値での検出結果を可視化"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 設定読み込み
    with open('config_improved_training.json', 'r') as f:
        config = json.load(f)
    
    # 最適閾値の読み込み
    with open('optimal_thresholds.json', 'r') as f:
        optimal = json.load(f)
    
    # モデルを読み込む
    model = AITECTDetector(
        num_classes=config['model']['num_classes'],
        grid_size=config['model']['grid_size'],
        num_anchors=config['model']['num_anchors']
    ).to(device)
    model.load_state_dict(torch.load(optimal['best_model']['path']))
    model.eval()
    
    optimal_threshold = optimal['best_model']['optimal_threshold']
    
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
    
    # 結果を分類
    perfect_detections = []  # TP多く、FP少ない
    good_detections = []     # TPあり、FPもある
    missed_detections = []   # FN多い
    false_detections = []    # FP多い
    
    print("Analyzing all validation images...")
    
    for idx in range(len(val_dataset)):
        image, target = val_dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(image_batch)
        
        # 最適閾値での予測
        processed = postprocess_predictions(
            predictions,
            conf_threshold=optimal_threshold,
            nms_threshold=0.5
        )[0]
        
        pred_boxes = processed['boxes'].cpu()
        gt_boxes = target['boxes']
        
        # 精度計算
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
        
        # スコア計算
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        result = {
            'idx': idx,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'pred_boxes': processed,
            'gt_boxes': target['boxes']
        }
        
        # 分類
        if tp > 0 and fp == 0 and fn == 0:
            perfect_detections.append(result)
        elif tp > 0 and fp <= tp:
            good_detections.append(result)
        elif fn > tp:
            missed_detections.append(result)
        elif fp > tp:
            false_detections.append(result)
    
    # 各カテゴリから例を選択
    categories = [
        ("Perfect Detection (100% accuracy)", perfect_detections[:3]),
        ("Good Detection (High TP, Low FP)", sorted(good_detections, key=lambda x: x['f1'], reverse=True)[:3]),
        ("Missed Detection (High FN)", sorted(missed_detections, key=lambda x: x['fn'], reverse=True)[:3]),
        ("False Detection (High FP)", sorted(false_detections, key=lambda x: x['fp'], reverse=True)[:3])
    ]
    
    # 大きな図を作成
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Detection Results with Optimal Threshold ({optimal_threshold:.3f})', fontsize=20)
    
    plot_idx = 1
    
    for cat_name, examples in categories:
        for example in examples:
            if plot_idx > 12:
                break
                
            ax = plt.subplot(4, 3, plot_idx)
            
            # 元画像を読み込み
            image_info = val_dataset.image_info[example['idx']]
            image_path = f"{val_dataset.image_dir}/{image_info['file_name'].split('/')[-1]}"
            orig_image = Image.open(image_path)
            
            ax.imshow(orig_image)
            
            # スケーリング係数
            scale_x = orig_image.width / 512
            scale_y = orig_image.height / 512
            
            # Ground Truth（緑）を描画
            for box in example['gt_boxes']:
                x1, y1, x2, y2 = box.cpu().numpy() if torch.is_tensor(box) else box
                x1, x2 = x1 * scale_x, x2 * scale_x
                y1, y2 = y1 * scale_y, y2 * scale_y
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=3, edgecolor='lime', facecolor='none',
                    label='GT' if plot_idx == 1 else None
                )
                ax.add_patch(rect)
            
            # 予測（色分け）を描画
            pred_boxes = example['pred_boxes']['boxes']
            pred_scores = example['pred_boxes']['scores']
            
            if len(pred_boxes) > 0 and len(example['gt_boxes']) > 0:
                gt_boxes_tensor = example['gt_boxes']
                if torch.is_tensor(gt_boxes_tensor):
                    gt_boxes_tensor = gt_boxes_tensor.cpu()
                else:
                    gt_boxes_tensor = torch.tensor(gt_boxes_tensor)
                ious = box_iou(pred_boxes.cpu(), gt_boxes_tensor)
                max_ious, _ = ious.max(dim=1)
            else:
                max_ious = torch.zeros(len(pred_boxes))
            
            for i, (box, score, iou) in enumerate(zip(pred_boxes, pred_scores, max_ious)):
                x1, y1, x2, y2 = box.cpu().numpy()
                x1, x2 = x1 * scale_x, x2 * scale_x
                y1, y2 = y1 * scale_y, y2 * scale_y
                
                # IoUに基づいて色を決定
                if iou > 0.5:
                    color = 'blue'  # True Positive
                    label = 'TP'
                else:
                    color = 'red'   # False Positive
                    label = 'FP'
                
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor=color, facecolor='none',
                    linestyle='--' if iou <= 0.5 else '-'
                )
                ax.add_patch(rect)
                ax.text(x1, y1-5, f'{score:.2f}', color=color, fontsize=10, weight='bold')
            
            # タイトル
            title = f'{cat_name.split("(")[0].strip()}\n'
            title += f'TP:{example["tp"]} FP:{example["fp"]} FN:{example["fn"]}'
            title += f' | F1:{example["f1"]:.3f}'
            ax.set_title(title, fontsize=11)
            ax.axis('off')
            
            plot_idx += 1
    
    # 凡例を追加
    legend_elements = [
        patches.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='lime', facecolor='none', label='Ground Truth'),
        patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='blue', facecolor='none', label='True Positive (TP)'),
        patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='False Positive (FP)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.savefig('detection_results_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: detection_results_visualization.png")
    
    # 統計サマリーも作成
    create_detection_statistics(perfect_detections, good_detections, missed_detections, false_detections)

def create_detection_statistics(perfect, good, missed, false_det):
    """検出統計の可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # カテゴリ分布
    ax = axes[0, 0]
    categories = ['Perfect\nDetection', 'Good\nDetection', 'Missed\nDetection', 'False\nDetection']
    counts = [len(perfect), len(good), len(missed), len(false_det)]
    colors = ['green', 'blue', 'orange', 'red']
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.7)
    ax.set_ylabel('Number of Images')
    ax.set_title('Detection Quality Distribution')
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom')
    
    # F1スコア分布
    ax = axes[0, 1]
    all_results = perfect + good + missed + false_det
    f1_scores = [r['f1'] for r in all_results]
    
    ax.hist(f1_scores, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(f1_scores), color='red', linestyle='--', label=f'Mean: {np.mean(f1_scores):.3f}')
    ax.set_xlabel('F1 Score')
    ax.set_ylabel('Count')
    ax.set_title('F1 Score Distribution')
    ax.legend()
    
    # TP/FP/FN統計
    ax = axes[1, 0]
    total_tp = sum(r['tp'] for r in all_results)
    total_fp = sum(r['fp'] for r in all_results)
    total_fn = sum(r['fn'] for r in all_results)
    
    labels = ['True Positives', 'False Positives', 'False Negatives']
    values = [total_tp, total_fp, total_fn]
    colors = ['green', 'red', 'orange']
    
    ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Overall Detection Statistics')
    
    # 信頼度スコア分析
    ax = axes[1, 1]
    all_scores = []
    tp_scores = []
    fp_scores = []
    
    for result in all_results:
        pred_boxes = result['pred_boxes']['boxes']
        pred_scores = result['pred_boxes']['scores']
        
        if len(pred_boxes) > 0 and len(result['gt_boxes']) > 0:
            gt_boxes_tensor = result['gt_boxes']
            if torch.is_tensor(gt_boxes_tensor):
                gt_boxes_tensor = gt_boxes_tensor.cpu()
            else:
                gt_boxes_tensor = torch.tensor(gt_boxes_tensor)
            ious = box_iou(pred_boxes.cpu(), gt_boxes_tensor)
            max_ious, _ = ious.max(dim=1)
            
            for score, iou in zip(pred_scores, max_ious):
                all_scores.append(score.item())
                if iou > 0.5:
                    tp_scores.append(score.item())
                else:
                    fp_scores.append(score.item())
        elif len(pred_boxes) > 0:
            for score in pred_scores:
                all_scores.append(score.item())
                fp_scores.append(score.item())
    
    if tp_scores:
        ax.hist(tp_scores, bins=20, alpha=0.5, label='TP scores', color='green', density=True)
    if fp_scores:
        ax.hist(fp_scores, bins=20, alpha=0.5, label='FP scores', color='red', density=True)
    
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Score Distribution by Type')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('detection_statistics.png', dpi=150, bbox_inches='tight')
    print("Statistics saved to: detection_statistics.png")
    
    # サマリー出力
    print("\n=== Detection Summary ===")
    print(f"Total images analyzed: {len(all_results)}")
    print(f"Perfect detections: {len(perfect)} ({len(perfect)/len(all_results)*100:.1f}%)")
    print(f"Good detections: {len(good)} ({len(good)/len(all_results)*100:.1f}%)")
    print(f"Missed detections: {len(missed)} ({len(missed)/len(all_results)*100:.1f}%)")
    print(f"False detections: {len(false_det)} ({len(false_det)/len(all_results)*100:.1f}%)")
    print(f"\nTotal TP: {total_tp}")
    print(f"Total FP: {total_fp}")
    print(f"Total FN: {total_fn}")
    print(f"Average F1: {np.mean(f1_scores):.3f}")

if __name__ == "__main__":
    visualize_detections_with_optimal_threshold()