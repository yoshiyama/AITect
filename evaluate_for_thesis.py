import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CocoDataset
from model_whiteline import WhiteLineDetector
from utils.bbox import box_iou, nms
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import os
from collections import defaultdict
import pandas as pd
from tabulate import tabulate

def load_config(config_path="config.json"):
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class ComprehensiveEvaluator:
    """卒業研究用の包括的評価クラス"""
    
    def __init__(self, model, device, conf_thresholds=[0.3, 0.5, 0.7], iou_thresholds=[0.5, 0.75]):
        self.model = model
        self.device = device
        self.conf_thresholds = conf_thresholds
        self.iou_thresholds = iou_thresholds
        self.results = defaultdict(list)
        
    def evaluate_single_image(self, image, target):
        """単一画像の評価"""
        self.model.eval()
        
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(self.device)
            output = self.model(image_tensor)[0]  # [N, 5]
            
        scores = torch.sigmoid(output[:, 4])
        boxes = output[:, :4]
        
        # xywhからxyxyに変換
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        pred_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        
        gt_boxes = target['boxes'].to(self.device)
        
        return {
            'pred_boxes': pred_boxes_xyxy,
            'pred_scores': scores,
            'gt_boxes': gt_boxes,
            'raw_output': output
        }
    
    def calculate_metrics_at_threshold(self, predictions, gt_boxes, conf_threshold, iou_threshold):
        """特定の閾値でのメトリクス計算"""
        # 信頼度でフィルタリング
        mask = predictions['pred_scores'] > conf_threshold
        
        if mask.sum() == 0:
            return {
                'tp': 0, 'fp': 0, 'fn': len(gt_boxes),
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'ap': 0.0, 'ious': []
            }
        
        filtered_boxes = predictions['pred_boxes'][mask]
        filtered_scores = predictions['pred_scores'][mask]
        
        # NMS適用
        keep = nms(filtered_boxes, filtered_scores, 0.5)
        final_boxes = filtered_boxes[keep]
        final_scores = filtered_scores[keep]
        
        if len(gt_boxes) == 0:
            return {
                'tp': 0, 'fp': len(final_boxes), 'fn': 0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'ap': 0.0, 'ious': []
            }
        
        # IoU計算
        ious = box_iou(final_boxes, gt_boxes)
        
        # マッチング
        tp = 0
        fp = 0
        matched_gts = set()
        ious_list = []
        
        if len(final_boxes) > 0:
            # 各予測に対して最もIoUが高いGTを見つける
            max_ious, matched_gt_indices = ious.max(dim=1)
            
            for i, (iou, gt_idx) in enumerate(zip(max_ious, matched_gt_indices)):
                if iou >= iou_threshold and gt_idx.item() not in matched_gts:
                    tp += 1
                    matched_gts.add(gt_idx.item())
                    ious_list.append(iou.item())
                else:
                    fp += 1
        
        fn = len(gt_boxes) - len(matched_gts)
        
        # メトリクス計算
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'ious': ious_list
        }
    
    def calculate_ap(self, all_predictions, all_gt_boxes, iou_threshold=0.5):
        """Average Precision計算"""
        # すべての予測を信頼度でソート
        all_scores = []
        all_labels = []
        all_ious = []
        
        for pred, gt_boxes in zip(all_predictions, all_gt_boxes):
            if len(pred['pred_boxes']) == 0:
                continue
                
            if len(gt_boxes) == 0:
                # GTなし - すべてFP
                all_scores.extend(pred['pred_scores'].cpu().numpy())
                all_labels.extend([0] * len(pred['pred_scores']))
                continue
            
            # IoU計算
            ious = box_iou(pred['pred_boxes'], gt_boxes)
            max_ious, matched_gt = ious.max(dim=1)
            
            # 各予測のラベル付け
            for score, max_iou in zip(pred['pred_scores'], max_ious):
                all_scores.append(score.cpu().item())
                all_labels.append(1 if max_iou >= iou_threshold else 0)
                all_ious.append(max_iou.cpu().item())
        
        if len(all_scores) == 0:
            return 0.0
        
        # 信頼度で降順ソート
        sorted_indices = np.argsort(all_scores)[::-1]
        sorted_labels = np.array(all_labels)[sorted_indices]
        
        # Precision-Recall計算
        tp_cumsum = np.cumsum(sorted_labels)
        fp_cumsum = np.cumsum(1 - sorted_labels)
        
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / sum(len(gt) for gt in all_gt_boxes)
        
        # AP計算（11点補間）
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        return ap
    
    def evaluate_dataset(self, dataset, num_samples=None):
        """データセット全体の評価"""
        if num_samples is None:
            num_samples = len(dataset)
        else:
            num_samples = min(num_samples, len(dataset))
        
        all_predictions = []
        all_gt_boxes = []
        
        print(f"評価中... (サンプル数: {num_samples})")
        
        for i in range(num_samples):
            if i % 10 == 0:
                print(f"  進捗: {i}/{num_samples}")
            
            image, target = dataset[i]
            result = self.evaluate_single_image(image, target)
            
            all_predictions.append({
                'pred_boxes': result['pred_boxes'].cpu(),
                'pred_scores': result['pred_scores'].cpu(),
            })
            all_gt_boxes.append(result['gt_boxes'].cpu())
        
        # 各閾値での評価
        results_by_threshold = {}
        
        for conf_thresh in self.conf_thresholds:
            results_by_threshold[f'conf_{conf_thresh}'] = {}
            
            for iou_thresh in self.iou_thresholds:
                total_tp = 0
                total_fp = 0
                total_fn = 0
                all_ious = []
                
                for pred, gt_boxes in zip(all_predictions, all_gt_boxes):
                    metrics = self.calculate_metrics_at_threshold(
                        pred, gt_boxes, conf_thresh, iou_thresh
                    )
                    total_tp += metrics['tp']
                    total_fp += metrics['fp']
                    total_fn += metrics['fn']
                    all_ious.extend(metrics['ious'])
                
                # 全体のメトリクス
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
                    mean_iou = std_iou = min_iou = max_iou = median_iou = 0
                
                results_by_threshold[f'conf_{conf_thresh}'][f'iou_{iou_thresh}'] = {
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
                    'iou_samples': len(all_ious),
                    'all_ious': all_ious  # 分布プロット用に保存
                }
        
        # AP計算
        ap_scores = {}
        for iou_thresh in self.iou_thresholds:
            ap = self.calculate_ap(all_predictions, all_gt_boxes, iou_thresh)
            ap_scores[f'AP@{iou_thresh}'] = ap
        
        # mAP
        mAP = np.mean(list(ap_scores.values()))
        
        return {
            'results_by_threshold': results_by_threshold,
            'ap_scores': ap_scores,
            'mAP': mAP,
            'total_images': num_samples,
            'total_gt_boxes': sum(len(gt) for gt in all_gt_boxes),
            'all_predictions': all_predictions,
            'all_gt_boxes': all_gt_boxes
        }

def create_evaluation_report(results, save_path='thesis_evaluation_report.txt'):
    """卒業研究用の評価レポート作成"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("物体検出モデル評価レポート\n")
        f.write(f"作成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # 基本情報
        f.write("【評価概要】\n")
        f.write(f"評価画像数: {results['total_images']} 枚\n")
        f.write(f"総GT数: {results['total_gt_boxes']} 個\n\n")
        
        # AP/mAP
        f.write("【Average Precision (AP)】\n")
        for key, value in results['ap_scores'].items():
            f.write(f"{key}: {value:.4f}\n")
        f.write(f"mAP: {results['mAP']:.4f}\n\n")
        
        # 各閾値での結果
        f.write("【信頼度閾値・IoU閾値別の評価結果】\n\n")
        
        for conf_key, iou_results in results['results_by_threshold'].items():
            conf_value = conf_key.split('_')[1]
            f.write(f"■ 信頼度閾値: {conf_value}\n")
            
            # テーブル形式で出力
            headers = ['IoU閾値', 'Precision', 'Recall', 'F1-Score', 'TP', 'FP', 'FN', '平均IoU', '標準偏差', '最小IoU', '最大IoU']
            table_data = []
            
            for iou_key, metrics in iou_results.items():
                iou_value = iou_key.split('_')[1]
                row = [
                    iou_value,
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1']:.4f}",
                    metrics['tp'],
                    metrics['fp'],
                    metrics['fn'],
                    f"{metrics['mean_iou']:.4f}",
                    f"{metrics['std_iou']:.4f}",
                    f"{metrics['min_iou']:.4f}",
                    f"{metrics['max_iou']:.4f}"
                ]
                table_data.append(row)
            
            f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
            f.write("\n\n")
        
        # 推奨設定
        f.write("【推奨設定】\n")
        
        # F1スコアが最大の設定を見つける
        best_f1 = 0
        best_config = None
        for conf_key, iou_results in results['results_by_threshold'].items():
            for iou_key, metrics in iou_results.items():
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_config = (conf_key.split('_')[1], iou_key.split('_')[1])
        
        if best_config:
            f.write(f"最高F1スコア設定: 信頼度閾値={best_config[0]}, IoU閾値={best_config[1]}, F1={best_f1:.4f}\n")
        
        # 用途別推奨
        f.write("\n用途別推奨設定:\n")
        f.write("- 高精度重視（誤検出最小化）: 信頼度閾値=0.7, IoU閾値=0.75\n")
        f.write("- バランス重視: 信頼度閾値=0.5, IoU閾値=0.5\n")
        f.write("- 高再現率重視（見逃し最小化）: 信頼度閾値=0.3, IoU閾値=0.5\n")

def create_visualization_plots(results, save_dir='thesis_evaluation_plots'):
    """評価結果の可視化"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Precision-Recall曲線
    plt.figure(figsize=(10, 8))
    
    for iou_thresh in [0.5, 0.75]:
        precisions = []
        recalls = []
        
        for conf_key, iou_results in results['results_by_threshold'].items():
            metrics = iou_results[f'iou_{iou_thresh}']
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
        
        # Recallでソート
        sorted_indices = np.argsort(recalls)
        recalls = np.array(recalls)[sorted_indices]
        precisions = np.array(precisions)[sorted_indices]
        
        plt.plot(recalls, precisions, 'o-', linewidth=2, markersize=8, 
                label=f'IoU={iou_thresh}')
    
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(f'{save_dir}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. F1スコアのヒートマップ
    conf_thresholds = [0.3, 0.5, 0.7]
    iou_thresholds = [0.5, 0.75]
    
    f1_matrix = np.zeros((len(conf_thresholds), len(iou_thresholds)))
    
    for i, conf in enumerate(conf_thresholds):
        for j, iou in enumerate(iou_thresholds):
            f1_matrix[i, j] = results['results_by_threshold'][f'conf_{conf}'][f'iou_{iou}']['f1']
    
    plt.figure(figsize=(8, 6))
    plt.imshow(f1_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='F1 Score')
    
    # ラベルとアノテーション
    plt.xticks(range(len(iou_thresholds)), [f'IoU={t}' for t in iou_thresholds])
    plt.yticks(range(len(conf_thresholds)), [f'Conf={t}' for t in conf_thresholds])
    
    # 値を表示
    for i in range(len(conf_thresholds)):
        for j in range(len(iou_thresholds)):
            plt.text(j, i, f'{f1_matrix[i, j]:.3f}', 
                    ha='center', va='center', color='black', fontsize=12)
    
    plt.xlabel('IoU Threshold', fontsize=14)
    plt.ylabel('Confidence Threshold', fontsize=14)
    plt.title('F1 Score Heatmap', fontsize=16)
    plt.savefig(f'{save_dir}/f1_score_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 検出統計
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # TP/FP/FN棒グラフ
    ax = axes[0, 0]
    conf_labels = []
    tp_values = []
    fp_values = []
    fn_values = []
    
    for conf in [0.3, 0.5, 0.7]:
        metrics = results['results_by_threshold'][f'conf_{conf}']['iou_0.5']
        conf_labels.append(f'Conf={conf}')
        tp_values.append(metrics['tp'])
        fp_values.append(metrics['fp'])
        fn_values.append(metrics['fn'])
    
    x = np.arange(len(conf_labels))
    width = 0.25
    
    ax.bar(x - width, tp_values, width, label='TP', color='green')
    ax.bar(x, fp_values, width, label='FP', color='red')
    ax.bar(x + width, fn_values, width, label='FN', color='orange')
    
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Count')
    ax.set_title('Detection Statistics (IoU=0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(conf_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # AP棒グラフ
    ax = axes[0, 1]
    ap_labels = list(results['ap_scores'].keys())
    ap_values = list(results['ap_scores'].values())
    
    bars = ax.bar(ap_labels + ['mAP'], ap_values + [results['mAP']], 
                   color=['blue', 'navy', 'darkblue'])
    
    # 値を表示
    for bar, value in zip(bars, ap_values + [results['mAP']]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('Average Precision')
    ax.set_title('Average Precision Scores')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # IoU分布
    ax = axes[1, 0]
    
    # 代表的な設定でのIoU分布を表示
    iou_data = results['results_by_threshold']['conf_0.5']['iou_0.5']['all_ious']
    
    if iou_data:
        ax.hist(iou_data, bins=20, range=(0.5, 1.0), alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.mean(iou_data), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(iou_data):.3f}')
        ax.axvline(np.median(iou_data), color='green', linestyle='--', linewidth=2, label=f'Median={np.median(iou_data):.3f}')
        ax.legend()
    
    ax.set_title('IoU Distribution of True Positives (Conf=0.5, IoU≥0.5)')
    ax.set_xlabel('IoU Value')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    
    # 混同行列風の表示
    ax = axes[1, 1]
    ax.axis('off')
    
    # 代表的な設定での結果を表示
    metrics = results['results_by_threshold']['conf_0.5']['iou_0.5']
    
    summary_text = f"""
    代表的な設定での結果 (Conf=0.5, IoU=0.5)
    
    Precision: {metrics['precision']:.3f}
    Recall: {metrics['recall']:.3f}
    F1-Score: {metrics['f1']:.3f}
    
    True Positives: {metrics['tp']}
    False Positives: {metrics['fp']}
    False Negatives: {metrics['fn']}
    
    平均IoU (TP only): {metrics['mean_iou']:.3f}
    IoU標準偏差: {metrics['std_iou']:.3f}
    IoU範囲: [{metrics['min_iou']:.3f}, {metrics['max_iou']:.3f}]
    IoU中央値: {metrics['median_iou']:.3f}
    """
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/evaluation_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可視化結果を {save_dir}/ に保存しました")

def visualize_detection_samples(model, dataset, device, save_dir='thesis_detection_samples', num_samples=5):
    """検出結果のサンプル画像"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # 代表的な閾値
    conf_threshold = 0.5
    
    for i in range(min(num_samples, len(dataset))):
        image, target = dataset[i]
        
        # 予測
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)[0]
        
        scores = torch.sigmoid(output[:, 4])
        mask = scores > conf_threshold
        
        if mask.sum() == 0:
            continue
        
        filtered_boxes = output[mask, :4]
        filtered_scores = scores[mask]
        
        # xyxy変換
        x1 = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2
        y1 = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2
        x2 = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2
        y2 = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2
        pred_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        
        # NMS
        keep = nms(pred_boxes_xyxy, filtered_scores, 0.5)
        final_boxes = pred_boxes_xyxy[keep]
        final_scores = filtered_scores[keep]
        
        # 可視化
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # 画像を表示（0-1に正規化）
        img_display = image.permute(1, 2, 0).cpu().numpy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        ax.imshow(img_display)
        
        # GT描画（緑）
        gt_boxes = target['boxes']
        for gt_box in gt_boxes:
            # 512x512にスケール
            x1 = gt_box[0] * 512 / image.shape[2]
            y1 = gt_box[1] * 512 / image.shape[1]
            w = (gt_box[2] - gt_box[0]) * 512 / image.shape[2]
            h = (gt_box[3] - gt_box[1]) * 512 / image.shape[1]
            
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=3, edgecolor='green', facecolor='none',
                linestyle='-', label='Ground Truth'
            )
            ax.add_patch(rect)
        
        # 予測描画（赤）
        for box, score in zip(final_boxes, final_scores):
            x1, y1, x2, y2 = box.cpu().numpy()
            w = x2 - x1
            h = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=3, edgecolor='red', facecolor='none',
                linestyle='-'
            )
            ax.add_patch(rect)
            
            # スコア表示
            ax.text(x1, y1-5, f'{score.item():.2f}', 
                   color='red', fontsize=16, weight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.set_title(f'Sample {i+1}: GT={len(gt_boxes)}, Pred={len(final_boxes)}', fontsize=16)
        ax.axis('off')
        
        # 凡例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=3, label='Ground Truth'),
            Line2D([0], [0], color='red', lw=3, label='Prediction')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/detection_sample_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """卒業研究用の包括的評価メイン関数"""
    print("="*60)
    print("卒業研究用 物体検出モデル評価プログラム")
    print("="*60)
    
    # 設定
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデル読み込み
    model = WhiteLineDetector(grid_size=8, num_anchors=3).to(device)
    model_path = "result/aitect_model_improved.pth"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\nモデルを読み込みました: {model_path}")
    except:
        print(f"\nエラー: {model_path} が見つかりません")
        print("先に python train_with_improvements.py で学習を実行してください")
        return
    
    # データセット
    val_image_dir = config['paths']['val_images']
    val_annotation_path = config['paths']['val_annotations']
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    print(f"\n検証データセットを読み込み中...")
    val_dataset = CocoDataset(val_image_dir, val_annotation_path, transform=transform)
    print(f"検証データ数: {len(val_dataset)} 枚")
    
    # 評価実行
    print("\n包括的評価を実行中...")
    evaluator = ComprehensiveEvaluator(
        model, device,
        conf_thresholds=[0.3, 0.5, 0.7],
        iou_thresholds=[0.5, 0.75]
    )
    
    results = evaluator.evaluate_dataset(val_dataset)
    
    # レポート作成
    print("\n評価レポートを作成中...")
    create_evaluation_report(results, 'thesis_evaluation_report.txt')
    print("評価レポートを thesis_evaluation_report.txt に保存しました")
    
    # 可視化
    print("\n評価結果を可視化中...")
    create_visualization_plots(results, 'thesis_evaluation_plots')
    
    # 検出サンプル
    print("\n検出サンプル画像を生成中...")
    visualize_detection_samples(model, val_dataset, device, 'thesis_detection_samples', num_samples=10)
    
    # 結果のサマリー表示
    print("\n" + "="*60)
    print("評価結果サマリー")
    print("="*60)
    print(f"総画像数: {results['total_images']}")
    print(f"総GT数: {results['total_gt_boxes']}")
    print(f"\n【Average Precision】")
    for key, value in results['ap_scores'].items():
        print(f"  {key}: {value:.4f}")
    print(f"  mAP: {results['mAP']:.4f}")
    
    print(f"\n【代表的な設定での結果 (Conf=0.5, IoU=0.5)】")
    metrics = results['results_by_threshold']['conf_0.5']['iou_0.5']
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    
    print("\n" + "="*60)
    print("評価完了！")
    print("生成されたファイル:")
    print("  - thesis_evaluation_report.txt (詳細レポート)")
    print("  - thesis_evaluation_plots/ (グラフ・図表)")
    print("  - thesis_detection_samples/ (検出結果サンプル)")
    print("="*60)

if __name__ == "__main__":
    main()