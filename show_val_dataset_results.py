import torch
from model import AITECTDetector
from dataset import CocoDataset
from torchvision import transforms
from utils.postprocess import postprocess_predictions
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import os

def show_validation_results():
    """検証データセットの全画像に対する検出結果を表示"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 最適化されたモデルと設定を使用
    with open('optimal_thresholds.json', 'r') as f:
        optimal = json.load(f)
    
    model_path = optimal['best_model']['path']
    threshold = optimal['best_model']['optimal_threshold']
    
    # モデル読み込み
    model = AITECTDetector(num_classes=1, grid_size=16, num_anchors=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # データセット
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    val_dataset = CocoDataset(
        "datasets/inaoka/val/JPEGImages",
        "datasets/inaoka/val/annotations.json",
        transform=transform
    )
    
    print(f"検証データセット内の画像数: {len(val_dataset)}")
    print(f"使用モデル: {model_path}")
    print(f"信頼度閾値: {threshold}")
    print("\n各画像の検出結果を生成中...")
    
    # 結果を保存するディレクトリ
    output_dir = "val_detection_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 統計情報
    total_tp = 0
    total_fp = 0
    total_fn = 0
    results_summary = []
    
    # 各画像を処理
    for idx in range(len(val_dataset)):
        # データ読み込み
        image, target = val_dataset[idx]
        image_info = val_dataset.image_info[idx]
        image_name = image_info['file_name'].split('/')[-1]
        
        # 元画像
        image_path = f"{val_dataset.image_dir}/{image_name}"
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
        
        # 精度計算
        pred_boxes = processed['boxes'].cpu()
        pred_scores = processed['scores'].cpu()
        gt_boxes = target['boxes']
        
        tp, fp, fn = 0, 0, len(gt_boxes)
        
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            from torchvision.ops import box_iou
            ious = box_iou(pred_boxes, gt_boxes)
            max_ious, _ = ious.max(dim=1)
            tp = (max_ious > 0.5).sum().item()
            fp = (max_ious <= 0.5).sum().item()
            gt_detected = (ious.max(dim=0)[0] > 0.5)
            fn = (~gt_detected).sum().item()
        elif len(pred_boxes) > 0:
            fp = len(pred_boxes)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # 結果を記録
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results_summary.append({
            'image': image_name,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_gt': len(gt_boxes),
            'num_pred': len(pred_boxes)
        })
        
        # 画像を作成
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(orig_image)
        
        # スケーリング係数
        scale_x = orig_image.width / 512
        scale_y = orig_image.height / 512
        
        # Ground Truth (緑)
        for box in gt_boxes:
            x1, y1, x2, y2 = box.cpu().numpy() if torch.is_tensor(box) else box
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
        
        # 予測 (青=TP, 赤=FP)
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)
            max_ious, _ = ious.max(dim=1)
        else:
            max_ious = torch.zeros(len(pred_boxes))
        
        for i, (box, score, iou) in enumerate(zip(pred_boxes, pred_scores, max_ious)):
            x1, y1, x2, y2 = box.numpy()
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            
            color = 'blue' if iou > 0.5 else 'red'
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=color, facecolor='none',
                linestyle='-' if iou > 0.5 else '--'
            )
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{score:.2f}', color=color, fontsize=10, weight='bold',
                   bbox=dict(facecolor='white', alpha=0.7))
        
        # タイトルと情報
        ax.set_title(f'{image_name}\nTP:{tp} FP:{fp} FN:{fn} | F1:{f1:.3f}', fontsize=14)
        ax.axis('off')
        
        # 凡例
        legend_elements = [
            patches.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='lime', facecolor='none', label='Ground Truth'),
            patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='blue', facecolor='none', label='True Positive'),
            patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='False Positive')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 保存
        output_path = os.path.join(output_dir, f'{idx:03d}_{image_name}')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        if (idx + 1) % 10 == 0:
            print(f"  処理済み: {idx + 1}/{len(val_dataset)}")
    
    # 全体の統計
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print("\n=== 全体の検出結果 ===")
    print(f"Total True Positives: {total_tp}")
    print(f"Total False Positives: {total_fp}")
    print(f"Total False Negatives: {total_fn}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    
    # サマリーファイルを保存
    with open(os.path.join(output_dir, 'results_summary.json'), 'w') as f:
        json.dump({
            'overall': {
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1
            },
            'per_image': results_summary
        }, f, indent=2)
    
    print(f"\n結果は '{output_dir}' フォルダに保存されました。")
    print(f"- 各画像の検出結果: {output_dir}/XXX_image_name.jpg")
    print(f"- 統計サマリー: {output_dir}/results_summary.json")
    
    # 上位・下位の結果を表示
    create_best_worst_summary(results_summary, val_dataset, model, threshold, device)

def create_best_worst_summary(results_summary, val_dataset, model, threshold, device):
    """最良・最悪の検出結果をまとめた画像を作成"""
    # F1スコアでソート
    sorted_results = sorted(results_summary, key=lambda x: x['f1'], reverse=True)
    
    # 上位5件と下位5件を選択
    best_5 = sorted_results[:5]
    worst_5 = sorted_results[-5:]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Best and Worst Detection Results', fontsize=16)
    
    for col, result in enumerate(best_5):
        ax = axes[0, col]
        idx = next(i for i, r in enumerate(results_summary) if r['image'] == result['image'])
        
        # 画像表示
        image_info = val_dataset.image_info[idx]
        image_path = f"{val_dataset.image_dir}/{result['image']}"
        orig_image = Image.open(image_path)
        ax.imshow(orig_image)
        
        ax.set_title(f"Best #{col+1}\nF1:{result['f1']:.3f} (TP:{result['tp']} FP:{result['fp']} FN:{result['fn']})", 
                    fontsize=10, color='green')
        ax.axis('off')
    
    for col, result in enumerate(worst_5):
        ax = axes[1, col]
        idx = next(i for i, r in enumerate(results_summary) if r['image'] == result['image'])
        
        # 画像表示
        image_info = val_dataset.image_info[idx]
        image_path = f"{val_dataset.image_dir}/{result['image']}"
        orig_image = Image.open(image_path)
        ax.imshow(orig_image)
        
        ax.set_title(f"Worst #{col+1}\nF1:{result['f1']:.3f} (TP:{result['tp']} FP:{result['fp']} FN:{result['fn']})", 
                    fontsize=10, color='red')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('val_best_worst_summary.png', dpi=150, bbox_inches='tight')
    print(f"\nBest/Worst サマリー画像: val_best_worst_summary.png")

if __name__ == "__main__":
    show_validation_results()