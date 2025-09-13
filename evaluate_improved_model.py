import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CocoDataset
from model_whiteline import WhiteLineDetector
from utils.bbox import box_iou, nms
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import os

def load_config(config_path="config.json"):
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def post_process_predictions(predictions, conf_threshold=0.5, nms_threshold=0.5):
    """予測の後処理（信頼度フィルタリング + NMS）"""
    # predictions: [N, 5] (x, y, w, h, conf)
    
    # 信頼度でフィルタリング
    scores = torch.sigmoid(predictions[:, 4])
    mask = scores > conf_threshold
    
    if mask.sum() == 0:
        return torch.zeros((0, 5)), torch.zeros(0)
    
    filtered_preds = predictions[mask]
    filtered_scores = scores[mask]
    
    # xywhからxyxyに変換
    boxes_xywh = filtered_preds[:, :4]
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
    
    # NMS適用
    keep_indices = nms(boxes_xyxy, filtered_scores, nms_threshold)
    
    return filtered_preds[keep_indices], filtered_scores[keep_indices]

def evaluate_on_validation(model, val_dataset, device, num_samples=10):
    """検証データで評価"""
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(val_dataset))):
            image, target = val_dataset[i]
            image_tensor = image.unsqueeze(0).to(device)
            
            # 予測
            output = model(image_tensor)[0]  # [192, 5]
            
            # 後処理
            filtered_preds, filtered_scores = post_process_predictions(
                output, conf_threshold=0.5, nms_threshold=0.5
            )
            
            # GTボックスを取得
            gt_boxes = target['boxes'].to(device)
            
            # 結果を保存
            results.append({
                'image_idx': i,
                'predictions': filtered_preds.cpu(),
                'scores': filtered_scores.cpu(),
                'gt_boxes': gt_boxes.cpu(),
                'num_predictions': len(filtered_preds),
                'num_gt': len(gt_boxes)
            })
    
    return results

def calculate_metrics(results):
    """評価メトリクスを計算"""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = []
    
    for result in results:
        preds = result['predictions']
        gt_boxes = result['gt_boxes']
        
        if len(preds) == 0:
            total_fn += len(gt_boxes)
            continue
        
        if len(gt_boxes) == 0:
            total_fp += len(preds)
            continue
        
        # 予測をxyxy形式に変換
        pred_x1 = preds[:, 0] - preds[:, 2] / 2
        pred_y1 = preds[:, 1] - preds[:, 3] / 2
        pred_x2 = preds[:, 0] + preds[:, 2] / 2
        pred_y2 = preds[:, 1] + preds[:, 3] / 2
        pred_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
        
        # IoU計算
        ious = box_iou(pred_boxes_xyxy, gt_boxes)
        
        # 各予測に対して最もIoUが高いGTを見つける
        max_ious, matched_gt = ious.max(dim=1)
        
        # TP, FPをカウント（IoU > 0.5）
        tp_mask = max_ious > 0.5
        total_tp += tp_mask.sum().item()
        total_fp += (~tp_mask).sum().item()
        
        # 各GTがマッチしたかチェック
        matched_gt_set = set(matched_gt[tp_mask].tolist())
        total_fn += len(gt_boxes) - len(matched_gt_set)
        
        # IoUを記録
        total_iou.extend(max_ious[tp_mask].tolist())
    
    # メトリクス計算
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = np.mean(total_iou) if total_iou else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn
    }

def visualize_predictions(model, dataset, device, save_dir='evaluation_results', num_images=5):
    """予測結果を可視化"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    for i in range(min(num_images, len(dataset))):
        image, target = dataset[i]
        
        # 元の画像を取得（変換前）
        original_image = Image.open(dataset.image_paths[i]).convert('RGB')
        
        # 予測
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)[0]
            
        # 後処理
        filtered_preds, filtered_scores = post_process_predictions(
            output, conf_threshold=0.5, nms_threshold=0.5
        )
        
        # 可視化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 左: GT
        ax1.imshow(original_image)
        ax1.set_title('Ground Truth', fontsize=16)
        
        # GTボックスを描画
        gt_boxes = target['boxes']
        for box in gt_boxes:
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=3, edgecolor='green', facecolor='none'
            )
            ax1.add_patch(rect)
        
        # 右: 予測
        ax2.imshow(original_image)
        ax2.set_title('Predictions', fontsize=16)
        
        # 予測ボックスを描画（座標をリサイズ）
        scale_x = original_image.width / 512
        scale_y = original_image.height / 512
        
        for pred, score in zip(filtered_preds, filtered_scores):
            # xywhからxyxyに変換してスケール調整
            x_center = pred[0].item() * scale_x
            y_center = pred[1].item() * scale_y
            width = pred[2].item() * scale_x
            height = pred[3].item() * scale_y
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=3, edgecolor='red', facecolor='none'
            )
            ax2.add_patch(rect)
            
            # スコアを表示
            ax2.text(x1, y1-5, f'{score.item():.2f}', 
                    color='red', fontsize=12, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax1.axis('off')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/prediction_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"画像 {i+1}: GT={len(gt_boxes)}個, 予測={len(filtered_preds)}個")

def main():
    """改善されたモデルの評価"""
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルを読み込み
    model = WhiteLineDetector(grid_size=8, num_anchors=3).to(device)
    model_path = "result/aitect_model_improved.pth"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"改善されたモデルを読み込みました: {model_path}")
    except:
        print(f"エラー: {model_path} が見つかりません")
        return
    
    # 検証データセット
    val_image_dir = config['paths']['val_images']
    val_annotation_path = config['paths']['val_annotations']
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    val_dataset = CocoDataset(val_image_dir, val_annotation_path, transform=transform)
    
    print(f"\n検証データ数: {len(val_dataset)}")
    
    # 1. 定量的評価
    print("\n=== 定量的評価 ===")
    results = evaluate_on_validation(model, val_dataset, device, num_samples=20)
    
    # 統計情報
    total_preds = sum(r['num_predictions'] for r in results)
    total_gt = sum(r['num_gt'] for r in results)
    print(f"総予測数: {total_preds}")
    print(f"総GT数: {total_gt}")
    
    # メトリクス計算
    metrics = calculate_metrics(results)
    print(f"\n評価メトリクス (IoU閾値=0.5):")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    print(f"平均IoU (TP only): {metrics['mean_iou']:.3f}")
    print(f"TP: {metrics['total_tp']}, FP: {metrics['total_fp']}, FN: {metrics['total_fn']}")
    
    # 2. 定性的評価（画像で確認）
    print("\n=== 予測結果の可視化 ===")
    visualize_predictions(model, val_dataset, device, 
                         save_dir='evaluation_results_improved', 
                         num_images=10)
    print("結果を evaluation_results_improved/ に保存しました")
    
    # 3. 改善前との比較
    print("\n=== 改善前モデルとの比較 ===")
    try:
        # 改善前のモデル
        model_old = WhiteLineDetector(grid_size=8, num_anchors=3).to(device)
        model_old.load_state_dict(torch.load("result/aitect_model.pth", map_location=device))
        model_old.eval()
        
        # サンプルで比較
        image, _ = val_dataset[0]
        image_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_old = model_old(image_tensor)[0]
            output_new = model(image_tensor)[0]
        
        scores_old = torch.sigmoid(output_old[:, 4])
        scores_new = torch.sigmoid(output_new[:, 4])
        
        print(f"改善前: 最大スコア={scores_old.max().item():.4f}, 高信頼度数={(scores_old > 0.5).sum().item()}")
        print(f"改善後: 最大スコア={scores_new.max().item():.4f}, 高信頼度数={(scores_new > 0.5).sum().item()}")
        
    except:
        print("改善前モデルとの比較はスキップ")

if __name__ == "__main__":
    main()