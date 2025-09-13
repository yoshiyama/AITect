import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CocoDataset
from model_whiteline import WhiteLineDetector
from loss_improved import detection_loss_improved
import json
import matplotlib.pyplot as plt
import numpy as np

def load_config(config_path="config.json"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_loss_components_detail(model, dataloader, device, config):
    """損失の各コンポーネントと予測の関係を詳細分析"""
    
    model.eval()
    
    # 損失関数の設定
    loss_type = config['training'].get('loss_type', 'mixed')
    iou_weight = config['training'].get('iou_weight', 2.0)
    l1_weight = config['training'].get('l1_weight', 0.5)
    use_focal = config['model'].get('use_focal_loss', True)
    
    results = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= 5:  # 5バッチ分析
                break
            
            images = images.to(device)
            batch_size = images.shape[0]
            
            for t in targets:
                t["boxes"] = t["boxes"].to(device)
                t["labels"] = t["labels"].to(device)
            
            # 予測
            preds = model(images)
            
            # バッチ内の各サンプルを分析
            for b in range(batch_size):
                pred = preds[b]  # [192, 5]
                target = targets[b]
                
                pred_boxes_xywh = pred[:, :4]
                pred_scores = pred[:, 4]
                
                # 予測ボックスを[x1, y1, x2, y2]形式に変換
                pred_x1 = pred_boxes_xywh[:, 0] - pred_boxes_xywh[:, 2] / 2
                pred_y1 = pred_boxes_xywh[:, 1] - pred_boxes_xywh[:, 3] / 2
                pred_x2 = pred_boxes_xywh[:, 0] + pred_boxes_xywh[:, 2] / 2
                pred_y2 = pred_boxes_xywh[:, 1] + pred_boxes_xywh[:, 3] / 2
                pred_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
                
                gt_boxes = target["boxes"]
                
                if gt_boxes.numel() == 0:
                    # GTなし
                    result = {
                        'has_gt': False,
                        'num_gt': 0,
                        'pred_scores': pred_scores.cpu().numpy(),
                        'pred_scores_sigmoid': torch.sigmoid(pred_scores).cpu().numpy(),
                        'all_negative_labels': True
                    }
                else:
                    # GTあり - 正例の割り当てを分析
                    from utils.bbox import box_iou
                    
                    # IoUベースのマッチング（改善版損失関数と同じロジック）
                    ious_all = box_iou(pred_boxes_xyxy, gt_boxes)  # [192, M]
                    max_ious, matched_gt_idx = ious_all.max(dim=1)  # [192]
                    
                    # 正例マスク
                    positive_mask = max_ious > 0.1
                    
                    # 各GTに対して最もIoUが高い予測も正例に
                    best_pred_per_gt, _ = ious_all.max(dim=0)  # [M]
                    for m in range(gt_boxes.shape[0]):
                        if best_pred_per_gt[m] > 0.05:
                            best_pred_idx = ious_all[:, m].argmax()
                            positive_mask[best_pred_idx] = True
                    
                    # ターゲット信頼度
                    target_conf = positive_mask.float()
                    
                    # 信頼度損失の計算（各予測ごと）
                    conf_losses = nn.functional.binary_cross_entropy_with_logits(
                        pred_scores, target_conf, reduction='none'
                    )
                    
                    result = {
                        'has_gt': True,
                        'num_gt': gt_boxes.shape[0],
                        'pred_scores': pred_scores.cpu().numpy(),
                        'pred_scores_sigmoid': torch.sigmoid(pred_scores).cpu().numpy(),
                        'target_conf': target_conf.cpu().numpy(),
                        'max_ious': max_ious.cpu().numpy(),
                        'conf_losses': conf_losses.cpu().numpy(),
                        'num_positive': positive_mask.sum().item(),
                        'positive_indices': torch.where(positive_mask)[0].cpu().numpy()
                    }
                
                results.append(result)
    
    return results

def visualize_loss_prediction_relationship(results):
    """損失と予測の関係を可視化"""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # 1. 予測スコア（sigmoid前）の分布
    ax = axes[0, 0]
    all_scores = []
    for r in results:
        all_scores.extend(r['pred_scores'])
    ax.hist(all_scores, bins=50, alpha=0.7)
    ax.set_xlabel('Raw Prediction Score (logits)')
    ax.set_ylabel('Count')
    ax.set_title('Raw Score Distribution')
    ax.axvline(0, color='red', linestyle='--', label='0 (sigmoid=0.5)')
    ax.legend()
    
    # 2. 予測スコア（sigmoid後）の分布
    ax = axes[0, 1]
    all_scores_sigmoid = []
    for r in results:
        all_scores_sigmoid.extend(r['pred_scores_sigmoid'])
    ax.hist(all_scores_sigmoid, bins=50, alpha=0.7)
    ax.set_xlabel('Prediction Score (after sigmoid)')
    ax.set_ylabel('Count')
    ax.set_title('Sigmoid Score Distribution')
    ax.axvline(0.5, color='red', linestyle='--', label='Threshold')
    ax.legend()
    
    # 3. 正例と負例のスコア分布
    ax = axes[0, 2]
    positive_scores = []
    negative_scores = []
    
    for r in results:
        if r['has_gt'] and 'target_conf' in r:
            pos_mask = r['target_conf'] > 0.5
            positive_scores.extend(r['pred_scores_sigmoid'][pos_mask])
            negative_scores.extend(r['pred_scores_sigmoid'][~pos_mask])
    
    if positive_scores:
        ax.hist(positive_scores, bins=30, alpha=0.5, label=f'Positive ({len(positive_scores)})', color='green')
    if negative_scores:
        ax.hist(negative_scores, bins=30, alpha=0.5, label=f'Negative ({len(negative_scores)})', color='red')
    ax.set_xlabel('Prediction Score')
    ax.set_ylabel('Count')
    ax.set_title('Score Distribution by Label')
    ax.legend()
    
    # 4. IoUと予測スコアの関係
    ax = axes[1, 0]
    ious = []
    scores = []
    for r in results:
        if r['has_gt'] and 'max_ious' in r:
            ious.extend(r['max_ious'])
            scores.extend(r['pred_scores_sigmoid'])
    
    if ious:
        ax.scatter(ious, scores, alpha=0.3, s=1)
        ax.set_xlabel('Max IoU with GT')
        ax.set_ylabel('Prediction Score')
        ax.set_title('IoU vs Prediction Score')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        ax.axvline(0.1, color='green', linestyle='--', alpha=0.5, label='Positive Threshold')
        ax.legend()
    
    # 5. 正例の数
    ax = axes[1, 1]
    positive_counts = [r['num_positive'] for r in results if 'num_positive' in r]
    if positive_counts:
        ax.hist(positive_counts, bins=20)
        ax.set_xlabel('Number of Positive Samples')
        ax.set_ylabel('Count')
        ax.set_title('Positive Sample Distribution')
        mean_pos = np.mean(positive_counts)
        ax.axvline(mean_pos, color='red', linestyle='--', label=f'Mean: {mean_pos:.1f}')
        ax.legend()
    
    # 6. 信頼度損失の分布
    ax = axes[1, 2]
    all_conf_losses = []
    for r in results:
        if 'conf_losses' in r:
            all_conf_losses.extend(r['conf_losses'])
    
    if all_conf_losses:
        ax.hist(all_conf_losses, bins=50, alpha=0.7)
        ax.set_xlabel('Confidence Loss')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Loss Distribution')
        mean_loss = np.mean(all_conf_losses)
        ax.axvline(mean_loss, color='red', linestyle='--', label=f'Mean: {mean_loss:.3f}')
        ax.legend()
    
    # 7. GT数と正例数の関係
    ax = axes[2, 0]
    gt_counts = []
    pos_counts = []
    for r in results:
        if r['has_gt'] and 'num_positive' in r:
            gt_counts.append(r['num_gt'])
            pos_counts.append(r['num_positive'])
    
    if gt_counts:
        ax.scatter(gt_counts, pos_counts, alpha=0.6)
        ax.set_xlabel('Number of GT Boxes')
        ax.set_ylabel('Number of Positive Predictions')
        ax.set_title('GT Count vs Positive Predictions')
        ax.grid(True)
    
    # 8. 正例の予測スコア
    ax = axes[2, 1]
    positive_pred_scores = []
    for r in results:
        if 'positive_indices' in r and len(r['positive_indices']) > 0:
            pos_idx = r['positive_indices']
            positive_pred_scores.extend(r['pred_scores_sigmoid'][pos_idx])
    
    if positive_pred_scores:
        ax.hist(positive_pred_scores, bins=30, alpha=0.7, color='green')
        ax.set_xlabel('Prediction Score')
        ax.set_ylabel('Count')
        ax.set_title('Positive Sample Prediction Scores')
        mean_score = np.mean(positive_pred_scores)
        ax.axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.3f}')
        ax.axvline(0.5, color='blue', linestyle='--', label='Threshold')
        ax.legend()
    
    # 9. サマリー統計
    ax = axes[2, 2]
    total_predictions = sum(len(r['pred_scores']) for r in results)
    total_positive = sum(r.get('num_positive', 0) for r in results)
    total_gt = sum(r.get('num_gt', 0) for r in results)
    
    stats_text = f"""Analysis Summary:
    
Total Predictions: {total_predictions}
Total Positive Labels: {total_positive}
Positive Ratio: {100*total_positive/total_predictions:.1f}%
Total GT Boxes: {total_gt}

Mean Scores:
- All: {np.mean(all_scores_sigmoid):.3f}
- Positive: {np.mean(positive_scores) if positive_scores else 0:.3f}
- Negative: {np.mean(negative_scores) if negative_scores else 0:.3f}

High Conf (>0.5): {sum(1 for s in all_scores_sigmoid if s > 0.5)}
Low Conf (<0.1): {sum(1 for s in all_scores_sigmoid if s < 0.1)}
"""
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontfamily='monospace')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('loss_prediction_relationship.png', dpi=150)
    plt.close()

def main():
    """損失と予測の関係を詳細分析"""
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデル（正しい設定で）
    model = WhiteLineDetector(grid_size=8, num_anchors=3).to(device)
    model_path = config['paths']['model_save_path']
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # データセット
    image_dir = config['paths']['val_images']
    annotation_path = config['paths']['val_annotations']
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    dataset = CocoDataset(image_dir, annotation_path, transform=transform)
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        return images, list(targets)
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    print("損失と予測の関係を分析中...")
    
    # 分析実行
    results = analyze_loss_components_detail(model, dataloader, device, config)
    
    # 可視化
    visualize_loss_prediction_relationship(results)
    
    print("\n分析完了！")
    print("結果を loss_prediction_relationship.png に保存しました")
    
    # 診断
    print("\n=== 診断結果 ===")
    all_scores = []
    for r in results:
        all_scores.extend(r['pred_scores_sigmoid'])
    
    mean_score = np.mean(all_scores)
    max_score = np.max(all_scores)
    
    if max_score < 0.1:
        print("❌ 重大な問題: すべての予測スコアが極めて低い")
        print("   → モデルが「物体なし」を学習してしまっている可能性")
        print("   → 正例と負例のバランス、損失の重み付けを確認")
    elif mean_score < 0.01:
        print("⚠️  問題: 予測スコアが全体的に低すぎる")
        print("   → 信頼度の学習が不十分")

if __name__ == "__main__":
    main()