import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CocoDataset
from model import AITECTDetector
from loss import detection_loss
from loss_improved import detection_loss_improved
import json
import matplotlib.pyplot as plt
import numpy as np

def load_config(config_path="config.json"):
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_loss_components(preds, targets, config):
    """損失の各コンポーネントを分析"""
    batch_size = preds.shape[0]
    
    total_conf_loss = 0
    total_reg_loss = 0
    positive_count = 0
    negative_count = 0
    
    for b in range(batch_size):
        pred = preds[b]  # [N, 5]
        target = targets[b]  # dict
        
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
            conf_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_scores, torch.zeros_like(pred_scores), reduction='none'
            )
            total_conf_loss += conf_loss.mean().item()
            negative_count += pred_scores.shape[0]
            continue
        
        # 距離ベースのマッチング
        N, M = pred_boxes_xywh.size(0), gt_boxes.size(0)
        pred_centers = pred_boxes_xywh[:, :2]
        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        
        distances = torch.cdist(pred_centers, gt_centers, p=2)
        min_distances, nearest_gt_idx = distances.min(dim=1)
        
        distance_threshold = 50.0
        close_mask = min_distances < distance_threshold
        
        # 正例と負例の数をカウント
        positive_count += close_mask.sum().item()
        negative_count += (~close_mask).sum().item()
        
        # 信頼度損失の計算
        target_conf = close_mask.float()
        conf_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_scores, target_conf, reduction='none'
        )
        total_conf_loss += conf_loss.mean().item()
        
        # 回帰損失（正例のみ）
        if close_mask.sum() > 0:
            pos_pred_boxes = pred_boxes_xywh[close_mask]
            pos_gt_boxes_xyxy = gt_boxes[nearest_gt_idx[close_mask]]
            
            # GTボックスを[x, y, w, h]形式に変換
            pos_gt_x = (pos_gt_boxes_xyxy[:, 0] + pos_gt_boxes_xyxy[:, 2]) / 2
            pos_gt_y = (pos_gt_boxes_xyxy[:, 1] + pos_gt_boxes_xyxy[:, 3]) / 2
            pos_gt_w = pos_gt_boxes_xyxy[:, 2] - pos_gt_boxes_xyxy[:, 0]
            pos_gt_h = pos_gt_boxes_xyxy[:, 3] - pos_gt_boxes_xyxy[:, 1]
            pos_gt_boxes_xywh = torch.stack([pos_gt_x, pos_gt_y, pos_gt_w, pos_gt_h], dim=1)
            
            reg_loss = torch.nn.functional.l1_loss(pos_pred_boxes, pos_gt_boxes_xywh, reduction='mean')
            total_reg_loss += reg_loss.item()
    
    return {
        'conf_loss': total_conf_loss / batch_size,
        'reg_loss': total_reg_loss / batch_size,
        'positive_ratio': positive_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0,
        'positive_count': positive_count,
        'negative_count': negative_count
    }

def analyze_predictions(preds, targets):
    """予測の統計情報を分析"""
    batch_size = preds.shape[0]
    
    all_scores = []
    all_boxes = []
    
    for b in range(batch_size):
        pred = preds[b]  # [N, 5]
        pred_boxes = pred[:, :4]
        pred_scores = pred[:, 4]
        
        all_scores.append(pred_scores)
        all_boxes.append(pred_boxes)
    
    all_scores = torch.cat(all_scores)
    all_boxes = torch.cat(all_boxes)
    
    # スコアの統計
    scores_sigmoid = torch.sigmoid(all_scores)
    
    # ボックスの統計
    box_centers = all_boxes[:, :2]
    box_sizes = all_boxes[:, 2:]
    
    stats = {
        'score_mean': scores_sigmoid.mean().item(),
        'score_std': scores_sigmoid.std().item(),
        'score_min': scores_sigmoid.min().item(),
        'score_max': scores_sigmoid.max().item(),
        'high_conf_count': (scores_sigmoid > 0.5).sum().item(),
        'center_x_mean': box_centers[:, 0].mean().item(),
        'center_y_mean': box_centers[:, 1].mean().item(),
        'width_mean': box_sizes[:, 0].mean().item(),
        'height_mean': box_sizes[:, 1].mean().item(),
        'width_std': box_sizes[:, 0].std().item(),
        'height_std': box_sizes[:, 1].std().item(),
    }
    
    return stats

def main():
    """損失と予測の詳細分析"""
    config = load_config()
    
    # 設定の取得
    image_dir = config['paths']['train_images']
    annotation_path = config['paths']['train_annotations']
    batch_size = 8
    image_size = config['training']['image_size']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # データセットとローダ
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = CocoDataset(image_dir, annotation_path, transform=transform)
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        return images, list(targets)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # モデル
    model = AITECTDetector().to(device)
    
    # 学習済みモデルがあれば読み込む
    model_path = config['paths']['model_save_path']
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"学習済みモデルを読み込みました: {model_path}")
        except:
            print("学習済みモデルが見つかりません。ランダム初期化で分析します。")
    
    model.eval()
    
    print("損失コンポーネントと予測の分析を開始します...\n")
    
    # 分析結果を保存
    loss_components_history = []
    prediction_stats_history = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= 10:  # 最初の10バッチのみ分析
                break
            
            images = images.to(device)
            for t in targets:
                t["boxes"] = t["boxes"].to(device)
                t["labels"] = t["labels"].to(device)
            
            # 順伝播
            preds = model(images)
            
            # 損失コンポーネントの分析
            loss_components = analyze_loss_components(preds, targets, config)
            loss_components_history.append(loss_components)
            
            # 予測の統計分析
            pred_stats = analyze_predictions(preds, targets)
            prediction_stats_history.append(pred_stats)
            
            print(f"バッチ {batch_idx + 1}:")
            print(f"  信頼度損失: {loss_components['conf_loss']:.4f}")
            print(f"  回帰損失: {loss_components['reg_loss']:.4f}")
            print(f"  正例の割合: {loss_components['positive_ratio']:.2%} ({loss_components['positive_count']}/{loss_components['positive_count'] + loss_components['negative_count']})")
            print(f"  予測スコア: 平均={pred_stats['score_mean']:.3f}, 標準偏差={pred_stats['score_std']:.3f}")
            print(f"  高信頼度予測数: {pred_stats['high_conf_count']}")
            print(f"  予測ボックス中心: x={pred_stats['center_x_mean']:.1f}, y={pred_stats['center_y_mean']:.1f}")
            print(f"  予測ボックスサイズ: w={pred_stats['width_mean']:.1f}±{pred_stats['width_std']:.1f}, h={pred_stats['height_mean']:.1f}±{pred_stats['height_std']:.1f}")
            print()
    
    # 統計の可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 損失コンポーネント
    ax = axes[0, 0]
    conf_losses = [lc['conf_loss'] for lc in loss_components_history]
    reg_losses = [lc['reg_loss'] for lc in loss_components_history]
    ax.plot(conf_losses, label='Confidence Loss')
    ax.plot(reg_losses, label='Regression Loss')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True)
    
    # 正例の割合
    ax = axes[0, 1]
    positive_ratios = [lc['positive_ratio'] for lc in loss_components_history]
    ax.plot(positive_ratios)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Positive Ratio')
    ax.set_title('Positive Sample Ratio')
    ax.grid(True)
    
    # 予測スコアの分布
    ax = axes[1, 0]
    score_means = [ps['score_mean'] for ps in prediction_stats_history]
    score_stds = [ps['score_std'] for ps in prediction_stats_history]
    ax.errorbar(range(len(score_means)), score_means, yerr=score_stds, fmt='o-')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Prediction Score')
    ax.set_title('Prediction Confidence Distribution')
    ax.grid(True)
    
    # ボックスサイズの分布
    ax = axes[1, 1]
    width_means = [ps['width_mean'] for ps in prediction_stats_history]
    height_means = [ps['height_mean'] for ps in prediction_stats_history]
    ax.plot(width_means, label='Width')
    ax.plot(height_means, label='Height')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Size (pixels)')
    ax.set_title('Predicted Box Sizes')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('loss_analysis.png')
    plt.close()
    
    print("\n分析結果を loss_analysis.png に保存しました。")
    
    # サマリー統計
    print("\n=== サマリー統計 ===")
    avg_conf_loss = np.mean(conf_losses)
    avg_reg_loss = np.mean(reg_losses)
    avg_positive_ratio = np.mean(positive_ratios)
    
    print(f"平均信頼度損失: {avg_conf_loss:.4f}")
    print(f"平均回帰損失: {avg_reg_loss:.4f}")
    print(f"平均正例割合: {avg_positive_ratio:.2%}")
    
    if avg_positive_ratio < 0.01:
        print("\n警告: 正例の割合が非常に低いです。")
        print("考えられる原因:")
        print("- 距離閾値が厳しすぎる")
        print("- グリッドとGTのミスマッチ")
        print("- アンカーサイズの不適合")

if __name__ == "__main__":
    main()