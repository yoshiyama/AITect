import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CocoDataset
from model_whiteline import WhiteLineDetector
from loss import detection_loss
from loss_improved import detection_loss_improved
import json
import matplotlib.pyplot as plt
import numpy as np
from utils.bbox import box_iou

def load_config(config_path="config.json"):
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def detect_model_config(model_path, device):
    """保存されたモデルの設定を検出"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # grid_xまたはgrid_yのサイズからgrid_sizeを推定
    for key in checkpoint.keys():
        if 'grid_x' in key or 'grid_y' in key:
            shape = checkpoint[key].shape
            grid_size = shape[2]  # [1, 1, grid_size, grid_size]
            print(f"検出されたgrid_size: {grid_size}")
            break
    
    # head.3.weightのサイズからnum_anchorsを推定
    for key in checkpoint.keys():
        if 'head.3.weight' in key or 'head.6.weight' in key:
            shape = checkpoint[key].shape
            num_outputs = shape[0]
            num_anchors = num_outputs // 5  # 5 = x, y, w, h, conf
            print(f"検出されたnum_anchors: {num_anchors}")
            break
    
    return grid_size, num_anchors

def analyze_prediction_distribution(model, dataloader, device, num_batches=10):
    """予測の分布を詳細分析"""
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    grid_activation_map = torch.zeros(model.grid_size, model.grid_size)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            images = images.to(device)
            batch_size = images.shape[0]
            
            # 予測
            preds = model(images)  # [B, N, 5]
            
            for b in range(batch_size):
                pred = preds[b]  # [N, 5]
                target = targets[b]
                
                # デバイスに移動
                target["boxes"] = target["boxes"].to(device)
                target["labels"] = target["labels"].to(device)
                
                pred_boxes_xywh = pred[:, :4]
                pred_scores = pred[:, 4]
                
                # 高信頼度予測の位置を記録
                high_conf_mask = torch.sigmoid(pred_scores) > 0.5
                
                # グリッド位置を計算
                grid_size = model.grid_size
                num_anchors = model.num_anchors
                
                for idx, is_high_conf in enumerate(high_conf_mask):
                    if is_high_conf:
                        # インデックスからグリッド位置を計算
                        anchor_idx = idx % num_anchors
                        cell_idx = idx // num_anchors
                        grid_y = cell_idx // grid_size
                        grid_x = cell_idx % grid_size
                        
                        grid_activation_map[grid_y, grid_x] += 1
                
                # 予測とGTを保存
                all_predictions.append({
                    'boxes': pred_boxes_xywh.cpu(),
                    'scores': torch.sigmoid(pred_scores).cpu(),
                    'image_idx': batch_idx * batch_size + b
                })
                
                all_targets.append({
                    'boxes': target["boxes"].cpu(),
                    'image_idx': batch_idx * batch_size + b
                })
    
    return all_predictions, all_targets, grid_activation_map

def analyze_loss_behavior(model, dataloader, device, config):
    """損失の振る舞いを分析"""
    
    model.eval()
    
    loss_distribution = {
        'total_losses': [],
        'conf_losses': [],
        'reg_losses': [],
        'positive_ratios': [],
        'gt_counts': [],
        'high_conf_counts': []
    }
    
    # 損失関数の設定
    use_improved = config['model'].get('use_improved_loss', False)
    loss_type = config['training'].get('loss_type', 'mixed')
    iou_weight = config['training'].get('iou_weight', 2.0)
    l1_weight = config['training'].get('l1_weight', 0.5)
    use_focal = config['model'].get('use_focal_loss', True)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= 20:  # 20バッチ分析
                break
            
            images = images.to(device)
            for t in targets:
                t["boxes"] = t["boxes"].to(device)
                t["labels"] = t["labels"].to(device)
            
            # 予測
            preds = model(images)
            
            # 損失計算
            if use_improved:
                loss = detection_loss_improved(preds, targets, 
                                             loss_type=loss_type,
                                             iou_weight=iou_weight, 
                                             l1_weight=l1_weight,
                                             use_focal=use_focal)
            else:
                loss = detection_loss(preds, targets, 
                                    loss_type=loss_type,
                                    iou_weight=iou_weight, 
                                    l1_weight=l1_weight,
                                    use_focal=use_focal)
            
            loss_distribution['total_losses'].append(loss.item())
            
            # バッチ内の統計
            batch_size = preds.shape[0]
            total_gt = sum(len(t["boxes"]) for t in targets)
            high_conf_count = (torch.sigmoid(preds[:, :, 4]) > 0.5).sum().item()
            
            loss_distribution['gt_counts'].append(total_gt)
            loss_distribution['high_conf_counts'].append(high_conf_count)
    
    return loss_distribution

def visualize_comprehensive_analysis(predictions, targets, grid_map, loss_dist, model):
    """包括的な分析結果を可視化"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. グリッドアクティベーションマップ
    ax1 = plt.subplot(3, 3, 1)
    im = ax1.imshow(grid_map.numpy(), cmap='hot', interpolation='nearest')
    ax1.set_title('Grid Activation Map (High Conf Predictions)')
    ax1.set_xlabel('Grid X')
    ax1.set_ylabel('Grid Y')
    plt.colorbar(im, ax=ax1)
    
    # 2. 予測スコアのヒストグラム
    ax2 = plt.subplot(3, 3, 2)
    all_scores = []
    for pred in predictions:
        all_scores.extend(pred['scores'].numpy().flatten())
    ax2.hist(all_scores, bins=50, alpha=0.7)
    ax2.set_xlabel('Prediction Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Score Distribution')
    ax2.axvline(0.5, color='red', linestyle='--', label='Threshold')
    
    # 3. 予測ボックスの中心位置分布
    ax3 = plt.subplot(3, 3, 3)
    pred_centers_x = []
    pred_centers_y = []
    for pred in predictions:
        boxes = pred['boxes']
        scores = pred['scores']
        high_conf = scores > 0.5
        if high_conf.any():
            centers = boxes[high_conf, :2]
            pred_centers_x.extend(centers[:, 0].numpy())
            pred_centers_y.extend(centers[:, 1].numpy())
    
    if pred_centers_x:
        ax3.scatter(pred_centers_x, pred_centers_y, alpha=0.5, s=10)
    ax3.set_xlim(0, 512)
    ax3.set_ylim(0, 512)
    ax3.invert_yaxis()
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('High Confidence Prediction Centers')
    ax3.grid(True)
    
    # 4. GT vs 予測数
    ax4 = plt.subplot(3, 3, 4)
    gt_counts = []
    pred_counts = []
    for i, pred in enumerate(predictions):
        # 対応するGTを見つける
        gt = next((t for t in targets if t['image_idx'] == pred['image_idx']), None)
        if gt:
            gt_counts.append(len(gt['boxes']))
            pred_counts.append((pred['scores'] > 0.5).sum().item())
    
    if gt_counts:
        ax4.scatter(gt_counts, pred_counts, alpha=0.6)
        max_count = max(max(gt_counts), max(pred_counts)) + 1
        ax4.plot([0, max_count], [0, max_count], 'r--', label='Perfect')
        ax4.set_xlabel('GT Box Count')
        ax4.set_ylabel('Predicted Box Count (>0.5)')
        ax4.set_title('GT vs Predicted Box Counts')
        ax4.legend()
        ax4.grid(True)
    
    # 5. 損失の分布
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(loss_dist['total_losses'], bins=30, alpha=0.7)
    ax5.set_xlabel('Loss Value')
    ax5.set_ylabel('Count')
    ax5.set_title('Loss Distribution')
    mean_loss = np.mean(loss_dist['total_losses'])
    ax5.axvline(mean_loss, color='red', linestyle='--', 
                label=f'Mean: {mean_loss:.3f}')
    ax5.legend()
    
    # 6. 予測ボックスサイズ分布
    ax6 = plt.subplot(3, 3, 6)
    pred_widths = []
    pred_heights = []
    for pred in predictions:
        boxes = pred['boxes']
        scores = pred['scores']
        high_conf = scores > 0.5
        if high_conf.any():
            sizes = boxes[high_conf, 2:]
            pred_widths.extend(sizes[:, 0].numpy())
            pred_heights.extend(sizes[:, 1].numpy())
    
    if pred_widths:
        ax6.scatter(pred_widths, pred_heights, alpha=0.5, s=10)
        ax6.set_xlabel('Width')
        ax6.set_ylabel('Height')
        ax6.set_title('Predicted Box Sizes (High Conf)')
        ax6.grid(True)
    
    # 7. GT数と損失の関係
    ax7 = plt.subplot(3, 3, 7)
    if loss_dist['gt_counts'] and loss_dist['total_losses']:
        ax7.scatter(loss_dist['gt_counts'], loss_dist['total_losses'], alpha=0.6)
        ax7.set_xlabel('GT Count in Batch')
        ax7.set_ylabel('Total Loss')
        ax7.set_title('GT Count vs Loss')
        ax7.grid(True)
    
    # 8. 高信頼度予測数の推移
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(loss_dist['high_conf_counts'])
    ax8.set_xlabel('Batch Index')
    ax8.set_ylabel('High Conf Predictions')
    ax8.set_title('High Confidence Predictions per Batch')
    ax8.grid(True)
    
    # 9. モデル情報
    ax9 = plt.subplot(3, 3, 9)
    ax9.text(0.1, 0.9, f'Model: WhiteLineDetector', transform=ax9.transAxes)
    ax9.text(0.1, 0.8, f'Grid Size: {model.grid_size}x{model.grid_size}', transform=ax9.transAxes)
    ax9.text(0.1, 0.7, f'Num Anchors: {model.num_anchors}', transform=ax9.transAxes)
    ax9.text(0.1, 0.6, f'Total Predictions: {model.grid_size * model.grid_size * model.num_anchors}', transform=ax9.transAxes)
    ax9.text(0.1, 0.5, f'Mean Loss: {np.mean(loss_dist["total_losses"]):.4f}', transform=ax9.transAxes)
    ax9.text(0.1, 0.4, f'Analyzed Images: {len(predictions)}', transform=ax9.transAxes)
    ax9.axis('off')
    
    plt.tight_layout()
    plt.savefig('trained_model_analysis.png', dpi=150)
    plt.close()

def main():
    """学習済みモデルの包括的分析"""
    config = load_config()
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルパス
    model_path = config['paths']['model_save_path']
    
    # モデルの設定を自動検出
    print("モデルの設定を検出中...")
    try:
        grid_size, num_anchors = detect_model_config(model_path, device)
    except Exception as e:
        print(f"モデル設定の検出に失敗: {e}")
        print("デフォルト設定を使用します")
        grid_size = 8
        num_anchors = 3
    
    # モデルを正しい設定で作成
    model = WhiteLineDetector(grid_size=grid_size, num_anchors=num_anchors).to(device)
    
    # モデルを読み込む
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"学習済みモデルを正常に読み込みました")
        print(f"Grid size: {grid_size}, Num anchors: {num_anchors}")
    except Exception as e:
        print(f"モデルの読み込みエラー: {e}")
        return
    
    # データセット準備
    image_dir = config['paths']['val_images']
    annotation_path = config['paths']['val_annotations']
    batch_size = 4
    image_size = config['training']['image_size']
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = CocoDataset(image_dir, annotation_path, transform=transform)
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        return images, list(targets)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print("\n学習済みモデルの包括的分析を開始します...")
    
    # 1. 予測分布の分析
    print("予測分布を分析中...")
    predictions, targets, grid_map = analyze_prediction_distribution(model, dataloader, device)
    
    # 2. 損失の振る舞いを分析
    print("損失の振る舞いを分析中...")
    loss_dist = analyze_loss_behavior(model, dataloader, device, config)
    
    # 3. 結果の可視化
    print("結果を可視化中...")
    visualize_comprehensive_analysis(predictions, targets, grid_map, loss_dist, model)
    
    # サマリー統計
    print("\n=== 分析サマリー ===")
    print(f"平均損失: {np.mean(loss_dist['total_losses']):.4f}")
    print(f"グリッドアクティベーション:")
    print(f"  最大活性化数: {grid_map.max().item():.0f}")
    print(f"  活性化セル数: {(grid_map > 0).sum().item()}/{grid_size * grid_size}")
    
    high_conf_total = sum((pred['scores'] > 0.5).sum().item() for pred in predictions)
    print(f"高信頼度予測総数: {high_conf_total}")
    
    print("\n分析結果を trained_model_analysis.png に保存しました")

if __name__ == "__main__":
    main()