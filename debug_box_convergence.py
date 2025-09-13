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
from utils.bbox import box_iou

def load_config(config_path="config.json"):
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_box_predictions(model, dataloader, device, num_batches=5):
    """予測ボックスとGTボックスの関係を詳細分析"""
    
    model.eval()
    
    all_results = []
    
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
                
                # 予測ボックスを[x1, y1, x2, y2]形式に変換
                pred_x1 = pred_boxes_xywh[:, 0] - pred_boxes_xywh[:, 2] / 2
                pred_y1 = pred_boxes_xywh[:, 1] - pred_boxes_xywh[:, 3] / 2
                pred_x2 = pred_boxes_xywh[:, 0] + pred_boxes_xywh[:, 2] / 2
                pred_y2 = pred_boxes_xywh[:, 1] + pred_boxes_xywh[:, 3] / 2
                pred_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
                
                gt_boxes = target["boxes"]  # [M, 4] in xyxy format
                
                if gt_boxes.numel() == 0:
                    continue
                
                # 各GTボックスに対して最も近い予測を見つける
                for gt_idx in range(gt_boxes.shape[0]):
                    gt_box = gt_boxes[gt_idx]
                    
                    # GTの中心とサイズ
                    gt_cx = (gt_box[0] + gt_box[2]) / 2
                    gt_cy = (gt_box[1] + gt_box[3]) / 2
                    gt_w = gt_box[2] - gt_box[0]
                    gt_h = gt_box[3] - gt_box[1]
                    
                    # 全予測との距離を計算
                    pred_centers = pred_boxes_xywh[:, :2]
                    gt_center = torch.tensor([gt_cx, gt_cy], device=device)
                    distances = torch.norm(pred_centers - gt_center, dim=1)
                    
                    # 最も近い予測を選択
                    min_dist_idx = distances.argmin()
                    closest_pred = pred_boxes_xywh[min_dist_idx]
                    closest_score = torch.sigmoid(pred_scores[min_dist_idx]).item()
                    
                    # IoUを計算
                    iou = box_iou(pred_boxes_xyxy[min_dist_idx:min_dist_idx+1], 
                                  gt_box.unsqueeze(0))[0, 0].item()
                    
                    # 予測の中心とサイズ
                    pred_cx = closest_pred[0].item()
                    pred_cy = closest_pred[1].item()
                    pred_w = closest_pred[2].item()
                    pred_h = closest_pred[3].item()
                    
                    # エラーを計算
                    center_error = distances[min_dist_idx].item()
                    width_error = abs(pred_w - gt_w.item())
                    height_error = abs(pred_h - gt_h.item())
                    
                    result = {
                        'batch_idx': batch_idx,
                        'image_idx': b,
                        'gt_idx': gt_idx,
                        'gt_cx': gt_cx.item(),
                        'gt_cy': gt_cy.item(),
                        'gt_w': gt_w.item(),
                        'gt_h': gt_h.item(),
                        'pred_cx': pred_cx,
                        'pred_cy': pred_cy,
                        'pred_w': pred_w,
                        'pred_h': pred_h,
                        'pred_score': closest_score,
                        'center_error': center_error,
                        'width_error': width_error,
                        'height_error': height_error,
                        'iou': iou,
                        'distance': distances[min_dist_idx].item()
                    }
                    
                    all_results.append(result)
    
    return all_results

def visualize_convergence_analysis(results):
    """収束分析の可視化"""
    
    if not results:
        print("分析するデータがありません")
        return
    
    # データを整理
    center_errors = [r['center_error'] for r in results]
    width_errors = [r['width_error'] for r in results]
    height_errors = [r['height_error'] for r in results]
    ious = [r['iou'] for r in results]
    scores = [r['pred_score'] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 中心位置のエラー分布
    ax = axes[0, 0]
    ax.hist(center_errors, bins=30, alpha=0.7, color='blue')
    ax.axvline(np.mean(center_errors), color='red', linestyle='--', 
               label=f'Mean: {np.mean(center_errors):.1f}')
    ax.set_xlabel('Center Position Error (pixels)')
    ax.set_ylabel('Count')
    ax.set_title('Center Position Error Distribution')
    ax.legend()
    
    # 2. 幅のエラー分布
    ax = axes[0, 1]
    ax.hist(width_errors, bins=30, alpha=0.7, color='green')
    ax.axvline(np.mean(width_errors), color='red', linestyle='--',
               label=f'Mean: {np.mean(width_errors):.1f}')
    ax.set_xlabel('Width Error (pixels)')
    ax.set_ylabel('Count')
    ax.set_title('Width Error Distribution')
    ax.legend()
    
    # 3. 高さのエラー分布
    ax = axes[0, 2]
    ax.hist(height_errors, bins=30, alpha=0.7, color='orange')
    ax.axvline(np.mean(height_errors), color='red', linestyle='--',
               label=f'Mean: {np.mean(height_errors):.1f}')
    ax.set_xlabel('Height Error (pixels)')
    ax.set_ylabel('Count')
    ax.set_title('Height Error Distribution')
    ax.legend()
    
    # 4. IoU分布
    ax = axes[1, 0]
    ax.hist(ious, bins=30, alpha=0.7, color='purple')
    ax.axvline(np.mean(ious), color='red', linestyle='--',
               label=f'Mean: {np.mean(ious):.3f}')
    ax.set_xlabel('IoU')
    ax.set_ylabel('Count')
    ax.set_title('IoU Distribution')
    ax.legend()
    
    # 5. スコアとIoUの関係
    ax = axes[1, 1]
    ax.scatter(scores, ious, alpha=0.5)
    ax.set_xlabel('Prediction Score')
    ax.set_ylabel('IoU')
    ax.set_title('Score vs IoU Correlation')
    ax.grid(True)
    
    # 6. 予測vs実際のサイズ
    ax = axes[1, 2]
    gt_sizes = [(r['gt_w'], r['gt_h']) for r in results]
    pred_sizes = [(r['pred_w'], r['pred_h']) for r in results]
    
    gt_areas = [w * h for w, h in gt_sizes]
    pred_areas = [w * h for w, h in pred_sizes]
    
    ax.scatter(gt_areas, pred_areas, alpha=0.5)
    ax.plot([0, max(gt_areas)], [0, max(gt_areas)], 'r--', label='Perfect Prediction')
    ax.set_xlabel('GT Box Area')
    ax.set_ylabel('Predicted Box Area')
    ax.set_title('GT vs Predicted Box Areas')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('box_convergence_analysis.png', dpi=150)
    plt.close()
    
    # 詳細な統計を出力
    print("\n=== ボックス予測の詳細分析 ===")
    print(f"分析したGTボックス数: {len(results)}")
    print(f"\n位置エラー:")
    print(f"  平均中心エラー: {np.mean(center_errors):.1f} pixels")
    print(f"  中央値中心エラー: {np.median(center_errors):.1f} pixels")
    print(f"  最大中心エラー: {np.max(center_errors):.1f} pixels")
    
    print(f"\nサイズエラー:")
    print(f"  平均幅エラー: {np.mean(width_errors):.1f} pixels")
    print(f"  平均高さエラー: {np.mean(height_errors):.1f} pixels")
    
    print(f"\nIoU統計:")
    print(f"  平均IoU: {np.mean(ious):.3f}")
    print(f"  IoU > 0.5: {sum(1 for iou in ious if iou > 0.5)}/{len(ious)} ({100*sum(1 for iou in ious if iou > 0.5)/len(ious):.1f}%)")
    print(f"  IoU > 0.3: {sum(1 for iou in ious if iou > 0.3)}/{len(ious)} ({100*sum(1 for iou in ious if iou > 0.3)/len(ious):.1f}%)")
    
    print(f"\n予測信頼度:")
    print(f"  平均スコア: {np.mean(scores):.3f}")
    print(f"  高スコア(>0.5)予測: {sum(1 for s in scores if s > 0.5)}/{len(scores)}")
    
    # 問題の診断
    print("\n=== 問題の診断 ===")
    if np.mean(center_errors) > 100:
        print("❌ 中心位置の予測が大きくずれています")
    elif np.mean(center_errors) > 50:
        print("⚠️  中心位置の予測にやや問題があります")
    else:
        print("✓ 中心位置の予測は比較的良好です")
    
    if np.mean(width_errors) > 100 or np.mean(height_errors) > 100:
        print("❌ サイズ予測が大きくずれています")
    elif np.mean(width_errors) > 50 or np.mean(height_errors) > 50:
        print("⚠️  サイズ予測にやや問題があります")
    else:
        print("✓ サイズ予測は比較的良好です")
    
    if np.mean(ious) < 0.1:
        print("❌ IoUが非常に低く、予測がGTとほとんど重なっていません")
    elif np.mean(ious) < 0.3:
        print("⚠️  IoUが低く、予測精度に問題があります")
    else:
        print("✓ IoUは許容範囲内です")
    
    # 相関分析
    score_iou_corr = np.corrcoef(scores, ious)[0, 1]
    print(f"\nスコアとIoUの相関係数: {score_iou_corr:.3f}")
    if abs(score_iou_corr) < 0.3:
        print("⚠️  予測信頼度とIoUの相関が低い - 信頼度が精度を反映していない可能性")

def main():
    """ボックス収束の詳細分析"""
    config = load_config()
    
    # 設定の取得
    image_dir = config['paths']['train_images']
    annotation_path = config['paths']['train_annotations']
    batch_size = 4
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
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # モデル
    model = AITECTDetector().to(device)
    
    # 学習済みモデルを読み込む
    model_path = config['paths']['model_save_path']
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"学習済みモデルを読み込みました: {model_path}")
    except Exception as e:
        print(f"モデルの読み込みに失敗: {e}")
        print("ランダム初期化のモデルで分析を続行します")
    
    print("\nボックス収束の分析を開始します...")
    
    # 分析実行
    results = analyze_box_predictions(model, dataloader, device, num_batches=10)
    
    if results:
        # 結果の可視化
        visualize_convergence_analysis(results)
        print("\n分析結果を box_convergence_analysis.png に保存しました")
    else:
        print("分析可能なデータが見つかりませんでした")

if __name__ == "__main__":
    main()