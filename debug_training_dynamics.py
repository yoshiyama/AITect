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

def track_predictions_during_training(model, dataloader, device, config, num_steps=50):
    """学習中の予測の変化を追跡"""
    
    # 最初のバッチを固定して使用
    fixed_batch = next(iter(dataloader))
    fixed_images, fixed_targets = fixed_batch
    fixed_images = fixed_images.to(device)
    
    for t in fixed_targets:
        t["boxes"] = t["boxes"].to(device)
        t["labels"] = t["labels"].to(device)
    
    # 特定のGTボックスを追跡（最初の画像の最初のGT）
    if len(fixed_targets) > 0 and fixed_targets[0]["boxes"].shape[0] > 0:
        tracked_gt = fixed_targets[0]["boxes"][0]  # [x1, y1, x2, y2]
        tracked_gt_cx = (tracked_gt[0] + tracked_gt[2]) / 2
        tracked_gt_cy = (tracked_gt[1] + tracked_gt[3]) / 2
        tracked_gt_w = tracked_gt[2] - tracked_gt[0]
        tracked_gt_h = tracked_gt[3] - tracked_gt[1]
    else:
        print("GTボックスが見つかりません")
        return None
    
    # オプティマイザ
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # 損失関数の設定
    use_improved = config['model'].get('use_improved_loss', False)
    loss_type = config['training'].get('loss_type', 'mixed')
    iou_weight = config['training'].get('iou_weight', 2.0)
    l1_weight = config['training'].get('l1_weight', 0.5)
    use_focal = config['model'].get('use_focal_loss', True)
    
    # 追跡データ
    tracking_data = {
        'step': [],
        'loss': [],
        'closest_pred_cx': [],
        'closest_pred_cy': [],
        'closest_pred_w': [],
        'closest_pred_h': [],
        'closest_pred_score': [],
        'center_error': [],
        'iou': [],
        'conf_loss_component': [],
        'reg_loss_component': []
    }
    
    print("学習中の予測変化を追跡します...")
    print(f"追跡するGTボックス: cx={tracked_gt_cx:.1f}, cy={tracked_gt_cy:.1f}, w={tracked_gt_w:.1f}, h={tracked_gt_h:.1f}")
    
    model.train()
    
    for step in range(num_steps):
        # 順伝播
        preds = model(fixed_images)
        
        # 損失計算
        if use_improved:
            loss = detection_loss_improved(preds, fixed_targets, 
                                         loss_type=loss_type,
                                         iou_weight=iou_weight, 
                                         l1_weight=l1_weight,
                                         use_focal=use_focal)
        else:
            loss = detection_loss(preds, fixed_targets, 
                                loss_type=loss_type,
                                iou_weight=iou_weight, 
                                l1_weight=l1_weight,
                                use_focal=use_focal)
        
        # 最初の画像の予測を分析
        pred = preds[0]  # [N, 5]
        pred_boxes_xywh = pred[:, :4]
        pred_scores = pred[:, 4]
        
        # 予測ボックスを[x1, y1, x2, y2]形式に変換
        pred_x1 = pred_boxes_xywh[:, 0] - pred_boxes_xywh[:, 2] / 2
        pred_y1 = pred_boxes_xywh[:, 1] - pred_boxes_xywh[:, 3] / 2
        pred_x2 = pred_boxes_xywh[:, 0] + pred_boxes_xywh[:, 2] / 2
        pred_y2 = pred_boxes_xywh[:, 1] + pred_boxes_xywh[:, 3] / 2
        pred_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
        
        # 追跡中のGTに最も近い予測を見つける
        pred_centers = pred_boxes_xywh[:, :2]
        gt_center = torch.tensor([tracked_gt_cx, tracked_gt_cy], device=device)
        distances = torch.norm(pred_centers - gt_center, dim=1)
        min_idx = distances.argmin()
        
        closest_pred = pred_boxes_xywh[min_idx]
        closest_score = torch.sigmoid(pred_scores[min_idx]).item()
        
        # IoU計算
        iou = box_iou(pred_boxes_xyxy[min_idx:min_idx+1], 
                      tracked_gt.unsqueeze(0))[0, 0].item()
        
        # データ記録
        tracking_data['step'].append(step)
        tracking_data['loss'].append(loss.item())
        tracking_data['closest_pred_cx'].append(closest_pred[0].item())
        tracking_data['closest_pred_cy'].append(closest_pred[1].item())
        tracking_data['closest_pred_w'].append(closest_pred[2].item())
        tracking_data['closest_pred_h'].append(closest_pred[3].item())
        tracking_data['closest_pred_score'].append(closest_score)
        tracking_data['center_error'].append(distances[min_idx].item())
        tracking_data['iou'].append(iou)
        
        # 損失コンポーネントの推定（簡易版）
        conf_target = 1.0 if distances[min_idx].item() < 50 else 0.0
        conf_loss = nn.functional.binary_cross_entropy_with_logits(
            pred_scores[min_idx].unsqueeze(0), 
            torch.tensor([conf_target], device=device)
        ).item()
        
        if conf_target > 0.5:
            reg_loss = nn.functional.l1_loss(
                closest_pred,
                torch.tensor([tracked_gt_cx, tracked_gt_cy, tracked_gt_w, tracked_gt_h], device=device)
            ).item()
        else:
            reg_loss = 0.0
        
        tracking_data['conf_loss_component'].append(conf_loss)
        tracking_data['reg_loss_component'].append(reg_loss)
        
        # 逆伝播と更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: Loss={loss.item():.4f}, Center Error={distances[min_idx].item():.1f}, IoU={iou:.3f}")
    
    return tracking_data, tracked_gt_cx.item(), tracked_gt_cy.item(), tracked_gt_w.item(), tracked_gt_h.item()

def visualize_training_dynamics(tracking_data, gt_cx, gt_cy, gt_w, gt_h):
    """学習ダイナミクスの可視化"""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # 1. 損失の推移
    ax = axes[0, 0]
    ax.plot(tracking_data['step'], tracking_data['loss'])
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Total Loss')
    ax.set_title('Loss During Training')
    ax.grid(True)
    
    # 2. 中心位置の推移
    ax = axes[0, 1]
    ax.plot(tracking_data['step'], tracking_data['closest_pred_cx'], label='Pred X', alpha=0.7)
    ax.plot(tracking_data['step'], tracking_data['closest_pred_cy'], label='Pred Y', alpha=0.7)
    ax.axhline(gt_cx, color='red', linestyle='--', label=f'GT X={gt_cx:.1f}')
    ax.axhline(gt_cy, color='blue', linestyle='--', label=f'GT Y={gt_cy:.1f}')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Center Coordinate')
    ax.set_title('Box Center Position During Training')
    ax.legend()
    ax.grid(True)
    
    # 3. サイズの推移
    ax = axes[0, 2]
    ax.plot(tracking_data['step'], tracking_data['closest_pred_w'], label='Pred W', alpha=0.7)
    ax.plot(tracking_data['step'], tracking_data['closest_pred_h'], label='Pred H', alpha=0.7)
    ax.axhline(gt_w, color='red', linestyle='--', label=f'GT W={gt_w:.1f}')
    ax.axhline(gt_h, color='blue', linestyle='--', label=f'GT H={gt_h:.1f}')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Size')
    ax.set_title('Box Size During Training')
    ax.legend()
    ax.grid(True)
    
    # 4. 中心エラーの推移
    ax = axes[1, 0]
    ax.plot(tracking_data['step'], tracking_data['center_error'])
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Center Error (pixels)')
    ax.set_title('Center Position Error')
    ax.grid(True)
    
    # 5. IoUの推移
    ax = axes[1, 1]
    ax.plot(tracking_data['step'], tracking_data['iou'])
    ax.set_xlabel('Training Step')
    ax.set_ylabel('IoU')
    ax.set_title('IoU with GT During Training')
    ax.set_ylim([0, 1])
    ax.grid(True)
    
    # 6. 予測信頼度の推移
    ax = axes[1, 2]
    ax.plot(tracking_data['step'], tracking_data['closest_pred_score'])
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Prediction Score')
    ax.set_title('Confidence Score During Training')
    ax.set_ylim([0, 1])
    ax.grid(True)
    
    # 7. 損失コンポーネント
    ax = axes[2, 0]
    ax.plot(tracking_data['step'], tracking_data['conf_loss_component'], label='Conf Loss', alpha=0.7)
    ax.plot(tracking_data['step'], tracking_data['reg_loss_component'], label='Reg Loss', alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss Component')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True)
    
    # 8. 位置の軌跡（2D）
    ax = axes[2, 1]
    ax.scatter(tracking_data['closest_pred_cx'], tracking_data['closest_pred_cy'], 
               c=tracking_data['step'], cmap='viridis', alpha=0.6, s=30)
    ax.scatter([gt_cx], [gt_cy], color='red', marker='x', s=200, label='GT')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Predicted Center Trajectory')
    ax.legend()
    ax.grid(True)
    
    # 9. サイズの軌跡（2D）
    ax = axes[2, 2]
    ax.scatter(tracking_data['closest_pred_w'], tracking_data['closest_pred_h'], 
               c=tracking_data['step'], cmap='viridis', alpha=0.6, s=30)
    ax.scatter([gt_w], [gt_h], color='red', marker='x', s=200, label='GT')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title('Predicted Size Trajectory')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_dynamics.png', dpi=150)
    plt.close()
    
    # 診断結果
    print("\n=== 学習ダイナミクスの診断 ===")
    
    # 損失の減少
    initial_loss = tracking_data['loss'][0]
    final_loss = tracking_data['loss'][-1]
    print(f"損失の変化: {initial_loss:.4f} → {final_loss:.4f} (減少率: {100*(1-final_loss/initial_loss):.1f}%)")
    
    # 中心位置の収束
    initial_error = tracking_data['center_error'][0]
    final_error = tracking_data['center_error'][-1]
    print(f"中心エラーの変化: {initial_error:.1f} → {final_error:.1f} pixels")
    
    # IoUの改善
    initial_iou = tracking_data['iou'][0]
    final_iou = tracking_data['iou'][-1]
    print(f"IoUの変化: {initial_iou:.3f} → {final_iou:.3f}")
    
    # 収束の診断
    if final_error > initial_error * 0.8:
        print("\n❌ 予測位置が正解に収束していません！")
        print("   → 学習率が不適切、または損失関数に問題がある可能性")
    
    if final_iou < 0.1:
        print("❌ IoUが非常に低いままです！")
        print("   → 回帰損失が効いていない可能性")
    
    # 振動の検出
    errors = tracking_data['center_error']
    if len(errors) > 10:
        recent_std = np.std(errors[-10:])
        if recent_std > 10:
            print("⚠️  予測が振動しています（収束していない）")

def main():
    """学習ダイナミクスの詳細分析"""
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
    
    # モデル（新規初期化）
    model = AITECTDetector().to(device)
    
    print("学習ダイナミクスの分析を開始します...")
    print("注意: この分析は新規初期化されたモデルから開始します")
    
    # 学習中の予測を追跡
    result = track_predictions_during_training(model, dataloader, device, config, num_steps=100)
    
    if result is not None:
        tracking_data, gt_cx, gt_cy, gt_w, gt_h = result
        
        # 結果の可視化
        visualize_training_dynamics(tracking_data, gt_cx, gt_cy, gt_w, gt_h)
        print("\n分析結果を training_dynamics.png に保存しました")
    else:
        print("分析を実行できませんでした")

if __name__ == "__main__":
    main()