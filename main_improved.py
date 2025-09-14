import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CocoDataset
from model_improved_v2 import ImprovedDetector
from loss_improved_v2 import detection_loss_improved_v2
from loss_improved_v2_with_logging import ImprovedDetectionLossWithLogging
from utils.postprocess import postprocess_predictions, adjust_confidence_threshold
from utils.monitor import TrainingMonitor
from utils.validation import validate_detection
from utils.metrics import calculate_metrics_batch
from utils.logger import DetailedLogger
import json
import argparse
import os
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)

def load_config(config_path="config_improved.json"):
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_transform(train=True, config=None):
    """データ変換を取得"""
    transforms_list = []
    
    if train and config.get('augmentation', {}).get('enabled', False):
        aug_config = config['augmentation']
        
        # Random horizontal flip
        if aug_config.get('random_flip', 0) > 0:
            transforms_list.append(T.RandomHorizontalFlip(p=aug_config['random_flip']))
        
        # Color jitter
        if 'color_jitter' in aug_config:
            cj = aug_config['color_jitter']
            transforms_list.append(
                T.ColorJitter(
                    brightness=cj.get('brightness', 0),
                    contrast=cj.get('contrast', 0),
                    saturation=cj.get('saturation', 0),
                    hue=cj.get('hue', 0)
                )
            )
    
    # Resize and convert to tensor
    transforms_list.extend([
        T.Resize((config['training']['image_size'], config['training']['image_size'])),
        T.ToTensor(),
    ])
    
    return T.Compose(transforms_list)

def train(args):
    # 設定ファイルを読み込み
    config = load_config(args.config)
    
    # コマンドライン引数で上書き
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    
    # 設定の取得
    train_config = config['training']
    model_config = config['model']
    loss_config = config['loss']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # データセットとローダ
    train_transform = get_transform(train=True, config=config)
    val_transform = get_transform(train=False, config=config)
    
    train_dataset = CocoDataset(
        config['paths']['train_images'],
        config['paths']['train_annotations'],
        transform=train_transform
    )
    val_dataset = CocoDataset(
        config['paths']['val_images'],
        config['paths']['val_annotations'],
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    print("\n=== 改善された物体検出モデルの学習 ===")
    print(f"モデル: ImprovedDetector (ResNet50 + FPN)")
    print(f"アンカー数: {model_config['num_anchors']} per location")
    print(f"損失関数: Improved v2 (IoU-based assignment + GIoU + Focal Loss)")
    print(f"学習データ数: {len(train_dataset)}")
    print(f"検証データ数: {len(val_dataset)}")
    print(f"バッチサイズ: {train_config['batch_size']}")
    print(f"エポック数: {train_config['num_epochs']}")
    print(f"学習率: {train_config['learning_rate']}")
    print(f"デバイス: {device}")
    print("=====================================\n")
    
    # モデルの初期化
    model = ImprovedDetector(
        num_classes=model_config['num_classes'],
        num_anchors=model_config['num_anchors'],
        pretrained=model_config.get('pretrained_backbone', True)
    ).to(device)
    
    # オプティマイザとスケジューラ
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=0.0001
    )
    
    scheduler = None
    if train_config.get('lr_scheduler', {}).get('enabled', False):
        scheduler_config = train_config['lr_scheduler']
        if scheduler_config['type'] == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=train_config['num_epochs'] - scheduler_config.get('warmup_epochs', 0),
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
    
    # モニタリング
    monitor = TrainingMonitor()
    logger = DetailedLogger(log_dir="logs", experiment_name=f"improved_v2_{time.strftime('%Y%m%d_%H%M%S')}")
    logger.save_config(config)
    best_val_f1 = 0.0
    
    # 損失関数をログ機能付きに変更
    loss_params = {k: v for k, v in loss_config.items() if k not in ['type', 'comment']}
    loss_fn = ImprovedDetectionLossWithLogging(
        num_classes=model_config['num_classes'],
        **loss_params
    )
    
    # 学習ループ
    for epoch in range(train_config['num_epochs']):
        model.train()
        total_loss = 0.0
        epoch_start_time = time.time()
        
        # Warmup
        if epoch < train_config.get('lr_scheduler', {}).get('warmup_epochs', 0):
            warmup_lr = train_config['learning_rate'] * (epoch + 1) / train_config['lr_scheduler']['warmup_epochs']
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            for t in targets:
                t["boxes"] = t["boxes"].to(device)
                t["labels"] = t["labels"].to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss with logging
            loss = loss_fn(predictions, targets)
            stats = loss_fn.get_last_stats()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # ログ記録
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_training_step(
                epoch + 1, batch_idx, len(train_loader), 
                loss.item(), current_lr,
                loss_components={
                    'cls_loss': stats['cls_loss'],
                    'reg_loss': stats['reg_loss'],
                    'total_loss': stats['total_loss']
                },
                anchor_stats={
                    'positive_ratio': stats['positive_ratio'],
                    'negative_ratio': stats['negative_ratio'],
                    'ignore_ratio': stats['ignore_ratio'],
                    'avg_iou_positive': stats['avg_iou_positive']
                }
            )
            
            # Log iteration
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{train_config['num_epochs']}] "
                      f"Iter [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (cls: {stats['cls_loss']:.4f}, reg: {stats['reg_loss']:.4f}) "
                      f"Pos: {stats['num_pos']}, Neg: {stats['num_neg']}, "
                      f"Pos IoU: {stats['avg_iou_positive']:.3f}")
        
        # Epoch statistics
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")
        
        # エポックログ
        logger.log_epoch(epoch + 1, avg_loss, epoch_time, optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduling
        if scheduler and epoch >= train_config.get('lr_scheduler', {}).get('warmup_epochs', 0):
            scheduler.step()
        
        # Validation
        if (epoch + 1) % train_config['validation_interval'] == 0:
            print(f"\n[Validation at epoch {epoch+1}]")
            model.eval()
            
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                # Collect predictions
                for images, targets in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    
                    # Postprocess predictions
                    processed = postprocess_predictions(
                        outputs,
                        conf_threshold=config['postprocess']['conf_threshold'],
                        nms_threshold=config['postprocess']['nms_threshold'],
                        use_soft_nms=config['postprocess']['use_soft_nms']
                    )
                    
                    val_predictions.extend(processed)
                    val_targets.extend(targets)
                
                # Calculate metrics
                results = evaluate_predictions(val_predictions, val_targets)
                
                print(f"Validation Results:")
                print(f"  mAP@0.5: {results['map_50']:.4f}")
                print(f"  mAP@0.75: {results['map_75']:.4f}")
                print(f"  F1@0.5: {results['f1_50']:.4f}")
                print(f"  Precision@0.5: {results['precision_50']:.4f}")
                print(f"  Recall@0.5: {results['recall_50']:.4f}")
                
                # ログに記録
                logger.log_validation(epoch + 1, results)
                
                # Save best model
                if results['f1_50'] > best_val_f1:
                    best_val_f1 = results['f1_50']
                    torch.save(model.state_dict(), config['paths']['model_save_path'])
                    print(f"  New best model saved! F1: {best_val_f1:.4f}")
            
            model.train()
        
        # Save checkpoint
        if (epoch + 1) % train_config['save_interval'] == 0:
            checkpoint_path = config['paths']['model_save_path'].replace('.pth', f'_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_f1': best_val_f1,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    print(f"\n学習完了！最高検証F1スコア: {best_val_f1:.4f}")
    
    # 最終的なプロットとレポートを生成
    print("\nGenerating final reports and plots...")
    logger.save_all_plots()
    print(f"Training logs saved to: {logger.log_dir}")
    print(f"Summary report: {os.path.join(logger.log_dir, 'summary_report.txt')}")
    print(f"Training curves: {os.path.join(logger.log_dir, 'training_curves.png')}")
    print(f"Loss components: {os.path.join(logger.log_dir, 'loss_components.png')}")
    print(f"Anchor statistics: {os.path.join(logger.log_dir, 'anchor_statistics.png')}")

def evaluate_predictions(predictions, targets):
    """予測結果を評価"""
    from torchvision.ops import box_iou
    
    all_scores = []
    all_labels = []
    all_ious = []
    
    for pred, target in zip(predictions, targets):
        if len(pred['boxes']) == 0 or len(target['boxes']) == 0:
            continue
        
        # Calculate IoU matrix
        ious = box_iou(pred['boxes'].cpu(), target['boxes'].cpu())
        
        # For each prediction, find best matching GT
        max_ious, _ = ious.max(dim=1)
        
        all_scores.extend(pred['scores'].cpu().tolist())
        all_ious.extend(max_ious.tolist())
    
    # Calculate metrics at different IoU thresholds
    results = {}
    
    for iou_thresh in [0.5, 0.75]:
        # Count TP, FP
        tp = sum(1 for iou, score in zip(all_ious, all_scores) if iou >= iou_thresh)
        fp = len(all_ious) - tp
        fn = sum(len(t['boxes']) for t in targets) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[f'precision_{int(iou_thresh*100)}'] = precision
        results[f'recall_{int(iou_thresh*100)}'] = recall
        results[f'f1_{int(iou_thresh*100)}'] = f1
        results[f'map_{int(iou_thresh*100)}'] = precision  # Simplified mAP
    
    return results

def main():
    parser = argparse.ArgumentParser(description='改善された物体検出モデルの学習')
    parser.add_argument('--config', type=str, default='config_improved.json',
                        help='設定ファイルのパス')
    parser.add_argument('--epochs', type=int, default=None,
                        help='エポック数')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=None,
                        help='学習率')
    parser.add_argument('--resume', type=str, default=None,
                        help='チェックポイントから再開')
    
    args = parser.parse_args()
    
    # Set up matplotlib backend
    import matplotlib
    matplotlib.use('Agg')
    
    train(args)

if __name__ == "__main__":
    main()