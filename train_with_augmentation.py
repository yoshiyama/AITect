import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms as T
from dataset import CocoDataset
from model import AITECTDetector
from loss import detection_loss
import json
import os
import time
from collections import defaultdict

class RandomGaussianNoise:
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

def get_augmented_transform(config, train=True):
    """データ拡張を含む変換"""
    transforms_list = []
    
    if train and config.get('augmentation', {}).get('enabled', False):
        aug = config['augmentation']
        
        # ランダム水平反転
        if aug.get('horizontal_flip', 0) > 0:
            transforms_list.append(T.RandomHorizontalFlip(p=aug['horizontal_flip']))
        
        # ランダム垂直反転
        if aug.get('vertical_flip', 0) > 0:
            transforms_list.append(T.RandomVerticalFlip(p=aug['vertical_flip']))
        
        # 色調変更
        if 'color_jitter' in aug:
            cj = aug['color_jitter']
            transforms_list.append(T.ColorJitter(
                brightness=cj.get('brightness', 0),
                contrast=cj.get('contrast', 0),
                saturation=cj.get('saturation', 0),
                hue=cj.get('hue', 0)
            ))
    
    # リサイズとテンソル化は必須
    transforms_list.extend([
        T.Resize((config['training']['image_size'], config['training']['image_size'])),
        T.ToTensor(),
    ])
    
    # ガウシアンノイズ
    if train and config.get('augmentation', {}).get('gaussian_noise', 0) > 0:
        transforms_list.append(RandomGaussianNoise(std=config['augmentation']['gaussian_noise']))
    
    return T.Compose(transforms_list)

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)

def train_improved():
    # 設定読み込み
    with open('config_improved_training.json', 'r') as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # データセット
    train_transform = get_augmented_transform(config, train=True)
    val_transform = get_augmented_transform(config, train=False)
    
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
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # モデル
    model = AITECTDetector(
        num_classes=config['model']['num_classes'],
        grid_size=config['model']['grid_size'],
        num_anchors=config['model']['num_anchors']
    ).to(device)
    
    # 既存の重みがあれば読み込む（転移学習）
    if os.path.exists("result/aitect_model_simple.pth"):
        print("Loading pretrained weights from previous training...")
        try:
            # 部分的に重みを読み込む（アンカー数が異なる場合に対応）
            pretrained_dict = torch.load("result/aitect_model_simple.pth")
            model_dict = model.state_dict()
            
            # 互換性のある重みのみを選択
            compatible_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    compatible_dict[k] = v
                else:
                    print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model_dict[k].shape}")
            
            # 読み込める重みのみを更新
            model_dict.update(compatible_dict)
            model.load_state_dict(model_dict)
            print(f"Loaded {len(compatible_dict)}/{len(pretrained_dict)} weights from pretrained model")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            print("Starting from scratch...")
    
    # オプティマイザ
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=0.0001)
    
    # 学習率スケジューラ
    scheduler = None
    if config['training'].get('lr_scheduler', {}).get('enabled', False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=config['training']['lr_scheduler']['min_lr']
        )
    
    # 学習ログ
    history = defaultdict(list)
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n=== 改善された学習設定 ===")
    print(f"エポック数: {config['training']['num_epochs']}")
    print(f"バッチサイズ: {config['training']['batch_size']}")
    print(f"学習率: {config['training']['learning_rate']}")
    print(f"データ拡張: {'有効' if config['augmentation']['enabled'] else '無効'}")
    print(f"学習率スケジューラ: {'有効' if config['training']['lr_scheduler']['enabled'] else '無効'}")
    print(f"デバイス: {device}")
    print("=======================\n")
    
    # 学習ループ
    for epoch in range(config['training']['num_epochs']):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                t['labels'] = t['labels'].to(device)
            
            # Forward
            predictions = model(images)
            loss = detection_loss(
                predictions, targets,
                loss_type=config['training']['loss_type'],
                iou_weight=config['training']['iou_weight'],
                l1_weight=config['training']['l1_weight'],
                use_focal=config['model']['use_focal_loss']
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{config['training']['num_epochs']}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # エポック終了時の処理
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # 検証
        if (epoch + 1) % config['training']['validation_interval'] == 0:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    for t in targets:
                        t['boxes'] = t['boxes'].to(device)
                        t['labels'] = t['labels'].to(device)
                    
                    predictions = model(images)
                    loss = detection_loss(
                        predictions, targets,
                        loss_type=config['training']['loss_type'],
                        iou_weight=config['training']['iou_weight'],
                        l1_weight=config['training']['l1_weight'],
                        use_focal=config['model']['use_focal_loss']
                    )
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            print(f"\nEpoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss - config['training']['early_stopping']['min_delta']:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # ベストモデルを保存
                torch.save(model.state_dict(), config['paths']['model_save_path'].replace('.pth', '_best.pth'))
                print("New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= config['training']['early_stopping']['patience']:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # 学習率更新
        if scheduler:
            scheduler.step()
        
        # 定期保存
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = config['paths']['model_save_path'].replace('.pth', f'_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'history': history
            }, checkpoint_path)
    
    # 最終モデル保存
    torch.save(model.state_dict(), config['paths']['model_save_path'])
    print(f"\nTraining completed! Final model saved to {config['paths']['model_save_path']}")
    
    # 学習履歴を保存
    with open('training_history.json', 'w') as f:
        json.dump(history, f)

if __name__ == "__main__":
    train_improved()