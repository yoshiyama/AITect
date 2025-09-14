import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision.transforms as T
from model import AITECTDetector
from dataset import CocoDataset
from loss import detection_loss
import json
import os
import time
from collections import defaultdict
import matplotlib.pyplot as plt

class MultiClassAugmentation:
    """マルチクラス検出用のデータ拡張"""
    def __init__(self, config):
        self.transforms = []
        
        if config.get('horizontal_flip', 0) > 0:
            self.transforms.append(T.RandomHorizontalFlip(p=config['horizontal_flip']))
        
        if config.get('color_jitter'):
            cj = config['color_jitter']
            self.transforms.append(T.ColorJitter(
                brightness=cj.get('brightness', 0),
                contrast=cj.get('contrast', 0),
                saturation=cj.get('saturation', 0),
                hue=cj.get('hue', 0)
            ))
        
        self.transforms.extend([
            T.Resize((512, 512)),
            T.ToTensor()
        ])
        
        self.transform = T.Compose(self.transforms)
    
    def __call__(self, img):
        return self.transform(img)

def create_multiclass_config():
    """マルチクラス検出用の設定"""
    config = {
        "training": {
            "num_epochs": 30,
            "batch_size": 4,
            "learning_rate": 0.001,
            "lr_scheduler": {
                "enabled": True,
                "type": "step",
                "step_size": 10,
                "gamma": 0.1
            },
            "image_size": 512,
            "save_interval": 5,
            "validation_interval": 1
        },
        "model": {
            "num_classes": 10,  # 10カテゴリ
            "grid_size": 16,
            "num_anchors": 9,   # 3スケール × 3アスペクト比
            "backbone": "resnet18",
            "use_pretrained_backbone": True
        },
        "loss": {
            "type": "mixed",
            "iou_weight": 5.0,    # IoU損失を重視
            "l1_weight": 1.0,
            "use_focal": True,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0
        },
        "augmentation": {
            "horizontal_flip": 0.5,
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1
            }
        },
        "postprocess": {
            "conf_threshold": 0.3,
            "nms_threshold": 0.5,
            "max_detections": 100
        }
    }
    
    with open('config_multiclass.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    return config

def train_multiclass_model():
    """マルチクラス一般物体検出モデルの学習"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = create_multiclass_config()
    
    print("=== マルチクラス物体検出モデルの学習 ===")
    print(f"デバイス: {device}")
    print(f"カテゴリ数: {config['model']['num_classes']}")
    print(f"エポック数: {config['training']['num_epochs']}")
    
    # データ拡張
    train_transform = MultiClassAugmentation(config['augmentation'])
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    # データセット
    # 基本データセットは変換なしで読み込み
    from PIL import Image
    
    class MiniCocoDataset(torch.utils.data.Dataset):
        def __init__(self, img_dir, ann_file):
            with open(ann_file, 'r') as f:
                self.coco_data = json.load(f)
            self.img_dir = img_dir
            self.images = self.coco_data['images']
            self.annotations = self.coco_data['annotations']
            
            # 画像IDごとにアノテーションをグループ化
            self.img_to_anns = defaultdict(list)
            for ann in self.annotations:
                self.img_to_anns[ann['image_id']].append(ann)
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            img_info = self.images[idx]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            img = Image.open(img_path).convert('RGB')
            
            # アノテーション取得
            anns = self.img_to_anns[img_info['id']]
            
            boxes = []
            labels = []
            
            for ann in anns:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x+w, y+h])
                labels.append(ann['category_id'])
            
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            }
            
            return img, target
    
    dataset = MiniCocoDataset(
        "datasets/mini_coco/images",
        "datasets/mini_coco/annotations.json"
    )
    
    # 80:20で分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # インデックスで分割してから変換を適用
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    
    # カスタムサブセット
    class TransformSubset(torch.utils.data.Subset):
        def __init__(self, dataset, indices, transform):
            super().__init__(dataset, indices)
            self.transform = transform
            
        def __getitem__(self, idx):
            img, target = self.dataset[self.indices[idx]]
            if self.transform:
                img = self.transform(img)
            else:
                # 変換がない場合もテンソルに変換
                img = transforms.ToTensor()(img)
            return img, target
    
    train_dataset = TransformSubset(dataset, train_indices, train_transform)
    val_dataset = TransformSubset(dataset, val_indices, val_transform)
    
    # データローダー
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        return images, list(targets)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # マルチプロセスを無効化
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # マルチプロセスを無効化
    )
    
    # モデル（新規作成）
    model = AITECTDetector(
        num_classes=config['model']['num_classes'],
        grid_size=config['model']['grid_size'],
        num_anchors=config['model']['num_anchors']
    ).to(device)
    
    # 最適化
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=0.0005
    )
    
    # 学習率スケジューラ
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['lr_scheduler']['step_size'],
        gamma=config['training']['lr_scheduler']['gamma']
    )
    
    # 学習記録
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    
    print(f"\n訓練サンプル数: {len(train_dataset)}")
    print(f"検証サンプル数: {len(val_dataset)}")
    print("\n学習開始...\n")
    
    # 学習ループ
    for epoch in range(config['training']['num_epochs']):
        # 訓練
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                t['labels'] = t['labels'].to(device)
            
            # Forward
            predictions = model(images)
            
            # 損失計算
            loss = detection_loss(
                predictions, targets,
                loss_type=config['loss']['type'],
                iou_weight=config['loss']['iou_weight'],
                l1_weight=config['loss']['l1_weight'],
                use_focal=config['loss']['use_focal']
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{config['training']['num_epochs']}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        # エポック終了
        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # 検証
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                for t in targets:
                    t['boxes'] = t['boxes'].to(device)
                    t['labels'] = t['labels'].to(device)
                
                predictions = model(images)
                loss = detection_loss(
                    predictions, targets,
                    loss_type=config['loss']['type'],
                    iou_weight=config['loss']['iou_weight'],
                    l1_weight=config['loss']['l1_weight'],
                    use_focal=config['loss']['use_focal']
                )
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        history['val_loss'].append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # ベストモデル保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "result/aitect_multiclass_best.pth")
            print("  ✅ Best model saved!")
        
        # 学習率更新
        scheduler.step()
        
        # 定期保存
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = f"result/aitect_multiclass_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_train_loss,
                'config': config,
                'history': history
            }, checkpoint_path)
        
        print("-" * 60)
    
    # 最終モデル保存
    torch.save(model.state_dict(), "result/aitect_multiclass_final.pth")
    print(f"\n✅ 学習完了!")
    print(f"最終モデル: result/aitect_multiclass_final.pth")
    print(f"ベストモデル: result/aitect_multiclass_best.pth (Val Loss: {best_val_loss:.4f})")
    
    # 学習履歴を保存
    with open('multiclass_training_history.json', 'w') as f:
        json.dump(history, f)
    
    # 学習曲線をプロット
    plot_training_history(history)
    
    return model

def plot_training_history(history):
    """学習履歴のプロット"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 損失
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 学習率
    ax2.plot(epochs, history['learning_rate'], 'g-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('multiclass_training_history.png', dpi=150)
    print("\n📊 学習履歴: multiclass_training_history.png")

if __name__ == "__main__":
    # マルチクラス検出モデルの学習
    model = train_multiclass_model()
    
    print("\n" + "="*60)
    print("次のステップ:")
    print("1. python evaluate_multiclass_model.py  # モデル評価")
    print("2. python visualize_multiclass_results.py  # 結果の可視化")