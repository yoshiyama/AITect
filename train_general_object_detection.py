import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms as T
from model import AITECTDetector
from dataset import CocoDataset
from loss import detection_loss
import json
import os
import time
from collections import defaultdict
import numpy as np

def setup_voc_dataset():
    """Pascal VOC 2012データセットのセットアップ"""
    print("=== Pascal VOC 2012 Dataset Setup ===")
    
    data_dir = "./datasets/voc2012"
    os.makedirs(data_dir, exist_ok=True)
    
    # VOC 2012をダウンロード（物体検出の標準データセット）
    try:
        # 訓練データ
        train_dataset = torchvision.datasets.VOCDetection(
            root=data_dir,
            year='2012',
            image_set='train',
            download=True,
            transform=transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
        )
        
        # 検証データ
        val_dataset = torchvision.datasets.VOCDetection(
            root=data_dir,
            year='2012', 
            image_set='val',
            download=True,
            transform=transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # VOCフォーマットをCOCO形式に変換
        voc_to_coco_format(train_dataset, f"{data_dir}/train_coco.json", "train")
        voc_to_coco_format(val_dataset, f"{data_dir}/val_coco.json", "val")
        
        return True
        
    except Exception as e:
        print(f"VOCダウンロードエラー: {e}")
        return False

def voc_to_coco_format(voc_dataset, output_path, split):
    """VOCフォーマットをCOCO形式に変換"""
    print(f"Converting {split} to COCO format...")
    
    # VOCクラス
    voc_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': i+1, 'name': cls} for i, cls in enumerate(voc_classes)]
    }
    
    ann_id = 0
    
    for idx in range(len(voc_dataset)):
        try:
            img, target = voc_dataset[idx]
            
            # 画像情報
            img_info = {
                'id': idx,
                'file_name': f"{split}_{idx:06d}.jpg",
                'width': 512,
                'height': 512
            }
            coco_data['images'].append(img_info)
            
            # アノテーション
            objects = target['annotation']['object']
            if not isinstance(objects, list):
                objects = [objects]
            
            for obj in objects:
                bbox = obj['bndbox']
                x1 = float(bbox['xmin'])
                y1 = float(bbox['ymin'])
                x2 = float(bbox['xmax'])
                y2 = float(bbox['ymax'])
                
                # 512x512にリサイズ
                orig_width = float(target['annotation']['size']['width'])
                orig_height = float(target['annotation']['size']['height'])
                
                x1 = x1 * 512 / orig_width
                x2 = x2 * 512 / orig_width
                y1 = y1 * 512 / orig_height
                y2 = y2 * 512 / orig_height
                
                ann = {
                    'id': ann_id,
                    'image_id': idx,
                    'category_id': voc_classes.index(obj['name']) + 1,
                    'bbox': [x1, y1, x2-x1, y2-y1],
                    'area': (x2-x1) * (y2-y1),
                    'iscrowd': 0
                }
                coco_data['annotations'].append(ann)
                ann_id += 1
                
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue
    
    # 保存
    with open(output_path, 'w') as f:
        json.dump(coco_data, f)
    
    print(f"Converted {len(coco_data['images'])} images with {len(coco_data['annotations'])} annotations")
    
def create_general_config():
    """一般物体検出用の設定を作成"""
    config = {
        "training": {
            "num_epochs": 50,
            "batch_size": 8,
            "learning_rate": 0.001,
            "lr_scheduler": {
                "enabled": True,
                "type": "cosine",
                "warmup_epochs": 3,
                "min_lr": 0.00001
            },
            "image_size": 512,
            "save_interval": 5,
            "validation_interval": 2,
            "gradient_accumulation_steps": 2
        },
        "model": {
            "num_classes": 20,  # VOCは20クラス
            "grid_size": 16,
            "num_anchors": 9,   # 3スケール × 3アスペクト比
            "use_focal_loss": True,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0
        },
        "loss": {
            "type": "mixed",
            "iou_weight": 2.0,
            "l1_weight": 1.0,
            "use_focal": True
        },
        "augmentation": {
            "enabled": True,
            "horizontal_flip": 0.5,
            "vertical_flip": 0.1,
            "color_jitter": {
                "brightness": 0.3,
                "contrast": 0.3,
                "saturation": 0.3,
                "hue": 0.1
            },
            "random_scale": {
                "enabled": True,
                "min_scale": 0.8,
                "max_scale": 1.2
            }
        },
        "postprocess": {
            "conf_threshold": 0.3,
            "nms_threshold": 0.5,
            "max_detections": 100
        }
    }
    
    with open('config_general_detection.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    return config

def train_general_detection_model():
    """一般物体検出モデルの学習"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 設定
    config = create_general_config()
    
    print("\n=== 一般物体検出モデルの学習 ===")
    print(f"デバイス: {device}")
    print(f"エポック数: {config['training']['num_epochs']}")
    print(f"バッチサイズ: {config['training']['batch_size']}")
    print(f"クラス数: {config['model']['num_classes']}")
    
    # データ拡張
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor()
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    # データセット
    # まずはSimple Shapesで動作確認
    if os.path.exists("datasets/simple_shapes"):
        print("\n1. Simple Shapesで動作確認...")
        train_dataset = CocoDataset(
            "datasets/simple_shapes/images",
            "datasets/simple_shapes/annotations.json",
            transform=train_transform
        )
        val_dataset = train_dataset  # 簡易的に同じものを使用
        
    # VOCがあれば使用
    elif os.path.exists("datasets/voc2012/train_coco.json"):
        print("\n2. Pascal VOCで学習...")
        train_dataset = CocoDataset(
            "datasets/voc2012/VOCdevkit/VOC2012/JPEGImages",
            "datasets/voc2012/train_coco.json",
            transform=train_transform
        )
        val_dataset = CocoDataset(
            "datasets/voc2012/VOCdevkit/VOC2012/JPEGImages",
            "datasets/voc2012/val_coco.json",
            transform=val_transform
        )
    else:
        print("データセットが見つかりません。まずsetup_voc_dataset()を実行してください。")
        return
    
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
    
    # 最適化
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=0.0001
    )
    
    # 学習率スケジューラ
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training']['lr_scheduler']['min_lr']
    )
    
    # 学習ログ
    history = defaultdict(list)
    best_val_loss = float('inf')
    
    print(f"\n訓練サンプル数: {len(train_dataset)}")
    print(f"検証サンプル数: {len(val_dataset)}")
    print("\n学習開始...")
    
    # 学習ループ
    for epoch in range(config['training']['num_epochs']):
        # 訓練
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
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{config['training']['num_epochs']}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # エポック終了
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
                        loss_type=config['loss']['type'],
                        iou_weight=config['loss']['iou_weight'],
                        l1_weight=config['loss']['l1_weight'],
                        use_focal=config['loss']['use_focal']
                    )
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            print(f"\nEpoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # ベストモデル保存
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "result/aitect_general_detection_best.pth")
                print("Best model saved!")
        
        # 学習率更新
        scheduler.step()
        
        # 定期保存
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = f"result/aitect_general_detection_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'config': config
            }, checkpoint_path)
    
    # 最終モデル保存
    torch.save(model.state_dict(), "result/aitect_general_detection_final.pth")
    print(f"\n学習完了! モデル保存: result/aitect_general_detection_final.pth")
    
    # 学習履歴を保存
    with open('general_detection_history.json', 'w') as f:
        json.dump(dict(history), f)
    
    return model

if __name__ == "__main__":
    print("一般物体検出モデルの学習準備")
    print("="*60)
    
    # VOCデータセットのセットアップを試みる
    if not os.path.exists("datasets/voc2012/train_coco.json"):
        print("\nVOCデータセットをダウンロード中...")
        if setup_voc_dataset():
            print("✅ VOCデータセットの準備完了")
        else:
            print("❌ VOCダウンロード失敗。Simple Shapesを使用します。")
    
    # 学習開始
    print("\n" + "="*60)
    model = train_general_detection_model()