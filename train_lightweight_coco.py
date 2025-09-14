import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import os
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from model_lightweight import create_lightweight_model
from utils.postprocess import postprocess_predictions

class COCODatasetLightweight(Dataset):
    """COCO Dataset for lightweight detection"""
    
    def __init__(self, img_dir, ann_file, transform=None, target_classes=None, input_size=416):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.transform = transform
        self.input_size = input_size
        
        # クラスのフィルタリング
        if target_classes:
            # 指定されたクラスのみ使用
            self.cat_ids = []
            self.class_names = []
            for class_name in target_classes:
                cat_ids = self.coco.getCatIds(catNms=[class_name])
                if cat_ids:
                    self.cat_ids.extend(cat_ids)
                    self.class_names.append(class_name)
            
            # 該当する画像IDを取得
            self.img_ids = []
            for cat_id in self.cat_ids:
                self.img_ids.extend(self.coco.getImgIds(catIds=[cat_id]))
            self.img_ids = list(set(self.img_ids))
            
            # クラスIDのマッピング（0から始まる連番に）
            self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
            
        else:
            # 全クラス使用
            self.img_ids = self.coco.getImgIds()
            self.cat_ids = self.coco.getCatIds()
            self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
            self.class_names = [cat['name'] for cat in self.coco.loadCats(self.cat_ids)]
        
        print(f"Dataset initialized:")
        print(f"  Images: {len(self.img_ids)}")
        print(f"  Classes: {len(self.cat_ids)} - {self.class_names}")
        
        # デフォルト変換
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((self.input_size, self.input_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        # 画像読み込み
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            # 画像が見つからない場合は黒画像
            img = Image.new('RGB', (self.input_size, self.input_size))
        
        # アノテーション取得
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)
        
        # バウンディングボックスとラベル
        boxes = []
        labels = []
        
        orig_w, orig_h = img.size
        
        for ann in anns:
            if ann['area'] > 0:  # 有効なアノテーションのみ
                x, y, w, h = ann['bbox']
                
                # 正規化（0-1の範囲に）
                x1 = x / orig_w
                y1 = y / orig_h
                x2 = (x + w) / orig_w
                y2 = (y + h) / orig_h
                
                # クリッピング
                x1 = max(0, min(1, x1))
                y1 = max(0, min(1, y1))
                x2 = max(0, min(1, x2))
                y2 = max(0, min(1, y2))
                
                if x2 > x1 and y2 > y1:  # 有効なボックスのみ
                    boxes.append([x1, y1, x2, y2])
                    labels.append(self.cat_id_to_label[ann['category_id']])
        
        # 画像変換
        img = self.transform(img)
        
        # ターゲット作成
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) * self.input_size,  # ピクセル座標に変換
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id])
        }
        
        return img, target

def collate_fn(batch):
    """バッチ処理用の関数"""
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(targets)

def compute_loss(predictions, targets, num_classes=1):
    """簡易的な損失関数"""
    device = predictions[0].device
    batch_size = len(targets)
    
    total_loss = 0
    
    # 各スケールで損失を計算
    for pred in predictions:
        grid_size = pred.size(1)
        num_anchors = pred.size(3)
        
        # 予測を展開
        pred = pred.view(batch_size, -1, 5 + num_classes)
        
        # とりあえずL2損失（簡易版）
        loss = torch.mean(pred ** 2)  # ダミー損失
        
        total_loss += loss
    
    return total_loss

def train_one_epoch(model, dataloader, optimizer, device):
    """1エポックの学習"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, targets in pbar:
        images = images.to(device)
        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)
        
        # Forward
        outputs = model(images)
        
        # 損失計算（簡易版）
        loss = compute_loss(outputs, targets, model.num_classes)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """評価"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                t['labels'] = t['labels'].to(device)
            
            outputs = model(images)
            loss = compute_loss(outputs, targets, model.num_classes)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description='Train lightweight COCO detector')
    parser.add_argument('--data_dir', default='./datasets/coco2017', help='COCO dataset directory')
    parser.add_argument('--classes', nargs='+', default=['person'], help='Classes to detect')
    parser.add_argument('--model_size', default='tiny', choices=['tiny', 'small', 'medium'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--input_size', type=int, default=416)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_dir', default='./result', help='Save directory')
    
    args = parser.parse_args()
    
    # デバイス
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # データセット準備
    print("\n=== Dataset Setup ===")
    
    # COCOデータセットが存在するかチェック
    train_ann = f"{args.data_dir}/annotations/instances_train2017.json"
    val_ann = f"{args.data_dir}/annotations/instances_val2017.json"
    
    if not os.path.exists(train_ann):
        print(f"COCO annotations not found at {train_ann}")
        print("\nPlease run first:")
        print("python setup_coco_training.py")
        return
    
    # データ拡張
    train_transform = T.Compose([
        T.RandomResizedCrop(args.input_size, scale=(0.8, 1.2)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # データセット
    train_dataset = COCODatasetLightweight(
        f"{args.data_dir}/train2017",
        train_ann,
        transform=train_transform,
        target_classes=args.classes,
        input_size=args.input_size
    )
    
    val_dataset = COCODatasetLightweight(
        f"{args.data_dir}/val2017",
        val_ann,
        transform=val_transform,
        target_classes=args.classes,
        input_size=args.input_size
    )
    
    # データローダー
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # モデル作成
    print("\n=== Model Setup ===")
    num_classes = len(train_dataset.cat_ids)
    model = create_lightweight_model(
        num_classes=num_classes,
        model_size=args.model_size,
        pretrained=True
    ).to(device)
    
    # オプティマイザ
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 学習ループ
    print("\n=== Training ===")
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 学習
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # 評価
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        
        # 学習率更新
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # モデル保存
        if (epoch + 1) % 10 == 0:
            save_path = f"{args.save_dir}/coco_{'_'.join(args.classes)}_{args.model_size}_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'classes': args.classes,
                'num_classes': num_classes
            }, save_path)
            print(f"Model saved to {save_path}")
    
    # 最終モデル保存
    final_path = f"{args.save_dir}/coco_{'_'.join(args.classes)}_{args.model_size}_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': args.classes,
        'num_classes': num_classes,
        'model_size': args.model_size,
        'input_size': args.input_size
    }, final_path)
    print(f"\nFinal model saved to {final_path}")
    
    # 学習曲線のプロット
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Progress - {args.model_size.upper()} model on {", ".join(args.classes)}')
    plt.legend()
    plt.savefig(f"{args.save_dir}/training_curve_{'_'.join(args.classes)}_{args.model_size}.png")
    plt.close()

if __name__ == "__main__":
    # 使用例を表示
    print("\n=== Lightweight COCO Object Detection Training ===")
    print("\n使用例:")
    print("1. 人物検出のみ（最軽量）:")
    print("   python train_lightweight_coco.py --classes person --model_size tiny --epochs 30")
    print("\n2. 複数クラス検出:")
    print("   python train_lightweight_coco.py --classes person car dog --model_size small")
    print("\n3. 全クラス検出（80クラス）:")
    print("   python train_lightweight_coco.py --model_size medium")
    print("\n" + "="*60 + "\n")
    
    # 実際の学習を開始
    # main()  # コメントアウト（実行時はコメントを外す）