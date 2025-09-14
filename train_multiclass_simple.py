import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import AITECTDetector
from loss import detection_loss
import json
import os
from PIL import Image
from collections import defaultdict

def train_multiclass_simple():
    """シンプルなマルチクラス学習（Mini COCO）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=== マルチクラス物体検出 - シンプル版 ===")
    
    # データセット読み込み
    class SimpleMiniCoco(torch.utils.data.Dataset):
        def __init__(self, transform=None):
            with open('datasets/mini_coco/annotations.json', 'r') as f:
                self.data = json.load(f)
            self.transform = transform or transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
            
            # 画像ごとのアノテーション
            self.img_anns = defaultdict(list)
            for ann in self.data['annotations']:
                self.img_anns[ann['image_id']].append(ann)
        
        def __len__(self):
            return len(self.data['images'])
        
        def __getitem__(self, idx):
            img_info = self.data['images'][idx]
            img_path = f"datasets/mini_coco/images/{img_info['file_name']}"
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            
            # ターゲット
            anns = self.img_anns[idx]
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
    
    # データセット
    dataset = SimpleMiniCoco()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        return images, list(targets)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # モデル
    model = AITECTDetector(num_classes=10, grid_size=16, num_anchors=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"訓練サンプル: {len(train_dataset)}, 検証サンプル: {len(val_dataset)}")
    print(f"カテゴリ数: 10")
    print("\n学習開始...")
    
    # 学習
    best_loss = float('inf')
    
    for epoch in range(20):
        # 訓練
        model.train()
        train_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                t['labels'] = t['labels'].to(device)
            
            predictions = model(images)
            loss = detection_loss(predictions, targets, use_focal=True)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}/20 - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        # 検証
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                for t in targets:
                    t['boxes'] = t['boxes'].to(device)
                    t['labels'] = t['labels'].to(device)
                
                predictions = model(images)
                loss = detection_loss(predictions, targets, use_focal=True)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1} - Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), "result/multiclass_simple_best.pth")
            print("✅ Best model saved!")
        
        print("-" * 50)
    
    print("\n学習完了!")
    return model

def evaluate_multiclass():
    """マルチクラスモデルの評価"""
    from utils.postprocess import postprocess_predictions
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデル読み込み
    model = AITECTDetector(num_classes=10, grid_size=16, num_anchors=3).to(device)
    model.load_state_dict(torch.load("result/multiclass_simple_best.pth"))
    model.eval()
    
    # テストデータ
    dataset = SimpleMiniCoco()
    
    # カテゴリ名
    with open('datasets/mini_coco/annotations.json', 'r') as f:
        coco_data = json.load(f)
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 評価と可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(4):
        img, target = dataset[i * 20]  # 均等に選択
        img_batch = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(img_batch)
        
        # 後処理
        processed = postprocess_predictions(predictions, conf_threshold=0.3, nms_threshold=0.5)[0]
        
        # 可視化
        ax = axes[i]
        img_np = img.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img_np)
        
        # GT（緑）
        for j, box in enumerate(target['boxes']):
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=3, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            label = categories[target['labels'][j].item()]
            ax.text(x1, y1-5, f'GT: {label}', color='lime', fontsize=10)
        
        # 予測（赤）
        for j, (box, score, label) in enumerate(zip(processed['boxes'], processed['scores'], processed['labels'])):
            x1, y1, x2, y2 = box.cpu().tolist()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            pred_label = categories.get(label.item(), f'cls{label.item()}')
            ax.text(x1, y2+15, f'{pred_label}: {score:.2f}', color='red', fontsize=10)
        
        ax.set_title(f'Sample {i+1}')
        ax.axis('off')
    
    plt.suptitle('Multiclass Object Detection Results', fontsize=16)
    plt.tight_layout()
    plt.savefig('multiclass_results.png', dpi=150)
    print("\n結果を保存: multiclass_results.png")

if __name__ == "__main__":
    # 学習
    model = train_multiclass_simple()
    
    # 評価
    print("\n" + "="*50)
    evaluate_multiclass()