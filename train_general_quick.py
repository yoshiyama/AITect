import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import AITECTDetector
from dataset import CocoDataset
from loss import detection_loss
import json
import os

def quick_train_general():
    """Simple Shapesで素早く一般物体検出の動作確認"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=== 一般物体検出モデル - クイック学習 ===")
    print(f"データセット: Simple Shapes")
    print(f"デバイス: {device}")
    
    # 設定
    config = {
        "num_epochs": 20,
        "batch_size": 4,
        "learning_rate": 0.001,
        "num_classes": 1,  # Simple Shapesは1クラス
        "grid_size": 16,
        "num_anchors": 3
    }
    
    # データセット
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    dataset = CocoDataset(
        "datasets/simple_shapes/images",
        "datasets/simple_shapes/annotations.json",
        transform=transform
    )
    
    # 80:20で分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        return images, list(targets)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 新しいモデル（白線検出の重みは使わない）
    model = AITECTDetector(
        num_classes=config['num_classes'],
        grid_size=config['grid_size'],
        num_anchors=config['num_anchors']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print(f"\n訓練開始: {train_size}サンプル")
    
    # 学習ループ
    train_losses = []
    val_losses = []
    
    for epoch in range(config['num_epochs']):
        # 訓練
        model.train()
        train_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                t['labels'] = t['labels'].to(device)
            
            predictions = model(images)
            loss = detection_loss(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
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
                loss = detection_loss(predictions, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} - "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # モデル保存
    torch.save(model.state_dict(), "result/aitect_general_shapes.pth")
    print(f"\n✅ 学習完了! モデル保存: result/aitect_general_shapes.pth")
    
    # 学習曲線を保存
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('General Object Detection Training (Simple Shapes)')
    plt.legend()
    plt.grid(True)
    plt.savefig('general_training_curve.png')
    print("学習曲線: general_training_curve.png")
    
    return model

def evaluate_general_model():
    """学習した一般物体検出モデルの評価"""
    from evaluate_on_public_dataset import evaluate_on_dataset
    
    # 一時的に設定を変更
    temp_config = {
        "best_model": {
            "path": "result/aitect_general_shapes.pth",
            "optimal_threshold": 0.3
        }
    }
    
    with open('optimal_thresholds.json', 'r') as f:
        original = json.load(f)
    
    with open('optimal_thresholds.json', 'w') as f:
        json.dump(temp_config, f)
    
    print("\n=== 一般物体検出モデルの評価 ===")
    evaluate_on_dataset("./datasets/simple_shapes", "Simple Shapes (General Model)")
    
    # 設定を戻す
    with open('optimal_thresholds.json', 'w') as f:
        json.dump(original, f)

if __name__ == "__main__":
    # 1. 一般物体検出として学習
    model = quick_train_general()
    
    # 2. 評価
    print("\n" + "="*60)
    evaluate_general_model()
    
    # 3. 白線検出モデルとの比較
    print("\n" + "="*60)
    print("=== モデル比較 ===")
    print("1. 白線検出モデル → Simple Shapes: F1=0.0057 (特化しすぎ)")
    print("2. 白線検出モデル + 適応学習: F1=0.2446 (転移学習)")
    print("3. 一般物体検出モデル: 上記の結果を確認")