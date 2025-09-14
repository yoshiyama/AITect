import torch
import torch.nn as nn
from model import AITECTDetector
from dataset import CocoDataset
from loss import detection_loss
from torchvision import transforms
import json
import os

def adapt_model_for_general_detection():
    """白線検出モデルを一般物体検出に適応"""
    print("=== モデルの一般物体検出への適応 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 事前学習済みモデルを読み込み
    print("\n1. 白線検出モデルを読み込み...")
    model = AITECTDetector(num_classes=1, grid_size=16, num_anchors=3).to(device)
    
    # 白線検出の重みを読み込み（特徴抽出部分のみ）
    pretrained_path = "result/aitect_model_improved_training_best.pth"
    if os.path.exists(pretrained_path):
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        
        # バックボーンの重みのみを選択（検出ヘッドは除外）
        backbone_dict = {k: v for k, v in pretrained_dict.items() 
                        if 'backbone' in k or 'conv' in k and 'head' not in k}
        
        model_dict.update(backbone_dict)
        model.load_state_dict(model_dict)
        print(f"  バックボーンの重みを転移: {len(backbone_dict)} layers")
    
    # 2. Simple Shapesデータセットで少量学習
    print("\n2. Simple Shapesデータセットで適応学習...")
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    # データセット分割
    dataset = CocoDataset(
        "datasets/simple_shapes/images",
        "datasets/simple_shapes/annotations.json",
        transform=transform
    )
    
    # 簡易的な学習（転移学習）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # 最適化設定（学習率を高めに）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 短期間の適応学習
    num_epochs = 10
    print(f"\n  エポック数: {num_epochs}")
    print(f"  訓練サンプル数: {train_size}")
    print(f"  検証サンプル数: {val_size}")
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 訓練
        model.train()
        train_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = torch.stack(images).to(device)
            targets_batch = []
            for t in targets:
                targets_batch.append({
                    'boxes': t['boxes'].to(device),
                    'labels': t['labels'].to(device)
                })
            
            predictions = model(images)
            loss = detection_loss(predictions, targets_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 検証
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = torch.stack(images).to(device)
                targets_batch = []
                for t in targets:
                    targets_batch.append({
                        'boxes': t['boxes'].to(device),
                        'labels': t['labels'].to(device)
                    })
                
                predictions = model(images)
                loss = detection_loss(predictions, targets_batch)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"  Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # ベストモデル保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "result/aitect_model_adapted_general.pth")
    
    print("\n✅ 適応学習完了!")
    print(f"適応モデル保存: result/aitect_model_adapted_general.pth")
    
    # 3. 適応設定を保存
    adapted_config = {
        "model_path": "result/aitect_model_adapted_general.pth",
        "original_model": "白線検出モデル",
        "adapted_for": "一般物体検出（Simple Shapes）",
        "training_epochs": num_epochs,
        "best_val_loss": best_val_loss,
        "recommended_threshold": 0.3
    }
    
    with open("adapted_model_config.json", "w") as f:
        json.dump(adapted_config, f, indent=2)
    
    return model

def test_adapted_model():
    """適応モデルのテスト"""
    from evaluate_on_public_dataset import evaluate_on_dataset
    
    # 適応モデル用の設定を一時的に作成
    temp_config = {
        "best_model": {
            "path": "result/aitect_model_adapted_general.pth",
            "optimal_threshold": 0.3
        }
    }
    
    # 元の設定をバックアップ
    with open('optimal_thresholds.json', 'r') as f:
        original_config = json.load(f)
    
    # 一時的に適応モデルの設定を使用
    with open('optimal_thresholds.json', 'w') as f:
        json.dump(temp_config, f)
    
    print("\n=== 適応モデルの評価 ===")
    results = evaluate_on_dataset("./datasets/simple_shapes", "Simple Shapes (Adapted Model)")
    
    # 元の設定を復元
    with open('optimal_thresholds.json', 'w') as f:
        json.dump(original_config, f)
    
    return results

if __name__ == "__main__":
    # モデルの適応
    adapted_model = adapt_model_for_general_detection()
    
    # 適応モデルのテスト
    print("\n" + "="*60)
    test_results = test_adapted_model()