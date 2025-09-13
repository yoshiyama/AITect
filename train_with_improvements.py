import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CocoDataset
from model_whiteline import WhiteLineDetector
from loss_yolo_style import yolo_style_detection_loss
import json
import os
import argparse

def load_config(config_path="config.json"):
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)

def train_improved(args):
    """改善された損失関数で学習"""
    config = load_config(args.config)
    
    # 設定の取得
    image_dir = config['paths']['train_images']
    annotation_path = config['paths']['train_annotations']
    num_epochs = args.epochs or config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    image_size = config['training']['image_size']
    save_interval = config['training']['save_interval']
    
    # 改善されたモデル保存パス
    model_save_path = "result/aitect_model_improved.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # データセットとローダ
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = CocoDataset(image_dir, annotation_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # モデル（正しい設定で新規作成）
    model = WhiteLineDetector(grid_size=8, num_anchors=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\n=== 改善された学習設定 ===")
    print(f"エポック数: {num_epochs}")
    print(f"バッチサイズ: {batch_size}")
    print(f"学習率: {learning_rate}")
    print(f"デバイス: {device}")
    print(f"損失関数: YOLOスタイル（正例重視）")
    print("=======================\n")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_positive = 0
        total_negative = 0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            for t in targets:
                t["boxes"] = t["boxes"].to(device)
                t["labels"] = t["labels"].to(device)
            
            # 順伝播
            preds = model(images)
            
            # 改善された損失関数を使用
            loss = yolo_style_detection_loss(
                preds, targets,
                lambda_coord=5.0,    # 座標損失を重視
                lambda_obj=1.0,      # 正例の重み
                lambda_noobj=0.5,    # 負例の重みを下げる
                use_focal=True,
                focal_alpha=0.25,
                focal_gamma=2.0
            )
            
            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 10バッチごとに進捗表示
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f}")
        
        # エポックごとの統計
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} 完了 - 平均損失: {avg_loss:.4f}")
        
        # ベストモデルの保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"ベストモデルを保存: {model_save_path}")
        
        # 定期保存
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = model_save_path.replace('.pth', f'_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"チェックポイント保存: {checkpoint_path}")
        
        # 簡易評価（最初の画像で予測を確認）
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample_image = next(iter(dataloader))[0][0:1].to(device)
                sample_pred = model(sample_image)[0]
                scores = torch.sigmoid(sample_pred[:, 4])
                high_conf = (scores > 0.5).sum().item()
                max_score = scores.max().item()
                print(f"サンプル予測 - 最大スコア: {max_score:.3f}, 高信頼度予測数: {high_conf}/192")
            model.train()
    
    print(f"\n学習完了！最終モデル: {model_save_path}")

def main():
    parser = argparse.ArgumentParser(description='改善された白線検出モデルの学習')
    parser.add_argument('--config', type=str, default='config.json', help='設定ファイル')
    parser.add_argument('--epochs', type=int, default=30, help='エポック数')
    
    args = parser.parse_args()
    train_improved(args)

if __name__ == "__main__":
    main()