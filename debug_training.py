#!/usr/bin/env python3
"""V2モデルの学習デバッグスクリプト"""

import torch
import json
from model import AITECTDetector
from dataset import CocoDataset
from torchvision import transforms
from loss import detection_loss

def debug_v2_training():
    print("=== V2モデル学習デバッグ ===\n")
    
    # 設定読み込み
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデル初期化
    grid_size = config['model']['grid_size']
    num_anchors = config['model']['num_anchors']
    model = AITECTDetector(grid_size=grid_size, num_anchors=num_anchors).to(device)
    
    # データセット準備（1バッチのみ）
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    image_dir = config['paths']['train_images']
    annotation_path = config['paths']['train_annotations']
    dataset = CocoDataset(image_dir, annotation_path, transform=transform)
    
    # 1つのサンプルで確認
    image, target = dataset[0]
    image = image.unsqueeze(0).to(device)
    target['boxes'] = target['boxes'].to(device)
    target['labels'] = target['labels'].to(device)
    
    # 順伝播
    model.eval()
    with torch.no_grad():
        output = model(image)
    
    print("モデル出力の統計:")
    pred_conf = output[0, :, 4]
    pred_conf_sigmoid = torch.sigmoid(pred_conf)
    
    print(f"  信頼度（生値）: min={pred_conf.min():.3f}, max={pred_conf.max():.3f}, mean={pred_conf.mean():.3f}")
    print(f"  信頼度（sigmoid後）: min={pred_conf_sigmoid.min():.3f}, max={pred_conf_sigmoid.max():.3f}, mean={pred_conf_sigmoid.mean():.3f}")
    
    # 閾値ごとの予測数
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        count = (pred_conf_sigmoid > thresh).sum().item()
        print(f"  閾値 {thresh}: {count}/{len(pred_conf_sigmoid)} 予測")
    
    # 損失計算
    targets = [target]
    
    print("\n損失計算:")
    # Focal Lossあり
    loss_focal = detection_loss(output, targets, use_focal=True)
    # Focal Lossなし
    loss_no_focal = detection_loss(output, targets, use_focal=False)
    
    print(f"  Focal Lossあり: {loss_focal.item():.4f}")
    print(f"  Focal Lossなし: {loss_no_focal.item():.4f}")
    
    # 学習済みモデルのチェック
    model_path = config['paths']['model_save_path']
    import os
    if os.path.exists(model_path):
        print(f"\n学習済みモデルが存在: {model_path}")
        # モデルを読み込んで再度確認
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  エポック: {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        with torch.no_grad():
            output_trained = model(image)
        
        pred_conf_trained = output_trained[0, :, 4]
        pred_conf_sigmoid_trained = torch.sigmoid(pred_conf_trained)
        
        print("\n学習済みモデルの出力:")
        print(f"  信頼度（sigmoid後）: min={pred_conf_sigmoid_trained.min():.3f}, max={pred_conf_sigmoid_trained.max():.3f}, mean={pred_conf_sigmoid_trained.mean():.3f}")
        
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            count = (pred_conf_sigmoid_trained > thresh).sum().item()
            print(f"  閾値 {thresh}: {count}/{len(pred_conf_sigmoid_trained)} 予測")

if __name__ == "__main__":
    debug_v2_training()