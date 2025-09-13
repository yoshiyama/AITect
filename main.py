import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CocoDataset
from model import AITECTDetector
from model_whiteline import WhiteLineDetector
from loss import detection_loss, detection_loss_improved
from auto_config import auto_configure
from utils.monitor import TrainingMonitor
from utils.validation import validate_detection
from utils.validation_fullsize import save_fullsize_detections
from utils.metrics import calculate_metrics_batch
from dataset import CocoDataset
import json
import argparse
import os

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)

def load_config(config_path="config.json"):
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def train(args):
    # 設定ファイルを読み込み
    config = load_config(args.config)
    
    # 自動設定を実行（--auto-configフラグが指定された場合）
    if args.auto_config:
        print("\n=== データセット分析に基づく自動設定を実行中 ===\n")
        config, stats = auto_configure(args.config)
        print("\n自動設定が完了しました。\n")
    
    # コマンドライン引数で上書き
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    
    # 設定の取得
    image_dir = config['paths']['train_images']
    annotation_path = config['paths']['train_annotations']
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    image_size = config['training']['image_size']
    save_interval = config['training']['save_interval']
    model_save_path = config['paths']['model_save_path']
    
    # 損失関数の設定
    loss_type = config['training'].get('loss_type', 'mixed')
    iou_weight = config['training'].get('iou_weight', 2.0)
    l1_weight = config['training'].get('l1_weight', 0.5)
    
    # モデルの設定
    grid_size = config['model'].get('grid_size', 13)
    num_anchors = config['model'].get('num_anchors', 3)
    use_focal = config['model'].get('use_focal_loss', True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセットとローダ
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = CocoDataset(image_dir, annotation_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    print("\n=== 学習設定 ===")
    print(f"エポック数: {num_epochs}")
    print(f"バッチサイズ: {batch_size}")
    print(f"学習率: {learning_rate}")
    print(f"画像サイズ: {image_size}x{image_size}")
    print(f"デバイス: {device}")
    print(f"データセットサイズ: {len(dataset)} 枚")
    print("================")
    
    # モニタリングツールの初期化
    monitor = TrainingMonitor()
    monitor.print_gpu_info()
    
    # 検証データセットの準備
    val_image_dir = config['paths']['val_images']
    val_annotation_path = config['paths']['val_annotations']
    val_dataset = CocoDataset(val_image_dir, val_annotation_path, transform=transform)
    val_check_interval = 5  # Check validation every 5 epochs

    # モデルと最適化
    model_type = config['model'].get('model_type', 'standard')
    if model_type == 'whiteline':
        # 白線検出モデル（自動設定されたアンカーを使用）
        auto_anchors = config['model'].get('auto_anchors', None)
        model = WhiteLineDetector(
            grid_size=grid_size, 
            num_anchors=num_anchors,
            auto_anchors=auto_anchors
        ).to(device)
    else:
        # 標準モデル
        model = AITECTDetector(grid_size=grid_size, num_anchors=num_anchors).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 保存ディレクトリの作成
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 学習ループ
    print(f"\n学習設定:")
    print(f"  デバイス: {device}")
    print(f"  エポック数: {num_epochs}")
    print(f"  バッチサイズ: {batch_size}")
    print(f"  学習率: {learning_rate}")
    print(f"  画像サイズ: {image_size}x{image_size}")
    print(f"  損失関数: {loss_type} (IoU重み: {iou_weight}, L1重み: {l1_weight})")
    print(f"  Focal Loss: {'有効' if use_focal else '無効'}")
    print(f"  グリッドサイズ: {grid_size}x{grid_size}")
    print(f"  アンカー数: {num_anchors}")
    print(f"  総予測数: {grid_size * grid_size * num_anchors}")
    print(f"  学習データ数: {len(dataset)}")
    print(f"  検証データ数: {len(val_dataset)}")
    print("\n学習を開始します...\n")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            for t in targets:
                t["boxes"] = t["boxes"].to(device)
                t["labels"] = t["labels"].to(device)

            # 順伝播
            preds = model(images)
            
            # 改善された損失関数を使用するかチェック
            use_improved = config['model'].get('use_improved_loss', False)
            if use_improved:
                loss = detection_loss_improved(preds, targets, 
                                             loss_type=loss_type,
                                             iou_weight=iou_weight, 
                                             l1_weight=l1_weight,
                                             use_focal=use_focal)
            else:
                loss = detection_loss(preds, targets, 
                                    loss_type=loss_type,
                                    iou_weight=iou_weight, 
                                    l1_weight=l1_weight,
                                    use_focal=use_focal)

            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            
            # イテレーションごとのログ
            monitor.log_iteration(epoch+1, batch_idx, len(dataloader), loss.item(), model, optimizer)
            
            optimizer.step()
            total_loss += loss.item()

        # エポックごとのログ
        avg_loss = total_loss / len(dataloader)
        monitor.log_epoch(epoch+1, avg_loss, model, optimizer)
        
        # 検証データで検出結果を確認
        if (epoch + 1) % val_check_interval == 0:
            print(f"\n  [検証] エポック {epoch+1} の評価を実施中...")
            
            # 検証データでメトリクス計算
            model.eval()
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for i in range(min(10, len(val_dataset))):  # 最初の10枚で評価
                    image, target = val_dataset[i]
                    image_tensor = image.unsqueeze(0).to(device)
                    output = model(image_tensor)[0]
                    val_predictions.append(output)
                    # targetのboxesもGPUに移動
                    target_gpu = {
                        'boxes': target['boxes'].to(device),
                        'labels': target['labels'].to(device),
                        'image_id': target['image_id']
                    }
                    val_targets.append(target_gpu)
            
            # メトリクス計算
            metrics = calculate_metrics_batch(val_predictions, val_targets)
            
            # 結果を表示
            print(f"\n  === 検証結果 (IoU闾値=0.5) ===")
            print(f"  Precision: {metrics['avg_precision']:.3f}")
            print(f"  Recall: {metrics['avg_recall']:.3f}")
            print(f"  F1 Score: {metrics['avg_f1']:.3f}")
            print(f"  平均IoU: {metrics['mean_iou']:.3f}")
            print(f"  GT総数: {metrics['total_gt']}, 予測総数: {metrics['total_pred']}")
            print(f"  ========================\n")
            
            # 画像を保存
            val_save_path = validate_detection(
                model, val_dataset, device, 
                num_samples=3, 
                save_dir=monitor.plot_dir,
                epoch=epoch+1
            )
            if val_save_path:
                print(f"  検出結果画像（512x512）: {val_save_path}")
            
            # フルサイズ画像も保存
            fullsize_dir = save_fullsize_detections(
                model, val_dataset, device,
                save_dir=monitor.plot_dir,
                epoch=epoch+1,
                num_samples=3
            )
            print(f"  フルサイズ検出結果: {fullsize_dir}")
            
            model.train()  # 学習モードに戻す
        
        # 定期的にモデルを保存
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = model_save_path.replace('.pth', f'_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  チェックポイント保存: {checkpoint_path}")

    # 最終モデル保存
    torch.save(model.state_dict(), model_save_path)
    print(f"\n最終モデル保存: {model_save_path}")
    
    # 学習サマリー表示
    monitor.print_summary()

def main():
    parser = argparse.ArgumentParser(description='AITect物体検出モデルの学習')
    parser.add_argument('--config', type=str, default='config.json',
                        help='設定ファイルのパス (デフォルト: config.json)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='エポック数 (設定ファイルの値を上書き)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='バッチサイズ (設定ファイルの値を上書き)')
    parser.add_argument('--lr', type=float, default=None,
                        help='学習率 (設定ファイルの値を上書き)')
    parser.add_argument('--auto-config', action='store_true',
                        help='データセット分析に基づいてパラメータを自動設定')
    
    args = parser.parse_args()
    
    # Import here to avoid circular dependency
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    train(args)

if __name__ == "__main__":
    main()
