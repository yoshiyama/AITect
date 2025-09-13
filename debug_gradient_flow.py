import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CocoDataset
from model import AITECTDetector
from loss import detection_loss
from loss_improved import detection_loss_improved
import json
import matplotlib.pyplot as plt
import numpy as np

def load_config(config_path="config.json"):
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_grad_flow(named_parameters, epoch, batch_idx):
    """勾配の流れを可視化"""
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
                max_grads.append(p.grad.abs().max().cpu().item())
            else:
                print(f"警告: {n} の勾配がNoneです")
    
    if not layers:
        print("勾配が計算されているレイヤーがありません")
        return
        
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.7, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.7, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title(f"Gradient flow - Epoch {epoch}, Batch {batch_idx}")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4)], ['mean-gradient', 'max-gradient'])
    plt.tight_layout()
    plt.savefig(f'gradient_flow_epoch{epoch}_batch{batch_idx}.png')
    plt.close()

def check_gradient_stats(model, loss, epoch, batch_idx):
    """勾配の統計情報を表示"""
    print(f"\n=== エポック {epoch}, バッチ {batch_idx} の勾配統計 ===")
    print(f"損失値: {loss.item():.6f}")
    
    total_grad_norm = 0
    param_count = 0
    zero_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            total_grad_norm += grad_norm ** 2
            param_count += 1
            
            # 勾配がゼロに近いパラメータをカウント
            if grad_norm < 1e-8:
                zero_grad_count += 1
            
            # 主要なレイヤーの勾配情報を表示
            if any(key in name for key in ['backbone.0', 'head', 'final']):
                print(f"{name:50s} | grad_norm: {grad_norm:.6f} | grad_mean: {param.grad.data.mean().item():.6f}")
    
    total_grad_norm = (total_grad_norm ** 0.5)
    print(f"\n全体の勾配ノルム: {total_grad_norm:.6f}")
    print(f"勾配がほぼゼロのパラメータ数: {zero_grad_count}/{param_count}")
    print("=" * 60)

def debug_gradient_flow():
    """勾配伝搬のデバッグ"""
    config = load_config()
    
    # 設定の取得
    image_dir = config['paths']['train_images']
    annotation_path = config['paths']['train_annotations']
    batch_size = 4
    image_size = config['training']['image_size']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # データセットとローダ
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = CocoDataset(image_dir, annotation_path, transform=transform)
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        return images, list(targets)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # モデルと最適化
    model = AITECTDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 損失関数の設定
    use_improved = config['model'].get('use_improved_loss', False)
    loss_type = config['training'].get('loss_type', 'mixed')
    iou_weight = config['training'].get('iou_weight', 2.0)
    l1_weight = config['training'].get('l1_weight', 0.5)
    use_focal = config['model'].get('use_focal_loss', True)
    
    print("勾配伝搬のデバッグを開始します...")
    print(f"使用する損失関数: {'改善版' if use_improved else '標準版'}")
    print(f"損失タイプ: {loss_type}")
    
    # パラメータの初期状態を保存
    initial_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.data.clone()
    
    # 数エポック分実行
    for epoch in range(3):
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= 3:  # 最初の3バッチのみ
                break
                
            images = images.to(device)
            for t in targets:
                t["boxes"] = t["boxes"].to(device)
                t["labels"] = t["labels"].to(device)
            
            # 順伝播
            preds = model(images)
            
            # 損失計算
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
            
            # 勾配をクリア
            optimizer.zero_grad()
            
            # 逆伝播
            loss.backward()
            
            # 勾配の統計情報を表示
            check_gradient_stats(model, loss, epoch + 1, batch_idx + 1)
            
            # 勾配フローを可視化
            if batch_idx == 0:  # 各エポックの最初のバッチのみ
                plot_grad_flow(model.named_parameters(), epoch + 1, batch_idx + 1)
            
            # パラメータ更新前後の変化を確認
            print("\n--- パラメータ更新の確認 ---")
            param_changed = 0
            for name, param in model.named_parameters():
                if param.requires_grad and name in initial_params:
                    change = (param.data - initial_params[name]).abs().max().item()
                    if change > 1e-6:
                        param_changed += 1
                        if 'head' in name or 'final' in name:  # 重要なレイヤーのみ表示
                            print(f"{name}: 最大変化量 = {change:.6f}")
            
            print(f"更新されたパラメータ数: {param_changed}/{len(initial_params)}")
            
            # 最適化ステップ
            optimizer.step()
            
            # 更新後のパラメータを保存
            for name, param in model.named_parameters():
                if param.requires_grad:
                    initial_params[name] = param.data.clone()
            
            print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    debug_gradient_flow()