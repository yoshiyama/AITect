import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CocoDataset
from model_whiteline import WhiteLineDetector
from utils.bbox import nms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

def visualize_single_prediction():
    """単一画像で予測を確認"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデル読み込み
    model = WhiteLineDetector(grid_size=8, num_anchors=3).to(device)
    model.load_state_dict(torch.load("result/aitect_model_improved.pth", map_location=device))
    model.eval()
    
    # データセット
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    with open('config.json') as f:
        config = json.load(f)
    
    val_dataset = CocoDataset(
        config['paths']['val_images'],
        config['paths']['val_annotations'],
        transform=transform
    )
    
    # 最初の画像で予測
    image, target = val_dataset[0]
    
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))[0]
    
    # 後処理
    scores = torch.sigmoid(output[:, 4])
    
    # 異なる閾値で結果を表示
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    thresholds = [0.3, 0.5, 0.7, 0.9]
    
    for idx, (ax, threshold) in enumerate(zip(axes.flat, thresholds)):
        # 画像を表示
        ax.imshow(image.permute(1, 2, 0))
        ax.set_title(f'Threshold: {threshold}')
        
        # 閾値でフィルタリング
        mask = scores > threshold
        filtered_boxes = output[mask, :4]
        filtered_scores = scores[mask]
        
        if len(filtered_boxes) > 0:
            # NMS適用
            x1 = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2
            y1 = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2
            x2 = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2
            y2 = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
            
            keep = nms(boxes_xyxy, filtered_scores, 0.5)
            
            # 描画
            for i in keep:
                box = filtered_boxes[i]
                score = filtered_scores[i]
                
                x = (box[0] - box[2]/2).cpu().item()
                y = (box[1] - box[3]/2).cpu().item()
                w = box[2].cpu().item()
                h = box[3].cpu().item()
                
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x, y-5, f'{score:.2f}', color='red', fontsize=10)
        
        # GT描画
        gt_boxes = target['boxes']
        for gt in gt_boxes:
            # GTは既にxyxy形式なので、xywh形式に変換して512x512にスケール
            x = gt[0] * 512 / image.shape[2]
            y = gt[1] * 512 / image.shape[1]
            w = (gt[2] - gt[0]) * 512 / image.shape[2]
            h = (gt[3] - gt[1]) * 512 / image.shape[1]
            
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
        
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.text(10, 30, f'Pred: {len(keep) if len(filtered_boxes) > 0 else 0}', 
                color='red', fontsize=12, bbox=dict(boxstyle="round", facecolor="white"))
        ax.text(10, 50, f'GT: {len(gt_boxes)}', 
                color='green', fontsize=12, bbox=dict(boxstyle="round", facecolor="white"))
    
    plt.tight_layout()
    plt.savefig('threshold_comparison.png', dpi=150)
    print("結果を threshold_comparison.png に保存しました")
    
    # スコア分布も確認
    plt.figure(figsize=(10, 6))
    plt.hist(scores.cpu().numpy(), bins=50, alpha=0.7)
    plt.axvline(0.5, color='red', linestyle='--', label='Default Threshold')
    plt.xlabel('Prediction Score')
    plt.ylabel('Count')
    plt.title('Score Distribution of All Predictions')
    plt.legend()
    plt.savefig('score_distribution_improved.png')
    print("スコア分布を score_distribution_improved.png に保存しました")
    
    # 統計情報
    print(f"\n予測統計:")
    print(f"最大スコア: {scores.max().item():.3f}")
    print(f"平均スコア: {scores.mean().item():.3f}")
    print(f"スコア > 0.9: {(scores > 0.9).sum().item()}")
    print(f"スコア > 0.7: {(scores > 0.7).sum().item()}")
    print(f"スコア > 0.5: {(scores > 0.5).sum().item()}")
    print(f"スコア > 0.3: {(scores > 0.3).sum().item()}")

if __name__ == "__main__":
    visualize_single_prediction()