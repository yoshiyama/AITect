import torch
from model import AITECTDetector
from dataset import CocoDataset
from torchvision import transforms
from utils.postprocess import postprocess_predictions
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルを読み込む
    model = AITECTDetector(num_classes=1, grid_size=16, num_anchors=3).to(device)
    
    # 最新のチェックポイントを探す
    checkpoint_path = "result/aitect_model_simple.pth"
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model from: {checkpoint_path}")
    except:
        print(f"Failed to load {checkpoint_path}, using latest checkpoint")
        # 他のチェックポイントを試す
        import glob
        checkpoints = glob.glob("result/aitect_model_simple*.pth")
        if checkpoints:
            checkpoint_path = max(checkpoints)
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded model from: {checkpoint_path}")
        else:
            print("No checkpoint found!")
            return
    
    model.eval()
    
    # 検証データセット
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    val_dataset = CocoDataset(
        "datasets/inaoka/val/JPEGImages",
        "datasets/inaoka/val/annotations.json",
        transform=transform
    )
    
    # 統計を収集
    all_scores = []
    tp, fp, fn = 0, 0, 0
    
    print("\n=== Model Evaluation ===")
    
    # 最初の5枚で詳細な評価
    for idx in range(min(5, len(val_dataset))):
        image, target = val_dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(image_batch)
        
        # 後処理
        processed = postprocess_predictions(
            predictions,
            conf_threshold=0.3,
            nms_threshold=0.5
        )[0]
        
        pred_boxes = processed['boxes'].cpu()
        pred_scores = processed['scores'].cpu()
        gt_boxes = target['boxes']
        
        print(f"\nImage {idx+1}:")
        print(f"  Ground truth boxes: {len(gt_boxes)}")
        print(f"  Predicted boxes: {len(pred_boxes)}")
        
        if len(pred_boxes) > 0:
            print(f"  Top 3 predictions:")
            for i in range(min(3, len(pred_boxes))):
                print(f"    Score: {pred_scores[i]:.4f}, Box: {pred_boxes[i].tolist()}")
            all_scores.extend(pred_scores.tolist())
        
        # 簡易的な精度計算（IoU > 0.5）
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            from torchvision.ops import box_iou
            ious = box_iou(pred_boxes, gt_boxes)
            
            # 各予測に対して最大IoUを計算
            max_ious, _ = ious.max(dim=1)
            tp += (max_ious > 0.5).sum().item()
            fp += (max_ious <= 0.5).sum().item()
            
            # 各GTに対して検出されたかチェック
            gt_detected = (ious.max(dim=0)[0] > 0.5)
            fn += (~gt_detected).sum().item()
        else:
            fp += len(pred_boxes)
            fn += len(gt_boxes)
    
    # 全体の統計
    print("\n=== Overall Statistics ===")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    if all_scores:
        print(f"\nConfidence score statistics:")
        print(f"  Min: {min(all_scores):.4f}")
        print(f"  Max: {max(all_scores):.4f}")
        print(f"  Mean: {np.mean(all_scores):.4f}")
        print(f"  Std: {np.std(all_scores):.4f}")
    
    # 可視化
    visualize_predictions(model, val_dataset, device)

def visualize_predictions(model, dataset, device, num_samples=3):
    """予測結果を可視化"""
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 6*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        # 元画像を読み込み
        image_info = dataset.image_info[idx]
        image_path = f"{dataset.image_dir}/{image_info['file_name'].split('/')[-1]}"
        orig_image = Image.open(image_path)
        
        # モデル用の画像
        image, target = dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(image_batch)
        
        processed = postprocess_predictions(
            predictions,
            conf_threshold=0.3,
            nms_threshold=0.5
        )[0]
        
        # Ground Truth
        ax = axes[idx, 0]
        ax.imshow(orig_image)
        ax.set_title(f'Ground Truth (Image {idx+1})')
        
        # GTボックスを描画（元画像座標系で）
        for box in target['boxes']:
            # 512x512から元画像サイズにスケール
            scale_x = orig_image.width / 512
            scale_y = orig_image.height / 512
            x1, y1, x2, y2 = box.cpu().numpy() if torch.is_tensor(box) else box
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='g', facecolor='none'
            )
            ax.add_patch(rect)
        
        # Predictions
        ax = axes[idx, 1]
        ax.imshow(orig_image)
        ax.set_title(f'Predictions (Image {idx+1})')
        
        # 予測ボックスを描画
        for box, score in zip(processed['boxes'], processed['scores']):
            # 512x512から元画像サイズにスケール
            scale_x = orig_image.width / 512
            scale_y = orig_image.height / 512
            x1, y1, x2, y2 = box.cpu().numpy() if torch.is_tensor(box) else box
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150)
    print(f"\nVisualization saved to: evaluation_results.png")

if __name__ == "__main__":
    evaluate()