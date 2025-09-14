import torch
from model import AITECTDetector
from dataset import CocoDataset
from torchvision import transforms
from utils.postprocess import postprocess_predictions
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def evaluate_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルを読み込む
    model = AITECTDetector(num_classes=1, grid_size=16, num_anchors=3).to(device)
    model.load_state_dict(torch.load("result/aitect_model_simple.pth"))
    model.eval()
    
    # 最適な閾値
    optimal_threshold = 0.39
    
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
    
    # 全体の統計
    all_tp, all_fp, all_fn = 0, 0, 0
    
    # 可視化用のサンプル
    num_samples = 6
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 5*num_samples))
    
    for idx in range(num_samples):
        # 元画像を読み込み
        image_info = val_dataset.image_info[idx]
        image_path = f"{val_dataset.image_dir}/{image_info['file_name'].split('/')[-1]}"
        orig_image = Image.open(image_path)
        
        # モデル用の画像
        image, target = val_dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(image_batch)
        
        # 通常の閾値（0.3）での結果
        processed_normal = postprocess_predictions(
            predictions,
            conf_threshold=0.3,
            nms_threshold=0.5
        )[0]
        
        # 最適化された閾値での結果
        processed_optimal = postprocess_predictions(
            predictions,
            conf_threshold=optimal_threshold,
            nms_threshold=0.5
        )[0]
        
        # 精度計算（最適閾値）
        pred_boxes = processed_optimal['boxes'].cpu()
        gt_boxes = target['boxes']
        
        tp, fp, fn = 0, 0, len(gt_boxes)
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)
            max_ious, _ = ious.max(dim=1)
            tp = (max_ious > 0.5).sum().item()
            fp = (max_ious <= 0.5).sum().item()
            gt_detected = (ious.max(dim=0)[0] > 0.5)
            fn = (~gt_detected).sum().item()
        elif len(pred_boxes) > 0:
            fp = len(pred_boxes)
        
        all_tp += tp
        all_fp += fp
        all_fn += fn
        
        # 左側：通常の閾値（0.3）
        ax = axes[idx, 0]
        ax.imshow(orig_image)
        ax.set_title(f'Threshold=0.3 (Pred: {len(processed_normal["boxes"])}, GT: {len(gt_boxes)})')
        
        # GTボックス（緑）
        for box in target['boxes']:
            scale_x = orig_image.width / 512
            scale_y = orig_image.height / 512
            x1, y1, x2, y2 = box.cpu().numpy() if torch.is_tensor(box) else box
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
        
        # 予測ボックス（赤）
        for box, score in zip(processed_normal['boxes'][:10], processed_normal['scores'][:10]):
            scale_x = orig_image.width / 512
            scale_y = orig_image.height / 512
            x1, y1, x2, y2 = box.cpu().numpy()
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=8)
        
        # 右側：最適化された閾値
        ax = axes[idx, 1]
        ax.imshow(orig_image)
        ax.set_title(f'Threshold={optimal_threshold} (TP:{tp}, FP:{fp}, FN:{fn})')
        
        # GTボックス（緑）
        for box in target['boxes']:
            scale_x = orig_image.width / 512
            scale_y = orig_image.height / 512
            x1, y1, x2, y2 = box.cpu().numpy() if torch.is_tensor(box) else box
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
        
        # 予測ボックス（最適閾値）
        for box, score in zip(processed_optimal['boxes'], processed_optimal['scores']):
            scale_x = orig_image.width / 512
            scale_y = orig_image.height / 512
            x1, y1, x2, y2 = box.cpu().numpy()
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='blue', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{score:.2f}', color='blue', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('optimal_threshold_comparison.png', dpi=150, bbox_inches='tight')
    
    # 全体の統計を出力
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n=== Performance with Optimal Threshold ({optimal_threshold}) ===")
    print(f"Total TP: {all_tp}")
    print(f"Total FP: {all_fp}")
    print(f"Total FN: {all_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nVisualization saved to: optimal_threshold_comparison.png")

if __name__ == "__main__":
    evaluate_and_visualize()