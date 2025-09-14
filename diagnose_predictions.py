import torch
from model_improved_v2 import ImprovedDetector
from dataset import CocoDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

def diagnose():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 最新のチェックポイントを読み込む
    checkpoint_paths = [
        "result/aitect_model_emergency_epoch5.pth",
        "result/aitect_model_improved_v2_epoch20.pth",
        "result/aitect_model_improved_v2_epoch10.pth",
        "result/aitect_model_emergency.pth",
        "result/aitect_model_improved_v2_fixed.pth",
        "result/aitect_model_improved_v2.pth"
    ]
    
    model_loaded = False
    for path in checkpoint_paths:
        try:
            model = ImprovedDetector(num_classes=1, num_anchors=3).to(device)
            # チェックポイントまたは状態辞書を読み込む
            checkpoint = torch.load(path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from checkpoint: {path}")
                print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
                print(f"  Loss: {checkpoint.get('loss', 'unknown')}")
            else:
                model.load_state_dict(checkpoint)
                print(f"Loaded model state dict from: {path}")
            model.eval()
            model_loaded = True
            break
        except Exception as e:
            print(f"Failed to load {path}: {str(e)}")
            continue
    
    if not model_loaded:
        print("No checkpoint found, using random weights")
        model = ImprovedDetector(num_classes=1, num_anchors=3).to(device)
        model.eval()
    
    # データセット準備
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    val_dataset = CocoDataset(
        "datasets/inaoka/val/JPEGImages",
        "datasets/inaoka/val/annotations.json",
        transform=transform
    )
    
    # 最初の画像でテスト
    image, target = val_dataset[0]
    image_batch = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image_batch)
    
    print(f"\n=== Raw Predictions Analysis ===")
    print(f"Predictions shape: {predictions.shape}")
    
    # Objectnessスコアの分析
    objectness = predictions[0, :, 4]
    print(f"\nObjectness scores (raw):")
    print(f"  Min: {objectness.min().item():.4f}")
    print(f"  Max: {objectness.max().item():.4f}")
    print(f"  Mean: {objectness.mean().item():.4f}")
    print(f"  Std: {objectness.std().item():.4f}")
    
    # Sigmoid後
    obj_sigmoid = torch.sigmoid(objectness)
    print(f"\nObjectness scores (sigmoid):")
    print(f"  Min: {obj_sigmoid.min().item():.4f}")
    print(f"  Max: {obj_sigmoid.max().item():.4f}")
    print(f"  Mean: {obj_sigmoid.mean().item():.4f}")
    
    # 上位スコアを持つ予測
    top_k = 20
    top_scores, top_indices = obj_sigmoid.topk(top_k)
    print(f"\nTop {top_k} confidence scores:")
    for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
        box = predictions[0, idx, :4]
        print(f"  {i+1}. Score: {score:.4f}, Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    # 異なる閾値での検出数
    thresholds = [0.5, 0.3, 0.1, 0.05, 0.01, 0.001]
    print(f"\nDetections at different thresholds:")
    for thresh in thresholds:
        count = (obj_sigmoid > thresh).sum().item()
        print(f"  Threshold {thresh}: {count} detections")
    
    # 後処理のテスト
    from utils.postprocess import postprocess_predictions
    
    print(f"\n=== Postprocessing Test ===")
    for conf_thresh in [0.5, 0.1, 0.01, 0.001]:
        processed = postprocess_predictions(
            predictions, 
            conf_threshold=conf_thresh,
            nms_threshold=0.5
        )
        n_dets = len(processed[0]['boxes'])
        print(f"Conf threshold {conf_thresh}: {n_dets} detections after NMS")
        
        if n_dets > 0 and n_dets < 10:
            print("  Sample detections:")
            for i, (box, score) in enumerate(zip(processed[0]['boxes'][:5], processed[0]['scores'][:5])):
                print(f"    {i+1}. Box: {box.tolist()}, Score: {score:.4f}")
    
    # Ground truth
    print(f"\n=== Ground Truth ===")
    print(f"Number of GT boxes: {len(target['boxes'])}")
    for i, box in enumerate(target['boxes'][:5]):
        print(f"  GT {i+1}: {box.tolist()}")

if __name__ == "__main__":
    diagnose()