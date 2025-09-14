import torch
from model_improved_v2 import ImprovedDetector
from dataset import CocoDataset
from torchvision import transforms
import numpy as np

def debug_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルの初期化
    model = ImprovedDetector(num_classes=1, num_anchors=3).to(device)
    
    # ダミー入力
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print("=== Model Output Debug ===")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 各成分の統計
    boxes = output[0, :, :4]
    objectness = output[0, :, 4]
    
    print(f"\nBox predictions (x, y, w, h):")
    print(f"  X center range: [{boxes[:, 0].min():.1f}, {boxes[:, 0].max():.1f}]")
    print(f"  Y center range: [{boxes[:, 1].min():.1f}, {boxes[:, 1].max():.1f}]")
    print(f"  Width range: [{boxes[:, 2].min():.1f}, {boxes[:, 2].max():.1f}]")
    print(f"  Height range: [{boxes[:, 3].min():.1f}, {boxes[:, 3].max():.1f}]")
    
    print(f"\nObjectness scores (raw):")
    print(f"  Range: [{objectness.min():.4f}, {objectness.max():.4f}]")
    print(f"  Mean: {objectness.mean():.4f}")
    print(f"  Std: {objectness.std():.4f}")
    
    # シグモイド後
    obj_sigmoid = torch.sigmoid(objectness)
    print(f"\nObjectness scores (after sigmoid):")
    print(f"  Range: [{obj_sigmoid.min():.4f}, {obj_sigmoid.max():.4f}]")
    print(f"  Mean: {obj_sigmoid.mean():.4f}")
    
    # 高スコアの予測数
    high_conf = (obj_sigmoid > 0.5).sum()
    print(f"\nPredictions with confidence > 0.5: {high_conf}/{len(objectness)}")
    
    # データセットから1枚読み込んでテスト
    print("\n=== Testing with real image ===")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    dataset = CocoDataset(
        "datasets/inaoka/train/JPEGImages",
        "datasets/inaoka/train/annotations.json",
        transform=transform
    )
    
    image, target = dataset[0]
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
    
    print(f"Ground truth boxes: {len(target['boxes'])}")
    print(f"Model predictions shape: {output.shape}")
    
    # 予測の可視化
    from utils.postprocess import postprocess_predictions
    processed = postprocess_predictions(output, conf_threshold=0.3)
    print(f"Predictions after NMS: {len(processed[0]['boxes'])}")
    
    if len(processed[0]['boxes']) > 0:
        print("Top predictions:")
        for i, (box, score) in enumerate(zip(processed[0]['boxes'][:5], processed[0]['scores'][:5])):
            print(f"  {i+1}. Box: {box.tolist()}, Score: {score:.4f}")

if __name__ == "__main__":
    debug_model()