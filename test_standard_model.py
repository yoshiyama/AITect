import torch
from model import AITECTDetectorV1
from dataset import CocoDataset
from torchvision import transforms
from loss import detection_loss
from utils.postprocess import postprocess_predictions

def test_standard_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 標準モデル（動作実績あり）
    model = AITECTDetectorV1(num_classes=1, grid_size=16, num_anchors=3).to(device)
    
    # データ準備
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    dataset = CocoDataset(
        "datasets/inaoka/train/JPEGImages",
        "datasets/inaoka/train/annotations.json",
        transform=transform
    )
    
    # バッチで学習をシミュレート
    images = []
    targets = []
    for i in range(4):
        img, target = dataset[i]
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images).to(device)
    for t in targets:
        t['boxes'] = t['boxes'].to(device)
        t['labels'] = t['labels'].to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images)
    
    print(f"Model output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min():.2f}, {outputs.max():.2f}]")
    
    # 損失計算テスト
    loss = detection_loss(outputs, targets, use_focal=True)
    print(f"Loss: {loss.item():.4f}")
    
    # 後処理テスト
    processed = postprocess_predictions(outputs, conf_threshold=0.3)
    for i, dets in enumerate(processed):
        print(f"\nImage {i}: {len(dets['boxes'])} detections")
        if len(dets['boxes']) > 0:
            print(f"  Top detection score: {dets['scores'][0]:.4f}")

if __name__ == "__main__":
    test_standard_model()