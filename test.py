import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CocoDataset
from model import AITECTDetector
from utils.bbox import nms
from utils.visualize import draw_boxes
from utils.metrics import precision_recall
import numpy as np

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)

@torch.no_grad()
def test():
    # 検証データセットを使用
    val_image_dir = "datasets/inaoka/val/JPEGImages"
    val_annotation_path = "datasets/inaoka/val/annotations.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    val_dataset = CocoDataset(val_image_dir, val_annotation_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    model = AITECTDetector().to(device)
    model.load_state_dict(torch.load("result/aitect_model.pth", map_location=device))
    model.eval()
    
    # 評価メトリクス
    all_precisions = []
    all_recalls = []
    conf_thresh = 0.5
    
    print("評価を開始します...")
    
    for idx, (images, targets) in enumerate(val_loader):
        images = images.to(device)
        output = model(images)[0]  # [N, 5]
        
        boxes = output[:, :4]
        scores = output[:, 4].sigmoid()
        
        # NMS適用
        keep = nms(boxes, scores, iou_threshold=0.4)
        selected_boxes = boxes[keep]
        selected_scores = scores[keep]
        
        # Ground truthボックス
        gt_boxes = targets[0]["boxes"].to(device)
        
        # Precision/Recall計算
        precision, recall = precision_recall(
            selected_boxes, selected_scores, gt_boxes, 
            iou_threshold=0.5, conf_thresh=conf_thresh
        )
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        
        # 最初の3枚だけ可視化
        if idx < 3:
            print(f"\n画像 {idx+1}:")
            print(f"  検出数: {len(selected_boxes)}")
            print(f"  GT数: {len(gt_boxes)}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            
            # 可視化
            image = val_dataset[idx][0]
            draw_boxes(image, selected_boxes.cpu(), selected_scores.cpu(), score_thresh=conf_thresh)
    
    # 全体の評価結果
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-6)
    
    print("\n=== 評価結果 ===")
    print(f"平均 Precision: {avg_precision:.3f}")
    print(f"平均 Recall: {avg_recall:.3f}")
    print(f"F1 Score: {f1_score:.3f}")
    print(f"評価画像数: {len(val_dataset)}")

if __name__ == "__main__":
    test()
