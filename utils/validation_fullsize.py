import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import os

@torch.no_grad()
def save_fullsize_detections(model, val_dataset, device, save_dir, epoch, num_samples=3):
    """
    Save detection results on original size images
    """
    model.eval()
    
    # Create subdirectory for full-size images
    fullsize_dir = os.path.join(save_dir, f'fullsize_epoch_{epoch}')
    os.makedirs(fullsize_dir, exist_ok=True)
    
    # Random sample indices
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    
    for idx in range(num_samples):
        ax_idx = indices[idx]
        
        # Get original image and target
        image_data = val_dataset.image_info[ax_idx]
        file_name = image_data['file_name']
        if file_name.startswith('JPEGImages/'):
            file_name = file_name[len('JPEGImages/'):]
        img_path = os.path.join(val_dataset.image_dir, file_name)
        original_image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = original_image.size
        
        # Get transformed image for model
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(original_image).unsqueeze(0).to(device)
        
        # Model prediction
        output = model(image_tensor)[0]  # [N, 5]
        
        # Process predictions
        boxes = output[:, :4]
        scores = output[:, 4].sigmoid()
        
        # Filter by confidence threshold
        conf_thresh = 0.3
        keep = scores > conf_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        
        # Apply NMS
        from utils.bbox import nms
        if len(boxes) > 0:
            keep_nms = nms(boxes, scores, iou_threshold=0.5)
            boxes = boxes[keep_nms]
            scores = scores[keep_nms]
        
        # Convert boxes back to original image coordinates
        scale_x = orig_width / 512
        scale_y = orig_height / 512
        
        boxes_original = boxes.clone()
        boxes_original[:, [0, 2]] *= scale_x
        boxes_original[:, [1, 3]] *= scale_y
        
        # Get ground truth boxes (already in original coordinates)
        image_id = image_data['id']
        anns = val_dataset.image_id_to_annotations.get(image_id, [])
        gt_boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
        gt_boxes = np.array(gt_boxes) if gt_boxes else np.array([]).reshape(0, 4)
        
        # Create figure for full-size image
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.imshow(original_image)
        
        # Draw ground truth boxes (green)
        for box in gt_boxes:
            rect = patches.Rectangle(
                (box[0], box[1]), 
                box[2]-box[0], 
                box[3]-box[1],
                linewidth=3, edgecolor='#00FF00', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(box[0], box[1]-10, 'GT', 
                   color='#00FF00', fontsize=12, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
        
        # Draw predicted boxes (red)
        boxes_np = boxes_original.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        for box, score in zip(boxes_np, scores_np):
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=2, edgecolor='#FF0000', facecolor='none',
                alpha=max(0.3, float(score))
            )
            ax.add_patch(rect)
            # Add confidence score
            ax.text(box[0], box[1]-10, f'{score:.2f}', 
                   color='#FF0000', fontsize=12, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        # Add title and legend
        ax.set_title(f'Sample {idx+1} - Original Size ({orig_width}x{orig_height})\n'
                     f'Predictions: {len(boxes_np)}, Ground Truth: {len(gt_boxes)}', 
                     fontsize=16)
        ax.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='#00FF00', linewidth=3, label='Ground Truth'),
            Patch(facecolor='none', edgecolor='#FF0000', linewidth=2, label='Prediction')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        # Save individual image
        save_path = os.path.join(fullsize_dir, f'detection_sample_{idx+1}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Full-size detection saved: {save_path}")
    
    return fullsize_dir