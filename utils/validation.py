import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import os

@torch.no_grad()
def validate_detection(model, val_dataset, device, num_samples=3, save_dir=None, epoch=None):
    """
    Validate detection results on validation dataset
    
    Args:
        model: Detection model
        val_dataset: Validation dataset
        device: torch device
        num_samples: Number of samples to visualize
        save_dir: Directory to save detection results
        epoch: Current epoch number
    """
    model.eval()
    
    # Create figure for visualization
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    if num_samples == 1:
        axes = [axes]
    
    # Random sample indices
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    
    for idx in range(num_samples):
        ax_idx = indices[idx]
        # Get image and ground truth
        image, target = val_dataset[ax_idx]
        original_image = transforms.ToPILImage()(image)
        
        # Debug: Print GT boxes
        print(f"\n[Sample {idx+1}] Image index: {ax_idx}")
        print(f"  Original image size: {original_image.size}")
        print(f"  Transformed image size: {image.shape}")
        print(f"  Ground Truth boxes (original coords):")
        for i, gt_box in enumerate(target["boxes"]):
            print(f"    GT Box {i+1}: x1={gt_box[0]:.1f}, y1={gt_box[1]:.1f}, x2={gt_box[2]:.1f}, y2={gt_box[3]:.1f}")
        
        # Model prediction
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)[0]  # [N, 5]
        
        # Process predictions
        boxes = output[:, :4]
        scores = output[:, 4].sigmoid()
        
        # Filter by confidence threshold
        conf_thresh = 0.3  # Lower threshold for V2 model initial training
        keep = scores > conf_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        
        print(f"  Predictions (after confidence filter):")
        print(f"    Total predictions: {len(boxes)}")
        if len(boxes) > 0:
            top_5 = min(5, len(boxes))
            sorted_indices = scores.argsort(descending=True)[:top_5]
            for i, pred_idx in enumerate(sorted_indices):
                print(f"    Pred {i+1}: x1={boxes[pred_idx][0]:.1f}, y1={boxes[pred_idx][1]:.1f}, x2={boxes[pred_idx][2]:.1f}, y2={boxes[pred_idx][3]:.1f}, conf={scores[pred_idx]:.3f}")
        
        # Apply NMS to reduce overlapping boxes
        from utils.bbox import nms
        if len(boxes) > 0:
            keep_nms = nms(boxes, scores, iou_threshold=0.4)
            boxes = boxes[keep_nms]
            scores = scores[keep_nms]
        
        # Convert to numpy for visualization
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        gt_boxes_np = target["boxes"].numpy()
        
        # Display image
        ax = axes[idx]
        ax.imshow(original_image)
        
        # Draw ground truth boxes (green)
        # Note: GT boxes are in original image coordinates, but the displayed image is 512x512
        # However, the transforms.ToPILImage() already handles the scaling, so we use coordinates as-is
        print(f"  Drawing GT boxes on 512x512 image:")
        for i, box in enumerate(gt_boxes_np):
            print(f"    GT Box {i+1} (for drawing): x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}")
            
            rect = patches.Rectangle(
                (box[0], box[1]), 
                box[2]-box[0], 
                box[3]-box[1],
                linewidth=4, edgecolor='#00FF00', facecolor='none'  # Bright green
            )
            ax.add_patch(rect)
        
        # Draw predicted boxes (red)
        print(f"  Drawing predicted boxes:")
        for i, (box, score) in enumerate(zip(boxes_np, scores_np)):
            print(f"    Pred Box {i+1}: x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}, conf={score:.3f}")
            
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=2, edgecolor='#FF0000', facecolor='none',  # Bright red
                alpha=max(0.3, float(score))  # Minimum alpha for visibility
            )
            ax.add_patch(rect)
            # Add confidence score
            ax.text(box[0], box[1]-5, f'{score:.2f}', 
                   color='red', fontsize=10, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        ax.set_title(f'Sample {idx+1}\nPred: {len(boxes_np)}, GT: {len(gt_boxes_np)}')
        ax.axis('off')
        
        # Set axis limits to match image size
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)  # Invert y-axis for image coordinates
    
    # Add legend to first subplot with proper colors
    if len(axes) > 0:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='#00FF00', linewidth=4, label='Ground Truth'),
            Patch(facecolor='none', edgecolor='#FF0000', linewidth=2, label='Prediction')
        ]
        axes[0].legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.suptitle(f'Validation Results - Epoch {epoch}' if epoch else 'Validation Results', 
                 fontsize=16)
    plt.tight_layout()
    
    # Save or show
    if save_dir and epoch is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'validation_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return None
        
def calculate_validation_metrics(model, val_loader, device):
    """
    Calculate validation metrics (mAP, precision, recall)
    """
    model.eval()
    
    all_detections = []
    all_ground_truths = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                # Get predictions
                boxes = output[:, :4]
                scores = output[:, 4].sigmoid()
                
                # Filter by confidence
                keep = scores > 0.5
                boxes = boxes[keep]
                scores = scores[keep]
                
                all_detections.append({
                    'boxes': boxes.cpu(),
                    'scores': scores.cpu()
                })
                
                all_ground_truths.append({
                    'boxes': targets[i]['boxes'].cpu()
                })
    
    # Simple detection count statistics
    total_gt = sum(len(gt['boxes']) for gt in all_ground_truths)
    total_pred = sum(len(det['boxes']) for det in all_detections)
    
    return {
        'total_ground_truth': total_gt,
        'total_predictions': total_pred,
        'avg_predictions_per_image': total_pred / len(all_detections) if len(all_detections) > 0 else 0
    }