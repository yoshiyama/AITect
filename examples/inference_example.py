#!/usr/bin/env python3
"""
Example inference script for AITect models
"""

import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_lightweight import create_lightweight_model
from utils.postprocess import non_max_suppression


def load_model(checkpoint_path, device='cuda'):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = create_lightweight_model(
        num_classes=checkpoint.get('num_classes', 1),
        model_size=checkpoint.get('model_size', 'tiny'),
        pretrained=False
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('classes', ['object'])


def preprocess_image(image_path, input_size=416):
    """Preprocess image for inference"""
    # Load image
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size
    
    # Transform
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor, img, orig_size


def postprocess_predictions(predictions, orig_size, input_size=416, conf_threshold=0.5, nms_threshold=0.4):
    """Post-process model predictions"""
    all_boxes = []
    all_scores = []
    all_labels = []
    
    # Process predictions from all scales
    for pred in predictions:
        # Apply confidence threshold and NMS
        boxes, scores, labels = non_max_suppression(
            pred[0],  # Remove batch dimension
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold
        )
        
        if len(boxes) > 0:
            # Scale boxes back to original image size
            scale_x = orig_size[0] / input_size
            scale_y = orig_size[1] / input_size
            
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
    
    # Concatenate results from all scales
    if all_boxes:
        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
    else:
        all_boxes = torch.empty((0, 4))
        all_scores = torch.empty(0)
        all_labels = torch.empty(0)
    
    return all_boxes, all_scores, all_labels


def visualize_predictions(image, boxes, scores, labels, class_names, save_path=None):
    """Visualize detection results"""
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Define colors for different classes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
    
    # Draw bounding boxes
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Get color for this class
        color = colors[int(label) % len(colors)]
        
        # Create rectangle
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label and score
        class_name = class_names[int(label)] if int(label) < len(class_names) else f'class_{int(label)}'
        text = f'{class_name}: {score:.2f}'
        ax.text(
            x1, y1 - 5,
            text,
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
            color='white',
            fontweight='bold'
        )
    
    ax.axis('off')
    plt.title(f'Detections: {len(boxes)} objects found')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def main():
    """Example inference pipeline"""
    # Configuration
    model_path = 'path/to/your/model.pth'  # Update this
    image_path = 'path/to/your/image.jpg'  # Update this
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, class_names = load_model(model_path, device)
    print(f"Model loaded. Classes: {class_names}")
    
    # Load and preprocess image
    print(f"Processing image: {image_path}")
    img_tensor, orig_img, orig_size = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # Post-process predictions
    boxes, scores, labels = postprocess_predictions(
        predictions, 
        orig_size,
        conf_threshold=0.5,
        nms_threshold=0.4
    )
    
    print(f"Found {len(boxes)} objects")
    
    # Visualize results
    if len(boxes) > 0:
        visualize_predictions(
            orig_img,
            boxes.cpu(),
            scores.cpu(),
            labels.cpu(),
            class_names,
            save_path='detection_result.png'
        )
        
        # Print detailed results
        print("\nDetailed results:")
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            class_name = class_names[int(label)] if int(label) < len(class_names) else f'class_{int(label)}'
            print(f"{i+1}. {class_name} (conf: {score:.3f}) - Box: {box.tolist()}")
    else:
        print("No objects detected.")


if __name__ == "__main__":
    # Example usage
    print("AITect Inference Example")
    print("========================")
    print("\nUsage:")
    print("1. Update model_path to point to your trained model")
    print("2. Update image_path to point to your test image")
    print("3. Run: python inference_example.py")
    
    # Uncomment to run inference
    # main()