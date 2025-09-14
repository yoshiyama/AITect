import torch
import torchvision.ops as ops

def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    Args:
        boxes: [N, 4] tensor of boxes in xyxy format
        scores: [N] tensor of scores
        iou_threshold: IoU threshold for suppression
    Returns:
        keep: indices of boxes to keep
    """
    return ops.nms(boxes, scores, iou_threshold)


def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001):
    """
    Soft-NMS: decreases scores of overlapping boxes instead of removing them
    Args:
        boxes: [N, 4] tensor of boxes in xyxy format
        scores: [N] tensor of scores
        iou_threshold: IoU threshold
        sigma: parameter for score reduction
        score_threshold: minimum score to keep box
    Returns:
        keep: indices of boxes to keep
        scores: updated scores
    """
    keep = []
    idxs = scores.argsort(descending=True)
    
    while idxs.numel() > 0:
        # Pick the box with highest score
        i = idxs[0]
        keep.append(i.item())
        
        if idxs.numel() == 1:
            break
        
        # Compute IoU with remaining boxes
        other_idxs = idxs[1:]
        ious = ops.box_iou(boxes[i].unsqueeze(0), boxes[other_idxs]).squeeze(0)
        
        # Apply soft-NMS
        weight = torch.exp(-(ious ** 2) / sigma)
        scores[other_idxs] *= weight
        
        # Remove boxes with low scores
        remaining_idxs = other_idxs[scores[other_idxs] > score_threshold]
        idxs = remaining_idxs[scores[remaining_idxs].argsort(descending=True)]
    
    keep = torch.tensor(keep, dtype=torch.long)
    return keep, scores


def postprocess_predictions(predictions, 
                          conf_threshold=0.5,
                          nms_threshold=0.5,
                          max_detections=100,
                          use_soft_nms=False):
    """
    Post-process raw model predictions
    Args:
        predictions: [B, N, 5+num_classes] raw predictions
        conf_threshold: confidence threshold
        nms_threshold: NMS IoU threshold
        max_detections: maximum detections per image
        use_soft_nms: whether to use soft-NMS
    Returns:
        list of dicts with 'boxes', 'scores', 'labels' for each image
    """
    batch_size = predictions.shape[0]
    num_classes = predictions.shape[2] - 5
    
    results = []
    
    for b in range(batch_size):
        pred = predictions[b]  # [N, 5+num_classes]
        
        # Extract components
        boxes_xywh = pred[:, :4]
        objectness = torch.sigmoid(pred[:, 4])
        
        # Convert to xyxy format
        x_center = boxes_xywh[:, 0]
        y_center = boxes_xywh[:, 1]
        width = boxes_xywh[:, 2]
        height = boxes_xywh[:, 3]
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        
        # Handle single vs multi-class
        if num_classes <= 1:
            # Single class - use objectness as score
            scores = objectness
            labels = torch.zeros_like(scores, dtype=torch.long)
        else:
            # Multi-class
            class_probs = torch.softmax(pred[:, 5:], dim=1)
            scores, labels = class_probs.max(dim=1)
            scores = scores * objectness  # Combine with objectness
        
        # Filter by confidence
        keep = scores > conf_threshold
        if keep.sum() == 0:
            results.append({
                'boxes': torch.empty((0, 4)),
                'scores': torch.empty((0,)),
                'labels': torch.empty((0,), dtype=torch.long)
            })
            continue
        
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        # Apply NMS
        if use_soft_nms:
            keep_idxs, scores = soft_nms(boxes, scores, nms_threshold)
        else:
            keep_idxs = nms(boxes, scores, nms_threshold)
        
        # Limit number of detections
        if len(keep_idxs) > max_detections:
            keep_idxs = keep_idxs[:max_detections]
        
        boxes = boxes[keep_idxs]
        scores = scores[keep_idxs]
        labels = labels[keep_idxs]
        
        # Clip boxes to image bounds
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=512)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=512)
        
        results.append({
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        })
    
    return results


def adjust_confidence_threshold(predictions, targets, iou_threshold=0.5):
    """
    Dynamically adjust confidence threshold based on validation data
    Args:
        predictions: model predictions
        targets: ground truth
        iou_threshold: IoU threshold for matching
    Returns:
        optimal_threshold: recommended confidence threshold
    """
    thresholds = torch.linspace(0.1, 0.9, 20)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        # Process predictions with current threshold
        processed = postprocess_predictions(
            predictions, 
            conf_threshold=threshold,
            nms_threshold=0.5
        )
        
        # Calculate F1 score
        f1_scores = []
        for pred, target in zip(processed, targets):
            if len(pred['boxes']) == 0 or len(target['boxes']) == 0:
                f1_scores.append(0.0)
                continue
            
            # Calculate IoU matrix
            ious = ops.box_iou(pred['boxes'], target['boxes'])
            
            # Count matches
            matches = (ious > iou_threshold).any(dim=1).sum().item()
            
            precision = matches / len(pred['boxes']) if len(pred['boxes']) > 0 else 0
            recall = matches / len(target['boxes']) if len(target['boxes']) > 0 else 0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_threshold = threshold.item()
    
    return best_threshold