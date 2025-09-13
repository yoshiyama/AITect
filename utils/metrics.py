import torch
import numpy as np
from utils.bbox import box_iou

def precision_recall(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5, conf_thresh=0.5):
    if pred_boxes.numel() == 0:
        if gt_boxes.numel() == 0:
            return 1.0, 1.0, 1.0  # No predictions and no GT = perfect
        else:
            return 0.0, 0.0, 0.0  # No predictions but GT exists = all missed
    
    if gt_boxes.numel() == 0:
        return 0.0, 0.0, 0.0  # Predictions exist but no GT = all false positives

    # 信頼度でフィルタ
    keep = pred_scores > conf_thresh
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    
    if pred_boxes.numel() == 0:
        return 0.0, 0.0, 0.0

    matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
    tp, fp = 0, 0
    iou_list = []  # Store IoU values for analysis

    for pb in pred_boxes:
        ious = box_iou(pb.unsqueeze(0), gt_boxes)[0]
        max_iou, idx = ious.max(0)
        iou_list.append(max_iou.item())
        
        if max_iou >= iou_threshold and not matched_gt[idx]:
            tp += 1
            matched_gt[idx] = True
        else:
            fp += 1

    fn = (~matched_gt).sum().item()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    # Calculate average IoU for matched boxes
    avg_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0
    
    return precision, recall, avg_iou

def calculate_metrics_batch(predictions, targets, iou_threshold=0.5, conf_thresh=0.5):
    """Calculate metrics for a batch of predictions"""
    batch_size = len(predictions)
    metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'avg_iou': [],
        'total_gt': 0,
        'total_pred': 0,
        'total_tp': 0
    }
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred[:, :4]
        pred_scores = pred[:, 4].sigmoid()
        gt_boxes = target['boxes']
        
        # Filter by confidence
        keep = pred_scores > conf_thresh
        pred_boxes_filtered = pred_boxes[keep]
        pred_scores_filtered = pred_scores[keep]
        
        if pred_boxes_filtered.numel() > 0 or gt_boxes.numel() > 0:
            prec, rec, avg_iou = precision_recall(
                pred_boxes_filtered, pred_scores_filtered, gt_boxes,
                iou_threshold, conf_thresh
            )
            f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
        else:
            prec = rec = f1 = avg_iou = 0.0
            
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)
        metrics['avg_iou'].append(avg_iou)
        metrics['total_gt'] += len(gt_boxes)
        metrics['total_pred'] += len(pred_boxes_filtered)
        
    # Calculate averages
    metrics['avg_precision'] = np.mean(metrics['precision']) if metrics['precision'] else 0.0
    metrics['avg_recall'] = np.mean(metrics['recall']) if metrics['recall'] else 0.0
    metrics['avg_f1'] = np.mean(metrics['f1']) if metrics['f1'] else 0.0
    metrics['mean_iou'] = np.mean([iou for iou in metrics['avg_iou'] if iou > 0]) if any(iou > 0 for iou in metrics['avg_iou']) else 0.0
    
    return metrics