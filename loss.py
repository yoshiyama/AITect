import torch
import torch.nn.functional as F
from utils.bbox import box_iou
from loss_improved import detection_loss_improved

def compute_iou(boxes1, boxes2):
    """
    boxes1, boxes2: [N, 4] each with [x1, y1, x2, y2]
    """
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area.clamp(min=1e-6)
    return iou

def iou_loss(pred_boxes, gt_boxes):
    """
    IoU loss for bounding box regression
    pred_boxes, gt_boxes: [N, 4] in xyxy format
    Returns: scalar loss value (1 - mean IoU)
    """
    # Compute pairwise IoU between predictions and ground truths
    ious = box_iou(pred_boxes, gt_boxes)
    
    # If shapes match (1-to-1 correspondence), use diagonal
    if pred_boxes.shape[0] == gt_boxes.shape[0]:
        ious = ious.diag()
    else:
        # Otherwise, use max IoU for each prediction
        ious, _ = ious.max(dim=1)
    
    # IoU loss is 1 - IoU (so perfect overlap = 0 loss)
    return 1.0 - ious.mean()

def focal_loss(pred_scores, target_conf, alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance
    FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    
    Args:
        pred_scores: [N] raw scores (logits)
        target_conf: [N] binary targets (0 or 1)
        alpha: weighting factor for positive class
        gamma: focusing parameter
    """
    # Apply sigmoid to get probabilities
    pred_prob = torch.sigmoid(pred_scores)
    
    # Calculate focal weight
    p_t = torch.where(target_conf == 1, pred_prob, 1 - pred_prob)
    focal_weight = (1 - p_t) ** gamma
    
    # Calculate weighted BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(pred_scores, target_conf, reduction='none')
    
    # Apply alpha weighting
    alpha_t = torch.where(target_conf == 1, alpha, 1 - alpha)
    
    # Combine all weights
    focal_loss = alpha_t * focal_weight * bce_loss
    
    return focal_loss.mean()

def detection_loss(preds, targets, iou_threshold=0.5, loss_type='mixed', iou_weight=2.0, l1_weight=0.5, use_focal=True):
    """
    preds: [B, N, 5] → predicted [x, y, w, h, conf]
    targets: list of dicts with keys 'boxes' and 'labels'
    """
    batch_size = preds.shape[0]
    total_loss = 0.0

    for b in range(batch_size):
        pred = preds[b]  # [N, 5]
        target = targets[b]  # dict

        # 予測値を分離: [x, y, w, h]形式（中心座標+幅高さ）
        pred_boxes_xywh = pred[:, :4]
        pred_scores = pred[:, 4]

        # 予測ボックスを[x1, y1, x2, y2]形式に変換
        pred_x1 = pred_boxes_xywh[:, 0] - pred_boxes_xywh[:, 2] / 2
        pred_y1 = pred_boxes_xywh[:, 1] - pred_boxes_xywh[:, 3] / 2
        pred_x2 = pred_boxes_xywh[:, 0] + pred_boxes_xywh[:, 2] / 2
        pred_y2 = pred_boxes_xywh[:, 1] + pred_boxes_xywh[:, 3] / 2
        pred_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)

        gt_boxes = target["boxes"]  # [M, 4] already in [x1, y1, x2, y2] format

        if gt_boxes.numel() == 0:
            # No object in ground truth → conf should be 0
            conf_loss = F.binary_cross_entropy_with_logits(pred_scores, torch.zeros_like(pred_scores))
            total_loss += conf_loss
            continue

        # Compute distance from each prediction to nearest GT
        N, M = pred_boxes_xywh.size(0), gt_boxes.size(0)
        
        # Calculate center points
        pred_centers = pred_boxes_xywh[:, :2]  # 既に中心座標 [x, y]
        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # [M, 2]
        
        # Distance matrix [N, M]
        distances = torch.cdist(pred_centers, gt_centers, p=2)
        
        # Find nearest GT for each prediction
        min_distances, nearest_gt_idx = distances.min(dim=1)  # [N]
        
        # Assign positive labels to predictions close to GT (within certain pixels)
        # Use adaptive threshold based on image size (512x512)
        distance_threshold = 50.0  # pixels
        close_mask = min_distances < distance_threshold
        
        # Also compute IoU for close predictions (using xyxy format)
        ious = torch.zeros(N, device=pred_boxes_xywh.device)
        for i in range(N):
            if close_mask[i]:
                ious[i] = compute_iou(pred_boxes_xyxy[i].unsqueeze(0), gt_boxes[nearest_gt_idx[i]].unsqueeze(0))[0]
        
        # Target confidence: 1 if close to GT, else 0
        target_conf = close_mask.float()
        
        # Confidence loss (for all predictions)
        if use_focal:
            conf_loss = focal_loss(pred_scores, target_conf)
        else:
            conf_loss = F.binary_cross_entropy_with_logits(pred_scores, target_conf, reduction='mean')
        
        # Regression loss (only for positive predictions)
        positive_mask = close_mask
        if positive_mask.sum() > 0:
            pos_pred_boxes = pred_boxes_xywh[positive_mask]
            pos_gt_boxes_xyxy = gt_boxes[nearest_gt_idx[positive_mask]]
            
            # GTボックスを[x, y, w, h]形式に変換
            pos_gt_x = (pos_gt_boxes_xyxy[:, 0] + pos_gt_boxes_xyxy[:, 2]) / 2
            pos_gt_y = (pos_gt_boxes_xyxy[:, 1] + pos_gt_boxes_xyxy[:, 3]) / 2
            pos_gt_w = pos_gt_boxes_xyxy[:, 2] - pos_gt_boxes_xyxy[:, 0]
            pos_gt_h = pos_gt_boxes_xyxy[:, 3] - pos_gt_boxes_xyxy[:, 1]
            pos_gt_boxes_xywh = torch.stack([pos_gt_x, pos_gt_y, pos_gt_w, pos_gt_h], dim=1)
            
            # Convert positive predictions to xyxy format for IoU loss
            pos_pred_x1 = pos_pred_boxes[:, 0] - pos_pred_boxes[:, 2] / 2
            pos_pred_y1 = pos_pred_boxes[:, 1] - pos_pred_boxes[:, 3] / 2
            pos_pred_x2 = pos_pred_boxes[:, 0] + pos_pred_boxes[:, 2] / 2
            pos_pred_y2 = pos_pred_boxes[:, 1] + pos_pred_boxes[:, 3] / 2
            pos_pred_boxes_xyxy = torch.stack([pos_pred_x1, pos_pred_y1, pos_pred_x2, pos_pred_y2], dim=1)
            
            # Choose loss based on configuration
            if loss_type == 'iou_only':
                # Pure IoU loss
                reg_loss = iou_loss(pos_pred_boxes_xyxy, pos_gt_boxes_xyxy)
            elif loss_type == 'l1_only':
                # Pure L1 loss (original)
                reg_loss = F.l1_loss(pos_pred_boxes, pos_gt_boxes_xywh, reduction='mean')
            else:  # mixed (default)
                # IoU loss
                iou_reg_loss = iou_loss(pos_pred_boxes_xyxy, pos_gt_boxes_xyxy)
                # L1 loss
                l1_reg_loss = F.l1_loss(pos_pred_boxes, pos_gt_boxes_xywh, reduction='mean')
                # Weighted combination
                reg_loss = iou_reg_loss * iou_weight + l1_reg_loss * l1_weight
        else:
            # If no positive predictions, add penalty to encourage predictions near GT
            reg_loss = min_distances.mean() / 100.0  # Normalized distance penalty
        
        # Total loss with balanced weights
        total_loss += conf_loss + reg_loss * 2.0  # Weight regression loss more

    return total_loss / batch_size