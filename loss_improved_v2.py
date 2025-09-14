import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.bbox import box_iou

class ImprovedDetectionLoss(nn.Module):
    """Improved detection loss with better positive assignment and balance"""
    def __init__(self, num_classes=1, 
                 pos_iou_threshold=0.5,
                 neg_iou_threshold=0.4,
                 focal_loss=True,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 bbox_loss_weight=2.0,
                 use_giou=True):
        super().__init__()
        self.num_classes = num_classes
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.bbox_loss_weight = bbox_loss_weight
        self.use_giou = use_giou
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, N, 5+num_classes] raw predictions
            targets: list of dicts with 'boxes' and 'labels'
        Returns:
            loss: scalar loss value
        """
        batch_size = predictions.shape[0]
        device = predictions.device
        
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        num_pos = 0
        
        for b in range(batch_size):
            pred = predictions[b]  # [N, 5+num_classes]
            target = targets[b]
            
            # Extract predictions
            pred_boxes = self._decode_boxes(pred[:, :4])  # [N, 4] in xyxy format
            pred_obj = pred[:, 4]  # [N] objectness scores
            
            if self.num_classes > 1:
                pred_cls = pred[:, 5:]  # [N, num_classes]
            
            # Get ground truth
            gt_boxes = target['boxes']  # [M, 4] in xyxy format
            gt_labels = target['labels']  # [M]
            
            if gt_boxes.numel() == 0:
                # No objects - all predictions should be negative
                neg_loss = self._focal_loss(pred_obj, torch.zeros_like(pred_obj))
                total_cls_loss += neg_loss
                continue
            
            # Compute IoU between predictions and ground truth
            ious = box_iou(pred_boxes, gt_boxes)  # [N, M]
            
            # Assign predictions to ground truth
            # 1. For each GT, find best matching prediction (to ensure each GT is matched)
            best_pred_per_gt, best_pred_idx = ious.max(dim=0)  # [M]
            
            # 2. For each prediction, find best matching GT
            best_gt_per_pred, best_gt_idx = ious.max(dim=1)  # [N]
            
            # 3. Determine positive and negative samples
            # Positive: best match for each GT OR IoU > pos_threshold
            pos_mask = torch.zeros(pred.shape[0], dtype=torch.bool, device=device)
            
            # Mark best predictions for each GT as positive
            for gt_idx, pred_idx in enumerate(best_pred_idx):
                if best_pred_per_gt[gt_idx] > 0.1:  # Minimal IoU requirement
                    pos_mask[pred_idx] = True
            
            # Also mark predictions with high IoU as positive
            high_iou_mask = best_gt_per_pred > self.pos_iou_threshold
            pos_mask = pos_mask | high_iou_mask
            
            # Negative: IoU < neg_threshold with all GTs
            neg_mask = best_gt_per_pred < self.neg_iou_threshold
            
            # Ignore: between neg and pos thresholds
            ignore_mask = ~(pos_mask | neg_mask)
            
            # Classification loss
            num_pos_batch = pos_mask.sum().item()
            num_pos += num_pos_batch
            
            # Objectness targets
            obj_targets = torch.zeros_like(pred_obj)
            obj_targets[pos_mask] = 1.0
            
            # Only compute loss for pos and neg samples (ignore neutral)
            valid_mask = pos_mask | neg_mask
            if valid_mask.sum() > 0:
                if self.focal_loss:
                    cls_loss = self._focal_loss(
                        pred_obj[valid_mask], 
                        obj_targets[valid_mask]
                    )
                else:
                    cls_loss = F.binary_cross_entropy_with_logits(
                        pred_obj[valid_mask],
                        obj_targets[valid_mask],
                        reduction='mean'
                    )
                total_cls_loss += cls_loss
            
            # Multi-class classification loss (if applicable)
            if self.num_classes > 1 and pos_mask.sum() > 0:
                pos_gt_labels = gt_labels[best_gt_idx[pos_mask]]
                class_loss = F.cross_entropy(
                    pred_cls[pos_mask],
                    pos_gt_labels,
                    reduction='mean'
                )
                total_cls_loss += class_loss
            
            # Regression loss (only for positive samples)
            if pos_mask.sum() > 0:
                pos_pred_boxes = pred_boxes[pos_mask]
                pos_gt_boxes = gt_boxes[best_gt_idx[pos_mask]]
                
                if self.use_giou:
                    reg_loss = self._giou_loss(pos_pred_boxes, pos_gt_boxes)
                else:
                    # Convert to xywh for L1 loss
                    pos_pred_xywh = self._xyxy_to_xywh(pos_pred_boxes)
                    pos_gt_xywh = self._xyxy_to_xywh(pos_gt_boxes)
                    reg_loss = F.l1_loss(pos_pred_xywh, pos_gt_xywh, reduction='mean')
                
                total_reg_loss += reg_loss
        
        # Normalize by number of positive samples
        num_pos = max(num_pos, 1)
        total_cls_loss = total_cls_loss / batch_size
        total_reg_loss = total_reg_loss / num_pos * batch_size
        
        # Total loss
        total_loss = total_cls_loss + self.bbox_loss_weight * total_reg_loss
        
        return total_loss
    
    def _decode_boxes(self, boxes):
        """Convert predicted boxes from xywh to xyxy format"""
        x_center = boxes[:, 0]
        y_center = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def _xyxy_to_xywh(self, boxes):
        """Convert boxes from xyxy to xywh format"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        return torch.stack([cx, cy, w, h], dim=1)
    
    def _focal_loss(self, pred, target):
        """Focal loss for addressing class imbalance"""
        pred_prob = torch.sigmoid(pred)
        
        # Calculate focal weight
        p_t = torch.where(target == 1, pred_prob, 1 - pred_prob)
        focal_weight = (1 - p_t) ** self.focal_gamma
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Apply alpha weighting
        alpha_t = torch.where(target == 1, self.focal_alpha, 1 - self.focal_alpha)
        
        # Combine
        focal_loss = alpha_t * focal_weight * bce_loss
        
        return focal_loss.mean()
    
    def _giou_loss(self, pred_boxes, target_boxes):
        """Generalized IoU loss"""
        # Calculate IoU
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        
        union = pred_area + target_area - intersection
        iou = intersection / union.clamp(min=1e-6)
        
        # Calculate GIoU
        # Find enclosing box
        enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
        
        giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
        
        return 1 - giou.mean()


def detection_loss_improved_v2(predictions, targets, **kwargs):
    """Convenience function for improved detection loss"""
    loss_fn = ImprovedDetectionLoss(**kwargs)
    return loss_fn(predictions, targets)