import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalDetectionLoss(nn.Module):
    """
    YOLO/RetinaNet style detection loss with Focal Loss
    - Focal Loss for classification (handles class imbalance)
    - IoU Loss for bounding box regression
    - Binary Cross Entropy for objectness
    """
    
    def __init__(self, num_classes=20, 
                 focal_alpha=0.25, focal_gamma=2.0,
                 bbox_weight=5.0, obj_weight=1.0, cls_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.bbox_weight = bbox_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        
        # Anchor boxes for different scales
        self.anchors = [
            [(10, 13), (16, 30), (33, 23)],      # Small objects
            [(30, 61), (62, 45), (59, 119)],     # Medium objects
            [(116, 90), (156, 198), (373, 326)]  # Large objects
        ]
        
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal Loss for addressing class imbalance"""
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = alpha * (1 - p_t) ** gamma
        return (focal_weight * ce_loss).mean()
    
    def iou_loss(self, pred_boxes, target_boxes):
        """IoU Loss for better bounding box regression"""
        # Calculate IoU
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        
        union = pred_area + target_area - intersection
        iou = intersection / (union + 1e-6)
        
        # IoU loss = 1 - IoU
        return (1 - iou).mean()
    
    def build_targets(self, predictions, targets):
        """Build targets for each prediction scale"""
        device = predictions[0].device
        batch_size = predictions[0].size(0)
        
        obj_masks = []
        bbox_targets = []
        cls_targets = []
        
        for scale_idx, pred in enumerate(predictions):
            grid_size = pred.size(1)
            num_anchors = pred.size(3)
            
            # Initialize masks and targets
            obj_mask = torch.zeros(batch_size, grid_size, grid_size, num_anchors, dtype=torch.float32, device=device)
            bbox_target = torch.zeros(batch_size, grid_size, grid_size, num_anchors, 4, dtype=torch.float32, device=device)
            cls_target = torch.zeros(batch_size, grid_size, grid_size, num_anchors, self.num_classes, dtype=torch.float32, device=device)
            
            # Process each image in batch
            for b in range(batch_size):
                if b < len(targets) and len(targets[b]['boxes']) > 0:
                    # Scale targets to grid size
                    scaled_boxes = targets[b]['boxes'] / 416.0 * grid_size  # Assuming 416 input size
                    
                    for box_idx, box in enumerate(scaled_boxes):
                        if box_idx >= len(targets[b]['labels']):
                            continue
                            
                        # Get grid cell location
                        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                        grid_x, grid_y = int(cx), int(cy)
                        
                        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                            # Find best anchor
                            box_w, box_h = box[2] - box[0], box[3] - box[1]
                            best_anchor = 0
                            best_iou = 0
                            
                            for anchor_idx, (anchor_w, anchor_h) in enumerate(self.anchors[scale_idx]):
                                # Simple IoU with anchor
                                anchor_w_scaled = anchor_w / 416.0 * grid_size
                                anchor_h_scaled = anchor_h / 416.0 * grid_size
                                
                                min_w = min(box_w, anchor_w_scaled)
                                min_h = min(box_h, anchor_h_scaled)
                                intersection = min_w * min_h
                                union = box_w * box_h + anchor_w_scaled * anchor_h_scaled - intersection
                                iou = intersection / (union + 1e-6)
                                
                                if iou > best_iou:
                                    best_iou = iou
                                    best_anchor = anchor_idx
                            
                            # Set targets
                            obj_mask[b, grid_y, grid_x, best_anchor] = 1.0
                            
                            # Bounding box targets (relative to grid cell)
                            bbox_target[b, grid_y, grid_x, best_anchor] = torch.tensor([
                                cx - grid_x,  # x offset
                                cy - grid_y,  # y offset
                                box_w,        # width
                                box_h         # height
                            ])
                            
                            # Class target
                            label = targets[b]['labels'][box_idx]
                            if label < self.num_classes:
                                cls_target[b, grid_y, grid_x, best_anchor, label] = 1.0
            
            obj_masks.append(obj_mask)
            bbox_targets.append(bbox_target)
            cls_targets.append(cls_target)
        
        return obj_masks, bbox_targets, cls_targets
    
    def forward(self, predictions, targets):
        """
        Calculate total loss
        Args:
            predictions: list of model outputs [B, H, W, A, 5+num_classes]
            targets: list of dicts with 'boxes' and 'labels'
        """
        device = predictions[0].device
        
        # Build targets for all scales
        obj_masks, bbox_targets, cls_targets = self.build_targets(predictions, targets)
        
        total_loss = 0
        losses = {'obj': 0, 'bbox': 0, 'cls': 0}
        
        for scale_idx, pred in enumerate(predictions):
            batch_size = pred.size(0)
            
            # Split predictions
            bbox_pred = pred[..., :4]
            obj_pred = torch.sigmoid(pred[..., 4:5])
            cls_pred = torch.sigmoid(pred[..., 5:])
            
            # Get targets for this scale
            obj_mask = obj_masks[scale_idx]
            bbox_target = bbox_targets[scale_idx]
            cls_target = cls_targets[scale_idx]
            
            # Objectness loss (Focal Loss)
            obj_loss = self.focal_loss(
                obj_pred.reshape(-1), 
                obj_mask.reshape(-1),
                alpha=self.focal_alpha,
                gamma=self.focal_gamma
            )
            
            # Bounding box loss (only for positive samples)
            pos_mask = obj_mask > 0.5
            if pos_mask.sum() > 0:
                # Transform predictions
                bbox_pred_transformed = bbox_pred.clone()
                bbox_pred_transformed[..., :2] = torch.sigmoid(bbox_pred[..., :2])  # x,y offsets
                bbox_pred_transformed[..., 2:] = torch.exp(bbox_pred[..., 2:].clamp(max=10))  # w,h
                
                # Convert to x1,y1,x2,y2 format for IoU loss
                pred_boxes = torch.zeros_like(bbox_pred_transformed)
                pred_boxes[..., 0] = bbox_pred_transformed[..., 0] - bbox_pred_transformed[..., 2] / 2
                pred_boxes[..., 1] = bbox_pred_transformed[..., 1] - bbox_pred_transformed[..., 3] / 2
                pred_boxes[..., 2] = bbox_pred_transformed[..., 0] + bbox_pred_transformed[..., 2] / 2
                pred_boxes[..., 3] = bbox_pred_transformed[..., 1] + bbox_pred_transformed[..., 3] / 2
                
                target_boxes = torch.zeros_like(bbox_target)
                target_boxes[..., 0] = bbox_target[..., 0] - bbox_target[..., 2] / 2
                target_boxes[..., 1] = bbox_target[..., 1] - bbox_target[..., 3] / 2
                target_boxes[..., 2] = bbox_target[..., 0] + bbox_target[..., 2] / 2
                target_boxes[..., 3] = bbox_target[..., 1] + bbox_target[..., 3] / 2
                
                bbox_loss = self.iou_loss(
                    pred_boxes[pos_mask].reshape(-1, 4),
                    target_boxes[pos_mask].reshape(-1, 4)
                )
            else:
                bbox_loss = torch.tensor(0.0, device=device)
            
            # Classification loss (Focal Loss for multi-class)
            if self.num_classes > 1 and pos_mask.sum() > 0:
                cls_loss = self.focal_loss(
                    cls_pred[pos_mask].reshape(-1, self.num_classes),
                    cls_target[pos_mask].reshape(-1, self.num_classes),
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma
                )
            else:
                cls_loss = torch.tensor(0.0, device=device)
            
            # Combine losses
            scale_loss = (self.obj_weight * obj_loss + 
                         self.bbox_weight * bbox_loss + 
                         self.cls_weight * cls_loss)
            
            total_loss += scale_loss
            losses['obj'] += obj_loss.item()
            losses['bbox'] += bbox_loss.item()
            losses['cls'] += cls_loss.item()
        
        # Average over scales
        total_loss /= len(predictions)
        for k in losses:
            losses[k] /= len(predictions)
        
        return total_loss, losses