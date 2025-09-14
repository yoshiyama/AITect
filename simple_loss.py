import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLoss(nn.Module):
    """Simple loss function for multi-class pretraining"""
    
    def __init__(self, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets):
        """
        Simplified loss calculation
        Args:
            predictions: list of model outputs
            targets: list of target dicts
        Returns:
            loss value
        """
        device = predictions[0].device
        total_loss = 0
        
        # Process each scale
        for pred_scale in predictions:
            batch_size = pred_scale.size(0)
            grid_size = pred_scale.size(1)
            num_anchors = pred_scale.size(3)
            
            # Reshape predictions
            pred_scale = pred_scale.view(batch_size, grid_size, grid_size, num_anchors, 5 + self.num_classes)
            
            # Simple objectness loss (binary cross entropy)
            obj_pred = torch.sigmoid(pred_scale[..., 4])
            obj_target = torch.zeros_like(obj_pred)
            
            # Mark some cells as positive (simplified)
            for b in range(batch_size):
                if b < len(targets) and len(targets[b]['boxes']) > 0:
                    # Randomly mark some cells as positive
                    num_pos = min(5, len(targets[b]['boxes']))
                    for _ in range(num_pos):
                        i = torch.randint(0, grid_size, (1,)).item()
                        j = torch.randint(0, grid_size, (1,)).item()
                        a = torch.randint(0, num_anchors, (1,)).item()
                        obj_target[b, i, j, a] = 1.0
            
            # Objectness loss
            obj_loss = F.binary_cross_entropy(obj_pred, obj_target)
            
            # Bounding box loss (only for positive samples)
            pos_mask = obj_target > 0.5
            if pos_mask.sum() > 0:
                bbox_pred = pred_scale[..., :4][pos_mask]
                # Simple L2 loss for bounding boxes
                bbox_target = torch.rand_like(bbox_pred) * 10  # Random targets for now
                bbox_loss = self.mse_loss(bbox_pred, bbox_target)
            else:
                bbox_loss = torch.tensor(0.0).to(device)
            
            # Classification loss
            if self.num_classes > 1:
                cls_pred = pred_scale[..., 5:][pos_mask]
                if pos_mask.sum() > 0:
                    # Random class targets
                    cls_target = torch.randint(0, self.num_classes, (cls_pred.size(0),)).to(device)
                    cls_loss = self.ce_loss(cls_pred, cls_target)
                else:
                    cls_loss = torch.tensor(0.0).to(device)
            else:
                cls_loss = torch.tensor(0.0).to(device)
            
            # Combine losses
            scale_loss = obj_loss + 5.0 * bbox_loss + cls_loss
            total_loss += scale_loss
        
        return total_loss / len(predictions)