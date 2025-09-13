import torch
import torch.nn.functional as F
from utils.bbox import box_iou

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
    """IoU loss for bounding box regression"""
    ious = box_iou(pred_boxes, gt_boxes)
    
    if pred_boxes.shape[0] == gt_boxes.shape[0]:
        ious = ious.diag()
    else:
        ious, _ = ious.max(dim=1)
    
    return 1.0 - ious.mean()

def focal_loss(pred_scores, target_conf, alpha=0.25, gamma=2.0):
    """Focal Loss for addressing class imbalance"""
    pred_prob = torch.sigmoid(pred_scores)
    
    p_t = torch.where(target_conf == 1, pred_prob, 1 - pred_prob)
    focal_weight = (1 - p_t) ** gamma
    
    bce_loss = F.binary_cross_entropy_with_logits(pred_scores, target_conf, reduction='none')
    alpha_t = torch.where(target_conf == 1, alpha, 1 - alpha)
    
    focal_loss = alpha_t * focal_weight * bce_loss
    
    return focal_loss.mean()

def detection_loss_improved(preds, targets, iou_threshold=0.5, loss_type='mixed', 
                          iou_weight=3.0, l1_weight=0.5, use_focal=True):
    """
    改善された損失関数：IoUベースのマッチングを使用
    """
    batch_size = preds.shape[0]
    total_loss = 0.0

    for b in range(batch_size):
        pred = preds[b]  # [N, 5]
        target = targets[b]  # dict

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
            # No object in ground truth
            conf_loss = F.binary_cross_entropy_with_logits(pred_scores, torch.zeros_like(pred_scores))
            total_loss += conf_loss
            continue

        # IoUベースのマッチング（改善点）
        N, M = pred_boxes_xyxy.size(0), gt_boxes.size(0)
        
        # すべての予測とGTのIoUを計算
        ious_all = box_iou(pred_boxes_xyxy, gt_boxes)  # [N, M]
        
        # 各予測に対して最大IoUとそのGTインデックスを取得
        max_ious, matched_gt_idx = ious_all.max(dim=1)  # [N]
        
        # IoUベースで正例を決定（距離ではなくIoU閾値を使用）
        # さらに、各GTに対して最もIoUが高い予測も正例に含める
        positive_mask = max_ious > 0.1  # IoU > 0.1なら正例候補
        
        # 各GTに対して最もIoUが高い予測を見つける
        best_pred_per_gt, _ = ious_all.max(dim=0)  # [M]
        for m in range(M):
            if best_pred_per_gt[m] > 0.05:  # 最小限のIoUがあれば
                best_pred_idx = ious_all[:, m].argmax()
                positive_mask[best_pred_idx] = True
        
        # ターゲット信頼度
        target_conf = positive_mask.float()
        
        # 信頼度損失
        if use_focal:
            conf_loss = focal_loss(pred_scores, target_conf)
        else:
            conf_loss = F.binary_cross_entropy_with_logits(pred_scores, target_conf, reduction='mean')
        
        # 回帰損失（正例のみ）
        if positive_mask.sum() > 0:
            pos_pred_boxes = pred_boxes_xywh[positive_mask]
            pos_gt_boxes_xyxy = gt_boxes[matched_gt_idx[positive_mask]]
            
            # GTボックスを[x, y, w, h]形式に変換
            pos_gt_x = (pos_gt_boxes_xyxy[:, 0] + pos_gt_boxes_xyxy[:, 2]) / 2
            pos_gt_y = (pos_gt_boxes_xyxy[:, 1] + pos_gt_boxes_xyxy[:, 3]) / 2
            pos_gt_w = pos_gt_boxes_xyxy[:, 2] - pos_gt_boxes_xyxy[:, 0]
            pos_gt_h = pos_gt_boxes_xyxy[:, 3] - pos_gt_boxes_xyxy[:, 1]
            pos_gt_boxes_xywh = torch.stack([pos_gt_x, pos_gt_y, pos_gt_w, pos_gt_h], dim=1)
            
            # IoU損失を重視
            pos_pred_x1 = pos_pred_boxes[:, 0] - pos_pred_boxes[:, 2] / 2
            pos_pred_y1 = pos_pred_boxes[:, 1] - pos_pred_boxes[:, 3] / 2
            pos_pred_x2 = pos_pred_boxes[:, 0] + pos_pred_boxes[:, 2] / 2
            pos_pred_y2 = pos_pred_boxes[:, 1] + pos_pred_boxes[:, 3] / 2
            pos_pred_boxes_xyxy = torch.stack([pos_pred_x1, pos_pred_y1, pos_pred_x2, pos_pred_y2], dim=1)
            
            if loss_type == 'iou_only':
                reg_loss = iou_loss(pos_pred_boxes_xyxy, pos_gt_boxes_xyxy)
            elif loss_type == 'l1_only':
                reg_loss = F.l1_loss(pos_pred_boxes, pos_gt_boxes_xywh, reduction='mean')
            else:  # mixed
                iou_reg_loss = iou_loss(pos_pred_boxes_xyxy, pos_gt_boxes_xyxy)
                l1_reg_loss = F.l1_loss(pos_pred_boxes, pos_gt_boxes_xywh, reduction='mean')
                reg_loss = iou_reg_loss * iou_weight + l1_reg_loss * l1_weight
        else:
            # 正例がない場合、最もIoUが高い予測を促す
            reg_loss = (1 - max_ious.max()) * 2.0
        
        # 総損失（回帰損失の重みを増加）
        total_loss += conf_loss + reg_loss * 3.0

    return total_loss / batch_size