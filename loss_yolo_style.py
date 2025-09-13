import torch
import torch.nn.functional as F
from utils.bbox import box_iou

def yolo_style_detection_loss(preds, targets, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5,
                             use_focal=True, focal_alpha=0.25, focal_gamma=2.0):
    """
    YOLOスタイルの損失関数
    
    主な特徴：
    1. 正例と負例で異なる重み付け
    2. 各GTに責任を持つセルを明確に定義
    3. 座標損失は正例のみに適用
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
        
        gt_boxes = target["boxes"]  # [M, 4] in xyxy format
        
        if gt_boxes.numel() == 0:
            # GTなし - すべて負例として扱う
            if use_focal:
                noobj_loss = focal_loss(pred_scores, torch.zeros_like(pred_scores), 
                                      alpha=1-focal_alpha, gamma=focal_gamma)
            else:
                noobj_loss = F.binary_cross_entropy_with_logits(
                    pred_scores, torch.zeros_like(pred_scores), reduction='mean'
                )
            total_loss += noobj_loss * lambda_noobj
            continue
        
        # YOLOスタイルの正例割り当て
        N, M = pred_boxes_xyxy.size(0), gt_boxes.size(0)
        
        # 1. 各GTの中心を含むグリッドセルを特定（仮想的に）
        # ここでは簡略化のため、IoUベースで割り当て
        ious_all = box_iou(pred_boxes_xyxy, gt_boxes)  # [N, M]
        
        # 2. 各GTに対して最もIoUが高い予測を必ず正例にする（責任の割り当て）
        positive_mask = torch.zeros(N, dtype=torch.bool, device=pred.device)
        matched_gt_idx = torch.zeros(N, dtype=torch.long, device=pred.device)
        
        for m in range(M):
            # 各GTに対して最もIoUが高い予測を見つける
            best_iou, best_idx = ious_all[:, m].max(dim=0)
            positive_mask[best_idx] = True
            matched_gt_idx[best_idx] = m
            
            # 追加：IoU > 0.5の予測も正例に含める（YOLOv2以降の戦略）
            high_iou_mask = ious_all[:, m] > 0.5
            positive_mask |= high_iou_mask
            matched_gt_idx[high_iou_mask] = m
        
        # 3. Ignore maskの設定（IoUが中途半端な予測は無視）
        # IoUが0.3-0.5の範囲の予測は損失計算から除外
        max_ious, _ = ious_all.max(dim=1)
        ignore_mask = (max_ious > 0.3) & (max_ious < 0.5) & (~positive_mask)
        
        # 負例マスク（正例でもignoreでもない）
        negative_mask = (~positive_mask) & (~ignore_mask)
        
        # 4. 信頼度損失の計算
        target_conf = positive_mask.float()
        
        if use_focal:
            # Focal Lossを使用
            # 正例の損失
            if positive_mask.sum() > 0:
                obj_loss = focal_loss(pred_scores[positive_mask], 
                                    target_conf[positive_mask],
                                    alpha=focal_alpha, gamma=focal_gamma)
                obj_loss = obj_loss * lambda_obj
            else:
                obj_loss = 0.0
            
            # 負例の損失（重み付けを軽くする）
            if negative_mask.sum() > 0:
                noobj_loss = focal_loss(pred_scores[negative_mask], 
                                      target_conf[negative_mask],
                                      alpha=1-focal_alpha, gamma=focal_gamma)
                noobj_loss = noobj_loss * lambda_noobj
            else:
                noobj_loss = 0.0
            
            conf_loss = obj_loss + noobj_loss
        else:
            # 標準のBCE Loss
            # マスクを適用して損失を計算
            obj_loss = F.binary_cross_entropy_with_logits(
                pred_scores[positive_mask], 
                target_conf[positive_mask], 
                reduction='mean' if positive_mask.sum() > 0 else 'sum'
            ) * lambda_obj if positive_mask.sum() > 0 else 0.0
            
            noobj_loss = F.binary_cross_entropy_with_logits(
                pred_scores[negative_mask], 
                target_conf[negative_mask], 
                reduction='mean' if negative_mask.sum() > 0 else 'sum'
            ) * lambda_noobj if negative_mask.sum() > 0 else 0.0
            
            conf_loss = obj_loss + noobj_loss
        
        # 5. 座標回帰損失（正例のみ、重み付けを大きく）
        if positive_mask.sum() > 0:
            pos_pred_boxes = pred_boxes_xywh[positive_mask]
            pos_gt_boxes_xyxy = gt_boxes[matched_gt_idx[positive_mask]]
            
            # GTボックスを[x, y, w, h]形式に変換
            pos_gt_x = (pos_gt_boxes_xyxy[:, 0] + pos_gt_boxes_xyxy[:, 2]) / 2
            pos_gt_y = (pos_gt_boxes_xyxy[:, 1] + pos_gt_boxes_xyxy[:, 3]) / 2
            pos_gt_w = pos_gt_boxes_xyxy[:, 2] - pos_gt_boxes_xyxy[:, 0]
            pos_gt_h = pos_gt_boxes_xyxy[:, 3] - pos_gt_boxes_xyxy[:, 1]
            pos_gt_boxes_xywh = torch.stack([pos_gt_x, pos_gt_y, pos_gt_w, pos_gt_h], dim=1)
            
            # YOLOスタイル：中心座標と幅高さで異なる損失
            # 中心座標：MSE Loss
            center_loss = F.mse_loss(pos_pred_boxes[:, :2], pos_gt_boxes_xywh[:, :2])
            
            # 幅高さ：sqrt(w), sqrt(h)でMSE（小さいボックスへのペナルティを相対的に大きく）
            pred_wh_sqrt = torch.sqrt(pos_pred_boxes[:, 2:].clamp(min=1e-6))
            gt_wh_sqrt = torch.sqrt(pos_gt_boxes_xywh[:, 2:].clamp(min=1e-6))
            size_loss = F.mse_loss(pred_wh_sqrt, gt_wh_sqrt)
            
            coord_loss = (center_loss + size_loss) * lambda_coord
        else:
            coord_loss = 0.0
        
        # 6. 総損失
        loss_components = {
            'conf_loss': conf_loss,
            'coord_loss': coord_loss,
            'n_positive': positive_mask.sum().item(),
            'n_negative': negative_mask.sum().item(),
            'n_ignore': ignore_mask.sum().item()
        }
        
        total_loss += conf_loss + coord_loss
        
        # デバッグ情報を出力（最初のバッチのみ）
        if b == 0:
            print(f"[Loss Debug] Positive: {loss_components['n_positive']}, "
                  f"Negative: {loss_components['n_negative']}, "
                  f"Ignore: {loss_components['n_ignore']}")
    
    return total_loss / batch_size


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal Loss実装"""
    pred_sigmoid = torch.sigmoid(pred)
    
    # pt = p if y = 1 else 1-p
    pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
    
    # Focal weight
    focal_weight = (1 - pt).pow(gamma)
    
    # Binary cross entropy
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    
    # Alpha weight
    alpha_weight = torch.where(target == 1, alpha, 1 - alpha)
    
    # Final focal loss
    focal_loss = alpha_weight * focal_weight * bce
    
    return focal_loss.mean()


def giou_loss(pred_boxes, target_boxes):
    """
    Generalized IoU Loss（YOLOv4以降で使用）
    より精密なボックス回帰
    """
    # IoU計算
    iou = box_iou(pred_boxes, target_boxes)
    if pred_boxes.shape[0] == target_boxes.shape[0]:
        iou = iou.diag()
    
    # 最小外接矩形の計算
    x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    
    enclosing_area = (x2 - x1) * (y2 - y1)
    
    # GIoU = IoU - (enclosing_area - union_area) / enclosing_area
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union_area = pred_area + target_area - iou * target_area
    
    giou = iou - (enclosing_area - union_area) / enclosing_area
    
    return 1 - giou.mean()