"""グリッドベース検出のデコード関数"""

import torch

def decode_predictions(predictions, image_size=512, grid_size=16):
    """
    グリッドベースの予測を画像座標に変換
    
    Args:
        predictions: [B, N, 5] or [N, 5] - モデルの生の出力
        image_size: 入力画像のサイズ（正方形を仮定）
        grid_size: グリッドの一辺のセル数
    
    Returns:
        decoded_preds: 同じ形状、但し座標は画像座標系に変換済み
    """
    if predictions.dim() == 2:
        # [N, 5]の場合
        predictions = predictions.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, N, _ = predictions.shape
    assert N == grid_size * grid_size, f"予測数 {N} がグリッドサイズ {grid_size}x{grid_size} と一致しません"
    
    # グリッドのインデックスを生成
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_size, device=predictions.device),
        torch.arange(grid_size, device=predictions.device),
        indexing='ij'
    )
    grid_x = grid_x.reshape(-1)  # [N]
    grid_y = grid_y.reshape(-1)  # [N]
    
    # セルサイズ
    cell_size = image_size / grid_size
    
    # 予測を分解
    pred_x = predictions[:, :, 0]  # [B, N]
    pred_y = predictions[:, :, 1]  # [B, N]
    pred_w = predictions[:, :, 2]  # [B, N]
    pred_h = predictions[:, :, 3]  # [B, N]
    pred_conf = predictions[:, :, 4]  # [B, N]
    
    # 座標変換
    # x, yはセル内の相対位置（0-1）として扱い、グリッド位置を加算
    # sigmoid を適用して 0-1 の範囲に制限
    pred_x = (torch.sigmoid(pred_x) + grid_x.unsqueeze(0)) * cell_size
    pred_y = (torch.sigmoid(pred_y) + grid_y.unsqueeze(0)) * cell_size
    
    # 幅と高さは画像サイズに対する比率として扱う
    # exp を適用して正の値にし、適切にスケール
    pred_w = torch.exp(pred_w) * cell_size
    pred_h = torch.exp(pred_h) * cell_size
    
    # 予測を再構成
    decoded_preds = torch.stack([pred_x, pred_y, pred_w, pred_h, pred_conf], dim=2)
    
    if squeeze_output:
        decoded_preds = decoded_preds.squeeze(0)
    
    return decoded_preds


def decode_predictions_xyxy(predictions, image_size=512, grid_size=16):
    """
    グリッドベースの予測を画像座標（xyxy形式）に変換
    """
    decoded = decode_predictions(predictions, image_size, grid_size)
    
    # xywh から xyxy に変換
    if decoded.dim() == 2:
        # [N, 5]
        x_center = decoded[:, 0]
        y_center = decoded[:, 1]
        width = decoded[:, 2]
        height = decoded[:, 3]
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # 画像境界でクリップ
        x1 = torch.clamp(x1, 0, image_size)
        y1 = torch.clamp(y1, 0, image_size)
        x2 = torch.clamp(x2, 0, image_size)
        y2 = torch.clamp(y2, 0, image_size)
        
        decoded[:, 0] = x1
        decoded[:, 1] = y1
        decoded[:, 2] = x2
        decoded[:, 3] = y2
    else:
        # [B, N, 5]
        x_center = decoded[:, :, 0]
        y_center = decoded[:, :, 1]
        width = decoded[:, :, 2]
        height = decoded[:, :, 3]
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # 画像境界でクリップ
        x1 = torch.clamp(x1, 0, image_size)
        y1 = torch.clamp(y1, 0, image_size)
        x2 = torch.clamp(x2, 0, image_size)
        y2 = torch.clamp(y2, 0, image_size)
        
        decoded[:, :, 0] = x1
        decoded[:, :, 1] = y1
        decoded[:, :, 2] = x2
        decoded[:, :, 3] = y2
    
    return decoded