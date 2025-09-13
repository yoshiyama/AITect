import torch
import torch.nn as nn
import torchvision.models as models
from model_v2 import AITECTDetectorV2
from model_whiteline import WhiteLineDetector

# デフォルトは白線検出モデル
AITECTDetector = WhiteLineDetector

# 旧モデルを保持（互換性のため）
class AITECTDetectorV1(nn.Module):
    def __init__(self, num_classes=1, grid_size=16, num_anchors=1):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.image_size = 512  # 固定画像サイズ
        
        # 軽量CNNバックボーン（ResNet18ベース）
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 最後のavgpoolとfcは除く
        
        # 検出Head：各グリッドセルが予測する値
        # 5 = x_offset, y_offset, w, h, objectness
        # (num_classes は1なので、objectnessのみで十分)
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_anchors * 5, kernel_size=1)
        )
        
        # グリッド座標の事前計算
        self._make_grid()
        
    def _make_grid(self):
        """グリッド座標を事前計算"""
        # グリッドのx, y座標を生成
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.grid_size),
            torch.arange(self.grid_size),
            indexing='ij'
        )
        # [1, 1, grid_size, grid_size]の形状に
        self.register_buffer('grid_x', grid_x.unsqueeze(0).unsqueeze(0).float())
        self.register_buffer('grid_y', grid_y.unsqueeze(0).unsqueeze(0).float())
        
        # アンカーのスケール（グリッドセルサイズに対する倍率）
        self.register_buffer('anchor_w', torch.ones(1, self.num_anchors, 1, 1))
        self.register_buffer('anchor_h', torch.ones(1, self.num_anchors, 1, 1))
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 入力画像
        
        Returns:
            predictions: [B, N, 5] デコード済みの予測
                - N = grid_size * grid_size * num_anchors
                - 5 = [x_center, y_center, width, height, objectness]
                - 座標は画像座標系（0-512）
        """
        batch_size = x.size(0)
        
        # バックボーン特徴抽出
        features = self.backbone(x)  # [B, 512, grid_size, grid_size]
        
        # 検出ヘッド
        raw_pred = self.head(features)  # [B, num_anchors*5, grid_size, grid_size]
        
        # 形状変更: [B, num_anchors, 5, grid_size, grid_size]
        raw_pred = raw_pred.view(batch_size, self.num_anchors, 5, self.grid_size, self.grid_size)
        
        # 予測値を分解
        # sigmoidを適用してセル内相対位置(0-1)に
        x_offset = torch.sigmoid(raw_pred[:, :, 0, :, :])  # [B, num_anchors, H, W]
        y_offset = torch.sigmoid(raw_pred[:, :, 1, :, :])
        
        # 幅と高さは exp で正の値に（グリッドセルサイズに対する倍率）
        w_scale = torch.exp(raw_pred[:, :, 2, :, :].clamp(max=10))  # 爆発を防ぐ
        h_scale = torch.exp(raw_pred[:, :, 3, :, :].clamp(max=10))
        
        # オブジェクトネススコア（sigmoidは後で適用）
        objectness = raw_pred[:, :, 4, :, :]
        
        # グリッド座標を加算して画像座標に変換
        cell_size = self.image_size / self.grid_size
        
        # 中心座標の計算
        x_center = (x_offset + self.grid_x) * cell_size  # [B, num_anchors, H, W]
        y_center = (y_offset + self.grid_y) * cell_size
        
        # 幅と高さの計算（アンカーサイズ × スケール × セルサイズ）
        width = w_scale * self.anchor_w * cell_size
        height = h_scale * self.anchor_h * cell_size
        
        # 形状を [B, N, 5] に変更
        N = self.grid_size * self.grid_size * self.num_anchors
        x_center = x_center.permute(0, 1, 2, 3).contiguous().view(batch_size, N)
        y_center = y_center.permute(0, 1, 2, 3).contiguous().view(batch_size, N)
        width = width.permute(0, 1, 2, 3).contiguous().view(batch_size, N)
        height = height.permute(0, 1, 2, 3).contiguous().view(batch_size, N)
        objectness = objectness.permute(0, 1, 2, 3).contiguous().view(batch_size, N)
        
        # 予測を結合
        predictions = torch.stack([x_center, y_center, width, height, objectness], dim=2)
        
        return predictions
    
    def decode_output(self, predictions):
        """
        モデル出力をバウンディングボックスにデコード（互換性のため残す）
        新しいforwardメソッドでは既にデコード済みなので、そのまま返す
        """
        return predictions