import torch
import torch.nn as nn
import torchvision.models as models

class AITECTDetectorV2Improved(nn.Module):
    def __init__(self, num_classes=1, grid_size=13, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.image_size = 512
        
        # 軽量CNNバックボーン（ResNet18ベース）
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # ダウンサンプリングを調整
        self.adapt_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        
        # 検出Head
        self.head = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, self.num_anchors * 5, kernel_size=1)
        )
        
        # グリッド座標の事前計算
        self._make_grid()
        
        # 改善されたアンカーボックス（より細い電柱に最適化）
        self._init_anchors_improved()
        
    def _make_grid(self):
        """グリッド座標を事前計算"""
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.grid_size),
            torch.arange(self.grid_size),
            indexing='ij'
        )
        self.register_buffer('grid_x', grid_x.unsqueeze(0).unsqueeze(0).float())
        self.register_buffer('grid_y', grid_y.unsqueeze(0).unsqueeze(0).float())
    
    def _init_anchors_improved(self):
        """改善されたアンカーボックス（より現実的な電柱サイズ）"""
        # アンカーサイズ（グリッドセルに対する比率）
        # より細い電柱に最適化
        anchor_sizes = torch.tensor([
            [0.5, 2.0],   # 細い電柱（幅0.5、高さ2.0グリッドセル）
            [0.7, 3.0],   # 中程度の電柱
            [1.0, 4.0],   # 太い電柱
        ])
        
        self.register_buffer('anchor_w', anchor_sizes[:, 0].view(1, self.num_anchors, 1, 1))
        self.register_buffer('anchor_h', anchor_sizes[:, 1].view(1, self.num_anchors, 1, 1))
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 入力画像
        
        Returns:
            predictions: [B, N, 5] デコード済みの予測
        """
        batch_size = x.size(0)
        
        # バックボーン特徴抽出
        features = self.backbone(x)
        
        # グリッドサイズに調整
        features = self.adapt_pool(features)
        
        # 検出ヘッド
        raw_pred = self.head(features)
        
        # 形状変更: [B, num_anchors, 5, grid_size, grid_size]
        raw_pred = raw_pred.view(batch_size, self.num_anchors, 5, self.grid_size, self.grid_size)
        
        # 予測値を分解
        x_offset = torch.sigmoid(raw_pred[:, :, 0, :, :])
        y_offset = torch.sigmoid(raw_pred[:, :, 1, :, :])
        
        # 幅と高さのスケール（より制限的な範囲）
        w_scale = torch.exp(raw_pred[:, :, 2, :, :].clamp(min=-2, max=2))
        h_scale = torch.exp(raw_pred[:, :, 3, :, :].clamp(min=-2, max=2))
        
        # オブジェクトネススコア
        objectness = raw_pred[:, :, 4, :, :]
        
        # グリッド座標を加算して画像座標に変換
        cell_size = self.image_size / self.grid_size
        
        # 中心座標の計算
        x_center = (x_offset + self.grid_x) * cell_size
        y_center = (y_offset + self.grid_y) * cell_size
        
        # 幅と高さの計算
        width = w_scale * self.anchor_w * cell_size
        height = h_scale * self.anchor_h * cell_size
        
        # 幅と高さを妥当な範囲に制限
        width = torch.clamp(width, min=10, max=150)   # 10-150ピクセル
        height = torch.clamp(height, min=30, max=300)  # 30-300ピクセル
        
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
    
    def get_anchor_boxes(self):
        """アンカーボックスのサイズを返す（デバッグ用）"""
        cell_size = self.image_size / self.grid_size
        anchor_w_pixels = self.anchor_w.squeeze() * cell_size
        anchor_h_pixels = self.anchor_h.squeeze() * cell_size
        return torch.stack([anchor_w_pixels, anchor_h_pixels], dim=1)