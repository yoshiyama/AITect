import torch
import torch.nn as nn
import torchvision.models as models

class AITECTDetectorV2(nn.Module):
    def __init__(self, num_classes=1, grid_size=13, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size  # 13x13 (YOLOv2と同じ)
        self.num_anchors = num_anchors  # 3つのアンカーボックス
        self.image_size = 512
        
        # 軽量CNNバックボーン（ResNet18ベース）
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # ダウンサンプリングを調整（512 -> 13のためにはstride調整が必要）
        # 512 / 13 ≈ 39.4 なので、追加のプーリングが必要
        self.adapt_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        
        # 検出Head：各グリッドセルが予測する値
        # num_anchors * 5 = アンカーごとの予測 (x, y, w, h, objectness)
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
        
        # アンカーボックスの初期化（電柱検出に最適化）
        # 電柱は縦長なので、アスペクト比を考慮
        self._init_anchors()
        
    def _make_grid(self):
        """グリッド座標を事前計算"""
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.grid_size),
            torch.arange(self.grid_size),
            indexing='ij'
        )
        self.register_buffer('grid_x', grid_x.unsqueeze(0).unsqueeze(0).float())
        self.register_buffer('grid_y', grid_y.unsqueeze(0).unsqueeze(0).float())
    
    def _init_anchors(self):
        """アンカーボックスの初期化（電柱用に最適化）"""
        # アンカーサイズ（グリッドセルに対する比率）
        # 電柱は細くて縦長なので、異なるスケールとアスペクト比を用意
        anchor_sizes = torch.tensor([
            [1.0, 2.5],   # 小さい電柱（幅1.0、高さ2.5グリッドセル）
            [1.5, 3.5],   # 中程度の電柱
            [2.0, 5.0],   # 大きい電柱
        ])
        
        self.register_buffer('anchor_w', anchor_sizes[:, 0].view(1, self.num_anchors, 1, 1))
        self.register_buffer('anchor_h', anchor_sizes[:, 1].view(1, self.num_anchors, 1, 1))
    
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
        features = self.backbone(x)  # [B, 512, H', W']
        
        # グリッドサイズに調整
        features = self.adapt_pool(features)  # [B, 512, 13, 13]
        
        # 検出ヘッド
        raw_pred = self.head(features)  # [B, num_anchors*5, 13, 13]
        
        # 形状変更: [B, num_anchors, 5, grid_size, grid_size]
        raw_pred = raw_pred.view(batch_size, self.num_anchors, 5, self.grid_size, self.grid_size)
        
        # 予測値を分解
        # sigmoidを適用してセル内相対位置(0-1)に
        x_offset = torch.sigmoid(raw_pred[:, :, 0, :, :])
        y_offset = torch.sigmoid(raw_pred[:, :, 1, :, :])
        
        # 幅と高さはアンカーに対するスケール
        w_scale = torch.exp(raw_pred[:, :, 2, :, :].clamp(max=5))
        h_scale = torch.exp(raw_pred[:, :, 3, :, :].clamp(max=5))
        
        # オブジェクトネススコア
        objectness = raw_pred[:, :, 4, :, :]
        
        # グリッド座標を加算して画像座標に変換
        cell_size = self.image_size / self.grid_size
        
        # 中心座標の計算
        x_center = (x_offset + self.grid_x) * cell_size
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
    
    def get_anchor_boxes(self):
        """アンカーボックスのサイズを返す（デバッグ用）"""
        cell_size = self.image_size / self.grid_size
        anchor_w_pixels = self.anchor_w.squeeze() * cell_size
        anchor_h_pixels = self.anchor_h.squeeze() * cell_size
        return torch.stack([anchor_w_pixels, anchor_h_pixels], dim=1)