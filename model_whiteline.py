import torch
import torch.nn as nn
import torchvision.models as models

class WhiteLineDetector(nn.Module):
    """白線検出に最適化されたシンプルなモデル"""
    def __init__(self, num_classes=1, grid_size=10, num_anchors=1, auto_anchors=None):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size  # 10x10 = 100予測（シンプル）
        self.num_anchors = num_anchors  # 1アンカーで十分
        self.image_size = 512
        self.auto_anchors = auto_anchors  # 自動設定されたアンカー
        
        # 軽量CNNバックボーン（ResNet18ベース）
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # グリッドサイズに調整
        self.adapt_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        
        # シンプルな検出ヘッド
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_anchors * 5, kernel_size=1)
        )
        
        # グリッド座標の事前計算
        self._make_grid()
        
        # アンカーボックスの初期化
        if auto_anchors is not None:
            self._init_anchors_auto(auto_anchors)
        else:
            self._init_anchors_whiteline()
        
    def _make_grid(self):
        """グリッド座標を事前計算"""
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.grid_size),
            torch.arange(self.grid_size),
            indexing='ij'
        )
        self.register_buffer('grid_x', grid_x.unsqueeze(0).unsqueeze(0).float())
        self.register_buffer('grid_y', grid_y.unsqueeze(0).unsqueeze(0).float())
    
    def _init_anchors_whiteline(self):
        """白線検出用のアンカーボックス"""
        # 実データ分析に基づく設定
        # 中央値: 幅60.0px、高さ39.3px、アスペクト比1.9
        # グリッドセル = 512/10 = 51.2px
        anchor_w = 1.17  # 60.0 / 51.2 = 1.17グリッドセル分
        anchor_h = 0.77  # 39.3 / 51.2 = 0.77グリッドセル分
        
        self.register_buffer('anchor_w', torch.tensor([anchor_w]).view(1, 1, 1, 1))
        self.register_buffer('anchor_h', torch.tensor([anchor_h]).view(1, 1, 1, 1))
    
    def _init_anchors_auto(self, auto_anchors):
        """自動設定されたアンカーボックス"""
        if self.num_anchors == 1:
            anchor_w = auto_anchors[0][0]
            anchor_h = auto_anchors[0][1]
            self.register_buffer('anchor_w', torch.tensor([anchor_w]).view(1, 1, 1, 1))
            self.register_buffer('anchor_h', torch.tensor([anchor_h]).view(1, 1, 1, 1))
        else:
            # 複数アンカーの場合（将来の拡張用）
            anchor_ws = [a[0] for a in auto_anchors]
            anchor_hs = [a[1] for a in auto_anchors]
            self.register_buffer('anchor_w', torch.tensor(anchor_ws).view(1, -1, 1, 1))
            self.register_buffer('anchor_h', torch.tensor(anchor_hs).view(1, -1, 1, 1))
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 入力画像
        
        Returns:
            predictions: [B, N, 5] デコード済みの予測
                - N = grid_size * grid_size * num_anchors (100)
                - 5 = [x_center, y_center, width, height, objectness]
                - 座標は画像座標系（0-512）
        """
        batch_size = x.size(0)
        
        # バックボーン特徴抽出
        features = self.backbone(x)  # [B, 512, H', W']
        
        # グリッドサイズに調整
        features = self.adapt_pool(features)  # [B, 512, 10, 10]
        
        # 検出ヘッド
        raw_pred = self.head(features)  # [B, 5, 10, 10]
        
        # 形状変更: [B, num_anchors, 5, grid_size, grid_size]
        raw_pred = raw_pred.view(batch_size, self.num_anchors, 5, self.grid_size, self.grid_size)
        
        # 予測値を分解
        # sigmoidを適用してセル内相対位置(0-1)に
        x_offset = torch.sigmoid(raw_pred[:, :, 0, :, :])
        y_offset = torch.sigmoid(raw_pred[:, :, 1, :, :])
        
        # 幅と高さのスケール（制限付き）
        w_scale = torch.exp(raw_pred[:, :, 2, :, :].clamp(min=-1, max=1))
        h_scale = torch.exp(raw_pred[:, :, 3, :, :].clamp(min=-1, max=1))
        
        # オブジェクトネススコア（生の値）
        objectness = raw_pred[:, :, 4, :, :]
        
        # グリッド座標を加算して画像座標に変換
        cell_size = self.image_size / self.grid_size  # 51.2ピクセル/セル
        
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
    
    def get_anchor_info(self):
        """アンカーボックスの情報を返す（デバッグ用）"""
        cell_size = self.image_size / self.grid_size
        anchor_w_pixels = self.anchor_w.item() * cell_size
        anchor_h_pixels = self.anchor_h.item() * cell_size
        return {
            'anchor_width': anchor_w_pixels,
            'anchor_height': anchor_h_pixels,
            'aspect_ratio': anchor_w_pixels / anchor_h_pixels,
            'total_predictions': self.grid_size * self.grid_size * self.num_anchors
        }