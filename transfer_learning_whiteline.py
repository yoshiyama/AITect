import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json

class PretrainedWhiteLineDetector(nn.Module):
    """
    事前学習済みモデルを使用した白線検出器
    """
    def __init__(self, backbone='resnet50', pretrained=True, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        
        if backbone == 'resnet50':
            # ResNet50 + FPN (Faster R-CNNスタイル)
            self.setup_resnet_fpn(pretrained)
        elif backbone == 'resnet18':
            # 軽量版：ResNet18
            self.setup_resnet18(pretrained)
        elif backbone == 'efficientnet':
            # EfficientNet (要追加ライブラリ)
            self.setup_efficientnet(pretrained)
        elif backbone == 'yolov5':
            # YOLOv5バックボーン
            self.setup_yolov5_backbone(pretrained)
    
    def setup_resnet_fpn(self, pretrained):
        """Faster R-CNNベースのセットアップ"""
        # 事前学習済みFaster R-CNNを読み込み
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # 分類器を白線検出用に置き換え（背景 + 白線 = 2クラス）
        num_classes = 2  # background + whiteline
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        self.model = model
    
    def setup_resnet18(self, pretrained):
        """軽量なResNet18ベースのセットアップ"""
        # ImageNetで事前学習済みのResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # 特徴抽出器として使用（最後の層を除く）
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Feature Pyramid Network (簡易版)
        self.fpn = SimpleFPN(in_channels=512, out_channels=256)
        
        # 検出ヘッド（YOLOスタイル）
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 5, kernel_size=1)  # 5 = x, y, w, h, conf
        )
        
        # グリッドサイズの調整
        self.grid_size = 16
        self.adapt_pool = nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size))
    
    def setup_yolov5_backbone(self, pretrained):
        """YOLOv5のバックボーンを使用"""
        # YOLOv5のCSPDarknet53バックボーン構造を模倣
        self.backbone = self._create_csp_darknet()
        
        if pretrained:
            # COCOで事前学習済みの重みを読み込む
            # 実際にはyolov5のcheckpointから部分的に読み込む必要がある
            print("注意: YOLOv5の事前学習重みは別途ダウンロードが必要です")
        
        # YOLOv5スタイルの検出ヘッド
        self.detect = YOLODetectionHead(num_classes=1)
    
    def _create_csp_darknet(self):
        """CSPDarknet53の簡易実装"""
        return nn.Sequential(
            # Focus層
            Focus(3, 64, k=3),
            # Stage 1
            Conv(64, 128, 3, 2),
            C3(128, 128, n=3),
            # Stage 2
            Conv(128, 256, 3, 2),
            C3(256, 256, n=9),
            # Stage 3
            Conv(256, 512, 3, 2),
            C3(512, 512, n=9),
            # Stage 4
            Conv(512, 1024, 3, 2),
            C3(1024, 1024, n=3),
        )
    
    def forward(self, x):
        if hasattr(self, 'model'):
            # Faster R-CNNスタイル
            return self.model(x)
        else:
            # カスタムモデル
            features = self.backbone(x)
            
            if hasattr(self, 'fpn'):
                features = self.fpn(features)
            
            if hasattr(self, 'adapt_pool'):
                features = self.adapt_pool(features)
            
            output = self.detection_head(features)
            
            # [B, 5, H, W] -> [B, H*W, 5]
            B, C, H, W = output.shape
            output = output.permute(0, 2, 3, 1).contiguous()
            output = output.view(B, H * W, C)
            
            return output


class SimpleFPN(nn.Module):
    """簡易的なFeature Pyramid Network"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lateral = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        lateral = self.lateral(x)
        smooth = self.smooth(lateral)
        return smooth


# YOLOv5スタイルのモジュール
class Conv(nn.Module):
    """標準的な畳み込み層"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p or k // 2, groups=g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    """YOLOv5のFocus層"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g)
    
    def forward(self, x):
        # Space-to-depth
        return self.conv(torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
    
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class Bottleneck(nn.Module):
    """標準的なボトルネック層"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class YOLODetectionHead(nn.Module):
    """YOLOスタイルの検出ヘッド"""
    def __init__(self, num_classes=1, anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.output_channels = anchors * (5 + num_classes)  # x,y,w,h,obj,cls
        
        self.conv = nn.Conv2d(1024, self.output_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


def load_pretrained_and_finetune():
    """
    事前学習モデルを読み込んで白線検出用に微調整
    """
    # 1. COCO事前学習済みモデルを読み込み
    model = PretrainedWhiteLineDetector(backbone='resnet18', pretrained=True)
    
    # 2. バックボーンの重みを固定（オプション）
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # 3. 検出ヘッドのみ学習可能に
    for param in model.detection_head.parameters():
        param.requires_grad = True
    
    return model


def create_transfer_config():
    """転移学習用の設定"""
    config = {
        "transfer_learning": {
            "backbone": "resnet18",  # resnet18, resnet50, efficientnet, yolov5
            "pretrained": True,
            "freeze_backbone": True,
            "freeze_epochs": 10,  # 最初の10エポックはバックボーン固定
            "unfreeze_learning_rate": 0.0001,  # 解凍後の学習率
        },
        "training": {
            "initial_epochs": 10,  # バックボーン固定で学習
            "finetune_epochs": 40,  # 全体を微調整
            "batch_size": 8,
            "learning_rate": 0.001,
            "warmup_epochs": 3,
        },
        "augmentation": {
            "use_mosaic": True,
            "use_mixup": True,
            "use_copy_paste": True,
        }
    }
    
    with open('config_transfer.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    return config


if __name__ == "__main__":
    # 転移学習モデルのテスト
    print("転移学習モデルを作成中...")
    
    # 設定作成
    config = create_transfer_config()
    print("転移学習設定を config_transfer.json に保存しました")
    
    # モデル作成
    model = load_pretrained_and_finetune()
    print(f"モデルを作成しました")
    
    # テスト入力
    dummy_input = torch.randn(1, 3, 512, 512)
    output = model(dummy_input)
    print(f"出力形状: {output.shape}")
    
    # パラメータ数の確認
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"総パラメータ数: {total_params:,}")
    print(f"学習可能パラメータ数: {trainable_params:,}")
    print(f"固定パラメータ数: {total_params - trainable_params:,}")