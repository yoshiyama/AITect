import torch
import torch.nn as nn
import torchvision.models as models

class AITECTDetector(nn.Module):
    def __init__(self, backbone_out=512, num_anchors=100):
        super().__init__()
        # 軽量CNNバックボーン（ResNet18ベース）
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 最後のavgpoolとfcは除く

        # 検出Head：N個のアンカーポイントに対する回帰＋信頼度
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 5, kernel_size=1)  # [x, y, w, h, conf]
        )

    def forward(self, x):
        feat = self.backbone(x)        # [B, 512, H/32, W/32]
        out = self.head(feat)          # [B, 5, H/32, W/32]
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1).reshape(B, -1, 5)  # [B, N, 5]
        return out
