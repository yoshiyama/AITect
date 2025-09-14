import torch
import torch.nn as nn
import torchvision.models as models
import math

class LightweightBackbone(nn.Module):
    """軽量バックボーン（MobileNetV2 or ResNet18）"""
    
    def __init__(self, model_type="mobilenet_v2", pretrained=True):
        super().__init__()
        
        if model_type == "mobilenet_v2":
            # MobileNetV2 - 非常に軽量（~3.5M params）
            backbone = models.mobilenet_v2(pretrained=pretrained)
            # 特徴抽出層を取得
            self.features = backbone.features
            self.out_channels = [32, 96, 320]  # 各段階の出力チャンネル数
            
        elif model_type == "resnet18":
            # ResNet18 - 軽量（~11M params）
            backbone = models.resnet18(pretrained=pretrained)
            # 各段階の特徴を抽出
            self.conv1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
            self.out_channels = [64, 128, 256, 512]
            
        elif model_type == "shufflenet":
            # ShuffleNet - 超軽量（~2.3M params）
            backbone = models.shufflenet_v2_x1_0(pretrained=pretrained)
            self.features = backbone.conv1
            self.stage2 = backbone.stage2
            self.stage3 = backbone.stage3
            self.stage4 = backbone.stage4
            self.out_channels = [24, 116, 232, 464]
            
        self.model_type = model_type
    
    def forward(self, x):
        features = []
        
        if self.model_type == "mobilenet_v2":
            # MobileNetV2の特徴抽出
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in [3, 6, 13, 18]:  # 特定の層で特徴を保存
                    features.append(x)
            return features[-3:]  # 最後の3つの特徴マップ
            
        elif self.model_type == "resnet18":
            # ResNet18の特徴抽出
            x = self.conv1(x)
            x = self.layer1(x)
            features.append(x)
            x = self.layer2(x)
            features.append(x)
            x = self.layer3(x)
            features.append(x)
            x = self.layer4(x)
            features.append(x)
            return features[-3:]  # 最後の3つの特徴マップ
            
        elif self.model_type == "shufflenet":
            # ShuffleNetの特徴抽出
            x = self.features(x)
            x = self.stage2(x)
            features.append(x)
            x = self.stage3(x)
            features.append(x)
            x = self.stage4(x)
            features.append(x)
            return features

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution（軽量化）"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class LightweightDetectionHead(nn.Module):
    """軽量検出ヘッド"""
    
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        
        # Depthwise Separable Convolutionで軽量化
        self.conv1 = DepthwiseSeparableConv(in_channels, in_channels // 2)
        self.conv2 = DepthwiseSeparableConv(in_channels // 2, in_channels // 2)
        
        # 出力層（バウンディングボックス + 信頼度 + クラス）
        self.output = nn.Conv2d(
            in_channels // 2,
            num_anchors * (5 + num_classes),  # x, y, w, h, objectness, classes
            1
        )
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.output(x)
        
        # 出力を整形
        batch_size = x.size(0)
        grid_size = x.size(2)
        
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, grid_size, grid_size, self.num_anchors, 5 + self.num_classes)
        
        return x

class AITectLightweight(nn.Module):
    """軽量物体検出モデル"""
    
    def __init__(self, num_classes=1, model_size="tiny", pretrained_backbone=True, backbone_weights_path=None):
        super().__init__()
        
        # モデルサイズ設定
        configs = {
            "tiny": {
                "backbone": "mobilenet_v2",
                "head_channels": [128, 256, 512],
                "grid_sizes": [52, 26, 13],
                "num_anchors": 3
            },
            "small": {
                "backbone": "resnet18",
                "head_channels": [256, 512, 1024],
                "grid_sizes": [52, 26, 13],
                "num_anchors": 3
            },
            "medium": {
                "backbone": "shufflenet",
                "head_channels": [256, 512, 1024],
                "grid_sizes": [52, 26, 13],
                "num_anchors": 3
            }
        }
        
        config = configs[model_size]
        self.model_size = model_size
        self.num_classes = num_classes
        
        # バックボーン
        self.backbone = LightweightBackbone(
            model_type=config["backbone"],
            pretrained=pretrained_backbone
        )
        
        # FPN風の特徴統合（簡易版）
        backbone_channels = self.backbone.out_channels[-3:]
        self.fpn_convs = nn.ModuleList()
        
        for i, (in_ch, out_ch) in enumerate(zip(backbone_channels, config["head_channels"])):
            self.fpn_convs.append(
                DepthwiseSeparableConv(in_ch, out_ch)
            )
        
        # 検出ヘッド（マルチスケール）
        self.detection_heads = nn.ModuleList()
        for out_ch in config["head_channels"]:
            self.detection_heads.append(
                LightweightDetectionHead(out_ch, num_classes, config["num_anchors"])
            )
        
        # アンカーボックス
        self.anchors = self._generate_anchors(config["grid_sizes"], config["num_anchors"])
        
        # グリッド
        self.register_buffer('grid_x', torch.zeros(1))
        self.register_buffer('grid_y', torch.zeros(1))
        
        # パラメータ数を計算
        self._count_parameters()
        
        # Load backbone weights if provided
        if backbone_weights_path:
            self.load_backbone_weights(backbone_weights_path)
    
    def _generate_anchors(self, grid_sizes, num_anchors):
        """アンカーボックスの生成"""
        # COCO用の典型的なアンカーサイズ
        if num_anchors == 3:
            # 小、中、大のアンカー
            anchor_sizes = [
                [(10, 13), (16, 30), (33, 23)],      # 小物体用
                [(30, 61), (62, 45), (59, 119)],     # 中物体用
                [(116, 90), (156, 198), (373, 326)]  # 大物体用
            ]
        else:
            # デフォルトアンカー
            anchor_sizes = [[(10, 10)] * num_anchors] * 3
        
        return anchor_sizes
    
    def _count_parameters(self):
        """パラメータ数をカウント"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n=== Model: AITect-{self.model_size.upper()} ===")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    
    def forward(self, x):
        # バックボーン特徴抽出
        features = self.backbone(x)
        
        # FPN処理と検出
        outputs = []
        for i, (feat, fpn_conv, det_head) in enumerate(
            zip(features, self.fpn_convs, self.detection_heads)
        ):
            # FPN変換
            feat = fpn_conv(feat)
            
            # 検出ヘッド
            output = det_head(feat)
            outputs.append(output)
        
        return outputs
    
    def predict(self, x, conf_threshold=0.5):
        """推論用メソッド"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            # 各スケールの出力を処理
            all_predictions = []
            
            for output, anchors in zip(outputs, self.anchors):
                batch_size = output.size(0)
                grid_size = output.size(1)
                
                # グリッド作成
                if self.grid_x.size(2) != grid_size:
                    self.grid_x = torch.arange(grid_size).repeat(grid_size, 1).view(
                        1, grid_size, grid_size, 1, 1
                    ).float().to(output.device)
                    self.grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view(
                        1, grid_size, grid_size, 1, 1
                    ).float().to(output.device)
                
                # 予測値を変換
                pred_boxes = output[..., :4]
                pred_conf = torch.sigmoid(output[..., 4:5])
                pred_cls = torch.sigmoid(output[..., 5:])
                
                # バウンディングボックスをデコード
                pred_boxes[..., 0:2] = torch.sigmoid(pred_boxes[..., 0:2]) + self.grid_x
                pred_boxes[..., 2:4] = torch.exp(pred_boxes[..., 2:4])
                
                # 信頼度でフィルタリング
                mask = pred_conf > conf_threshold
                
                for b in range(batch_size):
                    masked_boxes = pred_boxes[b][mask[b, ..., 0]]
                    masked_conf = pred_conf[b][mask[b, ..., 0]]
                    masked_cls = pred_cls[b][mask[b, ..., 0]]
                    
                    if len(masked_boxes) > 0:
                        predictions = {
                            'boxes': masked_boxes.view(-1, 4),
                            'scores': masked_conf.view(-1),
                            'labels': masked_cls.argmax(dim=-1).view(-1) if self.num_classes > 1 else torch.zeros_like(masked_conf.view(-1))
                        }
                        all_predictions.append(predictions)
            
            return all_predictions
    
    def load_backbone_weights(self, weights_path):
        """Load only backbone weights from a checkpoint."""
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Extract backbone weights
        if 'backbone_state_dict' in checkpoint:
            backbone_state = checkpoint['backbone_state_dict']
        else:
            # Extract from full model state
            backbone_state = {}
            full_state = checkpoint.get('model_state_dict', checkpoint)
            for name, param in full_state.items():
                if 'backbone' in name:
                    backbone_state[name] = param
        
        # Load weights
        current_state = self.state_dict()
        matched_weights = {}
        
        for name, param in backbone_state.items():
            if name in current_state:
                if current_state[name].shape == param.shape:
                    matched_weights[name] = param
                else:
                    print(f"Warning: Shape mismatch for {name}")
        
        # Update model state
        current_state.update(matched_weights)
        self.load_state_dict(current_state, strict=False)
        
        print(f"Loaded {len(matched_weights)} backbone parameters from {weights_path}")
    
    def get_param_groups(self, backbone_lr=1e-5, fpn_lr=1e-4, head_lr=1e-3):
        """Get parameter groups with different learning rates for optimizer."""
        backbone_params = []
        fpn_params = []
        head_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'backbone' in name:
                backbone_params.append(param)
            elif 'fpn_convs' in name:
                fpn_params.append(param)
            elif 'detection_heads' in name:
                head_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = []
        
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': backbone_lr,
                'name': 'backbone'
            })
        
        if fpn_params:
            param_groups.append({
                'params': fpn_params,
                'lr': fpn_lr,
                'name': 'fpn'
            })
        
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': head_lr,
                'name': 'detection_heads'
            })
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': head_lr,
                'name': 'other'
            })
        
        return param_groups
    
    def freeze_backbone(self, freeze=True):
        """Freeze or unfreeze backbone parameters."""
        for name, param in self.named_parameters():
            if 'backbone' in name:
                param.requires_grad = not freeze
    
    def freeze_fpn(self, freeze=True):
        """Freeze or unfreeze FPN parameters."""
        for name, param in self.named_parameters():
            if 'fpn_convs' in name:
                param.requires_grad = not freeze
    
    def replace_detection_heads(self, num_classes):
        """Replace detection heads for different number of classes."""
        config = {
            "tiny": {
                "head_channels": [128, 256, 512],
                "num_anchors": 3
            },
            "small": {
                "head_channels": [256, 512, 1024],
                "num_anchors": 3
            },
            "medium": {
                "head_channels": [256, 512, 1024],
                "num_anchors": 3
            }
        }[self.model_size]
        
        # Create new detection heads
        self.detection_heads = nn.ModuleList()
        for out_ch in config["head_channels"]:
            self.detection_heads.append(
                LightweightDetectionHead(out_ch, num_classes, config["num_anchors"])
            )
        
        self.num_classes = num_classes
        
        # Move to same device as model
        device = next(self.parameters()).device
        self.detection_heads = self.detection_heads.to(device)


def create_lightweight_model(num_classes=1, model_size="tiny", pretrained=True, backbone_weights_path=None):
    """軽量モデルを作成する便利関数"""
    return AITectLightweight(
        num_classes=num_classes,
        model_size=model_size,
        pretrained_backbone=pretrained,
        backbone_weights_path=backbone_weights_path
    )


if __name__ == "__main__":
    # モデルサイズの比較
    print("Lightweight Object Detection Models Comparison")
    print("="*60)
    
    for size in ["tiny", "small"]:
        model = create_lightweight_model(num_classes=1, model_size=size)
        
        # ダミー入力でテスト
        x = torch.randn(1, 3, 416, 416)
        outputs = model(x)
        
        print(f"\nOutput shapes for {size}:")
        for i, out in enumerate(outputs):
            print(f"  Scale {i+1}: {out.shape}")