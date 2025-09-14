import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction"""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # Lateral connections (1x1 conv)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        
        # Output convolutions (3x3 conv)
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
    
    def forward(self, features):
        """
        Args:
            features: list of feature maps from backbone (low to high level)
        Returns:
            list of FPN feature maps
        """
        # Build laterals
        laterals = []
        for i, feature in enumerate(features):
            lateral = self.lateral_convs[i](feature)
            laterals.append(lateral)
        
        # Build top-down path
        fpn_features = []
        for i in range(len(laterals) - 1, -1, -1):
            if i == len(laterals) - 1:
                # Highest level
                fpn_features.append(self.fpn_convs[i](laterals[i]))
            else:
                # Upsample and add
                upsampled = F.interpolate(
                    fpn_features[-1], 
                    size=laterals[i].shape[-2:], 
                    mode='nearest'
                )
                merged = laterals[i] + upsampled
                fpn_features.append(self.fpn_convs[i](merged))
        
        # Reverse to get low-to-high order
        fpn_features = fpn_features[::-1]
        
        return fpn_features


class ImprovedDetector(nn.Module):
    """Improved object detector with FPN and multi-scale predictions"""
    def __init__(self, num_classes=1, num_anchors=9, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.image_size = 512
        
        # Backbone: ResNet50 for better feature extraction
        resnet = models.resnet50(pretrained=pretrained)
        
        # Extract feature maps at different scales
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # /4
        self.layer2 = resnet.layer2  # /8
        self.layer3 = resnet.layer3  # /16
        self.layer4 = resnet.layer4  # /32
        
        # Feature channels at each level
        feature_channels = [256, 512, 1024, 2048]
        
        # FPN
        self.fpn = FPN(feature_channels, 256)
        
        # Detection heads for each FPN level
        self.detection_heads = nn.ModuleList()
        for _ in range(4):  # 4 FPN levels
            self.detection_heads.append(self._make_detection_head())
        
        # Initialize anchors
        self._init_anchors()
        
    def _make_detection_head(self):
        """Create detection head for one FPN level"""
        # 5 = x, y, w, h, objectness
        # num_classes for classification (if > 1)
        output_channels = self.num_anchors * (5 + self.num_classes)
        
        return nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, output_channels, kernel_size=1)
        )
    
    def _init_anchors(self):
        """Initialize multi-scale, multi-aspect ratio anchors"""
        # Ensure num_anchors = len(scales) * len(ratios)
        if self.num_anchors == 9:
            # 3 scales x 3 ratios
            scales_per_level = [1.0, 1.25, 1.58]
            aspect_ratios = [0.5, 1.0, 2.0]
        elif self.num_anchors == 3:
            # 1 scale x 3 ratios (simpler)
            scales_per_level = [1.0]
            aspect_ratios = [0.5, 1.0, 2.0]
        else:
            # Default: single scale, multiple ratios
            scales_per_level = [1.0]
            aspect_ratios = [1.0] * self.num_anchors
        
        # Anchor scales for each FPN level
        self.anchor_scales = [
            [s * 1.0 for s in scales_per_level],   # P2 (/4) - small anchors
            [s * 2.0 for s in scales_per_level],   # P3 (/8)
            [s * 4.0 for s in scales_per_level],   # P4 (/16)
            [s * 8.0 for s in scales_per_level]    # P5 (/32) - large anchors
        ]
        
        self.aspect_ratios = aspect_ratios
        
        # Generate anchors for each level
        for level_idx, level_scales in enumerate(self.anchor_scales):
            level_anchors = []
            for scale in level_scales:
                for ratio in self.aspect_ratios:
                    w = scale * (ratio ** 0.5)
                    h = scale / (ratio ** 0.5)
                    level_anchors.append([w, h])
            
            # Ensure we have the correct number of anchors
            assert len(level_anchors) == self.num_anchors, \
                f"Expected {self.num_anchors} anchors, got {len(level_anchors)}"
            
            # Convert to tensor and register as buffer
            level_anchors = torch.tensor(level_anchors).float()
            self.register_buffer(f'anchors_level_{level_idx}', level_anchors)
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input images
        Returns:
            predictions: list of predictions for each FPN level
                Each element: [B, N, 5 + num_classes] where N = H*W*num_anchors
        """
        batch_size = x.size(0)
        
        # Extract multi-scale features
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        # FPN
        features = [c1, c2, c3, c4]
        fpn_features = self.fpn(features)
        
        # Generate predictions at each level
        all_predictions = []
        
        for level_idx, (feature, head) in enumerate(zip(fpn_features, self.detection_heads)):
            # Get predictions
            raw_pred = head(feature)  # [B, num_anchors*(5+num_classes), H, W]
            
            H, W = feature.shape[-2:]
            
            # Reshape predictions
            pred = raw_pred.view(
                batch_size, 
                self.num_anchors, 
                5 + self.num_classes,
                H, 
                W
            )
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()  # [B, num_anchors, H, W, 5+num_classes]
            
            # Decode predictions
            decoded_pred = self._decode_predictions(pred, level_idx)
            
            # Flatten spatial dimensions
            decoded_pred = decoded_pred.view(batch_size, -1, 5 + self.num_classes)
            
            all_predictions.append(decoded_pred)
        
        # Concatenate predictions from all levels
        predictions = torch.cat(all_predictions, dim=1)  # [B, total_anchors, 5+num_classes]
        
        return predictions
    
    def _decode_predictions(self, pred, level_idx):
        """Decode predictions to absolute coordinates"""
        batch_size, num_anchors, H, W, _ = pred.shape
        
        # Get stride for this level
        stride = 2 ** (level_idx + 2)  # 4, 8, 16, 32
        
        # Generate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=pred.device),
            torch.arange(W, device=pred.device),
            indexing='ij'
        )
        grid_x = grid_x.float()
        grid_y = grid_y.float()
        
        # Get anchors for this level
        anchors = getattr(self, f'anchors_level_{level_idx}')  # [num_anchors, 2]
        anchor_w = anchors[:, 0].view(1, num_anchors, 1, 1)
        anchor_h = anchors[:, 1].view(1, num_anchors, 1, 1)
        
        # Decode center coordinates
        x_center = (torch.sigmoid(pred[..., 0]) + grid_x.unsqueeze(0).unsqueeze(0)) * stride
        y_center = (torch.sigmoid(pred[..., 1]) + grid_y.unsqueeze(0).unsqueeze(0)) * stride
        
        # Decode width and height
        width = torch.exp(pred[..., 2].clamp(max=10)) * anchor_w * stride
        height = torch.exp(pred[..., 3].clamp(max=10)) * anchor_h * stride
        
        # Objectness (raw logits)
        objectness = pred[..., 4]
        
        # Class predictions (if multi-class)
        if self.num_classes > 1:
            class_pred = pred[..., 5:]
        else:
            class_pred = torch.zeros_like(pred[..., 5:])
        
        # Stack decoded predictions
        decoded = torch.stack([
            x_center, y_center, width, height, objectness
        ], dim=-1)
        
        if self.num_classes > 1:
            decoded = torch.cat([decoded, class_pred], dim=-1)
        
        return decoded
    
    def get_anchor_boxes(self):
        """Get all anchor boxes for visualization/debugging"""
        all_anchors = []
        
        for level_idx in range(4):
            stride = 2 ** (level_idx + 2)
            H = W = self.image_size // stride
            
            anchors = getattr(self, f'anchors_level_{level_idx}')
            
            # Generate all anchor boxes
            for y in range(H):
                for x in range(W):
                    cx = (x + 0.5) * stride
                    cy = (y + 0.5) * stride
                    
                    for anchor_w, anchor_h in anchors:
                        w = anchor_w * stride
                        h = anchor_h * stride
                        all_anchors.append([cx, cy, w, h])
        
        return torch.tensor(all_anchors)