import torch
from model_improved_v2 import ImprovedDetector
import numpy as np

def check_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデル初期化
    model = ImprovedDetector(num_classes=1, num_anchors=3).to(device)
    
    # パラメータ数を確認
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 各レベルのアンカーを確認
    print("\n=== Anchor Configuration ===")
    for level in range(4):
        anchors = getattr(model, f'anchors_level_{level}')
        print(f"\nLevel {level} (stride {2**(level+2)}):")
        print(f"  Number of anchors: {len(anchors)}")
        for i, (w, h) in enumerate(anchors):
            print(f"  Anchor {i}: width={w:.2f}, height={h:.2f}, aspect_ratio={w/h:.2f}")
    
    # ダミー入力でフォワードパスをテスト
    print("\n=== Forward Pass Test ===")
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    
    with torch.no_grad():
        # 各レイヤーの出力を確認
        c1 = model.layer1(dummy_input)
        print(f"C1 shape: {c1.shape}")
        
        c2 = model.layer2(c1)
        print(f"C2 shape: {c2.shape}")
        
        c3 = model.layer3(c2)
        print(f"C3 shape: {c3.shape}")
        
        c4 = model.layer4(c3)
        print(f"C4 shape: {c4.shape}")
        
        # FPN出力
        fpn_features = model.fpn([c1, c2, c3, c4])
        print(f"\nFPN outputs:")
        for i, feat in enumerate(fpn_features):
            print(f"  Level {i}: {feat.shape}")
        
        # 最終出力
        output = model(dummy_input)
        print(f"\nFinal output shape: {output.shape}")
        
        # 各レベルの予測数
        print(f"\nPredictions per level:")
        stride_to_size = {4: 128, 8: 64, 16: 32, 32: 16}
        total_preds = 0
        for stride, size in stride_to_size.items():
            level_preds = size * size * 3  # 3 anchors per location
            total_preds += level_preds
            print(f"  Stride {stride}: {size}x{size} grid = {level_preds} predictions")
        print(f"Total predictions: {total_preds}")
    
    # NaNやInfをチェック
    print("\n=== Numerical Stability Check ===")
    with torch.no_grad():
        output = model(dummy_input)
        has_nan = torch.isnan(output).any()
        has_inf = torch.isinf(output).any()
        print(f"Contains NaN: {has_nan}")
        print(f"Contains Inf: {has_inf}")
        
        # 出力の統計
        print(f"\nOutput statistics:")
        print(f"  Min: {output.min().item():.4f}")
        print(f"  Max: {output.max().item():.4f}")
        print(f"  Mean: {output.mean().item():.4f}")
        print(f"  Std: {output.std().item():.4f}")

if __name__ == "__main__":
    check_model()