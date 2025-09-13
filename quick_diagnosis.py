import torch
from model_whiteline import WhiteLineDetector
import json

def load_config(config_path="config.json"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def diagnose():
    """クイック診断"""
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 正しい設定でモデルを作成
    model = WhiteLineDetector(grid_size=8, num_anchors=3).to(device)
    
    # モデルを読み込む
    model_path = config['paths']['model_save_path']
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # ダミー入力でテスト
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"出力形状: {output.shape}")  # [1, 192, 5]が期待される
    
    # 予測を分析
    pred = output[0]  # [192, 5]
    pred_boxes = pred[:, :4]
    pred_scores = torch.sigmoid(pred[:, 4])
    
    print(f"\n予測統計:")
    print(f"スコア平均: {pred_scores.mean().item():.3f}")
    print(f"スコア最大: {pred_scores.max().item():.3f}")
    print(f"スコア最小: {pred_scores.min().item():.3f}")
    print(f"高信頼度(>0.5): {(pred_scores > 0.5).sum().item()}/{192}")
    
    print(f"\nボックス統計:")
    print(f"中心X平均: {pred_boxes[:, 0].mean().item():.1f}")
    print(f"中心Y平均: {pred_boxes[:, 1].mean().item():.1f}")
    print(f"幅平均: {pred_boxes[:, 2].mean().item():.1f}")
    print(f"高さ平均: {pred_boxes[:, 3].mean().item():.1f}")
    
    # 異常値チェック
    print(f"\n異常値チェック:")
    print(f"負の幅/高さ: {(pred_boxes[:, 2:] < 0).any().item()}")
    print(f"巨大なボックス(>500): {(pred_boxes[:, 2:] > 500).any().item()}")
    print(f"範囲外の中心(>512): {(pred_boxes[:, :2] > 512).any().item()}")

if __name__ == "__main__":
    diagnose()