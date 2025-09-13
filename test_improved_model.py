import torch
from model_whiteline import WhiteLineDetector
import matplotlib.pyplot as plt
import numpy as np

def test_improvements():
    """改善の効果をテスト"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 現在のモデル（改善前）
    model_old = WhiteLineDetector(grid_size=8, num_anchors=3).to(device)
    model_old.load_state_dict(torch.load("result/aitect_model.pth", map_location=device))
    model_old.eval()
    
    # 2. ダミー入力でテスト
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    
    with torch.no_grad():
        pred_old = model_old(dummy_input)[0]
        scores_old = torch.sigmoid(pred_old[:, 4])
    
    print("=== 現在のモデル（改善前）===")
    print(f"最大スコア: {scores_old.max().item():.4f}")
    print(f"平均スコア: {scores_old.mean().item():.4f}")
    print(f"高信頼度(>0.5): {(scores_old > 0.5).sum().item()}/192")
    print(f"中信頼度(>0.3): {(scores_old > 0.3).sum().item()}/192")
    print(f"低信頼度(>0.1): {(scores_old > 0.1).sum().item()}/192")
    
    # スコア分布のヒストグラム
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(scores_old.cpu().numpy(), bins=50, alpha=0.7)
    plt.title('改善前のスコア分布')
    plt.xlabel('予測スコア')
    plt.ylabel('頻度')
    plt.xlim(0, 1)
    
    # 改善後のモデルがあればそれも表示
    try:
        model_new = WhiteLineDetector(grid_size=8, num_anchors=3).to(device)
        model_new.load_state_dict(torch.load("result/aitect_model_improved.pth", map_location=device))
        model_new.eval()
        
        with torch.no_grad():
            pred_new = model_new(dummy_input)[0]
            scores_new = torch.sigmoid(pred_new[:, 4])
        
        print("\n=== 改善後のモデル ===")
        print(f"最大スコア: {scores_new.max().item():.4f}")
        print(f"平均スコア: {scores_new.mean().item():.4f}")
        print(f"高信頼度(>0.5): {(scores_new > 0.5).sum().item()}/192")
        print(f"中信頼度(>0.3): {(scores_new > 0.3).sum().item()}/192")
        print(f"低信頼度(>0.1): {(scores_new > 0.1).sum().item()}/192")
        
        plt.subplot(1, 2, 2)
        plt.hist(scores_new.cpu().numpy(), bins=50, alpha=0.7, color='green')
        plt.title('改善後のスコア分布')
        plt.xlabel('予測スコア')
        plt.ylabel('頻度')
        plt.xlim(0, 1)
        
    except:
        print("\n改善後のモデルはまだ作成されていません。")
        print("以下のコマンドで学習を開始してください：")
        print("python train_with_improvements.py --epochs 30")
    
    plt.tight_layout()
    plt.savefig('score_distribution_comparison.png')
    print("\nスコア分布を score_distribution_comparison.png に保存しました")

if __name__ == "__main__":
    test_improvements()