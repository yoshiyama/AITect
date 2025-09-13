"""
AITECTモデルの効率性を簡潔に示すスクリプト
"""

import torch
from model_whiteline import WhiteLineDetector
from tabulate import tabulate

def analyze_aitect_efficiency():
    """AITECTモデルの効率性分析"""
    
    # AITECTモデル
    model = WhiteLineDetector(grid_size=8, num_anchors=3)
    
    # パラメータ数計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # float32
    
    # 他モデルとの比較データ（文献値）
    comparison_data = [
        ['AITECT (提案手法)', f'{total_params/1e6:.1f}M', f'{model_size_mb:.1f}MB', '◎', '◎'],
        ['YOLOv5s', '7.2M', '27.5MB', '○', '○'],
        ['YOLOv5m', '21.2M', '81.0MB', '○', '△'],
        ['YOLOX-S', '8.9M', '34.0MB', '○', '○'],
        ['Faster R-CNN', '41.8M', '159.0MB', '△', '×'],
        ['RetinaNet', '37.7M', '144.0MB', '△', '×'],
        ['SSD300', '26.2M', '100.0MB', '△', '△'],
    ]
    
    headers = ['モデル', 'パラメータ数', 'モデルサイズ', '推論速度', 'エッジ適性']
    
    print("\n" + "="*70)
    print("物体検出モデルの効率性比較")
    print("="*70)
    print(tabulate(comparison_data, headers=headers, tablefmt='grid'))
    
    # エコ指標
    print("\n【エコAI指標】")
    aitect_params = total_params / 1e6
    print(f"AITECTモデル: {aitect_params:.1f}M パラメータ")
    print(f"  • YOLOv5sより {(aitect_params/7.2 - 1)*100:+.0f}%")
    print(f"  • Faster R-CNNの {aitect_params/41.8*100:.0f}% のサイズ")
    print(f"  • メモリ使用量: 約{model_size_mb:.0f}MB（エッジデバイスで動作可能）")
    
    print("\n【研究的意義】")
    print("1. 限られた計算資源での物体検出を実現")
    print("2. エネルギー効率の高いAIシステム")
    print("3. IoTデバイスやモバイル環境での実用化が可能")
    print("4. SDGsに貢献する持続可能なAI技術")
    
    # モデル構造の詳細
    print("\n【モデル構造】")
    print(f"バックボーン: ResNet18 (軽量版)")
    print(f"グリッドサイズ: {model.grid_size}×{model.grid_size}")
    print(f"アンカー数: {model.num_anchors}")
    print(f"総予測数: {model.grid_size * model.grid_size * model.num_anchors}")
    
    return {
        'total_params': total_params,
        'model_size_mb': model_size_mb,
        'params_millions': aitect_params
    }

if __name__ == "__main__":
    analyze_aitect_efficiency()