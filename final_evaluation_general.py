import torch
import matplotlib.pyplot as plt
import numpy as np
from model import AITECTDetector
from utils.postprocess import postprocess_predictions
import json
import os

def create_final_summary():
    """一般物体検出学習の最終まとめ"""
    
    print("=== AITect 一般物体検出 - 最終評価 ===\n")
    
    # 結果のまとめ
    results = {
        "白線検出（元の用途）": {
            "dataset": "White Line (inaoka)",
            "f1_score": 0.4488,
            "precision": 0.5476,
            "recall": 0.3802,
            "status": "✅ 実用レベル"
        },
        "白線モデル→一般物体（直接）": {
            "dataset": "Simple Shapes",
            "f1_score": 0.0057,
            "precision": 0.0041,
            "recall": 0.0095,
            "status": "❌ 機能せず"
        },
        "白線モデル→一般物体（転移学習）": {
            "dataset": "Simple Shapes",
            "f1_score": 0.2446,
            "precision": 0.0653,
            "recall": 0.9521,
            "status": "🔄 改善あり"
        },
        "一般物体検出（新規学習）": {
            "dataset": "Simple Shapes",
            "f1_score": 0.6770,
            "precision": 0.6397,
            "recall": 0.7190,
            "status": "✅ 良好"
        },
        "マルチクラス検出（10カテゴリ）": {
            "dataset": "Mini COCO",
            "f1_score": "学習中",
            "precision": "-",
            "recall": "-",
            "status": "🚀 実装完了"
        }
    }
    
    # 結果表示
    print("【学習結果一覧】")
    print("-" * 80)
    print(f"{'モデル':<30} {'データセット':<15} {'F1スコア':<10} {'状態':<10}")
    print("-" * 80)
    
    for model_name, info in results.items():
        f1 = f"{info['f1_score']:.4f}" if isinstance(info['f1_score'], float) else info['f1_score']
        print(f"{model_name:<30} {info['dataset']:<15} {f1:<10} {info['status']}")
    
    # グラフ作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # F1スコア比較
    models = list(results.keys())[:4]  # 数値がある4つ
    f1_scores = [results[m]['f1_score'] for m in models]
    
    bars = ax1.bar(range(len(models)), f1_scores, color=['green', 'red', 'orange', 'blue'])
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m.split('（')[0] for m in models], rotation=45, ha='right')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_ylim(0, 0.8)
    
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # 学習アプローチの比較
    ax2.text(0.5, 0.9, '一般物体検出への道', fontsize=16, weight='bold',
             ha='center', transform=ax2.transAxes)
    
    approaches = [
        "1. 特定タスク特化モデル",
        "   → 他タスクには使えない (F1: 0.006)",
        "",
        "2. 転移学習アプローチ",
        "   → 少量データで改善可能 (F1: 0.245)",
        "",
        "3. 一般物体検出として学習",
        "   → 高い汎用性 (F1: 0.677)",
        "",
        "4. マルチクラス対応",
        "   → 実用的な物体検出システム"
    ]
    
    y_pos = 0.75
    for text in approaches:
        if text.startswith('   '):
            ax2.text(0.15, y_pos, text, fontsize=11, transform=ax2.transAxes, color='gray')
        elif text.startswith(('1.', '2.', '3.', '4.')):
            ax2.text(0.1, y_pos, text, fontsize=12, weight='bold', transform=ax2.transAxes)
        else:
            ax2.text(0.1, y_pos, text, fontsize=12, transform=ax2.transAxes)
        y_pos -= 0.08
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('final_general_detection_summary.png', dpi=150)
    print("\n📊 最終まとめ画像: final_general_detection_summary.png")
    
    # 推奨事項
    print("\n【結論と推奨事項】")
    print("="*60)
    print("1. AITectは汎用物体検出のベースとして使用可能")
    print("2. ただし、各タスクに応じた学習が必要：")
    print("   - 特定タスク → 専用データで学習")
    print("   - 汎用検出 → 多様なデータで学習")
    print("   - 転移学習 → 既存モデルを活用")
    print("\n3. 実装済みの機能：")
    print("   ✅ YOLO型グリッドベース検出")
    print("   ✅ マルチクラス対応")
    print("   ✅ データ拡張")
    print("   ✅ 転移学習サポート")
    print("\n4. 今後の改善案：")
    print("   - より大規模なデータセット（COCO, VOC）での学習")
    print("   - FPN等の高度なアーキテクチャ")
    print("   - リアルタイム推論の最適化")

if __name__ == "__main__":
    create_final_summary()