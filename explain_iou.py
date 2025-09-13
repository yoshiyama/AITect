"""
IoU（Intersection over Union）を分かりやすく説明するスクリプト
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def calculate_iou(box1, box2):
    """IoUを計算（box format: [x1, y1, x2, y2]）"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def visualize_iou_examples():
    """IoUの例を可視化"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 例1: 完全一致（IoU = 1.0）
    ax = axes[0, 0]
    gt_box = [20, 20, 80, 80]
    pred_box = [20, 20, 80, 80]
    draw_boxes_with_iou(ax, gt_box, pred_box, "完全一致")
    
    # 例2: 良い検出（IoU ≈ 0.8）
    ax = axes[0, 1]
    gt_box = [20, 20, 80, 80]
    pred_box = [25, 25, 85, 85]
    draw_boxes_with_iou(ax, gt_box, pred_box, "良い検出")
    
    # 例3: まあまあの検出（IoU ≈ 0.5）
    ax = axes[0, 2]
    gt_box = [20, 20, 80, 80]
    pred_box = [40, 20, 100, 80]
    draw_boxes_with_iou(ax, gt_box, pred_box, "まあまあの検出")
    
    # 例4: 悪い検出（IoU ≈ 0.3）
    ax = axes[1, 0]
    gt_box = [20, 20, 80, 80]
    pred_box = [50, 50, 110, 110]
    draw_boxes_with_iou(ax, gt_box, pred_box, "悪い検出")
    
    # 例5: ほぼ重ならない（IoU ≈ 0.1）
    ax = axes[1, 1]
    gt_box = [20, 20, 80, 80]
    pred_box = [70, 70, 130, 130]
    draw_boxes_with_iou(ax, gt_box, pred_box, "ほぼ重ならない")
    
    # 例6: 全く重ならない（IoU = 0.0）
    ax = axes[1, 2]
    gt_box = [20, 20, 80, 80]
    pred_box = [90, 20, 150, 80]
    draw_boxes_with_iou(ax, gt_box, pred_box, "全く重ならない")
    
    plt.tight_layout()
    plt.savefig('iou_examples.png', dpi=300, bbox_inches='tight')
    plt.show()

def draw_boxes_with_iou(ax, gt_box, pred_box, title):
    """ボックスとIoUを描画"""
    # 背景
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 150)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # GT box（緑）
    x1, y1, x2, y2 = gt_box
    gt_rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=3, edgecolor='green', 
                               facecolor='green', alpha=0.3,
                               label='正解（GT）')
    ax.add_patch(gt_rect)
    
    # Pred box（赤）
    x1, y1, x2, y2 = pred_box
    pred_rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=3, edgecolor='red', 
                                 facecolor='red', alpha=0.3,
                                 label='予測')
    ax.add_patch(pred_rect)
    
    # IoU計算と表示
    iou = calculate_iou(gt_box, pred_box)
    ax.set_title(f'{title}\nIoU = {iou:.3f}', fontsize=14, weight='bold')
    
    # 判定基準を表示
    if iou >= 0.5:
        judgment = "✓ 正検出（TP）"
        color = 'green'
    else:
        judgment = "✗ 誤検出（FP）"
        color = 'red'
    
    ax.text(75, 140, judgment, fontsize=12, ha='center', 
            color=color, weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 凡例
    ax.legend(loc='upper left', fontsize=10)

def explain_iou_metrics():
    """IoU関連の評価指標を説明"""
    print("\n" + "="*60)
    print("IoU（Intersection over Union）の理解")
    print("="*60)
    
    print("\n【IoUとは？】")
    print("予測したボックスと正解ボックスの「重なり具合」を表す指標です。")
    print("計算式: IoU = 重なり部分の面積 ÷ 全体の面積（和集合）")
    
    print("\n【IoUの値の意味】")
    print("- IoU = 1.0: 完全一致（100%重なる）")
    print("- IoU ≥ 0.7: 高精度な検出")
    print("- IoU ≥ 0.5: 一般的に「正しい検出」とみなす")
    print("- IoU < 0.5: 検出失敗")
    print("- IoU = 0.0: 全く重ならない")
    
    print("\n【評価結果でのIoU統計の見方】")
    print("1. 平均IoU: すべてのTP（正検出）のIoUの平均")
    print("   → 高いほど予測が正確")
    print("2. 標準偏差: IoUのばらつき")
    print("   → 小さいほど安定した検出")
    print("3. 最小IoU: 最も悪いTPのIoU")
    print("   → 0.5以上なら、すべてのTPが基準を満たす")
    print("4. 最大IoU: 最も良いTPのIoU")
    print("   → 1.0に近いほど完璧な検出がある")
    
    print("\n【実際の評価結果の例】")
    print("```")
    print("IoU統計 (TP予測のみ):")
    print("  平均IoU: 0.7234    ← 平均的に良い重なり")
    print("  標準偏差: 0.1523   ← やや不安定")
    print("  最小IoU: 0.5012    ← ギリギリ正検出")
    print("  最大IoU: 0.9856    ← ほぼ完璧な検出もある")
    print("  中央値IoU: 0.7456  ← 半数以上が0.74以上")
    print("```")
    
    print("\n【問題診断】")
    print("- 平均IoUが低い（< 0.6）")
    print("  → ボックスの位置やサイズの予測が不正確")
    print("- 標準偏差が大きい（> 0.2）")
    print("  → 検出品質が不安定")
    print("- 最小IoUが0.5未満")
    print("  → 閾値設定に問題がある可能性")

def analyze_iou_distribution():
    """IoU分布の分析例"""
    # サンプルデータ（実際のモデルの典型的な分布）
    np.random.seed(42)
    
    # 良いモデルのIoU分布
    good_ious = np.concatenate([
        np.random.normal(0.75, 0.1, 300),  # 多くは0.75付近
        np.random.normal(0.9, 0.05, 50),   # 一部は非常に良い
    ])
    good_ious = np.clip(good_ious, 0.5, 1.0)  # 0.5-1.0に制限
    
    # 悪いモデルのIoU分布
    bad_ious = np.concatenate([
        np.random.normal(0.55, 0.15, 200),  # 0.55付近に集中
        np.random.uniform(0.5, 0.7, 100),   # 広く分布
    ])
    bad_ious = np.clip(bad_ious, 0.5, 1.0)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 良いモデル
    ax1.hist(good_ious, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax1.axvline(np.mean(good_ious), color='red', linestyle='--', linewidth=2,
                label=f'平均: {np.mean(good_ious):.3f}')
    ax1.axvline(np.median(good_ious), color='blue', linestyle='--', linewidth=2,
                label=f'中央値: {np.median(good_ious):.3f}')
    ax1.set_xlabel('IoU値', fontsize=12)
    ax1.set_ylabel('頻度', fontsize=12)
    ax1.set_title('良いモデルのIoU分布\n（高精度・安定）', fontsize=14, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 統計情報
    ax1.text(0.52, ax1.get_ylim()[1]*0.9, 
             f'標準偏差: {np.std(good_ious):.3f}\n'
             f'最小: {np.min(good_ious):.3f}\n'
             f'最大: {np.max(good_ious):.3f}',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 悪いモデル
    ax2.hist(bad_ious, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax2.axvline(np.mean(bad_ious), color='red', linestyle='--', linewidth=2,
                label=f'平均: {np.mean(bad_ious):.3f}')
    ax2.axvline(np.median(bad_ious), color='blue', linestyle='--', linewidth=2,
                label=f'中央値: {np.median(bad_ious):.3f}')
    ax2.set_xlabel('IoU値', fontsize=12)
    ax2.set_ylabel('頻度', fontsize=12)
    ax2.set_title('改善が必要なモデルのIoU分布\n（低精度・不安定）', fontsize=14, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 統計情報
    ax2.text(0.52, ax2.get_ylim()[1]*0.9, 
             f'標準偏差: {np.std(bad_ious):.3f}\n'
             f'最小: {np.min(bad_ious):.3f}\n'
             f'最大: {np.max(bad_ious):.3f}',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('iou_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_iou_interpretation_guide():
    """IoU解釈ガイドの作成"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    guide_text = """
    IoU評価結果の読み方ガイド
    
    1. Precision/Recall/F1と一緒に見る
       - 高Precision + 低い平均IoU = 検出は正確だが、位置がずれている
       - 高Recall + 低い平均IoU = 多く検出できるが、精度が低い
    
    2. IoU統計の組み合わせパターン
    
       パターンA: 理想的な状態
       - 平均IoU: 0.75以上
       - 標準偏差: 0.15以下
       - 最小IoU: 0.6以上
       → 安定して高精度な検出
    
       パターンB: 位置ずれ
       - 平均IoU: 0.55-0.65
       - 標準偏差: 0.1以下
       - すべてのIoUが似た値
       → 一貫して位置がずれている（アンカー調整が必要）
    
       パターンC: 不安定
       - 平均IoU: 0.65
       - 標準偏差: 0.2以上
       - 最小と最大の差が大きい
       → 検出品質にばらつき（学習不足の可能性）
    
    3. 改善のヒント
       - 平均IoU < 0.6: ボックス回帰の損失重みを増やす
       - 標準偏差 > 0.2: より多くのデータで学習
       - 最小IoU < 0.5: 信頼度閾値を上げる
    """
    
    ax.text(0.1, 0.5, guide_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.title('IoU評価結果の解釈ガイド', fontsize=18, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('iou_interpretation_guide.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("IoUの説明資料を生成中...")
    
    # 1. IoUの基本概念を説明
    explain_iou_metrics()
    
    # 2. IoUの視覚的な例
    print("\nIoUの視覚的な例を生成中...")
    visualize_iou_examples()
    
    # 3. IoU分布の比較
    print("\nIoU分布の比較を生成中...")
    analyze_iou_distribution()
    
    # 4. 解釈ガイド
    print("\nIoU解釈ガイドを生成中...")
    create_iou_interpretation_guide()
    
    print("\n完了！以下のファイルが生成されました：")
    print("- iou_examples.png: IoUの視覚的な例")
    print("- iou_distribution_comparison.png: 良い/悪いモデルのIoU分布")
    print("- iou_interpretation_guide.png: IoU評価結果の解釈ガイド")