"""
損失計算の可視化スクリプト
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_loss_components():
    """損失関数の構成要素を可視化"""
    
    # Figure 1: 損失の全体構成
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 損失の全体構成（円グラフ）
    ax1.set_title('損失関数の構成要素', fontsize=16, weight='bold')
    sizes = [60, 40]
    labels = ['信頼度損失\n(Confidence Loss)', '回帰損失\n(Regression Loss)']
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0.05)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 12})
    
    # 2. 正例・負例・無視の分布
    ax2.set_title('予測の分類（YOLOスタイル）', fontsize=16, weight='bold')
    categories = ['正例\n(Positive)', '負例\n(Negative)', '無視\n(Ignore)']
    counts = [5, 180, 7]  # 例：192予測中
    colors_bar = ['green', 'red', 'gray']
    
    bars = ax2.bar(categories, counts, color=colors_bar, alpha=0.7)
    ax2.set_ylabel('予測数', fontsize=12)
    
    # 値を表示
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=12, weight='bold')
    
    # 3. IoUに基づく分類の図解
    ax3.set_title('IoUによる予測の分類', fontsize=16, weight='bold')
    ax3.set_xlim(0, 150)
    ax3.set_ylim(0, 150)
    ax3.set_aspect('equal')
    
    # GTボックス（青）
    gt_box = patches.Rectangle((40, 40), 70, 70, linewidth=3, 
                              edgecolor='blue', facecolor='blue', alpha=0.3,
                              label='GT (Ground Truth)')
    ax3.add_patch(gt_box)
    
    # 正例（IoU > 0.5）- 緑
    pos_box = patches.Rectangle((50, 50), 70, 70, linewidth=2,
                               edgecolor='green', facecolor='green', alpha=0.2)
    ax3.add_patch(pos_box)
    ax3.text(85, 135, 'IoU > 0.5\n→ 正例', ha='center', color='green', 
             fontsize=11, weight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 無視（0.3 < IoU < 0.5）- 灰色
    ignore_box = patches.Rectangle((20, 60), 60, 60, linewidth=2,
                                  edgecolor='gray', facecolor='gray', alpha=0.2)
    ax3.add_patch(ignore_box)
    ax3.text(50, 15, '0.3 < IoU < 0.5\n→ 無視', ha='center', color='gray',
             fontsize=11, weight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 負例（IoU < 0.3）- 赤
    neg_box = patches.Rectangle((100, 20), 40, 40, linewidth=2,
                               edgecolor='red', facecolor='red', alpha=0.2)
    ax3.add_patch(neg_box)
    ax3.text(120, 10, 'IoU < 0.3\n→ 負例', ha='center', color='red',
             fontsize=11, weight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    # 4. 損失の重み付け
    ax4.set_title('損失の重み付け戦略', fontsize=16, weight='bold')
    ax4.axis('off')
    
    weights_text = """
    YOLOスタイルの重み付け:
    
    • λ_coord = 5.0 (座標損失を重視)
      → 位置の正確性が重要
    
    • λ_obj = 1.0 (物体ありの標準重み)
      → バランスの基準
    
    • λ_noobj = 0.5 (物体なしを軽視)
      → 背景が多いため
    
    総損失 = λ_obj × 正例損失
           + λ_noobj × 負例損失
           + λ_coord × 回帰損失
    """
    
    ax4.text(0.1, 0.5, weights_text, transform=ax4.transAxes,
            fontsize=13, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('loss_components_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_loss_calculation_flow():
    """損失計算のフローを可視化"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # タイトル
    ax.text(5, 9.5, '損失計算のフロー', fontsize=20, weight='bold', ha='center')
    
    # ステップ1: 入力
    rect1 = patches.FancyBboxPatch((0.5, 7.5), 2, 1.2, 
                                  boxstyle="round,pad=0.1",
                                  facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.5, 8.1, '予測\n[N, 5]', ha='center', va='center', fontsize=12, weight='bold')
    
    rect2 = patches.FancyBboxPatch((7.5, 7.5), 2, 1.2,
                                  boxstyle="round,pad=0.1",
                                  facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(rect2)
    ax.text(8.5, 8.1, 'GT\n[M, 4]', ha='center', va='center', fontsize=12, weight='bold')
    
    # 矢印
    ax.arrow(1.5, 7.4, 0, -0.8, head_width=0.2, head_length=0.1, fc='black', ec='black')
    ax.arrow(8.5, 7.4, 0, -0.8, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # ステップ2: マッチング
    rect3 = patches.FancyBboxPatch((3.5, 5.5), 3, 1,
                                  boxstyle="round,pad=0.1",
                                  facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(rect3)
    ax.text(5, 6, 'IoU計算 & マッチング', ha='center', va='center', fontsize=12)
    
    # 矢印（分岐）
    ax.arrow(4, 5.4, -1.5, -0.8, head_width=0.15, head_length=0.1, fc='green', ec='green')
    ax.arrow(5, 5.4, 0, -0.8, head_width=0.15, head_length=0.1, fc='gray', ec='gray')
    ax.arrow(6, 5.4, 1.5, -0.8, head_width=0.15, head_length=0.1, fc='red', ec='red')
    
    # ステップ3: 分類
    # 正例
    rect4 = patches.FancyBboxPatch((0.5, 3.5), 2, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax.add_patch(rect4)
    ax.text(1.5, 3.9, '正例\nIoU>0.5', ha='center', va='center', fontsize=11)
    
    # 無視
    rect5 = patches.FancyBboxPatch((4, 3.5), 2, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor='lightgray', edgecolor='gray', linewidth=2)
    ax.add_patch(rect5)
    ax.text(5, 3.9, '無視\n0.3<IoU<0.5', ha='center', va='center', fontsize=11)
    
    # 負例
    rect6 = patches.FancyBboxPatch((7.5, 3.5), 2, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(rect6)
    ax.text(8.5, 3.9, '負例\nIoU<0.3', ha='center', va='center', fontsize=11)
    
    # 矢印（損失へ）
    ax.arrow(1.5, 3.4, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    ax.arrow(8.5, 3.4, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # ステップ4: 損失計算
    # 信頼度損失
    rect7 = patches.FancyBboxPatch((0.5, 1.8), 2, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#ffcccc', edgecolor='black', linewidth=2)
    ax.add_patch(rect7)
    ax.text(1.5, 2.2, '信頼度損失\n×λ_obj', ha='center', va='center', fontsize=11)
    
    rect8 = patches.FancyBboxPatch((7.5, 1.8), 2, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#ffcccc', edgecolor='black', linewidth=2)
    ax.add_patch(rect8)
    ax.text(8.5, 2.2, '信頼度損失\n×λ_noobj', ha='center', va='center', fontsize=11)
    
    # 回帰損失（正例のみ）
    rect9 = patches.FancyBboxPatch((0.5, 0.8), 2, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#ccccff', edgecolor='black', linewidth=2)
    ax.add_patch(rect9)
    ax.text(1.5, 1.2, '回帰損失\n×λ_coord', ha='center', va='center', fontsize=11)
    
    # 最終的な総損失
    ax.arrow(1.5, 0.7, 2.5, -0.3, head_width=0.15, head_length=0.1, fc='black', ec='black')
    ax.arrow(8.5, 1.7, -4.5, -1.3, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    rect10 = patches.FancyBboxPatch((3.5, 0), 3, 0.6,
                                   boxstyle="round,pad=0.1",
                                   facecolor='gold', edgecolor='black', linewidth=3)
    ax.add_patch(rect10)
    ax.text(5, 0.3, '総損失', ha='center', va='center', fontsize=14, weight='bold')
    
    plt.savefig('loss_calculation_flow.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_loss_formula_visualization():
    """損失関数の数式を可視化"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    formulas = r"""
    物体検出の損失関数
    
    1. 標準実装（混合モード）
    $\mathcal{L}_{total} = \mathcal{L}_{conf} + \mathcal{L}_{reg}$
    
    $\mathcal{L}_{conf} = BCE(scores, targets)$
    $\mathcal{L}_{reg} = \alpha_{iou} \cdot (1 - IoU) + \alpha_{L1} \cdot ||pred - gt||_1$
    
    2. YOLOスタイル実装
    $\mathcal{L}_{total} = \lambda_{obj} \sum_{i \in pos} \mathcal{L}_{conf}^i + 
                          \lambda_{noobj} \sum_{j \in neg} \mathcal{L}_{conf}^j + 
                          \lambda_{coord} \sum_{i \in pos} \mathcal{L}_{reg}^i$
    
    3. Focal Loss（クラス不均衡対策）
    $FL(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$
    
    ここで:
    - $p_t$: 正解クラスの予測確率
    - $\alpha_t$: クラス重み（正例:0.25, 負例:0.75）
    - $\gamma$: 難易度調整（通常2.0）
    """
    
    ax.text(0.5, 0.5, formulas, transform=ax.transAxes,
            fontsize=14, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.savefig('loss_formulas.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("損失関数の可視化を生成中...")
    
    # 1. 損失の構成要素
    print("1. 損失の構成要素を可視化...")
    visualize_loss_components()
    
    # 2. 損失計算のフロー
    print("2. 損失計算のフローを可視化...")
    visualize_loss_calculation_flow()
    
    # 3. 数式の可視化
    print("3. 損失関数の数式を生成...")
    create_loss_formula_visualization()
    
    print("\n完了！以下のファイルが生成されました：")
    print("- loss_components_visualization.png: 損失の構成要素")
    print("- loss_calculation_flow.png: 損失計算のフロー")
    print("- loss_formulas.png: 損失関数の数式")