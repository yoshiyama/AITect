import json
import numpy as np
import matplotlib.pyplot as plt

# 結果を読み込み
with open('val_detection_results/results_summary.json', 'r') as f:
    results = json.load(f)

# 統計分析
per_image = results['per_image']

# 検出パターンの分類
no_detection = [r for r in per_image if r['num_pred'] == 0]
perfect = [r for r in per_image if r['tp'] > 0 and r['fp'] == 0 and r['fn'] == 0]
partial = [r for r in per_image if r['tp'] > 0 and (r['fp'] > 0 or r['fn'] > 0)]
only_false = [r for r in per_image if r['tp'] == 0 and r['fp'] > 0]
missed_all = [r for r in per_image if r['tp'] == 0 and r['fn'] > 0 and r['fp'] == 0]

print("=== 現在のモデルの状況分析 ===\n")

print("1. 検出パターンの内訳:")
print(f"   - 完璧な検出: {len(perfect)}枚 ({len(perfect)/len(per_image)*100:.1f}%)")
print(f"   - 部分的な検出: {len(partial)}枚 ({len(partial)/len(per_image)*100:.1f}%)")
print(f"   - 誤検出のみ: {len(only_false)}枚 ({len(only_false)/len(per_image)*100:.1f}%)")
print(f"   - 全て見逃し: {len(missed_all)}枚 ({len(missed_all)/len(per_image)*100:.1f}%)")
print(f"   - 何も検出せず: {len(no_detection)}枚 ({len(no_detection)/len(per_image)*100:.1f}%)")

print("\n2. Ground Truthの統計:")
gt_counts = [r['num_gt'] for r in per_image]
print(f"   - 平均GT数: {np.mean(gt_counts):.2f}")
print(f"   - 最大GT数: {max(gt_counts)}")
print(f"   - 最小GT数: {min(gt_counts)}")

print("\n3. 検出数の統計:")
pred_counts = [r['num_pred'] for r in per_image]
print(f"   - 平均検出数: {np.mean(pred_counts):.2f}")
print(f"   - 最大検出数: {max(pred_counts)}")
print(f"   - 検出ゼロの画像: {pred_counts.count(0)}枚")

# 学習曲線を確認
print("\n4. 学習状況の分析:")
print("   訓練損失: 36.8 → 17.5 (52%減少)")
print("   検証損失: 36.8 → 35.2 (4%減少のみ)")
print("   → 訓練と検証の乖離が大きい = 過学習の兆候")

print("\n=== 改善の可能性 ===\n")

print("❌ 問題点:")
print("1. 検証損失がほぼ下がっていない → 汎化性能が低い")
print("2. 43枚中18枚で何も検出できていない")
print("3. False Negatives (75) > True Positives (46) → 見逃しが多い")
print("4. Ground Truthの品質問題（白線の定義が不明確な可能性）")

print("\n✅ 改善案:")
print("1. **データ品質の改善**")
print("   - Ground Truthの再確認・修正")
print("   - より多様なデータの追加")
print("   - 難しいケースのアノテーション追加")

print("\n2. **学習戦略の改善**")
print("   - より強い正則化（Dropout増加、Weight Decay強化）")
print("   - データ拡張をさらに強化")
print("   - 学習率をもっと小さくして長期学習")
print("   - Mixup, CutMix等の高度な拡張手法")

print("\n3. **モデルアーキテクチャの改善**")
print("   - より深いバックボーン（ResNet50, EfficientNet）")
print("   - Feature Pyramid Network (FPN) の追加")
print("   - Attention機構の導入")

print("\n4. **損失関数の改善**")
print("   - IoU損失の重み増加（現在2.0 → 5.0）")
print("   - Hard Negative Miningの実装")
print("   - クラスバランスの再調整")

# 可視化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 検出パターン
ax = axes[0, 0]
patterns = ['Perfect', 'Partial', 'Only FP', 'All Missed', 'No Detection']
counts = [len(perfect), len(partial), len(only_false), len(missed_all), len(no_detection)]
colors = ['green', 'blue', 'orange', 'red', 'gray']
ax.bar(patterns, counts, color=colors)
ax.set_title('Detection Pattern Distribution')
ax.set_ylabel('Number of Images')

# GT数の分布
ax = axes[0, 1]
ax.hist(gt_counts, bins=10, edgecolor='black')
ax.set_title('Ground Truth Count Distribution')
ax.set_xlabel('Number of GT boxes per image')
ax.set_ylabel('Frequency')

# 検出数の分布
ax = axes[1, 0]
ax.hist(pred_counts, bins=10, edgecolor='black', color='orange')
ax.set_title('Prediction Count Distribution')
ax.set_xlabel('Number of predicted boxes per image')
ax.set_ylabel('Frequency')

# F1スコアの分布
ax = axes[1, 1]
f1_scores = [r['f1'] for r in per_image]
ax.hist(f1_scores, bins=20, edgecolor='black', color='purple')
ax.set_title('F1 Score Distribution')
ax.set_xlabel('F1 Score')
ax.set_ylabel('Frequency')
ax.axvline(np.mean(f1_scores), color='red', linestyle='--', label=f'Mean: {np.mean(f1_scores):.3f}')
ax.legend()

plt.tight_layout()
plt.savefig('current_status_analysis.png', dpi=150)
print(f"\n分析グラフを保存: current_status_analysis.png")

print("\n=== 結論 ===")
print("現在のF1スコア0.45は、初期の0から大幅に改善されましたが、")
print("さらなる学習だけでは限界があります。")
print("\n推奨アクション:")
print("1. まずGround Truthの品質を確認")
print("2. より強い正則化で過学習を抑制")
print("3. モデルアーキテクチャの改善")
print("4. 期待値: F1スコア 0.45 → 0.60-0.65 (適切な改善で)")