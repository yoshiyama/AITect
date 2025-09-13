#!/usr/bin/env python3
"""予測ボックスの数を改善するための設定変更案"""

print("=== 予測ボックス数の改善案 ===\n")

print("現在の問題:")
print("- 16x16グリッド = 256個の予測位置すべてが出力")
print("- 信頼度閾値0.3が低すぎる")
print("- 多くの重複ボックス\n")

print("改善案:")
print("\n1. 信頼度閾値の調整（推奨）")
print("   utils/validation.py の conf_thresh を変更:")
print("   - 現在: 0.3")
print("   - 推奨: 0.5 または 0.6")
print("   効果: 低信頼度の予測を除外\n")

print("2. NMS閾値の調整")
print("   utils/validation.py の iou_threshold を変更:")
print("   - 現在: 0.5")
print("   - 推奨: 0.3 または 0.4")
print("   効果: より積極的に重複を除去\n")

print("3. config.jsonでの評価閾値調整")
print("   evaluation.conf_threshold を変更:")
print("   - 現在: 0.5")
print("   - 推奨: 0.6 または 0.7")
print("   効果: 最終的な検出の信頼度を上げる\n")

print("4. 長期的な改善")
print("   - グリッドサイズを減らす（16x16 → 13x13）")
print("   - 背景クラスの導入")
print("   - Focal Lossの使用で背景予測を抑制\n")

print("即座に適用可能な修正:")
print("1. validation.pyのconf_threshを0.5に上げる")
print("2. NMS iou_thresholdを0.4に下げる")