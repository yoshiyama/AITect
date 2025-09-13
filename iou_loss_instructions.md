# 📘 AITECTモデル改善のための指示書：IoU損失導入編

この文書は、現状のAITECT物体検出器において、位置回帰にIoUベースの損失（IoU, GIoU, DIoU, CIoU）を導入するための手順をまとめたものです。

---

## ✅ 目的
- 現在の L1 Loss による回帰損失を、**IoUベースの損失関数**に置き換える
- 評価指標（IoU ≥ 0.5 で正解）と一致した学習目標にする
- 学習の安定性や精度向上を目指す

---

## 🧱 ステップ1：IoU関数の確認（すでに存在）

ファイル：`utils/bbox.py`

```python
def box_iou(boxes1, boxes2): ...
```

この関数を使って `N×M` のIoU行列が計算できる。

---

## 🧪 ステップ2：IoU損失関数の追加（例）

ファイル：`loss.py` に以下を追加：

```python
def iou_loss(pred_boxes, gt_boxes):
    ious = box_iou(pred_boxes, gt_boxes).diag()
    return 1.0 - ious.mean()
```

オプションで GIoU, DIoU, CIoU のバージョンも後述の参考実装から選択可。

---

## 🔁 ステップ3：既存の bbox 回帰損失を差し替え

ファイル：`loss.py`

以下のように書き換え：

```python
# 旧: L1損失
# reg_loss = F.l1_loss(matched_pred_boxes, matched_gt_boxes, reduction='mean')

# 新: IoU損失
reg_loss = iou_loss(matched_pred_boxes, matched_gt_boxes)
```

※ 両方使って加重平均してもよい： `loss = α*L1 + β*IoU`

---

## 📊 ステップ4：比較実験の設計

| 実験名 | 損失関数 | 比較ポイント |
|--------|----------|---------------|
| Baseline | L1 + BCE | 現状モデル |
| IoUOnly | IoU + BCE | 重なり優先の学習 |
| MixLoss | L1 + IoU + BCE | 両者の中間（安定性＋最適性） |

---

## 💡 ステップ5：拡張オプション（任意）

### GIoU（Generalized IoU）

```python
# 追加ライブラリなしで実装可能（外接矩形との面積比較）
```

### CIoU（Complete IoU）

```bash
pip install iouloss-pytorch
```

または自作実装も可能（中心距離 + aspect比の項を追加）

---

## ✅ まとめ

- 評価指標との一貫性を持たせるために IoU損失を採用
- 損失の差し替えは1〜2行で完了
- 比較実験により改善効果を可視化可能

---

## 📂 保存先
このファイルを `doc/iou_loss_instructions.md` などに保存し、記録として残してください。
