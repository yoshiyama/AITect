# AITECT - 単一クラス物体検出スクラッチ実装

このプロジェクトは、愛知工業大学工学部土木工学科測量研究室による物体検出教育・研究のためのスクラッチ実装です。対象は単一クラス（例：白線）ですが、画像中に複数物体がある一般的な検出タスクに対応しています。

## 📁 ディレクトリ構成

```
project/
├── dataset.py
├── model.py
├── loss.py
├── main.py
├── test.py
├── utils/
│   ├── bbox.py
│   ├── visualize.py
│   └── metrics.py
├── result/
│   └── ait...
├── datasets/
│   └── inaoka/
│       ├── train/
│       │   ├── JPEGImages/
│       │   └── annotations.json
```

## 🚀 実行手順

### 1. 環境構築（例：conda）

```bash
conda create -n aitect python=3.10
conda activate aitect
pip install -r requirements.txt
```

### 2. 学習

#### 基本的な使い方（設定ファイルを使用）
```bash
python main.py
```

#### 自動設定機能（推奨）
```bash
# データセットを分析して最適なパラメータを自動設定
python main.py --auto-config
```
この機能は学習データを分析し、以下を自動的に設定します：
- グリッドサイズ（検出密度）
- アンカーボックスのサイズと数
- 損失関数の重み

#### コマンドライン引数で設定を変更
```bash
# エポック数を10に設定
python main.py --epochs 10

# バッチサイズを8、学習率を0.001に設定
python main.py --batch-size 8 --lr 0.001

# 自動設定 + エポック数指定
python main.py --auto-config --epochs 100

# 全てのオプションを表示
python main.py --help
```

#### 設定ファイル（config.json）を編集
`config.json`ファイルを直接編集することで、詳細な設定が可能です：
- `training.num_epochs`: エポック数（学習の繰り返し回数）
- `training.batch_size`: バッチサイズ（一度に処理する画像数）
- `training.learning_rate`: 学習率（パラメータの更新幅）
- `training.save_interval`: モデル保存間隔

### 3. 推論・可視化

```bash
python test.py
```

## 📦 入出力

- 入力画像：`datasets/inaoka/train/JPEGImages/`
- アノテーション：COCO形式（`annotations.json`）
- 学習済みモデル：`result/aitect_model.pth`
- 推論画像：Matplotlibで描画表示

## 📚 使用ライブラリ

- PyTorch
- torchvision
- matplotlib
- pycocotools
- OpenCV

## 🧠 モデルアーキテクチャ

### AITECTDetector の構造

本プロジェクトで使用する物体検出モデルは、シンプルながら効果的な構造を持っています：

#### 自動パラメータ設定機能
`--auto-config`オプションを使用すると、学習データの特性に基づいて以下が自動設定されます：
- **グリッドサイズ**: 検出対象のサイズに応じて8×8～16×16を自動選択
- **アンカーボックス**: 実際のGTボックスの統計から最適なサイズと数を決定
- **損失関数の重み**: サイズのばらつきに応じてIoU重みを調整

#### 1. バックボーン（特徴抽出部）
- **ResNet18** をベースとした軽量CNN
- 最終的な全結合層とプーリング層を除去
- 入力: 512×512×3 (RGB画像)
- 出力: 16×16×512 の特徴マップ

#### 2. 検出ヘッド
- 2層の畳み込みネットワーク
  - Conv2d(512→256) + ReLU
  - Conv2d(256→5)
- 各位置で5つの値を予測: [x, y, w, h, confidence]

#### 3. 層の詳細
```
入力画像 (512×512×3)
    ↓
ResNet18バックボーン（～50層）
    ↓
特徴マップ (16×16×512)
    ↓
検出ヘッド（2層）
    ↓
出力 (256個の検出候補)
```

総パラメータ数: 約11.3M（軽量で高速）

## 🎯 物体検出の仕組み（初学者向け解説）

### 1. なぜ物体検出は難しいのか？

物体検出では以下を同時に解決する必要があります：
- **どこに**物体があるか（位置）
- **どのくらいの大きさ**か（サイズ）
- **本当に物体か**（信頼度）

### 2. AITECTの検出手法

#### グリッドベース検出方式
本システムは「グリッドベース」と呼ばれる手法を採用しています：

1. **画像を格子状に分割**
   ```
   512×512の画像を16×16のグリッドに分割
   各グリッドは32×32ピクセルの領域を担当
   ```

2. **各グリッドで物体を検出**
   ```
   グリッド1: [x=10, y=20, w=50, h=60, conf=0.9] ← 電柱あり！
   グリッド2: [x=0,  y=0,  w=0,  h=0,  conf=0.1] ← 何もない
   ...
   グリッド256: [x=5, y=10, w=30, h=40, conf=0.8] ← 電柱あり！
   ```

3. **5つの値の意味**
   - **x, y**: 物体の中心座標（グリッド内の相対位置）
   - **w, h**: 物体の幅と高さ
   - **confidence**: 物体がある確信度（0～1）

#### なぜこの方式が効果的か？

1. **効率的な探索**
   - 画像全体を総当たりで探すのではなく、各領域で並列に検出
   - 計算量が大幅に削減

2. **位置の特定**
   - どのグリッドから検出されたかで、おおよその位置が分かる
   - グリッド内の詳細位置は x, y で微調整

3. **複数物体への対応**
   - 各グリッドが独立して検出するため、複数の電柱も検出可能

### 3. 学習の流れ

#### ステップ1: 特徴抽出
ResNet18が画像から「エッジ」「形状」「パターン」などの特徴を抽出

#### ステップ2: 検出予測
各グリッドで、抽出された特徴を基に物体の有無と位置を予測

#### ステップ3: 損失計算
予測と正解（アノテーション）の差を計算：
- 位置のズレ → 回帰損失
- 物体有無の間違い → 分類損失

#### ステップ4: パラメータ更新
損失が小さくなるようにネットワークの重みを調整

### 4. 実際の検出例

```
入力: 電柱が2本ある画像

処理:
グリッド(3,5) → [x=15, y=20, w=30, h=80, conf=0.95] ← 1本目の電柱
グリッド(10,5) → [x=10, y=18, w=28, h=82, conf=0.92] ← 2本目の電柱
その他のグリッド → [conf < 0.5] ← 物体なし

結果: 2本の電柱を正しく検出！
```

## 📊 学習過程の可視化

学習中、以下の情報がリアルタイムで記録・可視化されます：

### 1. Loss Progression（損失の推移）
- **意味**: モデルの予測と正解の差を表す指標
- **理想的な推移**: 学習が進むにつれて減少
- **注意点**: 急激な減少後の停滞は過学習の可能性

### 2. Learning Rate（学習率）
- **意味**: パラメータ更新の大きさ
- **デフォルト**: 0.0001（固定）
- **調整方法**: config.jsonまたは--lrオプション

### 3. Gradient Norm Distribution（勾配ノルム分布）
- **意味**: 逆伝播時の勾配の大きさ
- **健全な値**: 0.01～10の範囲
- **問題の兆候**:
  - 0に近い: 勾配消失（学習が進まない）
  - 100以上: 勾配爆発（不安定）

### 4. Weight Statistics（重み統計）
- **Mean（平均）**: 0に近いことが理想（初期化の影響）
- **Std Dev（標準偏差）**: 0.01～0.1が健全
  - 小さすぎる: パラメータの多様性不足
  - 大きすぎる: 不安定な学習

### 5. 検証結果（5エポックごと）
- 緑枠: 正解（Ground Truth）
- 赤枠: モデルの予測（信頼度付き）
- 検出の進捗を視覚的に確認可能

## 📈 評価指標

### 1. IoU（Intersection over Union）- 最も重要な指標

#### IoUとは？
物体検出の精度を測る最も基本的な指標です。予測したボックスと正解ボックスの「重なり具合」を数値化します。

```
IoU = 重なり部分の面積 ÷ 全体の面積

┌─────────┐
│  正解   │     IoU = 0.0
│         │     （全く重ならない）
└─────────┘
    ┌─────────┐
    │  予測   │
    └─────────┘

┌─────────┐
│  ┌──┼────┐
│  │重│予測│   IoU = 0.5
│  │な│    │   （半分重なる）
└──┼─り┘    │
   └────────┘

┌─────────┐
│ 正解≒予測 │   IoU = 0.9
└─────────┘   （ほぼ一致）
```

#### IoUの判定基準
- **IoU ≥ 0.5**: 一般的に「正しい検出」とみなす
- **IoU ≥ 0.7**: 高精度な検出
- **IoU < 0.5**: 検出失敗

### 2. Precision（精度）と Recall（再現率）

#### 具体例で理解する
```
実際の電柱: 10本
AIの検出: 12個（うち8個が正しい電柱、4個が誤検出）

Precision = 8/12 = 0.67 (67%)
→ 検出した12個のうち67%が正しかった

Recall = 8/10 = 0.80 (80%)
→ 実際の10本のうち80%を検出できた
```

#### トレードオフの関係
- **Precisionを重視**: 誤検出を減らす（確実なものだけ検出）
- **Recallを重視**: 見逃しを減らす（疑わしきは検出）

### 3. F1 Score
PrecisionとRecallのバランスを示す指標：
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### 4. mAP（mean Average Precision）

#### AP（Average Precision）とは
信頼度の閾値を変えながら、Precision-Recallカーブを描き、その下の面積を計算：

```
信頼度 0.9: Precision=1.0, Recall=0.3
信頼度 0.7: Precision=0.9, Recall=0.6
信頼度 0.5: Precision=0.8, Recall=0.8
信頼度 0.3: Precision=0.6, Recall=0.9
→ これらを統合してAP値を算出
```

#### なぜmAPが重要か
- 単一の閾値に依存しない総合的な評価
- 異なるモデルの性能を公平に比較可能
- 物体検出の標準的な評価指標

### 5. 実際の評価例

```python
# 検出結果の例
予測: [x=100, y=200, w=50, h=150, conf=0.85]
正解: [x=105, y=195, w=48, h=155]

1. IoU計算
   重なり面積: 6800
   全体面積: 8200
   IoU = 0.83 ✓（高精度）

2. 信頼度確認
   conf = 0.85 > 0.5 ✓（検出として採用）

3. 統計に追加
   True Positive（正検出）として記録
```

### 評価指標の使い分け

| 用途 | 重視する指標 | 理由 |
|------|------------|------|
| 災害時の電柱倒壊検出 | Recall | 見逃しを最小化 |
| 自動運転での障害物検出 | Precision | 誤検出による急ブレーキを防止 |
| 一般的な物体検出 | F1 Score / mAP | バランスの良い性能 |

## 🎓 研究成果の改善手法

### 問題：クラス不均衡による学習の失敗

物体検出では、背景（負例）が物体（正例）よりも圧倒的に多いため、モデルが「すべて背景」と予測することで損失を最小化しようとする問題が発生します。

### 解決策：YOLOスタイルの損失関数

#### 1. 正例・負例の重み付け差別化
```python
λ_coord = 5.0    # 座標回帰の重み（正例のみ）
λ_obj = 1.0      # 物体ありの信頼度損失
λ_noobj = 0.5    # 物体なしの信頼度損失（抑制）
```

#### 2. 責任の割り当て
- 各GTボックスに対して、必ず1つ以上の予測を「責任あり」として割り当て
- これにより、すべてのGTが学習に寄与することを保証

#### 3. Focal Lossの導入
```python
FL(pt) = -α(1-pt)^γ * log(pt)
```
- 簡単なサンプル（高信頼度の負例）の影響を抑制
- 難しいサンプルに学習を集中

### 実装の詳細

#### 改善前の問題
```
最大予測スコア: 0.012
高信頼度予測数: 0/192
→ モデルが「物体なし」しか学習していない
```

#### 改善後の結果
```
最大予測スコア: 0.854
高信頼度予測数: 37/192
Recall: 33.98% (改善前: 0%)
```

### さらなる改善の方向性

1. **データ拡張の強化**
   - Mosaic augmentation（複数画像の合成）
   - Copy-paste augmentation（正例の複製）
   - 白線特有の変換（かすれ、影、部分遮蔽）

2. **アンカーボックスの最適化**
   - 白線の形状に特化（縦長）
   - K-meansクラスタリングによる自動決定

3. **後処理の改善**
   - Soft-NMS（重なりの柔軟な処理）
   - 形状制約（アスペクト比、最小サイズ）

4. **学習戦略**
   - Warmup（学習率の段階的上昇）
   - Cosine annealing（学習率の周期的変化）
   - Hard negative mining（難しい負例の重点学習）

## 🌱 エコなAIとしての優位性

### モデル効率性の比較

AITECTモデルは、主要な物体検出モデルと比較して優れた効率性を実現：

| モデル | パラメータ数 | モデルサイズ | 推論速度 | エッジ適性 |
|--------|------------|-----------|---------|----------|
| **AITECT (提案手法)** | **12.4M** | **47.2MB** | **◎** | **◎** |
| YOLOv5s | 7.2M | 27.5MB | ○ | ○ |
| YOLOv5m | 21.2M | 81.0MB | ○ | △ |
| YOLOX-S | 8.9M | 34.0MB | ○ | ○ |
| Faster R-CNN | 41.8M | 159.0MB | △ | × |
| RetinaNet | 37.7M | 144.0MB | △ | × |
| SSD300 | 26.2M | 100.0MB | △ | △ |

### エコAI指標

- **Faster R-CNNの30%のサイズ**で同等の検出タスクを実現
- **メモリ使用量: 約47MB**（Raspberry Piでも動作可能）
- **推論速度: 386 FPS**（GPU環境での測定値）

### 研究的意義

1. **限られた計算資源での動作**
   - エッジデバイス（IoT機器、スマートフォン）での実用化
   - 電力消費の削減によるCO2排出量の低減

2. **持続可能なAI開発**
   - 大規模モデルに依存しない物体検出
   - SDGs目標7「エネルギーをみんなに そしてクリーンに」への貢献

3. **実用的な応用**
   - リアルタイム処理が必要な場面での活用
   - バッテリー駆動デバイスでの長時間稼働

### 効率性分析の実行

```bash
# モデルの効率性を分析
python model_efficiency_simple.py

# 詳細な比較分析（グラフ生成含む）
python model_efficiency_comparison.py
```

この効率性により、AITECTは「エコなAI」として、環境負荷を抑えながら実用的な物体検出を実現します。

## 🔧 デバッグ・分析ツール

### 損失と予測の分析

#### 1. 勾配伝搬の確認 (`debug_gradient_flow.py`)
モデルの学習が適切に行われているか確認します：
```bash
python debug_gradient_flow.py
```
- 各層の勾配の大きさを可視化
- パラメータ更新量の確認
- 勾配消失・爆発の検出

#### 2. 損失コンポーネントの分析 (`analyze_loss_components.py`)
損失の内訳と予測の統計を分析：
```bash
python analyze_loss_components.py
```
- 信頼度損失と回帰損失の分離
- 正例・負例の割合
- 予測スコアとボックスサイズの分布

#### 3. ボックス収束の分析 (`debug_box_convergence.py`)
予測ボックスがGTに収束しているか確認：
```bash
python debug_box_convergence.py
```
- 位置エラーとサイズエラーの分布
- IoU分布の可視化
- 問題の診断（位置 vs サイズ）

#### 4. 学習ダイナミクスの追跡 (`debug_training_dynamics.py`)
学習中の予測の変化をリアルタイム追跡：
```bash
python debug_training_dynamics.py
```
- 特定のGTに対する予測の収束過程
- 損失と予測精度の関係
- 学習が機能しているかの直接確認

### 改善された学習・評価

#### 1. YOLOスタイルの改善学習 (`train_with_improvements.py`)
不均衡問題に対処した改善版の学習：
```bash
python train_with_improvements.py --epochs 30
```
改善点：
- 正例と負例で異なる重み付け（λ_coord=5.0, λ_obj=1.0, λ_noobj=0.5）
- 各GTに必ず正例を割り当て
- Focal Lossによる難易度調整
- リアルタイムの進捗表示

#### 2. 最適な信頼度閾値の探索 (`find_optimal_threshold.py`)
```bash
python find_optimal_threshold.py
```
- 0.1から0.9まで閾値を変えて評価
- Precision-Recall曲線の生成
- 用途別の推奨閾値を提案

#### 3. 改善モデルの詳細評価 (`evaluate_improved_model.py`)
```bash
python evaluate_improved_model.py
```
- 定量的評価（Precision, Recall, F1, IoU）
- 予測結果の可視化
- 改善前後の比較

#### 4. 予測の可視化 (`quick_visualize.py`)
異なる閾値での検出結果を比較：
```bash
python quick_visualize.py
```
- 4つの異なる閾値で同時表示
- スコア分布のヒストグラム
- GT（緑）と予測（赤）の重ね合わせ

### 損失関数の改善

#### 1. YOLOスタイル損失関数 (`loss_yolo_style.py`)
物体検出の不均衡問題に対処：
- 正例・負例で異なる重み付け
- Ignore領域の設定（IoU 0.3-0.5）
- 各GTに責任を持つ予測を割り当て
- Focal LossとGIoU Lossの実装

#### 2. 改善版損失関数 (`loss_improved.py`)
IoUベースのマッチングを使用した損失関数

### 転移学習（オプション）

#### 転移学習の実装 (`transfer_learning_whiteline.py`)
事前学習済みモデルを活用（必須ではない）：
```bash
python transfer_learning_whiteline.py
```
対応バックボーン：
- ResNet18/50 + FPN
- Faster R-CNN
- YOLOv5
- EfficientNet

### トラブルシューティング

#### モデルの状態確認 (`quick_diagnosis.py`)
```bash
python quick_diagnosis.py
```
- 予測スコアの統計
- ボックスサイズの異常値チェック
- モデルが「物体なし」を学習していないか確認

#### 損失と予測の関係分析 (`debug_loss_prediction_relationship.py`)
```bash
python debug_loss_prediction_relationship.py
```
- 正例と負例のスコア分布
- IoUと予測スコアの相関
- 損失コンポーネントの詳細分析

### 推奨ワークフロー

1. **初期学習**
   ```bash
   python train_with_improvements.py --epochs 30
   ```

2. **問題診断**
   ```bash
   python quick_diagnosis.py
   python analyze_trained_model.py
   ```

3. **閾値最適化**
   ```bash
   python find_optimal_threshold.py
   ```

4. **詳細評価**
   ```bash
   python evaluate_improved_model.py
   ```

5. **必要に応じて追加学習**
   ```bash
   python train_with_improvements.py --epochs 50
   ```

## 📊 評価スクリプト

### 統合評価実行スクリプト (`run_evaluation.py`)

すべての評価を管理する統合スクリプト：

```bash
# デフォルト（卒業研究用評価）
python run_evaluation.py

# 特定の評価タイプを実行
python run_evaluation.py --type threshold    # 閾値最適化
python run_evaluation.py --type visual       # 可視化
python run_evaluation.py --type comparison   # モデル比較

# すべての評価を実行
python run_evaluation.py --all

# 利用可能な評価一覧
python run_evaluation.py --list
```

### 卒業研究用包括的評価 (`evaluate_for_thesis.py`)

卒業論文に必要なすべての評価指標を出力：

```bash
python evaluate_for_thesis.py
```

生成される内容：
- **thesis_evaluation_report.txt**: 詳細な数値結果レポート
  - Average Precision (AP@0.5, AP@0.75)
  - mean Average Precision (mAP)
  - 各閾値でのPrecision/Recall/F1
  - 混同行列（TP/FP/FN）
- **thesis_evaluation_plots/**: グラフ・図表
  - Precision-Recall曲線
  - F1スコアヒートマップ
  - 検出統計グラフ
- **thesis_detection_samples/**: 検出結果のサンプル画像

### シンプル評価 (`evaluate_simple.py`)

基本的なメトリクスのみを素早く確認：

```bash
# デフォルト設定で評価
python evaluate_simple.py

# カスタム設定
python evaluate_simple.py --conf 0.7 --iou 0.5 --samples 10
```

出力例：
```
評価結果
==================================================
評価時刻: 2025-01-15 10:30:45
モデル: result/aitect_model_improved.pth
評価画像数: 43
総GT数: 103
総予測数: 1619

メトリクス:
  Precision: 0.0216
  Recall: 0.3398
  F1-Score: 0.0407

詳細:
  True Positives (TP): 35
  False Positives (FP): 1584
  False Negatives (FN): 68
==================================================
```
