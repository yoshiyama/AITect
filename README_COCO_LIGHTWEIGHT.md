# 軽量COCO物体検出モデル - AITect Lightweight

## 概要
COCOデータセットを使用した軽量物体検出モデルの学習システムです。
クラス数を自由に選択でき、エッジデバイスでも動作する軽量モデルを構築できます。

## 特徴
- 🚀 **軽量モデル**: MobileNetV2, ResNet18, ShuffleNetベース
- 🎯 **クラス選択可能**: 1クラス〜80クラスまで自由に選択
- 📊 **COCOデータセット対応**: 標準的なベンチマーク
- ⚡ **高速推論**: エッジデバイス対応

## モデルサイズ

| モデル | バックボーン | パラメータ数 | 推奨用途 |
|--------|------------|------------|---------|
| Tiny | MobileNetV2 | ~2M | エッジ/モバイル |
| Small | ResNet18 | ~11M | 一般用途 |
| Medium | ShuffleNet | ~5M | バランス型 |

## セットアップ

### 1. 依存関係のインストール
```bash
pip install pycocotools tqdm matplotlib
```

### 2. COCOデータセットのダウンロード

#### 人物検出のみ（軽量・推奨）
```python
from setup_coco_training import setup_coco_dataset

# 人物クラスのみ、検証データ100枚
setup_coco_dataset(
    selected_classes=['person'],
    dataset_type='val',
    max_images=100
)
```

#### 複数クラス
```python
# 5クラス（人、車、自転車、犬、猫）
setup_coco_dataset(
    selected_classes=['person', 'car', 'bicycle', 'dog', 'cat'],
    dataset_type='train',
    max_images=1000
)
```

## 学習方法

### 1. 単一クラス検出（人物のみ）
```bash
python train_lightweight_coco.py \
    --classes person \
    --model_size tiny \
    --epochs 30 \
    --batch_size 16
```

### 2. 複数クラス検出
```bash
python train_lightweight_coco.py \
    --classes person car dog \
    --model_size small \
    --epochs 50
```

### 3. カスタム設定
```bash
python train_lightweight_coco.py \
    --classes person bicycle car motorcycle \
    --model_size small \
    --input_size 320 \
    --batch_size 32 \
    --lr 0.001 \
    --epochs 100
```

## 利用可能なCOCOクラス

### 人物・動物
- person（人物）
- bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

### 乗り物
- bicycle, car, motorcycle, airplane, bus, train, truck, boat

### 日用品
- bottle, chair, couch, potted plant, bed, dining table, toilet
- tv, laptop, mouse, remote, keyboard, cell phone

### スポーツ用品
- frisbee, skis, snowboard, sports ball, kite, baseball bat
- baseball glove, skateboard, surfboard, tennis racket

### 食器・食べ物
- fork, knife, spoon, bowl, banana, apple, sandwich, orange
- broccoli, carrot, hot dog, pizza, donut, cake

## 推論方法

```python
import torch
from model_lightweight import create_lightweight_model
from PIL import Image
import torchvision.transforms as T

# モデル読み込み
checkpoint = torch.load('result/coco_person_tiny_final.pth')
model = create_lightweight_model(
    num_classes=checkpoint['num_classes'],
    model_size=checkpoint['model_size']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 画像準備
transform = T.Compose([
    T.Resize((416, 416)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open('test.jpg')
img_tensor = transform(img).unsqueeze(0)

# 推論
with torch.no_grad():
    predictions = model.predict(img_tensor, conf_threshold=0.5)
```

## パフォーマンス目安

| モデル | クラス数 | 推論速度(FPS) | mAP@0.5 |
|--------|---------|--------------|---------|
| Tiny | 1 | ~60 | ~0.45 |
| Tiny | 5 | ~55 | ~0.40 |
| Small | 1 | ~30 | ~0.55 |
| Small | 10 | ~25 | ~0.48 |

※ 値は目安です。実際の性能はハードウェアと実装に依存します。

## Tips

1. **データ量とエポック数**
   - 1クラス: 1000枚, 30エポック
   - 5クラス: 5000枚, 50エポック
   - 10クラス以上: 10000枚以上, 100エポック

2. **メモリ不足の場合**
   - batch_sizeを小さくする（8 or 4）
   - input_sizeを小さくする（320 or 256）
   - model_sizeをtinyにする

3. **精度向上のために**
   - データ拡張を強化
   - 学習率スケジューリング
   - より大きなモデル（small/medium）を使用

## トラブルシューティング

### COCOデータセットが見つからない
```bash
python setup_coco_training.py
```

### CUDA out of memory
- batch_sizeを減らす
- input_sizeを減らす
- num_workersを減らす

### 学習が収束しない
- 学習率を調整（0.0001〜0.01）
- データ拡張を調整
- エポック数を増やす