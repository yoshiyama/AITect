import os
import json
import requests
import zipfile
import tarfile
from tqdm import tqdm
import shutil
from pycocotools.coco import COCO

def download_file(url, dest_path, chunk_size=8192):
    """ファイルをダウンロード（プログレスバー付き）"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def setup_coco_dataset(data_dir="./datasets/coco2017", 
                      download_images=True,
                      dataset_type="val",  # "train" or "val"
                      selected_classes=None,  # None = all classes, or ["person", "car", "dog"]
                      max_images=1000):  # 最大画像数（軽量化のため）
    """
    COCO 2017データセットのセットアップ
    
    Args:
        data_dir: データセット保存先
        download_images: 画像をダウンロードするか
        dataset_type: "train" または "val"
        selected_classes: 使用するクラスのリスト（Noneで全クラス）
        max_images: 最大画像数
    """
    print("=== COCO 2017 Dataset Setup ===")
    
    os.makedirs(f"{data_dir}/annotations", exist_ok=True)
    os.makedirs(f"{data_dir}/{dataset_type}2017", exist_ok=True)
    
    # 1. アノテーションのダウンロード
    ann_url = f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_zip = f"{data_dir}/annotations_trainval2017.zip"
    
    if not os.path.exists(f"{data_dir}/annotations/instances_{dataset_type}2017.json"):
        print(f"\n1. Downloading annotations...")
        download_file(ann_url, ann_zip)
        
        print("   Extracting annotations...")
        with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(ann_zip)
    
    # 2. COCOアノテーションを読み込み
    ann_file = f"{data_dir}/annotations/instances_{dataset_type}2017.json"
    coco = COCO(ann_file)
    
    # 3. クラスの選択
    if selected_classes:
        cat_ids = []
        for class_name in selected_classes:
            cat_id = coco.getCatIds(catNms=[class_name])
            if cat_id:
                cat_ids.extend(cat_id)
        print(f"\nSelected classes: {selected_classes}")
        print(f"Category IDs: {cat_ids}")
    else:
        cat_ids = coco.getCatIds()
        print(f"\nUsing all {len(cat_ids)} classes")
    
    # 4. 画像の選択
    img_ids = []
    for cat_id in cat_ids:
        img_ids.extend(coco.getImgIds(catIds=[cat_id]))
    img_ids = list(set(img_ids))[:max_images]  # 重複削除と制限
    
    print(f"Selected {len(img_ids)} images")
    
    # 5. 画像のダウンロード（オプション）
    if download_images:
        print(f"\n2. Downloading images...")
        img_dir = f"{data_dir}/{dataset_type}2017"
        
        # 既にダウンロード済みの画像をチェック
        existing_images = set(os.listdir(img_dir))
        
        for i, img_id in enumerate(tqdm(img_ids, desc="Downloading")):
            img_info = coco.loadImgs(img_id)[0]
            img_filename = img_info['file_name']
            
            if img_filename in existing_images:
                continue
                
            img_url = img_info['coco_url']
            img_path = os.path.join(img_dir, img_filename)
            
            try:
                response = requests.get(img_url, timeout=30)
                with open(img_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"\nError downloading {img_filename}: {e}")
    
    # 6. フィルタリングされたアノテーションを作成
    create_filtered_annotations(coco, img_ids, cat_ids, data_dir, dataset_type, selected_classes)
    
    return data_dir

def create_filtered_annotations(coco, img_ids, cat_ids, data_dir, dataset_type, selected_classes):
    """選択されたクラスと画像のみのアノテーションファイルを作成"""
    
    print("\n3. Creating filtered annotations...")
    
    # フィルタリングされたデータ
    filtered_data = {
        'info': coco.dataset.get('info', {}),
        'licenses': coco.dataset.get('licenses', []),
        'categories': [],
        'images': [],
        'annotations': []
    }
    
    # カテゴリ情報
    for cat_id in cat_ids:
        cat_info = coco.loadCats(cat_id)[0]
        filtered_data['categories'].append(cat_info)
    
    # 画像とアノテーション
    for img_id in img_ids:
        # 画像情報
        img_info = coco.loadImgs(img_id)[0]
        filtered_data['images'].append(img_info)
        
        # アノテーション
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        filtered_data['annotations'].extend(anns)
    
    # 保存
    suffix = "_".join(selected_classes) if selected_classes else "all"
    output_file = f"{data_dir}/annotations/filtered_{dataset_type}2017_{suffix}.json"
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f)
    
    print(f"Saved filtered annotations: {output_file}")
    print(f"  Images: {len(filtered_data['images'])}")
    print(f"  Annotations: {len(filtered_data['annotations'])}")
    print(f"  Categories: {len(filtered_data['categories'])}")

def setup_lightweight_model_config(num_classes=1, model_size="small"):
    """軽量モデルの設定を作成"""
    
    configs = {
        "tiny": {
            "backbone": "mobilenet_v2",  # 軽量バックボーン
            "grid_size": 13,  # 小さいグリッド
            "num_anchors": 3,  # 少ないアンカー
            "channels": [32, 64, 128, 256],  # 少ないチャンネル
            "params": "~2M parameters"
        },
        "small": {
            "backbone": "resnet18",
            "grid_size": 16,
            "num_anchors": 5,
            "channels": [64, 128, 256, 512],
            "params": "~11M parameters"
        },
        "medium": {
            "backbone": "resnet34",
            "grid_size": 20,
            "num_anchors": 9,
            "channels": [64, 128, 256, 512],
            "params": "~21M parameters"
        }
    }
    
    config = configs[model_size]
    
    training_config = {
        "dataset": {
            "name": "COCO 2017",
            "num_classes": num_classes,
            "input_size": 416,  # 軽量化のため小さめ
        },
        "model": {
            "architecture": "AITect-Lite",
            "size": model_size,
            "backbone": config["backbone"],
            "grid_size": config["grid_size"],
            "num_anchors": config["num_anchors"],
            "channels": config["channels"],
            "num_classes": num_classes,
            "estimated_params": config["params"]
        },
        "training": {
            "batch_size": 16,
            "num_epochs": 50,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "weight_decay": 0.0005,
            "lr_scheduler": {
                "type": "cosine",
                "warmup_epochs": 3
            }
        },
        "augmentation": {
            "horizontal_flip": 0.5,
            "scale_jitter": [0.8, 1.2],
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1
            },
            "mosaic": 0.5  # YOLO風のモザイク拡張
        },
        "loss": {
            "bbox_loss": "ciou",  # Complete IoU loss
            "cls_loss": "focal",
            "obj_loss": "bce",
            "bbox_weight": 5.0,
            "cls_weight": 1.0,
            "obj_weight": 1.0
        }
    }
    
    return training_config

def main():
    """メインセットアップ関数"""
    print("COCO Dataset Setup for Lightweight Object Detection")
    print("="*60)
    
    # 設定
    print("\n利用可能なオプション:")
    print("1. 単一クラス検出 (例: person only)")
    print("2. 少数クラス検出 (例: person, car, dog)")
    print("3. 全クラス検出 (80 classes)")
    
    # 例: 人物検出のみ
    single_class_example = ["person"]
    
    # 例: 複数クラス
    multi_class_example = ["person", "car", "bicycle", "dog", "cat"]
    
    # COCOの主要クラス一覧
    major_classes = {
        "人物": ["person"],
        "乗り物": ["bicycle", "car", "motorcycle", "bus", "truck"],
        "動物": ["bird", "cat", "dog", "horse", "sheep", "cow"],
        "日用品": ["bottle", "chair", "couch", "dining table", "laptop", "cell phone"],
        "食べ物": ["banana", "apple", "sandwich", "orange", "pizza"]
    }
    
    print("\n主要なCOCOクラス:")
    for category, classes in major_classes.items():
        print(f"  {category}: {', '.join(classes)}")
    
    # セットアップ例
    print("\n" + "="*60)
    print("セットアップ例:")
    print("="*60)
    
    # 1. 単一クラス（人物検出）
    print("\n1. 人物検出のみ（軽量・高速）")
    config1 = setup_lightweight_model_config(num_classes=1, model_size="tiny")
    print(f"   モデルサイズ: {config1['model']['estimated_params']}")
    print(f"   入力サイズ: {config1['dataset']['input_size']}x{config1['dataset']['input_size']}")
    print(f"   推奨用途: エッジデバイス、リアルタイム処理")
    
    # 2. 複数クラス
    print("\n2. 5クラス検出（バランス型）")
    config2 = setup_lightweight_model_config(num_classes=5, model_size="small")
    print(f"   モデルサイズ: {config2['model']['estimated_params']}")
    print(f"   クラス: person, car, bicycle, dog, cat")
    print(f"   推奨用途: 一般的な物体検出")
    
    # 3. 実行コマンド
    print("\n" + "="*60)
    print("実行方法:")
    print("="*60)
    print("\n# 1. データセットのダウンロード（人物のみ、100枚）")
    print('python -c "from setup_coco_training import setup_coco_dataset; setup_coco_dataset(selected_classes=[\'person\'], max_images=100)"')
    
    print("\n# 2. 軽量モデルで学習")
    print("python train_lightweight_coco.py --classes person --model_size tiny")
    
    print("\n# 3. 推論テスト")
    print("python inference_lightweight.py --model result/coco_person_tiny.pth --image test.jpg")

if __name__ == "__main__":
    main()