import os
import torch
import torchvision
from torchvision import transforms
import requests
import zipfile
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def setup_coco_subset():
    """COCO val2017の一部をダウンロード（車両検出用）"""
    print("=== COCO Dataset Setup ===")
    
    # COCOの車両関連カテゴリ
    vehicle_categories = {
        3: 'car',
        6: 'bus', 
        8: 'truck',
        4: 'motorcycle',
        2: 'bicycle'
    }
    
    # torchvisionのCOCO APIを使用
    try:
        from pycocotools.coco import COCO
        
        # アノテーションファイルをダウンロード
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        
        print("Note: COCO datasetのセットアップには時間がかかります。")
        print("代わりに軽量なデータセットを推奨します。")
        
    except ImportError:
        print("pycocotoolsがインストールされていません。")
        print("pip install pycocotools")
    
    return False

def setup_pascal_voc():
    """Pascal VOC 2007をセットアップ（一般的な物体検出用）"""
    print("\n=== Pascal VOC 2007 Dataset Setup ===")
    
    data_dir = "./datasets/voc2007"
    os.makedirs(data_dir, exist_ok=True)
    
    # VOCデータセットをダウンロード
    try:
        voc_dataset = torchvision.datasets.VOCDetection(
            root=data_dir,
            year='2007',
            image_set='val',
            download=True
        )
        
        print(f"VOC 2007 validation set: {len(voc_dataset)} images")
        
        # サンプル表示
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i in range(6):
            img, target = voc_dataset[i]
            ax = axes[i]
            ax.imshow(img)
            
            # バウンディングボックスを描画
            objects = target['annotation']['object']
            if not isinstance(objects, list):
                objects = [objects]
                
            for obj in objects:
                bbox = obj['bndbox']
                x1 = int(bbox['xmin'])
                y1 = int(bbox['ymin'])
                x2 = int(bbox['xmax'])
                y2 = int(bbox['ymax'])
                
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1-5, obj['name'], color='red', fontsize=10)
            
            ax.set_title(f'Sample {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('voc_samples.png')
        print("VOCサンプル画像: voc_samples.png")
        
        return voc_dataset
        
    except Exception as e:
        print(f"VOCダウンロードエラー: {e}")
        return None

def setup_simple_shape_dataset():
    """簡単な図形検出データセット（デモ用）を作成"""
    print("\n=== Simple Shape Detection Dataset ===")
    
    import numpy as np
    from PIL import Image, ImageDraw
    
    data_dir = "./datasets/simple_shapes"
    os.makedirs(f"{data_dir}/images", exist_ok=True)
    os.makedirs(f"{data_dir}/annotations", exist_ok=True)
    
    annotations = []
    
    # 50枚のシンプルな画像を生成
    for i in range(50):
        # 512x512の白背景
        img = Image.new('RGB', (512, 512), 'white')
        draw = ImageDraw.Draw(img)
        
        boxes = []
        
        # ランダムに1-3個の図形を配置
        num_objects = np.random.randint(1, 4)
        
        for _ in range(num_objects):
            # ランダムな位置とサイズ
            x1 = np.random.randint(50, 350)
            y1 = np.random.randint(50, 350)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            x2 = min(x1 + w, 480)
            y2 = min(y1 + h, 480)
            
            # ランダムな色
            color = np.random.choice(['red', 'blue', 'green', 'yellow', 'black'])
            
            # 矩形か楕円をランダムに選択
            if np.random.random() > 0.5:
                draw.rectangle([x1, y1, x2, y2], fill=color)
            else:
                draw.ellipse([x1, y1, x2, y2], fill=color)
            
            boxes.append({
                'bbox': [x1, y1, x2, y2],
                'category_id': 1,  # すべて同じカテゴリ
                'area': (x2-x1) * (y2-y1)
            })
        
        # 画像保存
        img_name = f'shape_{i:04d}.png'
        img.save(f"{data_dir}/images/{img_name}")
        
        # アノテーション追加
        annotations.append({
            'file_name': img_name,
            'width': 512,
            'height': 512,
            'annotations': boxes
        })
    
    # COCO形式のJSONを作成
    coco_format = {
        'images': [
            {
                'id': i,
                'file_name': ann['file_name'],
                'width': ann['width'],
                'height': ann['height']
            }
            for i, ann in enumerate(annotations)
        ],
        'annotations': [
            {
                'id': ann_id,
                'image_id': img_id,
                'category_id': box['category_id'],
                'bbox': [box['bbox'][0], box['bbox'][1], 
                        box['bbox'][2]-box['bbox'][0], 
                        box['bbox'][3]-box['bbox'][1]],  # COCO format: [x,y,w,h]
                'area': box['area'],
                'iscrowd': 0
            }
            for img_id, ann in enumerate(annotations)
            for ann_id, box in enumerate(ann['annotations'])
        ],
        'categories': [
            {'id': 1, 'name': 'shape', 'supercategory': 'object'}
        ]
    }
    
    with open(f"{data_dir}/annotations.json", 'w') as f:
        json.dump(coco_format, f)
    
    print(f"Created {len(annotations)} simple shape images")
    
    # サンプル表示
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    
    for i in range(6):
        img = Image.open(f"{data_dir}/images/shape_{i:04d}.png")
        ax = axes[i]
        ax.imshow(img)
        
        # アノテーション表示
        for box in annotations[i]['annotations']:
            x1, y1, x2, y2 = box['bbox']
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
        
        ax.set_title(f'Sample {i+1} ({len(annotations[i]["annotations"])} objects)')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_shapes_samples.png')
    print("サンプル画像: simple_shapes_samples.png")
    
    return data_dir

def download_wider_face_subset():
    """WIDER FACE データセットの一部（顔検出用）"""
    print("\n=== WIDER FACE Dataset Info ===")
    print("WIDER FACEは顔検出用データセットです。")
    print("ダウンロードURL: http://shuoyang1213.me/WIDERFACE/")
    print("※サイズが大きいため、手動ダウンロードを推奨")
    
def suggest_datasets():
    """推奨データセット一覧"""
    print("\n=== 推奨される公開データセット ===")
    
    datasets = {
        "1. Pascal VOC": {
            "用途": "一般物体検出（20クラス）",
            "サイズ": "~1GB",
            "難易度": "中",
            "URL": "http://host.robots.ox.ac.uk/pascal/VOC/"
        },
        "2. COCO": {
            "用途": "一般物体検出（80クラス）",
            "サイズ": "~20GB",
            "難易度": "高",
            "URL": "https://cocodataset.org/"
        },
        "3. KITTI": {
            "用途": "自動運転（車、歩行者）",
            "サイズ": "~10GB",
            "難易度": "高",
            "URL": "http://www.cvlibs.net/datasets/kitti/"
        },
        "4. Cityscapes": {
            "用途": "都市景観（道路、車）",
            "サイズ": "~50GB",
            "難易度": "高",
            "URL": "https://www.cityscapes-dataset.com/"
        },
        "5. Simple Shapes": {
            "用途": "基本検証用",
            "サイズ": "生成可能",
            "難易度": "低",
            "URL": "ローカル生成"
        }
    }
    
    print("\nデータセット一覧:")
    for name, info in datasets.items():
        print(f"\n{name}")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    return datasets

if __name__ == "__main__":
    print("公開データセットのセットアップ")
    print("="*50)
    
    # 推奨データセット表示
    suggest_datasets()
    
    # 自動的にSimple Shapesを生成
    print("\n\n自動的にSimple Shapesデータセットを生成します...")
    shape_dir = setup_simple_shape_dataset()
    print("\n✅ Simple Shapesデータセット生成完了")
    print(f"データ場所: {shape_dir}")
    
    # Pascal VOCも試す
    print("\n\nPascal VOC 2007のダウンロードを試みます...")
    voc_dataset = setup_pascal_voc()
    if voc_dataset:
        print("\n✅ Pascal VOCのセットアップ完了")