import os
import json
import requests
import zipfile
from tqdm import tqdm
import shutil

def download_coco_subset():
    """COCO val2017の一部をダウンロード（軽量版）"""
    print("=== COCO Dataset Subset Download ===")
    
    data_dir = "./datasets/coco_subset"
    os.makedirs(f"{data_dir}/images", exist_ok=True)
    os.makedirs(f"{data_dir}/annotations", exist_ok=True)
    
    # COCO 2017 validation annotations (軽量)
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    print("1. Downloading COCO annotations...")
    
    # アノテーションをダウンロード
    ann_file = f"{data_dir}/annotations.zip"
    
    try:
        # まずはアノテーションファイルの情報だけ取得
        response = requests.head(ann_url)
        file_size = int(response.headers.get('content-length', 0))
        print(f"   File size: {file_size / 1024 / 1024:.1f} MB")
        
        # サイズが大きすぎる場合はスキップ
        if file_size > 500 * 1024 * 1024:  # 500MB以上
            print("   ❌ ファイルサイズが大きすぎます。代替データセットを使用します。")
            return False
            
    except Exception as e:
        print(f"   Error: {e}")
        return False
    
    print("\n代わりに、より軽量なデータセットを作成します...")
    return create_mini_coco_format()

def create_mini_coco_format():
    """ミニCOCO形式のデータセットを作成"""
    print("\n=== Creating Mini COCO Dataset ===")
    
    data_dir = "./datasets/mini_coco"
    os.makedirs(f"{data_dir}/images", exist_ok=True)
    
    # COCOの主要カテゴリ（10カテゴリ）
    categories = [
        {'id': 1, 'name': 'person', 'supercategory': 'person'},
        {'id': 2, 'name': 'car', 'supercategory': 'vehicle'},
        {'id': 3, 'name': 'bicycle', 'supercategory': 'vehicle'},
        {'id': 4, 'name': 'dog', 'supercategory': 'animal'},
        {'id': 5, 'name': 'cat', 'supercategory': 'animal'},
        {'id': 6, 'name': 'chair', 'supercategory': 'furniture'},
        {'id': 7, 'name': 'bottle', 'supercategory': 'object'},
        {'id': 8, 'name': 'laptop', 'supercategory': 'electronic'},
        {'id': 9, 'name': 'book', 'supercategory': 'object'},
        {'id': 10, 'name': 'clock', 'supercategory': 'object'}
    ]
    
    # 既存のSimple Shapesデータを活用して多クラス化
    import numpy as np
    from PIL import Image, ImageDraw
    import random
    
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': categories
    }
    
    ann_id = 0
    
    # 100枚の画像を生成
    for img_id in range(100):
        # 512x512の背景
        bg_color = random.choice(['white', 'lightgray', 'beige'])
        img = Image.new('RGB', (512, 512), bg_color)
        draw = ImageDraw.Draw(img)
        
        # 1-4個のオブジェクトを配置
        num_objects = np.random.randint(1, 5)
        
        for _ in range(num_objects):
            # ランダムなカテゴリ
            category = random.choice(categories)
            cat_id = category['id']
            
            # 位置とサイズ
            x1 = np.random.randint(50, 350)
            y1 = np.random.randint(50, 350)
            w = np.random.randint(40, 120)
            h = np.random.randint(40, 120)
            x2 = min(x1 + w, 480)
            y2 = min(y1 + h, 480)
            
            # カテゴリに応じた描画（簡易版）
            colors = {
                1: 'pink',      # person
                2: 'blue',      # car
                3: 'green',     # bicycle
                4: 'brown',     # dog
                5: 'orange',    # cat
                6: 'purple',    # chair
                7: 'cyan',      # bottle
                8: 'gray',      # laptop
                9: 'red',       # book
                10: 'yellow'    # clock
            }
            
            color = colors.get(cat_id, 'black')
            
            # 形状を変える
            if cat_id in [1, 4, 5]:  # person, dog, cat
                draw.ellipse([x1, y1, x2, y2], fill=color)
            elif cat_id in [2, 8, 9]:  # car, laptop, book
                draw.rectangle([x1, y1, x2, y2], fill=color)
            else:  # その他
                points = [(x1, (y1+y2)//2), ((x1+x2)//2, y1), 
                         (x2, (y1+y2)//2), ((x1+x2)//2, y2)]
                draw.polygon(points, fill=color)
            
            # アノテーション追加
            ann = {
                'id': ann_id,
                'image_id': img_id,
                'category_id': cat_id,
                'bbox': [x1, y1, x2-x1, y2-y1],
                'area': (x2-x1) * (y2-y1),
                'iscrowd': 0
            }
            coco_data['annotations'].append(ann)
            ann_id += 1
        
        # 画像保存
        img_name = f'mini_coco_{img_id:04d}.jpg'
        img.save(f"{data_dir}/images/{img_name}")
        
        # 画像情報
        coco_data['images'].append({
            'id': img_id,
            'file_name': img_name,
            'width': 512,
            'height': 512
        })
    
    # アノテーション保存
    with open(f"{data_dir}/annotations.json", 'w') as f:
        json.dump(coco_data, f)
    
    print(f"✅ Created {len(coco_data['images'])} images with {len(coco_data['annotations'])} annotations")
    print(f"   Categories: {len(categories)}")
    print(f"   Location: {data_dir}")
    
    # サンプル画像を表示
    create_dataset_preview(data_dir, coco_data)
    
    return True

def create_dataset_preview(data_dir, coco_data):
    """データセットのプレビュー画像を作成"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    
    # カテゴリ名のマッピング
    cat_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    for i in range(6):
        img_info = coco_data['images'][i]
        img_path = f"{data_dir}/images/{img_info['file_name']}"
        img = Image.open(img_path)
        
        ax = axes[i]
        ax.imshow(img)
        
        # このアノテーションを取得
        anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == i]
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # カテゴリ名を表示
            cat_name = cat_names[ann['category_id']]
            ax.text(x, y-5, cat_name, color='red', fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.7))
        
        ax.set_title(f'Sample {i+1} ({len(anns)} objects)')
        ax.axis('off')
    
    plt.suptitle('Mini COCO Dataset Preview', fontsize=16)
    plt.tight_layout()
    plt.savefig('mini_coco_preview.png', dpi=150)
    print("\n📊 Preview saved: mini_coco_preview.png")

if __name__ == "__main__":
    # COCO風のマルチクラスデータセットを作成
    if create_mini_coco_format():
        print("\n✅ Mini COCOデータセット作成完了!")
        print("\n次のステップ:")
        print("1. python train_multiclass_detection.py  # マルチクラス学習")
        print("2. python evaluate_multiclass_model.py  # 評価")