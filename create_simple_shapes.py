import os
import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_simple_shapes_dataset():
    """簡単な図形検出データセットを作成"""
    print("=== Simple Shape Detection Dataset Creation ===")
    
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
        'annotations': [],
        'categories': [
            {'id': 1, 'name': 'shape', 'supercategory': 'object'}
        ]
    }
    
    # アノテーションを追加
    ann_id = 0
    for img_id, ann in enumerate(annotations):
        for box in ann['annotations']:
            coco_format['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': box['category_id'],
                'bbox': [box['bbox'][0], box['bbox'][1], 
                        box['bbox'][2]-box['bbox'][0], 
                        box['bbox'][3]-box['bbox'][1]],  # COCO format: [x,y,w,h]
                'area': box['area'],
                'iscrowd': 0
            })
            ann_id += 1
    
    with open(f"{data_dir}/annotations.json", 'w') as f:
        json.dump(coco_format, f)
    
    print(f"Created {len(annotations)} simple shape images")
    print(f"Total annotations: {len(coco_format['annotations'])}")
    
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

if __name__ == "__main__":
    data_dir = create_simple_shapes_dataset()
    print(f"\n✅ データセット作成完了: {data_dir}")