# dataset.py

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class CocoDataset(Dataset):
    def __init__(self, image_dir, annotation_path, transform=None):
        self.image_dir = image_dir
        self.annotation_path = annotation_path
        self.transform = transform

        # アノテーション読み込み
        with open(annotation_path) as f:
            coco = json.load(f)

        self.image_info = coco['images']
        self.annotations = coco['annotations']
        self.categories = coco['categories']

        # image_id ごとの bbox 一覧を構築
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        image_data = self.image_info[idx]
        image_id = image_data['id']
        file_name = image_data['file_name']
        # file_nameが既に'JPEGImages/'を含んでいる場合は削除
        if file_name.startswith('JPEGImages/'):
            file_name = file_name[len('JPEGImages/'):]
        img_path = os.path.join(self.image_dir, file_name)
        image = Image.open(img_path).convert("RGB")
        
        # 元画像のサイズを保存
        orig_width, orig_height = image.size

        anns = self.image_id_to_annotations.get(image_id, [])
        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # COCO形式 → [x1, y1, x2, y2]
            labels.append(ann['category_id'])   # 単一クラスでも保持

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # 画像変換を適用
        if self.transform:
            image = self.transform(image)
            
            # 座標変換: 元画像座標 → 変換後座標（512×512）
            if len(boxes) > 0:
                # 変換後の画像サイズ（transformで512×512にリサイズされる）
                new_width = new_height = 512
                
                # スケール係数を計算
                scale_x = new_width / orig_width
                scale_y = new_height / orig_height
                
                # ボックス座標をスケーリング
                boxes[:, [0, 2]] *= scale_x  # x座標
                boxes[:, [1, 3]] *= scale_y  # y座標

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([image_id])}

        return image, target
