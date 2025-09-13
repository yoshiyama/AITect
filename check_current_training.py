#!/usr/bin/env python3
"""現在の学習状況を確認"""

import torch
import json
import os

# 最新の学習済みモデルを確認
model_paths = [
    "result/aitect_model.pth",
    "result/aitect_model_v2.pth"
]

for path in model_paths:
    if os.path.exists(path):
        print(f"\nモデルファイル: {path}")
        checkpoint = torch.load(path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            print(f"  エポック: {checkpoint.get('epoch', 'unknown')}")
            print(f"  損失: {checkpoint.get('loss', 'unknown')}")
            
            # モデルの構造を確認
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # グリッドサイズを確認
            for key in state_dict.keys():
                if 'grid_x' in key:
                    grid_shape = state_dict[key].shape
                    print(f"  グリッドサイズ: {grid_shape[-1]}x{grid_shape[-1]}")
                    break
                    
            # アンカー数を確認
            for key in state_dict.keys():
                if 'anchor_w' in key:
                    anchor_shape = state_dict[key].shape
                    print(f"  アンカー数: {anchor_shape[1]}")
                    break

# 現在の設定を確認
config_files = ["config.json", "config_v2.json"]
for config_file in config_files:
    if os.path.exists(config_file):
        print(f"\n設定ファイル: {config_file}")
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"  グリッドサイズ: {config['model'].get('grid_size', 'unknown')}")
        print(f"  アンカー数: {config['model'].get('num_anchors', 'unknown')}")
        print(f"  Focal Loss: {config['model'].get('use_focal_loss', 'unknown')}")
        print(f"  信頼度閾値: {config['evaluation'].get('conf_threshold', 'unknown')}")