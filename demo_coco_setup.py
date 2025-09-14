#!/usr/bin/env python3
"""
COCO軽量物体検出 - クイックスタートデモ
"""

import os
import sys

def main():
    print("="*60)
    print("COCO軽量物体検出 - セットアップガイド")
    print("="*60)
    
    print("\n1. pycocotoolsのインストール確認...")
    try:
        import pycocotools
        print("✅ pycocotools: インストール済み")
    except ImportError:
        print("❌ pycocotools: 未インストール")
        print("実行: pip install pycocotools")
        return
    
    print("\n2. 利用可能なオプション:")
    print("-"*40)
    
    options = [
        {
            "name": "超軽量・人物検出",
            "command": "python train_lightweight_coco.py --classes person --model_size tiny --epochs 20",
            "description": "最も軽量で高速。エッジデバイス向け",
            "params": "~2M",
            "speed": "60+ FPS"
        },
        {
            "name": "マルチクラス検出（5クラス）",
            "command": "python train_lightweight_coco.py --classes person car bicycle dog cat --model_size small",
            "description": "一般的な物体検出用",
            "params": "~11M", 
            "speed": "25-30 FPS"
        },
        {
            "name": "車両検出特化",
            "command": "python train_lightweight_coco.py --classes car truck bus bicycle motorcycle --model_size small",
            "description": "交通監視・自動運転向け",
            "params": "~11M",
            "speed": "30 FPS"
        }
    ]
    
    for i, opt in enumerate(options, 1):
        print(f"\n{i}. {opt['name']}")
        print(f"   用途: {opt['description']}")
        print(f"   モデルサイズ: {opt['params']}")
        print(f"   推論速度: {opt['speed']}")
        print(f"   コマンド: {opt['command']}")
    
    print("\n" + "="*60)
    print("セットアップ手順:")
    print("="*60)
    
    print("\n1. COCOデータセットのダウンロード（初回のみ）:")
    print("   python -c \"from setup_coco_training import setup_coco_dataset; setup_coco_dataset(selected_classes=['person'], max_images=100)\"")
    
    print("\n2. 学習の開始:")
    print("   python train_lightweight_coco.py --classes person --model_size tiny --epochs 10")
    
    print("\n3. 推論テスト:")
    print("   python inference_lightweight.py --model result/coco_person_tiny_final.pth --image test.jpg")
    
    print("\n" + "="*60)
    print("モデルアーキテクチャの比較:")
    print("="*60)
    
    # モデルサイズの比較を表示
    print("\n軽量モデルの比較:")
    from model_lightweight import create_lightweight_model
    
    for size in ["tiny", "small"]:
        print(f"\n--- {size.upper()} モデル ---")
        model = create_lightweight_model(num_classes=1, model_size=size)
        
    print("\n" + "="*60)
    print("推奨される使用方法:")
    print("="*60)
    
    print("\n🚀 初心者向け:")
    print("   - 人物検出のみ、Tinyモデル")
    print("   - 少ないデータで短時間学習")
    
    print("\n💡 実用向け:")
    print("   - 5-10クラス、Smallモデル")
    print("   - 1000枚以上のデータで学習")
    
    print("\n🏆 本格運用:")
    print("   - 必要なクラスを選択")
    print("   - 大規模データで長時間学習")
    print("   - ハイパーパラメータ調整")

if __name__ == "__main__":
    main()