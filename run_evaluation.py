#!/usr/bin/env python3
"""
物体検出モデルの評価実行スクリプト
使い方:
    python run_evaluation.py                    # デフォルト設定で評価
    python run_evaluation.py --model improved   # 改善モデルの評価
    python run_evaluation.py --quick            # クイック評価（10枚のみ）
    python run_evaluation.py --full             # フル評価（全データ）
"""

import argparse
import subprocess
import os
from datetime import datetime

def run_evaluation(evaluation_type, model_type='improved', quick=False):
    """指定された評価を実行"""
    
    print(f"\n{'='*60}")
    print(f"評価タイプ: {evaluation_type}")
    print(f"モデル: {model_type}")
    print(f"モード: {'クイック' if quick else 'フル'}")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    if evaluation_type == 'thesis':
        # 卒業研究用の包括的評価
        cmd = ['python', 'evaluate_for_thesis.py']
        if quick:
            print("注意: evaluate_for_thesis.pyは常に全データで評価します")
        
    elif evaluation_type == 'threshold':
        # 最適閾値の探索
        cmd = ['python', 'find_optimal_threshold.py']
        
    elif evaluation_type == 'visual':
        # 可視化のみ
        cmd = ['python', 'quick_visualize.py']
        
    elif evaluation_type == 'comparison':
        # 改善前後の比較
        cmd = ['python', 'test_improved_model.py']
        
    elif evaluation_type == 'convergence':
        # ボックス収束の分析
        cmd = ['python', 'debug_box_convergence.py']
        
    elif evaluation_type == 'metrics':
        # 簡易的なメトリクス計算（IoU統計含む）
        cmd = ['python', 'evaluate_simple.py']
        if model_type == 'original':
            cmd.extend(['--model', 'result/aitect_model.pth'])
        else:
            cmd.extend(['--model', 'result/aitect_model_improved.pth'])
        if quick:
            cmd.extend(['--samples', '10'])
        
    else:
        print(f"エラー: 不明な評価タイプ '{evaluation_type}'")
        return
    
    # コマンド実行
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ {evaluation_type}評価が正常に完了しました")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ エラーが発生しました: {e}")
    except FileNotFoundError:
        print(f"\n✗ 評価スクリプトが見つかりません: {' '.join(cmd)}")

def list_available_evaluations():
    """利用可能な評価の一覧を表示"""
    evaluations = {
        'thesis': '卒業研究用の包括的評価（AP, mAP, PR曲線など）',
        'threshold': '最適な信頼度閾値の探索',
        'visual': '検出結果の可視化',
        'comparison': '改善前後のモデル比較',
        'convergence': 'ボックス予測の収束分析',
        'metrics': '基本的なメトリクス評価（IoU統計含む）',
    }
    
    print("\n利用可能な評価タイプ:")
    for key, desc in evaluations.items():
        print(f"  {key:12s} : {desc}")

def check_model_exists(model_type):
    """モデルファイルの存在確認"""
    model_paths = {
        'original': 'result/aitect_model.pth',
        'improved': 'result/aitect_model_improved.pth',
    }
    
    model_path = model_paths.get(model_type)
    if model_path and os.path.exists(model_path):
        return True
    else:
        print(f"\n警告: モデルファイルが見つかりません: {model_path}")
        print("先に学習を実行してください:")
        if model_type == 'improved':
            print("  python train_with_improvements.py --epochs 30")
        else:
            print("  python main.py")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='物体検出モデルの評価実行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python run_evaluation.py                    # デフォルト（thesis評価）
  python run_evaluation.py --type threshold   # 閾値最適化
  python run_evaluation.py --list            # 利用可能な評価一覧
  python run_evaluation.py --all             # すべての評価を実行
        """
    )
    
    parser.add_argument('--type', '-t', 
                       choices=['thesis', 'threshold', 'visual', 'comparison', 'convergence', 'metrics'],
                       default='thesis',
                       help='評価タイプ（デフォルト: thesis）')
    
    parser.add_argument('--model', '-m',
                       choices=['original', 'improved'],
                       default='improved',
                       help='評価するモデル（デフォルト: improved）')
    
    parser.add_argument('--quick', '-q',
                       action='store_true',
                       help='クイック評価（少数サンプルのみ）')
    
    parser.add_argument('--all', '-a',
                       action='store_true',
                       help='すべての評価を実行')
    
    parser.add_argument('--list', '-l',
                       action='store_true',
                       help='利用可能な評価の一覧を表示')
    
    args = parser.parse_args()
    
    # 評価一覧の表示
    if args.list:
        list_available_evaluations()
        return
    
    # モデルの存在確認
    if not check_model_exists(args.model):
        return
    
    # すべての評価を実行
    if args.all:
        print("\nすべての評価を実行します...")
        evaluation_types = ['thesis', 'threshold', 'visual', 'metrics']
        for eval_type in evaluation_types:
            run_evaluation(eval_type, args.model, args.quick)
            print("\n" + "-"*60 + "\n")
    else:
        # 指定された評価を実行
        run_evaluation(args.type, args.model, args.quick)
    
    print("\n評価実行スクリプトが終了しました。")

if __name__ == "__main__":
    main()