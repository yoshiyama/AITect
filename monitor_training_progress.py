import os
import time
import torch
import json
from evaluate_improved_model import evaluate_model

def get_latest_checkpoint():
    """最新のチェックポイントを取得"""
    checkpoints = []
    for f in os.listdir('result'):
        if f.startswith('aitect_model_improved_training_epoch'):
            epoch = int(f.split('epoch')[1].split('.')[0])
            checkpoints.append((epoch, f'result/{f}'))
    
    if checkpoints:
        checkpoints.sort()
        return checkpoints[-1]
    return None, None

def monitor_training():
    """トレーニングの進捗をモニタリング"""
    print("=== Training Progress Monitor ===")
    print("Checking for latest checkpoints...\n")
    
    last_epoch = 0
    
    while True:
        # 最新のチェックポイントを確認
        epoch, checkpoint_path = get_latest_checkpoint()
        
        if epoch and epoch > last_epoch:
            print(f"\n{'='*60}")
            print(f"New checkpoint found: Epoch {epoch}")
            print(f"{'='*60}")
            
            # 評価を実行
            result = evaluate_model(checkpoint_path, 'config_improved_training.json', visualize=False)
            
            if result:
                print(f"\nEpoch {epoch} Summary:")
                print(f"  F1 Score: {result['f1']:.4f}")
                print(f"  Precision: {result['precision']:.4f}")
                print(f"  Recall: {result['recall']:.4f}")
                
                # 改善履歴を保存
                history_file = 'training_progress.json'
                if os.path.exists(history_file):
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                else:
                    history = []
                
                history.append({
                    'epoch': epoch,
                    'f1': result['f1'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'tp': result['tp'],
                    'fp': result['fp'],
                    'fn': result['fn']
                })
                
                with open(history_file, 'w') as f:
                    json.dump(history, f, indent=2)
                
                # 改善の傾向を表示
                if len(history) > 1:
                    prev = history[-2]
                    print(f"\nImprovement from epoch {prev['epoch']}:")
                    print(f"  F1 Score: {prev['f1']:.4f} → {result['f1']:.4f} ({(result['f1']-prev['f1']):.4f})")
            
            last_epoch = epoch
        
        # ベストモデルもチェック
        best_path = 'result/aitect_model_improved_training_best.pth'
        if os.path.exists(best_path):
            mtime = os.path.getmtime(best_path)
            if not hasattr(monitor_training, 'best_mtime') or mtime > monitor_training.best_mtime:
                monitor_training.best_mtime = mtime
                print(f"\n{'*'*60}")
                print("Best model updated!")
                print(f"{'*'*60}")
                result = evaluate_model(best_path, 'config_improved_training.json', visualize=False)
                if result:
                    print(f"Best Model Performance:")
                    print(f"  F1 Score: {result['f1']:.4f}")
                    print(f"  Precision: {result['precision']:.4f}")
                    print(f"  Recall: {result['recall']:.4f}")
        
        # トレーニングが終了したかチェック
        if os.path.exists('result/aitect_model_improved_training.pth'):
            print("\n" + "="*60)
            print("Training completed!")
            print("="*60)
            print("\nFinal model evaluation:")
            evaluate_model('result/aitect_model_improved_training.pth', 'config_improved_training.json', visualize=True)
            break
        
        print("\nWaiting for next checkpoint... (Press Ctrl+C to stop)")
        time.sleep(60)  # 1分ごとにチェック

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")