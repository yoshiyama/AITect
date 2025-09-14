import json
import glob
import os

def check_latest_log():
    # 最新のログディレクトリを見つける
    log_dirs = glob.glob("logs/improved_v2_*")
    if not log_dirs:
        print("No log directories found")
        return
    
    latest_dir = max(log_dirs, key=os.path.getmtime)
    print(f"Checking logs in: {latest_dir}")
    
    # training_log.jsonから最新のエントリを読む
    log_file = os.path.join(latest_dir, "training_log.json")
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                # 最後の数行を解析
                print("\n=== Latest Training Stats ===")
                for line in lines[-5:]:
                    data = json.loads(line)
                    print(f"Epoch {data['epoch']}, Batch {data['batch']}/{data['total_batches']}:")
                    print(f"  Total Loss: {data['loss']:.4f}")
                    if 'loss_components' in data:
                        print(f"  - Cls Loss: {data['loss_components']['cls_loss']:.4f}")
                        print(f"  - Reg Loss: {data['loss_components']['reg_loss']:.4f}")
                    if 'anchor_stats' in data:
                        print(f"  - Positive ratio: {data['anchor_stats']['positive_ratio']:.4f}")
                        print(f"  - Negative ratio: {data['anchor_stats']['negative_ratio']:.4f}")
                        print(f"  - Avg IoU (positive): {data['anchor_stats']['avg_iou_positive']:.4f}")
                    print()

if __name__ == "__main__":
    check_latest_log()