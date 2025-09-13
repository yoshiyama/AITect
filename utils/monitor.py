import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import json

class TrainingMonitor:
    """学習過程を監視・可視化するクラス"""
    
    def __init__(self, log_dir="result/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # ログファイルの準備
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.json")
        self.plot_dir = os.path.join(log_dir, f"plots_{timestamp}")
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # 記録用リスト
        self.losses = []
        self.learning_rates = []
        self.gradient_norms = []
        self.weight_stats = []
        self.gpu_memory = []
        
    def check_gpu_status(self):
        """GPU使用状況をチェック"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3   # GB
            return {
                "available": True,
                "name": gpu_name,
                "memory_allocated_gb": round(gpu_memory_allocated, 2),
                "memory_reserved_gb": round(gpu_memory_reserved, 2)
            }
        else:
            return {"available": False, "name": "CPU"}
    
    def print_gpu_info(self):
        """GPU情報を表示"""
        gpu_info = self.check_gpu_status()
        print("\n=== GPU情報 ===")
        if gpu_info["available"]:
            print(f"✓ GPU使用中: {gpu_info['name']}")
            print(f"  メモリ使用量: {gpu_info['memory_allocated_gb']} GB / {gpu_info['memory_reserved_gb']} GB")
        else:
            print("✗ CPU使用中（GPUが利用できません）")
        print("===============\n")
        
    def calculate_gradient_norm(self, model):
        """勾配のノルムを計算"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def get_weight_statistics(self, model):
        """重みの統計情報を取得"""
        all_weights = []
        for p in model.parameters():
            if p.requires_grad:
                all_weights.extend(p.data.cpu().numpy().flatten())
        
        weights_array = np.array(all_weights)
        return {
            "mean": float(np.mean(weights_array)),
            "std": float(np.std(weights_array)),
            "min": float(np.min(weights_array)),
            "max": float(np.max(weights_array))
        }
    
    def log_iteration(self, epoch, batch_idx, total_batches, loss, model, optimizer):
        """イテレーションごとのログ記録"""
        # 勾配ノルム計算
        grad_norm = self.calculate_gradient_norm(model)
        self.gradient_norms.append(grad_norm)
        
        # GPU メモリ
        gpu_info = self.check_gpu_status()
        if gpu_info["available"]:
            self.gpu_memory.append(gpu_info["memory_allocated_gb"])
        
        # 進捗表示
        progress = (batch_idx + 1) / total_batches * 100
        print(f"\r[Epoch {epoch}] 進捗: {progress:>5.1f}% | "
              f"損失: {loss:.4f} | 勾配ノルム: {grad_norm:.2f}", end="")
    
    def log_epoch(self, epoch, avg_loss, model, optimizer):
        """エポックごとのログ記録"""
        self.losses.append(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)
        
        # 重み統計
        weight_stats = self.get_weight_statistics(model)
        self.weight_stats.append(weight_stats)
        
        # コンソール出力
        print(f"\n[Epoch {epoch}] 平均損失: {avg_loss:.4f} | 学習率: {current_lr:.6f}")
        print(f"  重み統計 - 平均: {weight_stats['mean']:.4f}, "
              f"標準偏差: {weight_stats['std']:.4f}, "
              f"最小: {weight_stats['min']:.4f}, 最大: {weight_stats['max']:.4f}")
        
        # グラフ保存（5エポックごと）
        if epoch % 5 == 0:
            self.save_plots(epoch)
            print(f"  グラフ保存: {os.path.join(self.plot_dir, f'training_progress_epoch_{epoch}.png')}")
        
        # JSONログ保存
        self.save_json_log(epoch, avg_loss, current_lr, weight_stats)
    
    def save_plots(self, epoch):
        """学習過程のグラフを保存"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss progression
        axes[0, 0].plot(self.losses, 'b-', linewidth=2)
        axes[0, 0].set_title('Loss Progression', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[0, 1].plot(self.learning_rates, 'g-', linewidth=2)
        axes[0, 1].set_title('Learning Rate', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient norm distribution
        if len(self.gradient_norms) > 0:
            recent_grads = self.gradient_norms[-100:]  # Last 100
            axes[1, 0].hist(recent_grads, bins=20, alpha=0.7, color='orange')
            axes[1, 0].set_title('Gradient Norm Distribution (Recent)', fontsize=14)
            axes[1, 0].set_xlabel('Gradient Norm')
            axes[1, 0].set_ylabel('Frequency')
        
        # Weight statistics
        if len(self.weight_stats) > 0:
            means = [w['mean'] for w in self.weight_stats]
            stds = [w['std'] for w in self.weight_stats]
            epochs = range(1, len(means) + 1)
            
            axes[1, 1].plot(epochs, means, 'r-', label='Mean', linewidth=2)
            axes[1, 1].fill_between(epochs, 
                                   np.array(means) - np.array(stds),
                                   np.array(means) + np.array(stds),
                                   alpha=0.3, color='red')
            axes[1, 1].set_title('Weight Mean and Std Dev', fontsize=14)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plot_dir, f'training_progress_epoch_{epoch}.png')
        plt.savefig(plot_path, dpi=100)
        plt.close()
    
    def save_json_log(self, epoch, loss, lr, weight_stats):
        """JSONログファイルに保存"""
        log_entry = {
            "epoch": epoch,
            "loss": loss,
            "learning_rate": lr,
            "weight_stats": weight_stats,
            "gpu_info": self.check_gpu_status(),
            "timestamp": datetime.now().isoformat()
        }
        
        # 既存のログを読み込み
        logs = []
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
        
        logs.append(log_entry)
        
        # 保存
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def print_summary(self):
        """学習終了時のサマリー表示"""
        print("\n\n=== 学習完了サマリー ===")
        print(f"総エポック数: {len(self.losses)}")
        print(f"最小損失: {min(self.losses):.4f} (エポック {self.losses.index(min(self.losses)) + 1})")
        print(f"最終損失: {self.losses[-1]:.4f}")
        print(f"ログ保存先: {self.log_file}")
        print(f"グラフ保存先: {self.plot_dir}")
        print("======================")
        
        # 最終的な損失曲線を保存
        self.save_final_loss_curve()
    
    def save_final_loss_curve(self):
        """最終的な損失曲線を単独で保存"""
        if len(self.losses) == 0:
            return
            
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.losses) + 1)
        
        # 損失曲線をプロット
        plt.plot(epochs, self.losses, 'b-', linewidth=2, marker='o', markersize=4)
        
        # 最小値をマーク
        min_loss_idx = self.losses.index(min(self.losses))
        plt.plot(min_loss_idx + 1, self.losses[min_loss_idx], 'r*', markersize=15, 
                label=f'Min Loss: {self.losses[min_loss_idx]:.4f} (Epoch {min_loss_idx + 1})')
        
        plt.title('Training Loss Curve', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # y軸を対数スケールに（損失が急激に減少する場合に見やすい）
        if max(self.losses) / min(self.losses) > 100:
            plt.yscale('log')
            plt.ylabel('Loss (log scale)', fontsize=14)
        
        plt.tight_layout()
        
        # 保存
        final_plot_path = os.path.join(self.plot_dir, 'final_loss_curve.png')
        plt.savefig(final_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n最終損失曲線を保存: {final_plot_path}")