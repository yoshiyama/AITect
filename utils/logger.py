import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import torch

class DetailedLogger:
    """詳細なログ記録とモニタリングのためのクラス"""
    def __init__(self, log_dir="logs", experiment_name=None):
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # ログファイルのパス
        self.train_log_path = os.path.join(self.log_dir, "training_log.json")
        self.metrics_log_path = os.path.join(self.log_dir, "metrics_log.json")
        self.config_log_path = os.path.join(self.log_dir, "config.json")
        
        # ログデータの初期化
        self.train_history = {
            'loss': [],
            'learning_rate': [],
            'epoch_time': [],
            'gpu_memory': []
        }
        
        self.validation_history = {
            'loss': [],
            'map_50': [],
            'map_75': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        self.per_class_metrics = defaultdict(lambda: {
            'precision': [],
            'recall': [],
            'f1_score': []
        })
        
        # 損失の内訳
        self.loss_components = {
            'cls_loss': [],
            'reg_loss': [],
            'total_loss': []
        }
        
        # アンカー統計
        self.anchor_stats = {
            'positive_ratio': [],
            'negative_ratio': [],
            'ignore_ratio': [],
            'avg_iou_positive': []
        }
        
    def save_config(self, config):
        """設定を保存"""
        with open(self.config_log_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_training_step(self, epoch, batch_idx, total_batches, loss, 
                         lr, loss_components=None, anchor_stats=None):
        """学習ステップのログ"""
        log_entry = {
            'epoch': epoch,
            'batch': batch_idx,
            'total_batches': total_batches,
            'loss': float(loss),
            'learning_rate': float(lr),
            'timestamp': datetime.now().isoformat()
        }
        
        # GPU使用量
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            log_entry['gpu_memory_gb'] = gpu_memory
        
        # 損失の内訳
        if loss_components:
            log_entry['loss_components'] = loss_components
            for key, value in loss_components.items():
                if key not in self.loss_components:
                    self.loss_components[key] = []
                self.loss_components[key].append(float(value))
        
        # アンカー統計
        if anchor_stats:
            log_entry['anchor_stats'] = anchor_stats
            for key, value in anchor_stats.items():
                if key in self.anchor_stats:
                    self.anchor_stats[key].append(float(value))
        
        # ファイルに追記
        with open(self.train_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_epoch(self, epoch, avg_loss, epoch_time, lr):
        """エポックごとのログ"""
        self.train_history['loss'].append(float(avg_loss))
        self.train_history['learning_rate'].append(float(lr))
        self.train_history['epoch_time'].append(float(epoch_time))
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
            self.train_history['gpu_memory'].append(gpu_memory)
    
    def log_validation(self, epoch, metrics):
        """検証結果のログ"""
        log_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        # 履歴に追加
        for key in ['map_50', 'map_75', 'precision', 'recall', 'f1_score']:
            if key in metrics:
                self.validation_history[key].append(float(metrics[key]))
        
        # ファイルに保存
        with open(self.metrics_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def plot_training_curves(self):
        """学習曲線をプロット"""
        # 損失曲線
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 学習損失
        axes[0, 0].plot(self.train_history['loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # 2. 学習率
        axes[0, 1].plot(self.train_history['learning_rate'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('LR')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # 3. 検証メトリクス
        if self.validation_history['map_50']:
            epochs = range(len(self.validation_history['map_50']))
            axes[1, 0].plot(epochs, self.validation_history['map_50'], label='mAP@0.5')
            axes[1, 0].plot(epochs, self.validation_history['map_75'], label='mAP@0.75')
            axes[1, 0].set_title('Validation mAP')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mAP')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 4. Precision/Recall/F1
        if self.validation_history['f1_score']:
            epochs = range(len(self.validation_history['f1_score']))
            axes[1, 1].plot(epochs, self.validation_history['precision'], label='Precision')
            axes[1, 1].plot(epochs, self.validation_history['recall'], label='Recall')
            axes[1, 1].plot(epochs, self.validation_history['f1_score'], label='F1')
            axes[1, 1].set_title('Validation Metrics')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.close()
    
    def plot_loss_components(self):
        """損失成分をプロット"""
        if not self.loss_components['total_loss']:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(len(self.loss_components['total_loss']))
        for key, values in self.loss_components.items():
            if values:
                ax.plot(iterations[-len(values):], values, label=key)
        
        ax.set_title('Loss Components Over Training')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'loss_components.png'))
        plt.close()
    
    def plot_anchor_statistics(self):
        """アンカー統計をプロット"""
        if not self.anchor_stats['positive_ratio']:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        iterations = range(len(self.anchor_stats['positive_ratio']))
        
        # 1. アンカー割合
        ax1.plot(iterations, self.anchor_stats['positive_ratio'], label='Positive')
        ax1.plot(iterations, self.anchor_stats['negative_ratio'], label='Negative')
        ax1.plot(iterations, self.anchor_stats['ignore_ratio'], label='Ignore')
        ax1.set_title('Anchor Assignment Ratios')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Ratio')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 正例の平均IoU
        ax2.plot(iterations, self.anchor_stats['avg_iou_positive'])
        ax2.set_title('Average IoU of Positive Anchors')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('IoU')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'anchor_statistics.png'))
        plt.close()
    
    def generate_summary_report(self):
        """サマリーレポートを生成"""
        report_path = os.path.join(self.log_dir, 'summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Training Summary Report\n")
            f.write("Generated at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("=" * 80 + "\n\n")
            
            # 学習統計
            f.write("Training Statistics:\n")
            f.write("-" * 40 + "\n")
            if self.train_history['loss']:
                f.write(f"Final Loss: {self.train_history['loss'][-1]:.4f}\n")
                f.write(f"Best Loss: {min(self.train_history['loss']):.4f}\n")
                f.write(f"Average Epoch Time: {np.mean(self.train_history['epoch_time']):.2f}s\n")
                if self.train_history['gpu_memory']:
                    f.write(f"Peak GPU Memory: {max(self.train_history['gpu_memory']):.2f}GB\n")
            
            # 検証統計
            f.write("\n\nValidation Statistics:\n")
            f.write("-" * 40 + "\n")
            if self.validation_history['map_50']:
                f.write(f"Best mAP@0.5: {max(self.validation_history['map_50']):.4f}\n")
                f.write(f"Best mAP@0.75: {max(self.validation_history['map_75']):.4f}\n")
                f.write(f"Best F1 Score: {max(self.validation_history['f1_score']):.4f}\n")
                
                # 最良エポック
                best_f1_epoch = np.argmax(self.validation_history['f1_score']) + 1
                f.write(f"Best F1 Score at Epoch: {best_f1_epoch}\n")
            
            # チューニング推奨事項
            f.write("\n\nTuning Recommendations:\n")
            f.write("-" * 40 + "\n")
            
            # 学習率の推奨
            if self.train_history['loss']:
                recent_loss_trend = np.mean(np.diff(self.train_history['loss'][-10:]))
                if recent_loss_trend > -0.001:
                    f.write("- Learning rate might be too low (loss plateau detected)\n")
                elif recent_loss_trend < -0.1:
                    f.write("- Consider reducing learning rate decay (loss still decreasing rapidly)\n")
            
            # アンカー統計に基づく推奨
            if self.anchor_stats['positive_ratio']:
                avg_pos_ratio = np.mean(self.anchor_stats['positive_ratio'][-100:])
                if avg_pos_ratio < 0.01:
                    f.write("- Very low positive anchor ratio. Consider:\n")
                    f.write("  * Adjusting anchor sizes/ratios\n")
                    f.write("  * Lowering positive IoU threshold\n")
                    f.write("  * Adding more anchor scales\n")
                
                avg_iou = np.mean(self.anchor_stats['avg_iou_positive'][-100:])
                if avg_iou < 0.6:
                    f.write(f"- Low average IoU for positive anchors ({avg_iou:.3f}). Consider:\n")
                    f.write("  * Refining anchor configurations\n")
                    f.write("  * Using anchor optimization tools\n")
            
            # 検証メトリクスに基づく推奨
            if self.validation_history['precision'] and self.validation_history['recall']:
                final_precision = self.validation_history['precision'][-1]
                final_recall = self.validation_history['recall'][-1]
                
                if final_precision < 0.5:
                    f.write("- Low precision. Consider:\n")
                    f.write("  * Increasing confidence threshold\n")
                    f.write("  * Improving NMS threshold\n")
                    f.write("  * Adding hard negative mining\n")
                
                if final_recall < 0.5:
                    f.write("- Low recall. Consider:\n")
                    f.write("  * Decreasing confidence threshold\n")
                    f.write("  * Adding more anchors\n")
                    f.write("  * Data augmentation for small objects\n")
    
    def save_all_plots(self):
        """すべてのプロットを保存"""
        self.plot_training_curves()
        self.plot_loss_components()
        self.plot_anchor_statistics()
        self.generate_summary_report()