"""
Multi-class pretraining script for lightweight detection models.
Trains on multiple COCO classes to create a robust pretrained backbone
that can be fine-tuned for specific single-class detection tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
import numpy as np
import json
import os
from datetime import datetime
import logging
from tqdm import tqdm
import argparse

from model_lightweight import AITectLightweight, LightweightBackbone
from loss_improved_v2 import ImprovedYOLOLoss
from utils.logger import setup_logger
from utils.metrics import calculate_map
from utils.postprocess import non_max_suppression


class MultiClassPretrainer:
    """Handles pretraining of multi-class detection models."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger('pretrain_multi_class')
        
        # Setup directories
        self.setup_directories()
        
        # Initialize model
        self.model = self.create_model()
        
        # Initialize loss function
        self.criterion = ImprovedYOLOLoss(
            num_classes=config['num_classes'],
            lambda_coord=config.get('lambda_coord', 5.0),
            lambda_noobj=config.get('lambda_noobj', 0.5),
            lambda_obj=config.get('lambda_obj', 1.0),
            lambda_class=config.get('lambda_class', 1.0),
            use_focal_loss=config.get('use_focal_loss', True)
        )
        
        # Setup optimizer with different learning rates
        self.optimizer = self.setup_optimizer()
        
        # Setup scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=config.get('scheduler_T0', 10), 
            T_mult=config.get('scheduler_Tmult', 2)
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_map': [],
            'learning_rates': []
        }
        
    def setup_directories(self):
        """Create necessary directories for saving models and logs."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = f"pretrain_results/multi_class_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.save_dir}/backbone_weights", exist_ok=True)
        
    def create_model(self):
        """Create the multi-class detection model."""
        model = AITectLightweight(
            num_classes=self.config['num_classes'],
            model_size=self.config.get('model_size', 'tiny'),
            pretrained_backbone=self.config.get('pretrained_backbone', True)
        )
        
        # Initialize detection heads with proper weights
        self._initialize_detection_heads(model)
        
        return model.to(self.device)
    
    def _initialize_detection_heads(self, model):
        """Initialize detection head weights properly."""
        for head in model.detection_heads:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def setup_optimizer(self):
        """Setup optimizer with different learning rates for backbone and heads."""
        # Separate parameters
        backbone_params = []
        head_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            elif 'detection_heads' in name:
                head_params.append(param)
            else:
                other_params.append(param)
        
        # Different learning rates
        param_groups = [
            {
                'params': backbone_params, 
                'lr': self.config['backbone_lr'],
                'name': 'backbone'
            },
            {
                'params': head_params, 
                'lr': self.config['head_lr'],
                'name': 'detection_heads'
            },
            {
                'params': other_params, 
                'lr': self.config['other_lr'],
                'name': 'other'
            }
        ]
        
        # Create optimizer
        if self.config.get('optimizer', 'adamw') == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                weight_decay=self.config.get('weight_decay', 0.0005)
            )
        else:
            optimizer = optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 0.0005),
                nesterov=True
            )
        
        return optimizer
    
    def get_coco_dataloaders(self):
        """Create COCO dataloaders for selected classes."""
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Resize((self.config['input_size'], self.config['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.config['input_size'], self.config['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets (you would need to implement COCOSubsetDataset)
        # This is a placeholder - you'll need to implement proper COCO subset loading
        train_dataset = self.create_coco_subset(
            root=self.config['coco_root'],
            annFile=self.config['train_ann_file'],
            transform=train_transform,
            selected_classes=self.config['selected_classes']
        )
        
        val_dataset = self.create_coco_subset(
            root=self.config['coco_root'],
            annFile=self.config['val_ann_file'],
            transform=val_transform,
            selected_classes=self.config['selected_classes']
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def create_coco_subset(self, root, annFile, transform, selected_classes):
        """Create a subset of COCO dataset with selected classes."""
        # This is a simplified implementation
        # In practice, you'd filter COCO annotations for selected classes
        from dataset import ObjectDetectionDataset
        
        # Convert COCO format to your format for selected classes
        # This would require implementing proper COCO filtering
        return ObjectDetectionDataset(
            root_dir=root,
            ann_file=annFile,
            transform=transform,
            num_classes=len(selected_classes)
        )
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Training')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('grad_clip', 5.0)
            )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
            
            # Log learning rates periodically
            if batch_idx % 100 == 0:
                self._log_learning_rates()
        
        return total_loss / num_batches
    
    def validate(self, val_loader, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} - Validation')
            for images, targets in pbar:
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Get predictions
                predictions = self.model.predict(images, conf_threshold=0.3)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate mAP
        map_score = calculate_map(all_predictions, all_targets, self.config['num_classes'])
        
        return avg_loss, map_score
    
    def _log_learning_rates(self):
        """Log current learning rates for each parameter group."""
        for group in self.optimizer.param_groups:
            self.logger.info(f"{group['name']} LR: {group['lr']:.6f}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = f"{self.save_dir}/checkpoints/checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = f"{self.save_dir}/best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
        
        # Save backbone weights separately
        self.save_backbone_weights(epoch, is_best)
    
    def save_backbone_weights(self, epoch, is_best=False):
        """Save only the backbone weights for transfer learning."""
        backbone_state = {}
        
        # Extract backbone weights
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_state[name] = param.data.cpu()
        
        # Save backbone
        backbone_path = f"{self.save_dir}/backbone_weights/backbone_epoch_{epoch}.pth"
        torch.save({
            'backbone_state_dict': backbone_state,
            'backbone_config': {
                'model_type': self.model.backbone.model_type,
                'out_channels': self.model.backbone.out_channels
            },
            'epoch': epoch,
            'num_classes': self.config['num_classes']
        }, backbone_path)
        
        if is_best:
            best_backbone_path = f"{self.save_dir}/backbone_weights/best_backbone.pth"
            torch.save({
                'backbone_state_dict': backbone_state,
                'backbone_config': {
                    'model_type': self.model.backbone.model_type,
                    'out_channels': self.model.backbone.out_channels
                },
                'epoch': epoch,
                'num_classes': self.config['num_classes']
            }, best_backbone_path)
            self.logger.info(f"Saved best backbone to {best_backbone_path}")
    
    def train(self, num_epochs):
        """Main training loop."""
        train_loader, val_loader = self.get_coco_dataloaders()
        
        best_map = 0
        
        for epoch in range(1, num_epochs + 1):
            self.logger.info(f"\nEpoch {epoch}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_map = self.validate(val_loader, epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_map'].append(val_map)
            self.history['learning_rates'].append({
                group['name']: group['lr'] for group in self.optimizer.param_groups
            })
            
            # Log metrics
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Loss: {val_loss:.4f}")
            self.logger.info(f"Val mAP: {val_map:.4f}")
            
            # Save checkpoint
            is_best = val_map > best_map
            if is_best:
                best_map = val_map
            
            if epoch % self.config.get('save_interval', 5) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Save training history
            self.save_history()
    
    def save_history(self):
        """Save training history."""
        history_path = f"{self.save_dir}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def save_config(self):
        """Save training configuration."""
        config_path = f"{self.save_dir}/config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)


def get_common_coco_classes():
    """Get list of common COCO classes for pretraining."""
    # Top 30 most common/useful COCO classes
    common_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
        'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee'
    ]
    
    # COCO class IDs (0-indexed)
    coco_class_ids = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29
    ]
    
    return common_classes, coco_class_ids


def main():
    parser = argparse.ArgumentParser(description='Pretrain multi-class detection model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--model-size', type=str, default='tiny', choices=['tiny', 'small', 'medium'])
    parser.add_argument('--coco-root', type=str, required=True, help='Path to COCO dataset')
    
    args = parser.parse_args()
    
    # Get common COCO classes
    class_names, class_ids = get_common_coco_classes()
    
    # Default configuration
    config = {
        'num_classes': len(class_names),
        'selected_classes': class_ids,
        'class_names': class_names,
        'input_size': 416,
        'batch_size': args.batch_size,
        'model_size': args.model_size,
        'pretrained_backbone': True,
        
        # Learning rates
        'backbone_lr': 1e-4,
        'head_lr': 1e-3,
        'other_lr': 5e-4,
        
        # Training settings
        'num_epochs': args.epochs,
        'weight_decay': 0.0005,
        'grad_clip': 5.0,
        'optimizer': 'adamw',
        
        # Loss settings
        'lambda_coord': 5.0,
        'lambda_noobj': 0.5,
        'lambda_obj': 1.0,
        'lambda_class': 1.0,
        'use_focal_loss': True,
        
        # Scheduler settings
        'scheduler_T0': 10,
        'scheduler_Tmult': 2,
        
        # Data settings
        'coco_root': args.coco_root,
        'train_ann_file': os.path.join(args.coco_root, 'annotations/instances_train2017.json'),
        'val_ann_file': os.path.join(args.coco_root, 'annotations/instances_val2017.json'),
        'num_workers': 4,
        
        # Save settings
        'save_interval': 5
    }
    
    # Load custom config if provided
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
    
    # Create pretrainer
    pretrainer = MultiClassPretrainer(config)
    
    # Save configuration
    pretrainer.save_config()
    
    # Start training
    pretrainer.train(config['num_epochs'])
    
    print(f"\nPretraining completed! Results saved to: {pretrainer.save_dir}")


if __name__ == "__main__":
    main()