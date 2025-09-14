"""
Fine-tuning script for single-class detection using pretrained multi-class backbone.
Implements proper transfer learning strategies including layer freezing,
differential learning rates, and gradual unfreezing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import torchvision.transforms as transforms
import numpy as np
import json
import os
from datetime import datetime
import logging
from tqdm import tqdm
import argparse
from collections import OrderedDict

from model_lightweight import AITectLightweight
from loss_improved_v2 import ImprovedYOLOLoss
from dataset import ObjectDetectionDataset
from utils.logger import setup_logger
from utils.metrics import calculate_map, calculate_precision_recall_f1
from utils.postprocess import non_max_suppression
from utils.visualize import visualize_predictions


class SingleClassFineTuner:
    """Handles fine-tuning of pretrained models for single-class detection."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger('finetune_single_class')
        
        # Setup directories
        self.setup_directories()
        
        # Initialize model with pretrained backbone
        self.model = self.create_model_with_pretrained_backbone()
        
        # Initialize loss function
        self.criterion = ImprovedYOLOLoss(
            num_classes=1,  # Single class
            lambda_coord=config.get('lambda_coord', 5.0),
            lambda_noobj=config.get('lambda_noobj', 0.5),
            lambda_obj=config.get('lambda_obj', 1.0),
            lambda_class=config.get('lambda_class', 0.1),  # Lower for single class
            use_focal_loss=config.get('use_focal_loss', True)
        )
        
        # Setup fine-tuning stages
        self.current_stage = 0
        self.stages = self.setup_finetuning_stages()
        
        # Setup optimizer (will be updated per stage)
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_map': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rates': [],
            'stages': []
        }
    
    def setup_directories(self):
        """Create necessary directories for saving models and logs."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = f"finetune_results/{self.config['class_name']}_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.save_dir}/visualizations", exist_ok=True)
    
    def create_model_with_pretrained_backbone(self):
        """Create model and load pretrained backbone weights."""
        # Create model with single class
        model = AITectLightweight(
            num_classes=1,
            model_size=self.config.get('model_size', 'tiny'),
            pretrained_backbone=False  # We'll load our own pretrained weights
        )
        
        # Load pretrained backbone
        if self.config.get('pretrained_backbone_path'):
            self.logger.info(f"Loading pretrained backbone from: {self.config['pretrained_backbone_path']}")
            self.load_pretrained_backbone(model)
        
        # Initialize new detection heads
        self._initialize_detection_heads(model)
        
        return model.to(self.device)
    
    def load_pretrained_backbone(self, model):
        """Load pretrained backbone weights."""
        checkpoint = torch.load(
            self.config['pretrained_backbone_path'], 
            map_location=self.device
        )
        
        # Extract backbone weights
        backbone_state = checkpoint.get('backbone_state_dict', checkpoint)
        
        # Load backbone weights
        model_state = model.state_dict()
        pretrained_dict = {}
        
        for name, param in backbone_state.items():
            if name in model_state and 'backbone' in name:
                if model_state[name].shape == param.shape:
                    pretrained_dict[name] = param
                else:
                    self.logger.warning(f"Shape mismatch for {name}: "
                                      f"model {model_state[name].shape} vs "
                                      f"pretrained {param.shape}")
        
        # Update model state
        model_state.update(pretrained_dict)
        model.load_state_dict(model_state, strict=False)
        
        self.logger.info(f"Loaded {len(pretrained_dict)} backbone parameters")
    
    def _initialize_detection_heads(self, model):
        """Initialize detection head weights for single-class detection."""
        for head in model.detection_heads:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    # Special initialization for final layer
                    if m.out_channels == head.num_anchors * (5 + 1):  # 5 + 1 class
                        # Initialize objectness to positive bias
                        nn.init.constant_(m.bias[4::6], 1.0)
                        # Initialize class score to neutral
                        nn.init.constant_(m.bias[5::6], 0.0)
                    else:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def setup_finetuning_stages(self):
        """Define fine-tuning stages with different strategies."""
        stages = [
            {
                'name': 'Stage 1: Freeze Backbone, Train Heads',
                'epochs': self.config.get('stage1_epochs', 10),
                'freeze_backbone': True,
                'freeze_fpn': True,
                'learning_rates': {
                    'detection_heads': self.config.get('stage1_head_lr', 1e-3),
                    'fpn': 0,
                    'backbone': 0
                },
                'scheduler': 'cosine'
            },
            {
                'name': 'Stage 2: Unfreeze FPN, Lower LR',
                'epochs': self.config.get('stage2_epochs', 10),
                'freeze_backbone': True,
                'freeze_fpn': False,
                'learning_rates': {
                    'detection_heads': self.config.get('stage2_head_lr', 5e-4),
                    'fpn': self.config.get('stage2_fpn_lr', 1e-4),
                    'backbone': 0
                },
                'scheduler': 'cosine'
            },
            {
                'name': 'Stage 3: Unfreeze All, Very Low LR',
                'epochs': self.config.get('stage3_epochs', 20),
                'freeze_backbone': False,
                'freeze_fpn': False,
                'learning_rates': {
                    'detection_heads': self.config.get('stage3_head_lr', 1e-4),
                    'fpn': self.config.get('stage3_fpn_lr', 5e-5),
                    'backbone': self.config.get('stage3_backbone_lr', 1e-5)
                },
                'scheduler': 'onecycle'
            }
        ]
        
        return stages
    
    def setup_stage_optimizer(self, stage):
        """Setup optimizer for current stage with appropriate freezing and LRs."""
        # Freeze/unfreeze layers
        self._freeze_layers(stage)
        
        # Group parameters
        param_groups = self._get_parameter_groups(stage)
        
        # Create optimizer
        if self.config.get('optimizer', 'adamw') == 'adamw':
            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=self.config.get('weight_decay', 0.0005)
            )
        else:
            self.optimizer = optim.SGD(
                param_groups,
                momentum=0.937,
                weight_decay=self.config.get('weight_decay', 0.0005),
                nesterov=True
            )
        
        # Setup scheduler
        total_steps = stage['epochs'] * self.steps_per_epoch
        
        if stage['scheduler'] == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=[g['lr'] for g in param_groups],
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
        else:  # cosine
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=1e-6
            )
    
    def _freeze_layers(self, stage):
        """Freeze/unfreeze layers based on stage configuration."""
        # Freeze/unfreeze backbone
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = not stage['freeze_backbone']
            elif 'fpn_convs' in name:
                param.requires_grad = not stage['freeze_fpn']
            else:  # detection heads and other layers
                param.requires_grad = True
        
        # Log frozen/unfrozen parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        self.logger.info(f"Stage configuration:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Frozen parameters: {frozen_params:,}")
    
    def _get_parameter_groups(self, stage):
        """Get parameter groups with appropriate learning rates."""
        param_groups = []
        
        # Detection heads
        head_params = []
        for name, param in self.model.named_parameters():
            if 'detection_heads' in name and param.requires_grad:
                head_params.append(param)
        
        if head_params and stage['learning_rates']['detection_heads'] > 0:
            param_groups.append({
                'params': head_params,
                'lr': stage['learning_rates']['detection_heads'],
                'name': 'detection_heads'
            })
        
        # FPN layers
        fpn_params = []
        for name, param in self.model.named_parameters():
            if 'fpn_convs' in name and param.requires_grad:
                fpn_params.append(param)
        
        if fpn_params and stage['learning_rates']['fpn'] > 0:
            param_groups.append({
                'params': fpn_params,
                'lr': stage['learning_rates']['fpn'],
                'name': 'fpn'
            })
        
        # Backbone
        backbone_params = []
        for name, param in self.model.named_parameters():
            if 'backbone' in name and param.requires_grad:
                backbone_params.append(param)
        
        if backbone_params and stage['learning_rates']['backbone'] > 0:
            param_groups.append({
                'params': backbone_params,
                'lr': stage['learning_rates']['backbone'],
                'name': 'backbone'
            })
        
        return param_groups
    
    def get_dataloaders(self):
        """Create dataloaders for fine-tuning."""
        # Data augmentation
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.Resize((self.config['input_size'], self.config['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.config['input_size'], self.config['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = ObjectDetectionDataset(
            root_dir=self.config['train_root'],
            ann_file=self.config['train_ann_file'],
            transform=train_transform,
            num_classes=1
        )
        
        val_dataset = ObjectDetectionDataset(
            root_dir=self.config['val_root'],
            ann_file=self.config['val_ann_file'],
            transform=val_transform,
            num_classes=1
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        self.steps_per_epoch = len(train_loader)
        
        return train_loader, val_loader
    
    def collate_fn(self, batch):
        """Custom collate function for handling variable number of objects."""
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
        
        images = torch.stack(images, 0)
        
        return images, targets
    
    def train_epoch(self, train_loader, epoch, stage_name):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'{stage_name} - Epoch {epoch}')
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
            
            # Update scheduler
            if self.scheduler and hasattr(self.scheduler, 'step'):
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        return total_loss / num_batches
    
    def validate(self, val_loader, epoch, stage_name):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'{stage_name} - Validation')
            for images, targets in pbar:
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Get predictions
                predictions = self.model.predict(
                    images, 
                    conf_threshold=self.config.get('conf_threshold', 0.3)
                )
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        map_score = calculate_map(all_predictions, all_targets, num_classes=1)
        precision, recall, f1 = calculate_precision_recall_f1(
            all_predictions, all_targets, 
            conf_threshold=self.config.get('conf_threshold', 0.3)
        )
        
        # Visualize some predictions
        if epoch % self.config.get('vis_interval', 5) == 0:
            self.visualize_predictions(val_loader, epoch, stage_name)
        
        return avg_loss, map_score, precision, recall, f1
    
    def visualize_predictions(self, val_loader, epoch, stage_name):
        """Visualize predictions on validation set."""
        self.model.eval()
        
        # Get a few samples
        num_samples = min(5, len(val_loader))
        
        with torch.no_grad():
            for i, (images, targets) in enumerate(val_loader):
                if i >= num_samples:
                    break
                
                images = images.to(self.device)
                predictions = self.model.predict(images, conf_threshold=0.3)
                
                # Visualize first image in batch
                save_path = f"{self.save_dir}/visualizations/{stage_name}_epoch_{epoch}_sample_{i}.png"
                visualize_predictions(
                    images[0].cpu(),
                    predictions[0] if predictions else None,
                    targets[0],
                    save_path=save_path,
                    class_names=[self.config['class_name']]
                )
    
    def save_checkpoint(self, epoch, stage_idx, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'stage': self.stages[stage_idx],
            'stage_idx': stage_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'config': self.config
        }
        
        # Save stage checkpoint
        stage_name = self.stages[stage_idx]['name'].replace(' ', '_').replace(':', '')
        checkpoint_path = f"{self.save_dir}/checkpoints/{stage_name}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = f"{self.save_dir}/best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
        
        # Save final model for each stage
        stage = self.stages[stage_idx]
        if epoch == stage['epochs']:
            final_stage_path = f"{self.save_dir}/final_{stage_name}.pth"
            torch.save(checkpoint, final_stage_path)
    
    def finetune(self):
        """Main fine-tuning loop with multiple stages."""
        train_loader, val_loader = self.get_dataloaders()
        
        best_f1 = 0
        total_epochs = 0
        
        # Execute each stage
        for stage_idx, stage in enumerate(self.stages):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Starting {stage['name']}")
            self.logger.info(f"{'='*60}")
            
            # Setup optimizer for this stage
            self.setup_stage_optimizer(stage)
            
            # Train for stage epochs
            for epoch in range(1, stage['epochs'] + 1):
                total_epochs += 1
                
                # Training
                train_loss = self.train_epoch(train_loader, epoch, stage['name'])
                
                # Validation
                val_loss, val_map, precision, recall, f1 = self.validate(
                    val_loader, epoch, stage['name']
                )
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['val_map'].append(val_map)
                self.history['val_precision'].append(precision)
                self.history['val_recall'].append(recall)
                self.history['val_f1'].append(f1)
                self.history['learning_rates'].append({
                    group['name']: group['lr'] 
                    for group in self.optimizer.param_groups
                })
                self.history['stages'].append(stage['name'])
                
                # Log metrics
                self.logger.info(f"\nEpoch {epoch}/{stage['epochs']} - Total Epoch {total_epochs}")
                self.logger.info(f"Train Loss: {train_loss:.4f}")
                self.logger.info(f"Val Loss: {val_loss:.4f}")
                self.logger.info(f"Val mAP: {val_map:.4f}")
                self.logger.info(f"Val Precision: {precision:.4f}")
                self.logger.info(f"Val Recall: {recall:.4f}")
                self.logger.info(f"Val F1: {f1:.4f}")
                
                # Save checkpoint
                is_best = f1 > best_f1
                if is_best:
                    best_f1 = f1
                
                if epoch % self.config.get('save_interval', 5) == 0 or is_best or epoch == stage['epochs']:
                    self.save_checkpoint(total_epochs, stage_idx, is_best)
                
                # Save history
                self.save_history()
        
        self.logger.info(f"\nFine-tuning completed! Best F1: {best_f1:.4f}")
        self.logger.info(f"Results saved to: {self.save_dir}")
    
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


def main():
    parser = argparse.ArgumentParser(description='Fine-tune model for single-class detection')
    parser.add_argument('--pretrained-backbone', type=str, required=True,
                       help='Path to pretrained backbone weights')
    parser.add_argument('--class-name', type=str, required=True,
                       help='Name of the target class')
    parser.add_argument('--train-root', type=str, required=True,
                       help='Root directory for training data')
    parser.add_argument('--train-ann', type=str, required=True,
                       help='Training annotations file')
    parser.add_argument('--val-root', type=str, required=True,
                       help='Root directory for validation data')
    parser.add_argument('--val-ann', type=str, required=True,
                       help='Validation annotations file')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--model-size', type=str, default='tiny',
                       choices=['tiny', 'small', 'medium'])
    parser.add_argument('--config', type=str, help='Path to config file')
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'class_name': args.class_name,
        'pretrained_backbone_path': args.pretrained_backbone,
        'train_root': args.train_root,
        'train_ann_file': args.train_ann,
        'val_root': args.val_root,
        'val_ann_file': args.val_ann,
        'input_size': 416,
        'batch_size': args.batch_size,
        'model_size': args.model_size,
        
        # Stage configurations
        'stage1_epochs': 10,
        'stage1_head_lr': 1e-3,
        
        'stage2_epochs': 10,
        'stage2_head_lr': 5e-4,
        'stage2_fpn_lr': 1e-4,
        
        'stage3_epochs': 20,
        'stage3_head_lr': 1e-4,
        'stage3_fpn_lr': 5e-5,
        'stage3_backbone_lr': 1e-5,
        
        # Training settings
        'weight_decay': 0.0005,
        'grad_clip': 5.0,
        'optimizer': 'adamw',
        
        # Loss settings
        'lambda_coord': 5.0,
        'lambda_noobj': 0.5,
        'lambda_obj': 1.0,
        'lambda_class': 0.1,
        'use_focal_loss': True,
        
        # Evaluation settings
        'conf_threshold': 0.3,
        
        # Save settings
        'save_interval': 5,
        'vis_interval': 5,
        
        # Data settings
        'num_workers': 4
    }
    
    # Load custom config if provided
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
    
    # Create fine-tuner
    finetuner = SingleClassFineTuner(config)
    
    # Save configuration
    finetuner.save_config()
    
    # Start fine-tuning
    finetuner.finetune()


if __name__ == "__main__":
    main()