#!/usr/bin/env python3
"""
Multi-class pretraining with Focal Detection Loss
Improved version with proper object detection loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Import existing components
from train_lightweight_coco import COCODatasetLightweight, collate_fn
from model_lightweight import create_lightweight_model
from detection_loss_focal import FocalDetectionLoss
from utils.postprocess import postprocess_predictions


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch with detailed loss tracking"""
    model.train()
    total_loss = 0
    loss_components = {'obj': 0, 'bbox': 0, 'cls': 0}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')
    for images, targets in pbar:
        images = images.to(device)
        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss, losses = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        for k, v in losses.items():
            loss_components[k] += v
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'obj': f'{losses["obj"]:.4f}',
            'bbox': f'{losses["bbox"]:.4f}',
            'cls': f'{losses["cls"]:.4f}'
        })
    
    # Average losses
    avg_loss = total_loss / num_batches
    for k in loss_components:
        loss_components[k] /= num_batches
    
    return avg_loss, loss_components


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    total_loss = 0
    loss_components = {'obj': 0, 'bbox': 0, 'cls': 0}
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Validation')
        for images, targets in pbar:
            images = images.to(device)
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                t['labels'] = t['labels'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss, losses = criterion(outputs, targets)
            
            total_loss += loss.item()
            for k, v in losses.items():
                loss_components[k] += v
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'obj': f'{losses["obj"]:.4f}',
                'bbox': f'{losses["bbox"]:.4f}',
                'cls': f'{losses["cls"]:.4f}'
            })
    
    # Average losses
    avg_loss = total_loss / num_batches
    for k in loss_components:
        loss_components[k] /= num_batches
    
    return avg_loss, loss_components


def get_common_coco_classes():
    """Get list of common COCO classes for pretraining"""
    # Top 20 most common/useful COCO classes
    common_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'backpack', 'umbrella', 'handbag', 'bottle'
    ]
    
    return common_classes


def save_backbone_weights(model, save_path):
    """Save only the backbone and FPN weights"""
    backbone_state = {}
    fpn_state = {}
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_state[name] = param.data.cpu()
        elif 'fpn' in name:
            fpn_state[name] = param.data.cpu()
    
    torch.save({
        'backbone_state_dict': backbone_state,
        'fpn_state_dict': fpn_state,
        'model_size': model.model_size,
        'pretrained_classes': get_common_coco_classes()
    }, save_path)
    
    print(f"Saved backbone weights to {save_path}")


def plot_training_history(history, save_dir):
    """Plot and save training history"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Objectness loss
    axes[0, 1].plot(epochs, history['train_obj_loss'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_obj_loss'], 'r-', label='Val')
    axes[0, 1].set_title('Objectness Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Bbox loss
    axes[1, 0].plot(epochs, history['train_bbox_loss'], 'b-', label='Train')
    axes[1, 0].plot(epochs, history['val_bbox_loss'], 'r-', label='Val')
    axes[1, 0].set_title('Bounding Box Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Classification loss
    axes[1, 1].plot(epochs, history['train_cls_loss'], 'b-', label='Train')
    axes[1, 1].plot(epochs, history['val_cls_loss'], 'r-', label='Val')
    axes[1, 1].set_title('Classification Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_history.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Pretrain multi-class detection model with Focal Loss')
    parser.add_argument('--data_dir', default='./datasets/coco2017', help='COCO dataset directory')
    parser.add_argument('--model_size', default='small', choices=['tiny', 'small', 'medium'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_dir', default='./pretrain_focal_results', help='Save directory')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"{args.save_dir}/multi_class_focal_{args.model_size}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    
    # Get common classes
    selected_classes = get_common_coco_classes()
    print(f"\nPretraining with {len(selected_classes)} classes:")
    print(f"Classes: {', '.join(selected_classes)}")
    
    # Check if COCO data exists
    train_ann = f"{args.data_dir}/annotations/instances_train2017.json"
    val_ann = f"{args.data_dir}/annotations/instances_val2017.json"
    
    if not os.path.exists(train_ann):
        print(f"\nCOCO annotations not found at {train_ann}")
        print("\nPlease run first:")
        print("python setup_coco_training.py")
        return
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = COCODatasetLightweight(
        f"{args.data_dir}/train2017",
        train_ann,
        target_classes=selected_classes,
        input_size=416
    )
    
    val_dataset = COCODatasetLightweight(
        f"{args.data_dir}/val2017",
        val_ann,
        target_classes=selected_classes,
        input_size=416
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True  # Important for batch normalization
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    print(f"\nCreating {args.model_size} model...")
    num_classes = len(selected_classes)
    model = create_lightweight_model(
        num_classes=num_classes,
        model_size=args.model_size,
        pretrained=True
    ).to(device)
    
    # Loss function with Focal Loss
    criterion = FocalDetectionLoss(
        num_classes=num_classes,
        focal_alpha=0.25,
        focal_gamma=2.0,
        bbox_weight=5.0,
        obj_weight=1.0,
        cls_weight=1.0
    )
    
    # Optimizer with different learning rates
    backbone_params = []
    fpn_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'fpn' in name:
            fpn_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},   # Lower LR for pretrained backbone
        {'params': fpn_params, 'lr': args.lr * 0.5},        # Medium LR for FPN
        {'params': head_params, 'lr': args.lr}              # Full LR for detection heads
    ], weight_decay=0.0005)
    
    # Scheduler with warm restart
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_obj_loss': [], 'val_obj_loss': [],
        'train_bbox_loss': [], 'val_bbox_loss': [],
        'train_cls_loss': [], 'val_cls_loss': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    print("\n=== Starting Training with Focal Detection Loss ===")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_losses = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_losses = validate(model, val_loader, criterion, device, epoch)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_obj_loss'].append(train_losses['obj'])
        history['val_obj_loss'].append(val_losses['obj'])
        history['train_bbox_loss'].append(train_losses['bbox'])
        history['val_bbox_loss'].append(val_losses['bbox'])
        history['train_cls_loss'].append(train_losses['cls'])
        history['val_cls_loss'].append(val_losses['cls'])
        history['learning_rates'].append(current_lr)
        
        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train - Total: {train_loss:.4f}, Obj: {train_losses['obj']:.4f}, BBox: {train_losses['bbox']:.4f}, Cls: {train_losses['cls']:.4f}")
        print(f"Val   - Total: {val_loss:.4f}, Obj: {val_losses['obj']:.4f}, BBox: {val_losses['bbox']:.4f}, Cls: {val_losses['cls']:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save complete model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'classes': selected_classes,
                'num_classes': num_classes,
                'model_size': args.model_size
            }, f"{save_dir}/best_model.pth")
            
            # Save backbone weights only
            save_backbone_weights(
                model, 
                f"{save_dir}/checkpoints/pretrained_focal_{args.model_size}_best.pth"
            )
            
            print(f"✅ Saved best model (val_loss: {val_loss:.4f})")
        
        # Regular checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history
            }, f"{save_dir}/checkpoint_epoch_{epoch}.pth")
            
            # Plot training history
            plot_training_history(history, save_dir)
    
    # Save final model
    save_backbone_weights(
        model,
        f"{save_dir}/checkpoints/pretrained_focal_{args.model_size}_final.pth"
    )
    
    # Final plots
    plot_training_history(history, save_dir)
    
    # Save training summary
    summary = {
        'model_size': args.model_size,
        'num_classes': num_classes,
        'classes': selected_classes,
        'epochs': args.epochs,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_loss,
        'final_val_loss': val_loss,
        'final_losses': {
            'train': train_losses,
            'val': val_losses
        },
        'save_dir': save_dir
    }
    
    with open(f"{save_dir}/training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Training Completed ===")
    print(f"Results saved to: {save_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"\nLoss breakdown (final epoch):")
    print(f"  Objectness: {val_losses['obj']:.4f}")
    print(f"  Bounding Box: {val_losses['bbox']:.4f}")
    print(f"  Classification: {val_losses['cls']:.4f}")
    print(f"\nTo use for fine-tuning:")
    print(f"--pretrained_path {save_dir}/checkpoints/pretrained_focal_{args.model_size}_best.pth")


if __name__ == "__main__":
    main()