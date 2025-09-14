#!/usr/bin/env python3
"""
Simple multi-class pretraining script using the existing COCO dataset loader.
Trains on multiple COCO classes to create a robust pretrained backbone.
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

# Import existing components
from train_lightweight_coco import COCODatasetLightweight, collate_fn
from model_lightweight import create_lightweight_model
from simple_loss import SimpleLoss as ImprovedDetectionLoss
from utils.postprocess import postprocess_predictions


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')
    for images, targets in pbar:
        images = images.to(device)
        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Convert targets to the format expected by loss function
        batch_targets = []
        for t in targets:
            batch_targets.append({
                'boxes': t['boxes'],
                'labels': t['labels']
            })
        
        # Compute loss
        total_loss_value = criterion(outputs, batch_targets)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss_value.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += total_loss_value.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{total_loss_value.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}'
        })
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    total_loss = 0
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
            
            # Convert targets
            batch_targets = []
            for t in targets:
                batch_targets.append({
                    'boxes': t['boxes'],
                    'labels': t['labels']
                })
            
            # Compute loss
            total_loss_value = criterion(outputs, batch_targets)
            
            total_loss += total_loss_value.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{total_loss_value.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
    
    return total_loss / num_batches


def get_common_coco_classes():
    """Get list of common COCO classes for pretraining."""
    # Top 20 most common/useful COCO classes
    common_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'backpack', 'umbrella', 'handbag', 'bottle'
    ]
    
    return common_classes


def save_backbone_weights(model, save_path):
    """Save only the backbone weights."""
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


def main():
    parser = argparse.ArgumentParser(description='Pretrain multi-class detection model')
    parser.add_argument('--data_dir', default='./datasets/coco2017', help='COCO dataset directory')
    parser.add_argument('--model_size', default='small', choices=['tiny', 'small', 'medium'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_dir', default='./pretrain_results', help='Save directory')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"{args.save_dir}/multi_class_{args.model_size}_{timestamp}"
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
        pin_memory=True
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
    
    # Loss function
    criterion = ImprovedDetectionLoss(num_classes=num_classes)
    
    # Optimizer with different learning rates
    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for backbone
        {'params': other_params, 'lr': args.lr}
    ], weight_decay=0.0005)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Training loop
    print("\n=== Starting Pretraining ===")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, epoch)
        val_losses.append(val_loss)
        
        # Update scheduler
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save complete model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'classes': selected_classes,
                'num_classes': num_classes,
                'model_size': args.model_size
            }, f"{save_dir}/best_model.pth")
            
            # Save backbone weights only
            save_backbone_weights(
                model, 
                f"{save_dir}/checkpoints/pretrained_multi_class_{args.model_size}_best.pth"
            )
            
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Regular checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'classes': selected_classes,
                'num_classes': num_classes,
                'model_size': args.model_size
            }, f"{save_dir}/checkpoint_epoch_{epoch}.pth")
    
    # Save final model
    save_backbone_weights(
        model,
        f"{save_dir}/checkpoints/pretrained_multi_class_{args.model_size}_final.pth"
    )
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Pretraining Progress - {args.model_size.upper()} model on {num_classes} classes')
    plt.legend()
    plt.savefig(f"{save_dir}/training_curve.png")
    plt.close()
    
    # Save training summary
    summary = {
        'model_size': args.model_size,
        'num_classes': num_classes,
        'classes': selected_classes,
        'epochs': args.epochs,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'save_dir': save_dir
    }
    
    with open(f"{save_dir}/training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Pretraining Completed ===")
    print(f"Results saved to: {save_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"\nTo use for fine-tuning:")
    print(f"--pretrained_path {save_dir}/checkpoints/pretrained_multi_class_{args.model_size}_best.pth")


if __name__ == "__main__":
    main()