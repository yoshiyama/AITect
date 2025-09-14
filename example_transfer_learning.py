"""
Example script demonstrating the complete transfer learning pipeline:
1. Pretrain on multiple COCO classes
2. Fine-tune for single-class detection
"""

import torch
import json
import os
from datetime import datetime

from pretrain_multi_class import MultiClassPretrainer
from finetune_single_class import SingleClassFineTuner
from model_lightweight import create_lightweight_model


def example_pretraining():
    """Example of pretraining on multiple COCO classes."""
    print("="*60)
    print("STEP 1: Pretraining on Multiple COCO Classes")
    print("="*60)
    
    # Configuration for pretraining
    pretrain_config = {
        'num_classes': 30,  # 30 common COCO classes
        'selected_classes': list(range(30)),  # First 30 COCO classes
        'input_size': 416,
        'batch_size': 32,
        'model_size': 'tiny',
        'pretrained_backbone': True,
        
        # Learning rates
        'backbone_lr': 1e-4,
        'head_lr': 1e-3,
        'other_lr': 5e-4,
        
        # Training settings
        'num_epochs': 50,  # Reduced for example
        'weight_decay': 0.0005,
        'grad_clip': 5.0,
        'optimizer': 'adamw',
        
        # Loss settings
        'lambda_coord': 5.0,
        'lambda_noobj': 0.5,
        'lambda_obj': 1.0,
        'lambda_class': 1.0,
        'use_focal_loss': True,
        
        # Data settings (update these paths)
        'coco_root': '/path/to/coco/dataset',
        'train_ann_file': '/path/to/coco/annotations/instances_train2017.json',
        'val_ann_file': '/path/to/coco/annotations/instances_val2017.json',
        'num_workers': 4,
        
        # Save settings
        'save_interval': 5
    }
    
    # Create pretrainer
    # pretrainer = MultiClassPretrainer(pretrain_config)
    # pretrainer.save_config()
    # pretrainer.train(pretrain_config['num_epochs'])
    
    print("\nPretraining completed (example - not actually run)")
    print("Pretrained backbone saved to: pretrain_results/multi_class_*/backbone_weights/best_backbone.pth")
    
    # Return path to pretrained backbone (example)
    return "pretrain_results/multi_class_example/backbone_weights/best_backbone.pth"


def example_finetuning(pretrained_backbone_path):
    """Example of fine-tuning for single-class detection."""
    print("\n" + "="*60)
    print("STEP 2: Fine-tuning for Single-Class Detection")
    print("="*60)
    
    # Configuration for fine-tuning
    finetune_config = {
        'class_name': 'whiteline',
        'pretrained_backbone_path': pretrained_backbone_path,
        'train_root': 'datasets/inaoka/train/JPEGImages',
        'train_ann_file': 'datasets/inaoka/train/annotations.json',
        'val_root': 'datasets/inaoka/val',
        'val_ann_file': 'datasets/inaoka/val/annotations.json',
        'input_size': 416,
        'batch_size': 16,
        'model_size': 'tiny',
        
        # Stage configurations
        'stage1_epochs': 5,
        'stage1_head_lr': 1e-3,
        
        'stage2_epochs': 5,
        'stage2_head_lr': 5e-4,
        'stage2_fpn_lr': 1e-4,
        
        'stage3_epochs': 10,
        'stage3_head_lr': 1e-4,
        'stage3_fpn_lr': 5e-5,
        'stage3_backbone_lr': 1e-5,
        
        # Training settings
        'weight_decay': 0.0005,
        'grad_clip': 5.0,
        'optimizer': 'adamw',
        
        # Loss settings (adjusted for single class)
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
    
    # Create fine-tuner
    # finetuner = SingleClassFineTuner(finetune_config)
    # finetuner.save_config()
    # finetuner.finetune()
    
    print("\nFine-tuning completed (example - not actually run)")
    print("Fine-tuned model saved to: finetune_results/whiteline_*/best_model.pth")


def example_inference():
    """Example of using the fine-tuned model for inference."""
    print("\n" + "="*60)
    print("STEP 3: Using Fine-tuned Model for Inference")
    print("="*60)
    
    # Load fine-tuned model
    model = create_lightweight_model(
        num_classes=1,
        model_size='tiny',
        pretrained=False
    )
    
    # Load fine-tuned weights (example path)
    # checkpoint = torch.load('finetune_results/whiteline_example/best_model.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
    
    print("\nModel loaded and ready for inference!")
    
    # Example inference code
    print("\nExample inference code:")
    print("""
    # Prepare image
    image = transform(PIL_image).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        predictions = model.predict(image, conf_threshold=0.5)
    
    # Process predictions
    for pred in predictions:
        boxes = pred['boxes']
        scores = pred['scores']
        labels = pred['labels']
        
        # Draw boxes on image
        for box, score in zip(boxes, scores):
            print(f"Detected object with confidence {score:.2f}")
    """)


def demonstrate_model_capabilities():
    """Demonstrate the new model capabilities."""
    print("\n" + "="*60)
    print("Model Capabilities Demonstration")
    print("="*60)
    
    # Create model
    model = create_lightweight_model(num_classes=1, model_size='tiny')
    
    print("\n1. Loading pretrained backbone:")
    # model.load_backbone_weights('path/to/backbone.pth')
    print("   - Backbone weights loaded successfully")
    
    print("\n2. Getting parameter groups for optimizer:")
    param_groups = model.get_param_groups(
        backbone_lr=1e-5,
        fpn_lr=1e-4,
        head_lr=1e-3
    )
    for group in param_groups:
        print(f"   - {group['name']}: {len(group['params'])} parameters, LR={group['lr']}")
    
    print("\n3. Freezing/unfreezing layers:")
    model.freeze_backbone(True)
    print("   - Backbone frozen")
    model.freeze_fpn(True)
    print("   - FPN frozen")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   - Trainable: {trainable_params:,} / Total: {total_params:,}")
    
    print("\n4. Replacing detection heads:")
    model.replace_detection_heads(num_classes=10)
    print("   - Detection heads replaced for 10 classes")
    
    print("\n5. Model architecture summary:")
    print(f"   - Backbone: {model.backbone.model_type}")
    print(f"   - Model size: {model.model_size}")
    print(f"   - Number of classes: {model.num_classes}")


def print_transfer_learning_tips():
    """Print helpful tips for transfer learning."""
    print("\n" + "="*60)
    print("Transfer Learning Best Practices")
    print("="*60)
    
    tips = [
        "1. Start with a frozen backbone to prevent catastrophic forgetting",
        "2. Use different learning rates: backbone < FPN < detection heads",
        "3. Gradually unfreeze layers from top to bottom",
        "4. Monitor validation metrics closely to avoid overfitting",
        "5. Use data augmentation, but less aggressive than training from scratch",
        "6. Save checkpoints frequently during fine-tuning",
        "7. Consider using EMA (Exponential Moving Average) for stability",
        "8. Fine-tune batch normalization statistics in later stages",
        "9. Use gradient clipping to prevent training instability",
        "10. Validate on diverse data to ensure generalization"
    ]
    
    for tip in tips:
        print(f"\n{tip}")


def main():
    """Main function demonstrating the complete pipeline."""
    print("Transfer Learning Pipeline for Object Detection")
    print("=" * 60)
    
    # Step 1: Pretrain on multiple classes
    pretrained_backbone_path = example_pretraining()
    
    # Step 2: Fine-tune for single class
    example_finetuning(pretrained_backbone_path)
    
    # Step 3: Use for inference
    example_inference()
    
    # Demonstrate model capabilities
    demonstrate_model_capabilities()
    
    # Print tips
    print_transfer_learning_tips()
    
    print("\n" + "="*60)
    print("Pipeline demonstration completed!")
    print("="*60)
    
    print("\nTo actually run the pipeline:")
    print("1. Update data paths in the configuration")
    print("2. Uncomment the training code sections")
    print("3. Run: python example_transfer_learning.py")
    
    print("\nOr use the scripts directly:")
    print("- Pretraining: python pretrain_multi_class.py --coco-root /path/to/coco --epochs 100")
    print("- Fine-tuning: python finetune_single_class.py --pretrained-backbone path/to/backbone.pth \\")
    print("               --class-name whiteline --train-root datasets/inaoka/train/JPEGImages \\")
    print("               --train-ann datasets/inaoka/train/annotations.json")


if __name__ == "__main__":
    main()