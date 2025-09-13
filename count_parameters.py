#!/usr/bin/env python3
"""
Script to count parameters in AITect models.
Counts total parameters, trainable parameters, and shows breakdown by component.
"""

import torch
import torch.nn as nn
from model import AITECTDetectorV1
from model_v2 import AITECTDetectorV2
from model_v2_improved import AITECTDetectorV2Improved
from model_whiteline import WhiteLineDetector


def count_parameters(model, detailed=True):
    """
    Count parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        detailed: If True, show breakdown by component
    
    Returns:
        dict: Dictionary containing parameter counts
    """
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Breakdown by component
    component_params = {}
    
    if hasattr(model, 'backbone'):
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        component_params['backbone'] = backbone_params
    
    if hasattr(model, 'head'):
        head_params = sum(p.numel() for p in model.head.parameters())
        component_params['head'] = head_params
    
    if hasattr(model, 'adapt_pool'):
        adapt_pool_params = sum(p.numel() for p in model.adapt_pool.parameters())
        component_params['adapt_pool'] = adapt_pool_params
    
    # Count buffers (registered tensors that are not parameters)
    buffer_count = sum(b.numel() for b in model.buffers())
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
        'components': component_params,
        'buffers': buffer_count
    }


def format_number(num):
    """Format large numbers with commas and M/K suffixes."""
    if num >= 1e6:
        return f"{num:,} ({num/1e6:.2f}M)"
    elif num >= 1e3:
        return f"{num:,} ({num/1e3:.2f}K)"
    else:
        return f"{num:,}"


def print_model_summary(model_name, model_class, **kwargs):
    """Print parameter summary for a model."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    # Create model instance
    model = model_class(**kwargs)
    model.eval()
    
    # Count parameters
    param_info = count_parameters(model)
    
    # Print summary
    print(f"Total parameters:      {format_number(param_info['total'])}")
    print(f"Trainable parameters:  {format_number(param_info['trainable'])}")
    print(f"Non-trainable params:  {format_number(param_info['non_trainable'])}")
    print(f"Buffer elements:       {format_number(param_info['buffers'])}")
    
    # Print component breakdown
    if param_info['components']:
        print(f"\nComponent breakdown:")
        for component, count in param_info['components'].items():
            percentage = (count / param_info['total']) * 100
            print(f"  - {component:12s}: {format_number(count)} ({percentage:.1f}%)")
    
    # Model configuration
    print(f"\nModel configuration:")
    print(f"  - Grid size:    {model.grid_size}x{model.grid_size}")
    print(f"  - Num anchors:  {model.num_anchors}")
    print(f"  - Image size:   {model.image_size}x{model.image_size}")
    print(f"  - Output shape: [{model.grid_size * model.grid_size * model.num_anchors}, 5]")
    
    # Memory estimation (assuming float32)
    param_memory_mb = (param_info['total'] * 4) / (1024 * 1024)
    print(f"\nMemory usage (parameters only):")
    print(f"  - Float32: {param_memory_mb:.2f} MB")
    print(f"  - Float16: {param_memory_mb/2:.2f} MB")
    
    return param_info


def main():
    """Main function to count parameters for all AITect models."""
    print("AITect Model Parameter Count Analysis")
    print("=====================================")
    
    # Dictionary to store all model info
    model_configs = {
        "AITECTDetectorV1": {
            "class": AITECTDetectorV1,
            "kwargs": {"num_classes": 1, "grid_size": 16, "num_anchors": 1}
        },
        "AITECTDetectorV2": {
            "class": AITECTDetectorV2,
            "kwargs": {"num_classes": 1, "grid_size": 13, "num_anchors": 3}
        },
        "AITECTDetectorV2Improved": {
            "class": AITECTDetectorV2Improved,
            "kwargs": {"num_classes": 1, "grid_size": 13, "num_anchors": 3}
        },
        "WhiteLineDetector (Default)": {
            "class": WhiteLineDetector,
            "kwargs": {"num_classes": 1, "grid_size": 10, "num_anchors": 1}
        }
    }
    
    # Count parameters for each model
    results = {}
    for model_name, config in model_configs.items():
        param_info = print_model_summary(
            model_name, 
            config["class"], 
            **config["kwargs"]
        )
        results[model_name] = param_info
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("Model Comparison Summary")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Total Params':<15} {'Predictions':<15}")
    print(f"{'-'*60}")
    
    for model_name, config in model_configs.items():
        model = config["class"](**config["kwargs"])
        total_predictions = model.grid_size * model.grid_size * model.num_anchors
        total_params = results[model_name]['total']
        print(f"{model_name:<30} {format_number(total_params):<15} {total_predictions:<15}")
    
    # Find the most efficient model (params per prediction)
    print(f"\n{'='*60}")
    print("Efficiency Analysis (Parameters per Prediction)")
    print(f"{'='*60}")
    
    for model_name, config in model_configs.items():
        model = config["class"](**config["kwargs"])
        total_predictions = model.grid_size * model.grid_size * model.num_anchors
        total_params = results[model_name]['total']
        efficiency = total_params / total_predictions
        print(f"{model_name:<30} {efficiency:,.0f} params/prediction")


if __name__ == "__main__":
    main()