#!/usr/bin/env python3
"""
Test model initialization improvements.
"""

import torch
import sys
from model import GPTModel

def test_layer_uncertainty_symmetry_breaking():
    """Test that layer uncertainties start with different values in the actual model."""
    print("=== Testing Layer Uncertainty Symmetry Breaking ===")
    
    # Create model with layer supervision
    model = GPTModel(
        vocab_size=1000,
        dim=512,
        n_layers=12,
        n_heads=8,
        layer_supervision_frequency=4,
        task_names=['teacher_forcing', 'cocktail_party']
    )
    
    print(f"Model has {len(model.supervised_layer_indices)} supervised layers: {model.supervised_layer_indices}")
    
    # Check layer uncertainty parameters
    layer_uncertainties = {}
    for layer_idx in model.supervised_layer_indices:
        if hasattr(model.blocks[layer_idx], 'log_sigma'):
            log_sigma = model.blocks[layer_idx].log_sigma.item()
            sigma = torch.exp(model.blocks[layer_idx].log_sigma).item()
            layer_uncertainties[f'layer_{layer_idx}'] = {'log_sigma': log_sigma, 'sigma': sigma}
            print(f"   Layer {layer_idx}: log_sigma = {log_sigma:.6f}, sigma = {sigma:.6f}")
    
    # Check if they are different (symmetry broken)
    log_sigma_values = [params['log_sigma'] for params in layer_uncertainties.values()]
    all_different = len(set([round(v, 6) for v in log_sigma_values])) == len(log_sigma_values)
    
    print(f"\nSymmetry breaking successful: {all_different}")
    if all_different:
        print("✓ Layer uncertainties start with different values")
    else:
        print("✗ Layer uncertainties are still identical")
    
    return all_different

def test_task_level_uncertainties():
    """Test task-level uncertainty initialization."""
    print("\n=== Testing Task-Level Uncertainties ===")
    
    model = GPTModel(
        vocab_size=1000,
        dim=512,
        n_layers=12,
        n_heads=8,
        task_names=['teacher_forcing', 'cocktail_party']
    )
    
    print("Task-level uncertainties:")
    for task_name, log_sigma_param in model.log_sigmas.items():
        log_sigma = log_sigma_param.item()
        sigma = torch.exp(log_sigma_param).item()
        print(f"   {task_name}: log_sigma = {log_sigma:.6f}, sigma = {sigma:.6f}")
    
    # These should still be identical (zeros) since task-level doesn't need symmetry breaking
    return True

if __name__ == "__main__":
    print("Testing Model Initialization Improvements")
    print("=" * 50)
    
    result1 = test_layer_uncertainty_symmetry_breaking()
    result2 = test_task_level_uncertainties()
    
    print(f"\n{'='*50}")
    if result1:
        print("✓ Model initialization improvements working correctly!")
    else:
        print("✗ Model initialization needs further fixes")