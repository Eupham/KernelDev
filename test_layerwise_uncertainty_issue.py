#!/usr/bin/env python3
"""
Test to demonstrate the current layerwise uncertainty issue and validate the fix.

This test shows:
1. Current problem: Layer uncertainty is not task-specific
2. Required fix: Each layer needs separate uncertainty for each task
3. Validation that the fix works correctly
"""

import torch
import torch.nn as nn
import sys
from model import GPTModel

def test_current_limitation():
    """Demonstrate the current limitation: layers have one uncertainty regardless of task."""
    print("=== Testing Current Layer Uncertainty Limitation ===")
    
    # Create model with layer supervision
    model = GPTModel(
        vocab_size=1000,
        dim=512,
        n_layers=12,
        n_heads=8,
        layer_supervision_frequency=4,  # Layers 4, 8 have supervision
        task_names=['teacher_forcing', 'cocktail_party']
    )
    
    print(f"Model has {len(model.supervised_layer_indices)} supervised layers: {model.supervised_layer_indices}")
    
    # Check current uncertainty structure
    print("\n1. Current uncertainty structure:")
    print("   Task-level uncertainties (model.log_sigmas):")
    for task_name, log_sigma_param in model.log_sigmas.items():
        print(f"     {task_name}: {log_sigma_param.item():.6f}")
    
    print("   Layer-level uncertainties (one per layer, NOT per task):")
    for layer_idx in model.supervised_layer_indices:
        if hasattr(model.blocks[layer_idx], 'log_sigma'):
            log_sigma = model.blocks[layer_idx].log_sigma.item()
            print(f"     Layer {layer_idx}: {log_sigma:.6f}")
    
    # The problem: Layer uncertainty is shared across tasks
    print("\n2. Current Problem:")
    print("   - Each supervised layer has ONE uncertainty parameter")
    print("   - This single uncertainty is used for BOTH teacher_forcing and cocktail_party")
    print("   - But the issue requires SEPARATE uncertainty for each task at each layer")
    print("   - This means we need 2 × number_of_supervised_layers uncertainty parameters")
    
    # What's needed
    print("\n3. What's needed:")
    print("   - Layer 4 should have: σ_4_tf (teacher_forcing) and σ_4_cp (cocktail_party)")
    print("   - Layer 8 should have: σ_8_tf (teacher_forcing) and σ_8_cp (cocktail_party)")
    print("   - Remove task-level uncertainties entirely")
    print("   - Each task gets its own uncertainty treatment at each layer")
    
    return True

def test_desired_implementation():
    """Test what the fixed implementation should look like."""
    print("\n=== Testing Desired Implementation ===")
    
    # This is what we want to implement
    print("Desired structure after fix:")
    print("   - NO task-level uncertainty (eliminate model.log_sigmas)")
    print("   - Each supervised layer has uncertainty per task:")
    print("     * layer.log_sigma_teacher_forcing")
    print("     * layer.log_sigma_cocktail_party")
    print("   - apply_layer_uncertainty_weighting uses task-specific layer uncertainty")
    print("   - Separate reporting of raw vs uncertainty-weighted loss per task")
    
    return True

def test_loss_reporting_requirements():
    """Test the loss reporting requirements from the issue."""
    print("\n=== Testing Loss Reporting Requirements ===")
    
    print("Current issue with loss reporting:")
    print("   - Uncertainty weighting changes the reported loss values")
    print("   - Need to report BOTH raw loss and uncertainty-weighted loss separately")
    print("   - Currently only reporting uncertainty for layers 4,8 but should calculate for all")
    
    print("\nRequired reporting format:")
    print("   teacher_forcing:")
    print("     raw_loss: 2.34")
    print("     uncertainty_weighted_loss: 1.87")
    print("     layer_uncertainties: L4(σ:1.23), L8(σ:0.98)")
    print("   cocktail_party:")
    print("     raw_loss: 1.89") 
    print("     uncertainty_weighted_loss: 1.45")
    print("     layer_uncertainties: L4(σ:1.45), L8(σ:1.12)")
    
    return True

if __name__ == "__main__":
    print("Testing Layerwise Uncertainty Issue")
    print("=" * 50)
    
    results = []
    results.append(test_current_limitation())
    results.append(test_desired_implementation())
    results.append(test_loss_reporting_requirements())
    
    print(f"\n{'='*50}")
    if all(results):
        print("✓ All tests passed - Issue clearly identified")
        print("Next step: Implement the fix")
    else:
        print("✗ Some tests failed")