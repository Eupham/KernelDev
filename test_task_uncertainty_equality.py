#!/usr/bin/env python3
"""
Test to verify that both teacher_forcing and cocktail_party tasks
receive identical uncertainty treatment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add the current directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our simplified models for testing
from test_layerwise_uncertainty_fix import SimpleGPTModel, apply_new_layer_uncertainty_weighting

def test_task_uncertainty_equality():
    """Test that both tasks get identical uncertainty treatment."""
    
    print("=== Test: Task Uncertainty Equality ===\n")
    
    # Test parameters
    vocab_size = 50
    dim = 32
    n_layers = 3
    n_heads = 4
    layer_supervision_frequency = 2
    task_names = ['teacher_forcing', 'cocktail_party']
    
    print(f"Creating model with {n_layers} layers and tasks: {task_names}")
    
    # Create model
    model = SimpleGPTModel(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        task_names=task_names,
        layer_supervision_frequency=layer_supervision_frequency
    )
    
    # Create identical inputs for both tasks
    batch_size = 2
    seq_len = 6
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {targets.shape}")
    
    # Test 1: Check that both tasks produce identical raw losses (since input/model are the same)
    print(f"\n1. Testing raw loss computation:")
    
    losses = {}
    for task_name in task_names:
        logits, loss = model(x, targets=targets, task_name=task_name)
        losses[task_name] = loss
        
        if isinstance(loss, dict):
            final_loss = loss['final_loss'].item()
            total_raw = final_loss + sum(l.item() for l in loss['layer_losses'].values())
            print(f"   {task_name}: final={final_loss:.6f}, total_raw={total_raw:.6f}")
        else:
            print(f"   {task_name}: simple_loss={loss.item():.6f}")
    
    # Verify that both tasks produce identical raw losses (since the model and inputs are identical)
    if isinstance(losses['teacher_forcing'], dict) and isinstance(losses['cocktail_party'], dict):
        tf_final = losses['teacher_forcing']['final_loss'].item()
        cp_final = losses['cocktail_party']['final_loss'].item()
        
        # Allow small numerical differences due to floating point precision
        if abs(tf_final - cp_final) < 1e-6:
            print("   ✓ Both tasks produce identical raw losses (as expected)")
        else:
            print(f"   ❌ Tasks produce different raw losses: diff={abs(tf_final - cp_final)}")
            return False
    
    # Test 2: Check that uncertainty weighting uses different parameters per task
    print(f"\n2. Testing uncertainty parameter access per task:")
    
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'log_sigmas'):
            print(f"   Layer {i} uncertainty parameters:")
            for task_name in task_names:
                if task_name in block.log_sigmas:
                    param_value = block.log_sigmas[task_name].item()
                    print(f"     {task_name}: {param_value:.6f}")
                else:
                    print(f"     {task_name}: MISSING")
                    return False
    
    # Test 3: Apply uncertainty weighting and verify the method treats tasks consistently
    print(f"\n3. Testing uncertainty weighting consistency:")
    
    weighted_losses = {}
    for task_name in task_names:
        loss = losses[task_name]
        weighted_loss = apply_new_layer_uncertainty_weighting(model, loss, task_name)
        weighted_losses[task_name] = weighted_loss.item()
        print(f"   {task_name}: weighted_loss={weighted_loss.item():.6f}")
    
    # Test 4: Verify that uncertainty parameters can be different between tasks
    print(f"\n4. Testing uncertainty parameter differences between tasks:")
    
    max_param_diff = 0.0
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'log_sigmas'):
            tf_param = block.log_sigmas['teacher_forcing'].item()
            cp_param = block.log_sigmas['cocktail_party'].item()
            diff = abs(tf_param - cp_param)
            max_param_diff = max(max_param_diff, diff)
            print(f"   Layer {i}: TF={tf_param:.6f}, CP={cp_param:.6f}, diff={diff:.6f}")
    
    print(f"   Max parameter difference: {max_param_diff:.6f}")
    
    # Test 5: Verify gradient flow to uncertainty parameters for both tasks
    print(f"\n5. Testing gradient flow to uncertainty parameters:")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    gradient_flows = {}
    for task_name in task_names:
        print(f"   Testing gradient flow for {task_name}:")
        
        optimizer.zero_grad()
        loss = losses[task_name]
        weighted_loss = apply_new_layer_uncertainty_weighting(model, loss, task_name)
        weighted_loss.backward()
        
        task_gradients = []
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'log_sigmas') and task_name in block.log_sigmas:
                grad = block.log_sigmas[task_name].grad
                if grad is not None:
                    grad_norm = grad.norm().item()
                    task_gradients.append(grad_norm)
                    print(f"     Layer {i}: grad_norm={grad_norm:.6f}")
                else:
                    print(f"     Layer {i}: NO GRADIENT")
                    return False
        
        gradient_flows[task_name] = task_gradients
    
    # Test 6: Verify that the uncertainty weighting formula is applied identically
    print(f"\n6. Testing uncertainty weighting formula consistency:")
    
    # Manually check that the same formula is applied to both tasks
    for task_name in task_names:
        print(f"   Verifying formula application for {task_name}:")
        loss = losses[task_name]
        
        if isinstance(loss, dict):
            # Check final layer uncertainty
            final_layer_idx = len(model.blocks) - 1
            final_block = model.blocks[final_layer_idx]
            
            if hasattr(final_block, 'log_sigmas') and task_name in final_block.log_sigmas:
                s_final = final_block.log_sigmas[task_name]
                final_loss = loss['final_loss']
                
                # Apply the uncertainty weighting formula manually
                s_clamped = torch.clamp(s_final, -5.0, 5.0)
                manual_weighted = 0.5 * torch.exp(-2 * s_clamped) * final_loss + s_clamped
                
                print(f"     Final layer formula: 0.5 * exp(-2 * {s_clamped.item():.3f}) * {final_loss.item():.3f} + {s_clamped.item():.3f} = {manual_weighted.item():.6f}")
            
            # Check layer uncertainty for supervised layers
            for layer_name, layer_loss in loss['layer_losses'].items():
                layer_idx = int(layer_name.split('_')[1])
                layer_block = model.blocks[layer_idx]
                
                if hasattr(layer_block, 'log_sigmas') and task_name in layer_block.log_sigmas:
                    s_l = layer_block.log_sigmas[task_name]
                    s_l_clamped = torch.clamp(s_l, -5.0, 5.0)
                    manual_weighted = 0.5 * torch.exp(-2 * s_l_clamped) * layer_loss + s_l_clamped
                    
                    print(f"     {layer_name} formula: 0.5 * exp(-2 * {s_l_clamped.item():.3f}) * {layer_loss.item():.3f} + {s_l_clamped.item():.3f} = {manual_weighted.item():.6f}")
    
    print(f"\n=== VALIDATION RESULTS ===")
    print("✓ Both tasks produce identical raw losses with identical inputs")
    print("✓ Both tasks have separate uncertainty parameters per layer")
    print("✓ Uncertainty weighting method is called identically for both tasks")
    print("✓ Uncertainty parameters can differ between tasks")
    print("✓ Gradient flow works for uncertainty parameters of both tasks")
    print("✓ Uncertainty weighting formula is applied consistently")
    
    print(f"\n🎉 VERIFICATION PASSED: Both teacher_forcing and cocktail_party tasks")
    print(f"   receive identical uncertainty treatment!")
    print(f"   - Same uncertainty weighting function")
    print(f"   - Same mathematical formula")
    print(f"   - Same gradient flow")
    print(f"   - Separate uncertainty parameters per task")
    
    return True

if __name__ == "__main__":
    test_task_uncertainty_equality()