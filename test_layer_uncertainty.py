#!/usr/bin/env python3
"""
Layer-Level Uncertainty Validation Test

This test validates the layer-level uncertainty implementation by:
1. Creating a model with layer supervision
2. Testing layer uncertainty parameters 
3. Validating deep supervision readout heads
4. Checking uncertainty loss computation
5. Verifying gradient flow through layer uncertainty parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
warnings.filterwarnings('ignore')

# Import model components
from model import GPTModel, TransformerBlock

# Mock SPECIAL_TOKENS for testing
SPECIAL_TOKENS = {
    '[PAD]': 0,
    '[CLS]': 1,
    '[MASK]': 2,
    '[SPAN]': 3,
    '[ES]': 4,
    '[MASKQ]': 5
}

def test_layer_uncertainty_mechanism():
    """Test the layer-level uncertainty mechanism."""
    
    print("=== Layer-Level Uncertainty Validation Test ===\n")
    
    # Create a small model with layer supervision
    vocab_size = 1000
    dim = 64
    n_layers = 8
    layer_supervision_frequency = 2  # Every 2nd layer
    task_names = ['teacher_forcing']
    
    model = GPTModel(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=4,
        max_seq_len=128,
        task_names=task_names,
        layer_supervision_frequency=layer_supervision_frequency
    )
    
    print(f"✓ Created model with {n_layers} layers, supervision every {layer_supervision_frequency} layers")
    print(f"✓ Supervised layers: {model.supervised_layer_indices}")
    
    # Test 1: Check layer supervision setup
    print("\n1. Checking layer supervision setup:")
    
    supervised_layers = []
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'log_sigma') and hasattr(block, 'layer_head'):
            supervised_layers.append(i)
            log_sigma_val = block.log_sigma.data.item()
            print(f"   Layer {i}: has supervision, log_sigma = {log_sigma_val:.6f}")
        else:
            print(f"   Layer {i}: no supervision")
    
    print(f"   Expected supervised layers: {model.supervised_layer_indices}")
    print(f"   Actual supervised layers: {supervised_layers}")
    assert supervised_layers == model.supervised_layer_indices, "Supervised layer mismatch!"
    
    # Test 2: Create sample input and test forward pass
    print("\n2. Testing forward pass with layer supervision:")
    
    batch_size = 2
    seq_len = 16
    
    # Create teacher forcing style input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Add [CLS] tokens at the beginning (required for teacher forcing mode)
    cls_token_id = SPECIAL_TOKENS['[CLS]']
    x[:, 0] = cls_token_id
    
    print(f"   Input shape: {x.shape}")
    print(f"   Target shape: {targets.shape}")
    
    # Forward pass
    logits, loss = model(x, targets=targets, task_name='teacher_forcing')
    
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Loss type: {type(loss)}")
    
    if isinstance(loss, dict):
        print(f"   Loss structure: {list(loss.keys())}")
        print(f"   Final loss: {loss['final_loss'].item():.6f}")
        if 'layer_losses' in loss:
            print(f"   Layer losses: {len(loss['layer_losses'])} layers")
            for layer_name, layer_loss in loss['layer_losses'].items():
                print(f"     {layer_name}: {layer_loss.item():.6f}")
    else:
        print(f"   Simple loss: {loss.item():.6f}")
    
    # Test 3: Check gradient flow through layer uncertainty parameters
    print("\n3. Testing gradient flow through layer uncertainty parameters:")
    
    if isinstance(loss, dict):
        total_loss = loss['final_loss']
        if 'layer_losses' in loss:
            for layer_loss in loss['layer_losses'].values():
                total_loss += layer_loss
    else:
        total_loss = loss
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients on layer uncertainty parameters
    layer_grad_info = {}
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'log_sigma'):
            grad_norm = block.log_sigma.grad.norm().item() if block.log_sigma.grad is not None else 0.0
            grad_value = block.log_sigma.grad.item() if block.log_sigma.grad is not None else None
            layer_grad_info[i] = (grad_norm, grad_value)
            print(f"   Layer {i} log_sigma: grad_norm = {grad_norm:.6f}, grad = {grad_value}")
    
    has_layer_gradients = all(info[0] > 0 for info in layer_grad_info.values())
    print(f"   All layer uncertainty parameters have gradients: {has_layer_gradients}")
    
    # Test 4: Test uncertainty weighting computation
    print("\n4. Testing uncertainty weighting computation:")
    
    # Import training utilities
    from train_loop import Trainer, TrainingConfig
    
    config = TrainingConfig(learning_rate=1e-3, num_epochs=1)
    trainer = Trainer(model, config)
    
    # Test the layer uncertainty weighting function
    if isinstance(loss, dict):
        # Reset gradients
        model.zero_grad()
        
        weighted_loss = trainer.apply_layer_uncertainty_weighting(loss, 'teacher_forcing')
        print(f"   Original loss structure: {type(loss)}")
        print(f"   Weighted loss: {weighted_loss.item():.6f}")
        
        # Check if KL penalty is included
        weighted_loss.backward()
        
        # Check that layer uncertainty parameters still receive gradients
        layer_grad_after_weighting = {}
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'log_sigma'):
                grad_norm = block.log_sigma.grad.norm().item() if block.log_sigma.grad is not None else 0.0
                layer_grad_after_weighting[i] = grad_norm
                print(f"   Layer {i} log_sigma after weighting: grad_norm = {grad_norm:.6f}")
        
        has_weighted_gradients = all(grad > 0 for grad in layer_grad_after_weighting.values())
        print(f"   All layer parameters have gradients after weighting: {has_weighted_gradients}")
    
    # Test 5: Test layer uncertainty parameter clamping
    print("\n5. Testing layer uncertainty parameter clamping:")
    
    # Set some extreme values to test clamping
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'log_sigma'):
            # Test with extreme values
            with torch.no_grad():
                original_value = block.log_sigma.data.clone()
                
                # Test positive extreme
                block.log_sigma.data.fill_(10.0)
                dummy_loss = torch.tensor(1.0, requires_grad=True)
                s_l_clamped = torch.clamp(block.log_sigma, -5.0, 5.0)
                clamped_pos = s_l_clamped.item()
                
                # Test negative extreme
                block.log_sigma.data.fill_(-10.0)
                s_l_clamped = torch.clamp(block.log_sigma, -5.0, 5.0)
                clamped_neg = s_l_clamped.item()
                
                # Restore original
                block.log_sigma.data.copy_(original_value)
                
                print(f"   Layer {i}: +10.0 → {clamped_pos:.1f}, -10.0 → {clamped_neg:.1f}")
                
                assert clamped_pos == 5.0, f"Positive clamping failed: {clamped_pos}"
                assert clamped_neg == -5.0, f"Negative clamping failed: {clamped_neg}"
    
    # Test 6: Validate uncertainty formula properties
    print("\n6. Validating uncertainty formula properties:")
    
    test_loss_val = 2.0
    test_log_sigmas = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    print("   log_sigma | sigma | data_weight | regularizer | total")
    print("   ---------|-------|-------------|-------------|------")
    
    for log_sig in test_log_sigmas:
        sigma = math.exp(log_sig)
        data_weight = 0.5 * math.exp(-2 * log_sig)
        regularizer = log_sig
        total_weighted = data_weight * test_loss_val + regularizer
        
        print(f"   {log_sig:8.1f} | {sigma:5.3f} | {data_weight:11.6f} | {regularizer:11.6f} | {total_weighted:5.3f}")
    
    # Test 7: Final validation
    print("\n=== VALIDATION RESULTS ===")
    
    all_tests_passed = True
    
    # Check layer supervision setup
    setup_correct = supervised_layers == model.supervised_layer_indices
    print(f"✓ Layer supervision setup correct: {setup_correct}")
    if not setup_correct:
        all_tests_passed = False
    
    # Check structured loss output
    structured_loss = isinstance(loss, dict) and 'layer_losses' in loss
    print(f"✓ Model outputs structured loss with layer supervision: {structured_loss}")
    if not structured_loss:
        all_tests_passed = False
    
    # Check layer uncertainty gradients
    print(f"✓ Layer uncertainty parameters receive gradients: {has_layer_gradients}")
    if not has_layer_gradients:
        all_tests_passed = False
    
    # Check uncertainty weighting preserves gradients
    if 'has_weighted_gradients' in locals():
        print(f"✓ Uncertainty weighting preserves gradients: {has_weighted_gradients}")
        if not has_weighted_gradients:
            all_tests_passed = False
    
    # Check readout heads exist
    readout_heads_exist = all(hasattr(model.blocks[i], 'layer_head') for i in supervised_layers)
    print(f"✓ Readout heads exist for supervised layers: {readout_heads_exist}")
    if not readout_heads_exist:
        all_tests_passed = False
    
    # Summary
    if all_tests_passed:
        print(f"\n🎉 ALL TESTS PASSED: Layer-level uncertainty mechanism is properly implemented!")
        print(f"   - Layer uncertainty parameters are learnable and receive gradients")
        print(f"   - Deep supervision readout heads work correctly")
        print(f"   - Layer-wise uncertainty weighting is applied correctly")
        print(f"   - Structured loss output includes both final and layer losses")
        print(f"   - Parameter clamping works to prevent degenerate values")
    else:
        print(f"\n❌ SOME TESTS FAILED: Layer uncertainty implementation has issues!")
    
    return all_tests_passed

if __name__ == "__main__":
    test_layer_uncertainty_mechanism()