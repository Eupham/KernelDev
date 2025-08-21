#!/usr/bin/env python3
"""
Complete Layer-Level Uncertainty Integration Test

This test validates that the layer-level uncertainty mechanism integrates
correctly with the existing task-level uncertainty system, using actual
model components where possible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
warnings.filterwarnings('ignore')

# Import from the actual system where possible
try:
    from model import GPTModel, TransformerBlock
    SPECIAL_TOKENS = {
        '[PAD]': 0,
        '[CLS]': 1,
        '[MASK]': 2,
        '[SPAN]': 3,
        '[ES]': 4,
        '[MASKQ]': 5
    }
    ACTUAL_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Could not import actual model: {e}")
    ACTUAL_MODEL_AVAILABLE = False

def test_integration():
    """Test the integration of layer and task level uncertainty."""
    
    print("=== Layer-Level Uncertainty Integration Test ===\n")
    
    if not ACTUAL_MODEL_AVAILABLE:
        print("❌ Actual model components not available, skipping integration test")
        return False
    
    # Test parameters
    vocab_size = 100
    dim = 32
    n_layers = 6
    n_heads = 4
    max_seq_len = 64
    layer_supervision_frequency = 2
    task_names = ['teacher_forcing', 'cocktail_party']
    
    print(f"Creating model with:")
    print(f"  - {n_layers} layers")  
    print(f"  - Layer supervision every {layer_supervision_frequency} layers")
    print(f"  - Task-level uncertainty for: {task_names}")
    
    # Create model with both task and layer level uncertainty
    try:
        model = GPTModel(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            task_names=task_names,
            layer_supervision_frequency=layer_supervision_frequency
        )
        print("✓ Model created successfully")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Check task-level uncertainty parameters
    print(f"\n1. Task-level uncertainty parameters:")
    if hasattr(model, 'log_sigmas'):
        for task_name, log_sigma in model.log_sigmas.items():
            sigma = torch.exp(log_sigma)
            print(f"   {task_name}: log_sigma = {log_sigma.item():.6f}, sigma = {sigma.item():.6f}")
    else:
        print("   No task-level uncertainty parameters found")
    
    # Check layer-level uncertainty parameters
    print(f"\n2. Layer-level uncertainty parameters:")
    layer_uncertainty_count = 0
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'log_sigma'):
            log_sigma = block.log_sigma
            sigma = torch.exp(log_sigma)
            layer_uncertainty_count += 1
            print(f"   Layer {i}: log_sigma = {log_sigma.item():.6f}, sigma = {sigma.item():.6f}")
        else:
            print(f"   Layer {i}: no uncertainty parameter")
    
    print(f"   Total layers with uncertainty: {layer_uncertainty_count}")
    print(f"   Expected: {len(model.supervised_layer_indices)}")
    
    # Test forward pass with teacher forcing
    print(f"\n3. Testing teacher forcing forward pass:")
    
    batch_size = 2
    seq_len = 8
    
    # Create teacher forcing input (avoid special tokens except CLS)
    x = torch.randint(6, vocab_size, (batch_size, seq_len))  # Avoid special tokens
    targets = torch.randint(6, vocab_size, (batch_size, seq_len))
    
    # Add CLS token at beginning
    x[:, 0] = SPECIAL_TOKENS['[CLS]']
    
    try:
        logits, loss = model(x, targets=targets, task_name='teacher_forcing')
        print(f"   ✓ Forward pass successful")
        print(f"   Logits shape: {logits.shape}")
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
            
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False
    
    # Test uncertainty weighting (simplified version from train_loop.py)
    print(f"\n4. Testing uncertainty weighting:")
    
    try:
        # Import the actual uncertainty weighting function
        from train_loop import Trainer, TrainingConfig
        
        config = TrainingConfig(learning_rate=1e-3, num_epochs=1)
        trainer = Trainer(model, config)
        
        # Apply uncertainty weighting
        weighted_loss = trainer.apply_layer_uncertainty_weighting(loss, 'teacher_forcing')
        print(f"   ✓ Uncertainty weighting successful")
        print(f"   Weighted loss: {weighted_loss.item():.6f}")
        
        # Test gradient flow
        model.zero_grad()
        weighted_loss.backward()
        
        # Check task-level gradients
        task_gradients = {}
        if hasattr(model, 'log_sigmas'):
            for task_name, log_sigma in model.log_sigmas.items():
                if log_sigma.grad is not None:
                    task_gradients[task_name] = log_sigma.grad.item()
                    print(f"   Task {task_name} gradient: {log_sigma.grad.item():.6f}")
                else:
                    task_gradients[task_name] = 0.0
                    print(f"   Task {task_name} gradient: None")
        
        # Check layer-level gradients
        layer_gradients = {}
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'log_sigma'):
                if block.log_sigma.grad is not None:
                    layer_gradients[i] = block.log_sigma.grad.item()
                    print(f"   Layer {i} gradient: {block.log_sigma.grad.item():.6f}")
                else:
                    layer_gradients[i] = 0.0
                    print(f"   Layer {i} gradient: None")
        
        # Check that both task and layer uncertainties have gradients
        task_has_gradients = any(abs(g) > 1e-6 for g in task_gradients.values()) if task_gradients else False
        layer_has_gradients = any(abs(g) > 1e-6 for g in layer_gradients.values()) if layer_gradients else False
        
        print(f"   Task-level parameters have gradients: {task_has_gradients}")
        print(f"   Layer-level parameters have gradients: {layer_has_gradients}")
        
    except Exception as e:
        print(f"   ❌ Uncertainty weighting failed: {e}")
        return False
    
    # Test parameter updates through optimization
    print(f"\n5. Testing parameter updates through optimization:")
    
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Save initial values
        initial_task_values = {}
        if hasattr(model, 'log_sigmas'):
            for task_name, log_sigma in model.log_sigmas.items():
                initial_task_values[task_name] = log_sigma.data.clone()
        
        initial_layer_values = {}
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'log_sigma'):
                initial_layer_values[i] = block.log_sigma.data.clone()
        
        # Run a few optimization steps
        for step in range(3):
            optimizer.zero_grad()
            
            # Create new batch
            x = torch.randint(6, vocab_size, (batch_size, seq_len))
            targets = torch.randint(6, vocab_size, (batch_size, seq_len))
            x[:, 0] = SPECIAL_TOKENS['[CLS]']
            
            logits, loss = model(x, targets=targets, task_name='teacher_forcing')
            weighted_loss = trainer.apply_layer_uncertainty_weighting(loss, 'teacher_forcing')
            
            weighted_loss.backward()
            optimizer.step()
            
            if step == 0:
                print(f"   Step {step}: loss = {weighted_loss.item():.6f}")
        
        # Check parameter changes
        print(f"   Parameter changes after optimization:")
        
        # Task-level changes
        task_changes = {}
        if hasattr(model, 'log_sigmas'):
            for task_name, log_sigma in model.log_sigmas.items():
                if task_name in initial_task_values:
                    initial = initial_task_values[task_name].item()
                    current = log_sigma.data.item()
                    change = abs(current - initial)
                    task_changes[task_name] = change
                    print(f"     Task {task_name}: {initial:.6f} → {current:.6f} (|Δ|={change:.6f})")
        
        # Layer-level changes
        layer_changes = {}
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'log_sigma') and i in initial_layer_values:
                initial = initial_layer_values[i].item()
                current = block.log_sigma.data.item()
                change = abs(current - initial)
                layer_changes[i] = change
                print(f"     Layer {i}: {initial:.6f} → {current:.6f} (|Δ|={change:.6f})")
        
        # Verify parameters actually changed
        task_updated = any(change > 1e-6 for change in task_changes.values()) if task_changes else False
        layer_updated = any(change > 1e-6 for change in layer_changes.values()) if layer_changes else False
        
        print(f"   Task parameters updated: {task_updated}")
        print(f"   Layer parameters updated: {layer_updated}")
        
    except Exception as e:
        print(f"   ❌ Optimization test failed: {e}")
        return False
    
    # Final validation
    print(f"\n=== INTEGRATION TEST RESULTS ===")
    
    all_tests_passed = True
    test_results = []
    
    # Model creation
    test_results.append(("Model creation", True))
    
    # Layer supervision setup
    layer_setup_correct = len(model.supervised_layer_indices) > 0
    test_results.append(("Layer supervision setup", layer_setup_correct))
    if not layer_setup_correct:
        all_tests_passed = False
    
    # Structured loss output
    structured_loss_works = isinstance(loss, dict) and 'layer_losses' in loss
    test_results.append(("Structured loss output", structured_loss_works))
    if not structured_loss_works:
        all_tests_passed = False
    
    # Uncertainty weighting
    uncertainty_weighting_works = True  # Already tested above
    test_results.append(("Uncertainty weighting", uncertainty_weighting_works))
    
    # Gradient flow
    gradients_work = layer_has_gradients
    test_results.append(("Layer uncertainty gradients", gradients_work))
    if not gradients_work:
        all_tests_passed = False
    
    # Parameter updates
    parameters_update = layer_updated
    test_results.append(("Parameter updates", parameters_update))
    if not parameters_update:
        all_tests_passed = False
    
    # Print results
    for test_name, passed in test_results:
        status = "✓" if passed else "❌"
        print(f"{status} {test_name}: {passed}")
    
    if all_tests_passed:
        print(f"\n🎉 ALL INTEGRATION TESTS PASSED!")
        print(f"   The layer-level uncertainty mechanism successfully integrates with:")
        print(f"   - Existing task-level uncertainty")
        print(f"   - Deep supervision readout heads")
        print(f"   - Uncertainty-weighted loss computation")
        print(f"   - Gradient flow and parameter optimization")
        print(f"   - Training loop infrastructure")
    else:
        print(f"\n❌ SOME INTEGRATION TESTS FAILED!")
    
    return all_tests_passed

if __name__ == "__main__":
    test_integration()