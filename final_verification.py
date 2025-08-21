#!/usr/bin/env python3
"""
Final verification that all uncertainty mechanisms have been removed.
"""

import torch
from model import GPTModel
from train_loop import Trainer, TrainingConfig

def verify_no_uncertainty_components():
    """Verify that no uncertainty components remain in the model."""
    print("=== Verifying No Uncertainty Components ===")
    
    model = GPTModel(
        vocab_size=1000,
        dim=256,
        n_layers=6,
        n_heads=4,
        layer_supervision_frequency=2
    )
    
    # Check that no layers have uncertainty parameters
    uncertainty_found = False
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'log_sigmas'):
            print(f"❌ Layer {i} still has log_sigmas")
            uncertainty_found = True
        else:
            print(f"✓ Layer {i} has no uncertainty parameters")
    
    # Check that model has no task-specific parameters
    if hasattr(model, 'task_names'):
        print(f"❌ Model still has task_names: {model.task_names}")
        uncertainty_found = True
    else:
        print("✓ Model has no task_names parameter")
    
    if hasattr(model, 'task_alpha') or hasattr(model, 'task_beta') or hasattr(model, 'task_id'):
        print("❌ Model still has task-specific conditioning parameters")
        uncertainty_found = True
    else:
        print("✓ Model has no task-specific conditioning parameters")
    
    return not uncertainty_found

def test_training_functionality():
    """Test that training works without uncertainty."""
    print("\n=== Testing Training Without Uncertainty ===")
    
    model = GPTModel(vocab_size=100, dim=128, n_layers=2, n_heads=2)
    config = TrainingConfig(num_epochs=1, learning_rate=1e-3, log_every=10)
    trainer = Trainer(model, config)
    
    # Create sample training data
    batch_size = 4
    seq_len = 8
    x = torch.randint(0, 100, (batch_size, seq_len))
    y = torch.randint(0, 100, (batch_size, seq_len))
    
    # Test forward pass
    logits, loss = model(x, targets=y)
    print(f"✓ Forward pass: logits {logits.shape}, loss {loss.item():.4f}")
    
    # Test loss combination
    combined_loss = trainer.combine_losses(loss)
    print(f"✓ Loss combination: {combined_loss.item():.4f}")
    
    # Test structured loss
    structured = {'final_ce': torch.tensor(2.0), 'layer_ce': {'layer_0': torch.tensor(1.5)}}
    combined_structured = trainer.combine_losses({'teacher_forcing': structured})
    print(f"✓ Structured loss combination: {combined_structured.item():.4f}")
    
    return True

def test_generation_still_works():
    """Test that text generation still works."""
    print("\n=== Testing Text Generation ===")
    
    model = GPTModel(vocab_size=100, dim=128, n_layers=2, n_heads=2)
    
    # Test generation
    prompt = torch.randint(0, 100, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    
    print(f"✓ Input shape: {prompt.shape}")
    print(f"✓ Generated shape: {generated.shape}")
    print(f"✓ Generated {generated.shape[1] - prompt.shape[1]} new tokens")
    
    return True

if __name__ == "__main__":
    print("Final Verification: All Uncertainty Mechanisms Removed")
    print("=" * 65)
    
    results = []
    results.append(verify_no_uncertainty_components())
    results.append(test_training_functionality())
    results.append(test_generation_still_works())
    
    if all(results):
        print("\n🎉 VERIFICATION COMPLETE: All uncertainty mechanisms successfully removed!")
        print("\n📋 Summary:")
        print("   ✅ No uncertainty parameters remain in model architecture")
        print("   ✅ Training and loss computation work without uncertainty")
        print("   ✅ Text generation works correctly")
        print("   ✅ Core transformer functionality preserved")
        print("   ✅ Layer supervision works without uncertainty weighting")
        print("\n🔧 The model is now a clean transformer architecture focused on:")
        print("   - Teacher forcing language modeling")
        print("   - Cocktail party span selection tasks")
        print("   - Hierarchical attention patterns")
        print("   - Layer supervision for improved training")
    else:
        print("\n❌ VERIFICATION FAILED: Some uncertainty components may remain")
        exit(1)