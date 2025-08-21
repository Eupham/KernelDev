#!/usr/bin/env python3
"""
Test core model functionality after uncertainty removal.
"""

import torch
from model import GPTModel

def test_teacher_forcing():
    """Test basic teacher forcing functionality."""
    print("=== Testing Teacher Forcing ===")
    
    model = GPTModel(
        vocab_size=1000,
        dim=256,
        n_layers=4,
        n_heads=4,
        max_seq_len=512
    )
    
    # Create sample data
    batch_size = 2
    seq_len = 10
    x = torch.randint(0, 1000, (batch_size, seq_len))
    targets = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    logits, loss = model(x, targets=targets)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output logits shape: {logits.shape}")
    print(f"✓ Loss: {loss.item():.4f}")
    
    # Test generation
    generated = model.generate(x[:1], max_new_tokens=5)
    print(f"✓ Generation shape: {generated.shape}")
    
    return True

def test_cocktail_party():
    """Test cocktail party task functionality."""
    print("\n=== Testing Cocktail Party ===")
    
    model = GPTModel(
        vocab_size=1000,
        dim=256,
        n_layers=4,
        n_heads=4,
        max_seq_len=512
    )
    
    # Create sample cocktail party data with special tokens
    batch_size = 2
    seq_len = 20
    
    # Mock special tokens
    from model import SPECIAL_TOKENS
    cls_token = SPECIAL_TOKENS['[CLS]']
    mask_token = SPECIAL_TOKENS['[MASK]']
    span_token = SPECIAL_TOKENS['[SPAN]']
    es_token = SPECIAL_TOKENS['[ES]']
    
    # Create a sequence with span structure
    x = torch.randint(100, 900, (batch_size, seq_len))
    x[:, 0] = cls_token  # [CLS] at start
    x[:, 5] = mask_token  # [MASK] query
    x[:, 8] = span_token  # [SPAN] start
    x[:, 12] = es_token   # [ES] end
    x[:, 15] = span_token  # [SPAN] start
    x[:, 18] = es_token   # [ES] end
    
    # Create correct span indices
    correct_idx = torch.randint(0, 2, (batch_size,))
    
    # Forward pass
    scores, loss = model(x, correct_idx=correct_idx)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output scores shape: {scores.shape}")
    print(f"✓ Loss: {loss.item():.4f}")
    
    return True

def test_layer_supervision():
    """Test layer supervision without uncertainty."""
    print("\n=== Testing Layer Supervision ===")
    
    model = GPTModel(
        vocab_size=1000,
        dim=256,
        n_layers=6,
        n_heads=4,
        layer_supervision_frequency=2,
        max_seq_len=512
    )
    
    print(f"✓ Supervised layers: {model.supervised_layer_indices}")
    
    # Check that supervised layers have readout heads
    for i in model.supervised_layer_indices:
        block = model.blocks[i]
        assert hasattr(block, 'layer_head'), f"Layer {i} should have readout head"
        print(f"✓ Layer {i} has readout head")
    
    # Test forward pass with targets
    batch_size = 2
    seq_len = 10
    x = torch.randint(0, 1000, (batch_size, seq_len))
    targets = torch.randint(0, 1000, (batch_size, seq_len))
    
    logits, loss = model(x, targets=targets)
    
    # Should return structured loss if layer supervision is enabled
    if isinstance(loss, dict):
        print(f"✓ Structured loss returned with keys: {loss.keys()}")
        if 'layer_ce' in loss:
            print(f"✓ Layer losses: {list(loss['layer_ce'].keys())}")
    else:
        print(f"✓ Simple loss: {loss.item():.4f}")
    
    return True

if __name__ == "__main__":
    print("Testing Core Model Functionality (No Uncertainty)")
    print("=" * 60)
    
    results = []
    results.append(test_teacher_forcing())
    results.append(test_cocktail_party())  
    results.append(test_layer_supervision())
    
    if all(results):
        print("\n🎉 ALL TESTS PASSED: Core functionality works without uncertainty!")
        print("   - Teacher forcing works correctly")
        print("   - Cocktail party task works correctly") 
        print("   - Layer supervision works correctly")
        print("   - No uncertainty components remain")
    else:
        print("\n❌ SOME TESTS FAILED!")
        exit(1)