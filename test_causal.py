#!/usr/bin/env python3
"""
Test script to validate the toggleable causal mask functionality.
Tests both causal=True and causal=False modes.
"""

import torch
import torch.nn.functional as F
from original_kernel import flash_attention, flash_attention_reference

def test_causal_functionality():
    """Test that causal parameter works correctly."""
    print("Testing toggleable causal mask functionality...")
    
    # Test parameters
    batch_size = 2
    n_heads = 4
    seq_len = 128
    head_dim = 64
    
    # Create random inputs
    torch.manual_seed(42)
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16, requires_grad=True)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16, requires_grad=True)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16, requires_grad=True)
    
    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    
    # Test 1: Forward pass with causal=True
    print("\n1. Testing causal=True (default causal attention)")
    out_causal = flash_attention(q, k, v, causal=True)
    print(f"Causal output shape: {out_causal.shape}")
    
    # Test 2: Forward pass with causal=False
    print("\n2. Testing causal=False (bidirectional attention)")
    out_bidirectional = flash_attention(q, k, v, causal=False)
    print(f"Bidirectional output shape: {out_bidirectional.shape}")
    
    # Test 3: Compare outputs - they should be different
    print("\n3. Comparing causal vs bidirectional outputs")
    diff = torch.abs(out_causal - out_bidirectional).max().item()
    print(f"Max difference between causal and bidirectional: {diff:.6f}")
    
    if diff > 1e-6:
        print("✓ PASS: Outputs are different as expected")
    else:
        print("✗ FAIL: Outputs are too similar - causal masking may not be working")
        
    # Test 4: Test with reference implementation
    print("\n4. Testing against reference implementation")
    try:
        ref_causal = flash_attention_reference(q, k, v, causal=True)
        ref_bidirectional = flash_attention_reference(q, k, v, causal=False)
        
        causal_diff = torch.abs(out_causal - ref_causal).max().item()
        bidirectional_diff = torch.abs(out_bidirectional - ref_bidirectional).max().item()
        
        print(f"Causal vs reference diff: {causal_diff:.6f}")
        print(f"Bidirectional vs reference diff: {bidirectional_diff:.6f}")
        
        if causal_diff < 1e-2 and bidirectional_diff < 1e-2:
            print("✓ PASS: Kernel outputs match reference implementation")
        else:
            print("✗ FAIL: Kernel outputs don't match reference implementation")
    except Exception as e:
        print(f"Reference test failed: {e}")
    
    # Test 5: Backward pass
    print("\n5. Testing backward pass")
    loss_causal = out_causal.sum()
    loss_bidirectional = out_bidirectional.sum()
    
    try:
        loss_causal.backward(retain_graph=True)
        print("✓ PASS: Causal backward pass successful")
        
        # Clear gradients for bidirectional test
        q.grad = None
        k.grad = None
        v.grad = None
        
        loss_bidirectional.backward()
        print("✓ PASS: Bidirectional backward pass successful")
        
    except Exception as e:
        print(f"✗ FAIL: Backward pass failed: {e}")
    
    print("\n6. Testing attention patterns")
    # Create a simple case to visualize attention patterns
    simple_seq_len = 4
    simple_q = torch.randn(1, 1, simple_seq_len, head_dim, device='cuda', dtype=torch.float16)
    simple_k = torch.randn(1, 1, simple_seq_len, head_dim, device='cuda', dtype=torch.float16)
    simple_v = torch.randn(1, 1, simple_seq_len, head_dim, device='cuda', dtype=torch.float16)
    
    # Test both modes
    simple_causal = flash_attention(simple_q, simple_k, simple_v, causal=True)
    simple_bidirectional = flash_attention(simple_q, simple_k, simple_v, causal=False)
    
    print(f"Simple test - Causal output norm: {simple_causal.norm().item():.4f}")
    print(f"Simple test - Bidirectional output norm: {simple_bidirectional.norm().item():.4f}")
    
    print("\n✓ All tests completed!")

def test_model_integration():
    """Test the model integration with causal parameter."""
    print("\n" + "="*50)
    print("Testing model integration...")
    
    from model import GPTModel
    
    # Test with causal=True (default)
    model_causal = GPTModel(
        vocab_size=1000,
        dim=256,
        n_layers=2,
        n_heads=4,
        max_seq_len=128,
        causal=True
    ).cuda()
    
    # Test with causal=False
    model_bidirectional = GPTModel(
        vocab_size=1000,
        dim=256,
        n_layers=2,
        n_heads=4,
        max_seq_len=128,
        causal=False
    ).cuda()
    
    # Create test input
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device='cuda')
    
    print(f"Model input shape: {input_ids.shape}")
    
    # Test forward passes
    with torch.no_grad():
        out_causal = model_causal(input_ids)
        out_bidirectional = model_bidirectional(input_ids)
    
    print(f"Causal model output shape: {out_causal.shape}")
    print(f"Bidirectional model output shape: {out_bidirectional.shape}")
    
    # Compare outputs
    diff = torch.abs(out_causal - out_bidirectional).max().item()
    print(f"Max difference between causal and bidirectional models: {diff:.6f}")
    
    if diff > 1e-6:
        print("✓ PASS: Model outputs are different as expected")
    else:
        print("✗ FAIL: Model outputs are too similar")
    
    print("✓ Model integration tests completed!")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available. Starting tests...")
        test_causal_functionality()
        test_model_integration()
    else:
        print("CUDA is not available. Skipping tests.")
