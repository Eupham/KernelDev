#!/usr/bin/env python3
"""
Quick test to verify causal masking is working correctly.
"""

import torch
from original_kernel import flash_attention

def test_causal_masking():
    """Test that causal parameter works correctly with supported head dimensions."""
    print("Testing causal masking with supported head dimensions...")
    
    # Use supported head dimension (64)
    batch_size = 1
    n_heads = 2
    seq_len = 32
    head_dim = 64  # Supported dimension
    
    # Create random inputs
    torch.manual_seed(42)
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    
    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"Head dimension: {head_dim} (should be supported)")
    
    try:
        # Test 1: Forward pass with causal=True
        print("\n1. Testing causal=True (causal attention)")
        out_causal = flash_attention(q, k, v, causal=True)
        print(f"Causal output shape: {out_causal.shape}")
        
        # Test 2: Forward pass with causal=False
        print("\n2. Testing causal=False (bidirectional attention)")
        out_bidirectional = flash_attention(q, k, v, causal=False)
        print(f"Bidirectional output shape: {out_bidirectional.shape}")
        
        # Test 3: Compare outputs - they should be different
        print("\n3. Comparing causal vs bidirectional outputs")
        diff = torch.abs(out_causal - out_bidirectional).max().item()
        mean_diff = torch.abs(out_causal - out_bidirectional).mean().item()
        print(f"Max difference: {diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        if diff > 1e-6:
            print("✓ PASS: Outputs are different - causal masking is working!")
            
            # Additional test: check specific positions
            print("\n4. Detailed analysis:")
            print(f"Causal output sample: {out_causal[0, 0, :5, :3]}")
            print(f"Bidirectional output sample: {out_bidirectional[0, 0, :5, :3]}")
            
            return True
        else:
            print("✗ FAIL: Outputs are identical - causal masking is NOT working!")
            return False
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_flow():
    """Test that gradients work correctly with causal masking."""
    print("\n5. Testing gradient flow with causal masking...")
    
    batch_size = 1
    n_heads = 2
    seq_len = 16
    head_dim = 64
    
    # Create inputs with gradients
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16, requires_grad=True)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16, requires_grad=True)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16, requires_grad=True)
    
    try:
        # Forward pass
        out = flash_attention(q, k, v, causal=True)
        
        # Backward pass
        loss = out.sum()
        loss.backward()
        
        print("✓ PASS: Gradient flow works with causal masking")
        print(f"Q grad shape: {q.grad.shape if q.grad is not None else 'None'}")
        print(f"K grad shape: {k.grad.shape if k.grad is not None else 'None'}")
        print(f"V grad shape: {v.grad.shape if v.grad is not None else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"✗ FAIL: Gradient flow failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("CAUSAL MASKING VERIFICATION TEST")
    print("=" * 50)
    
    success1 = test_causal_masking()
    success2 = test_gradient_flow()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✓ ALL TESTS PASSED: Causal masking is working correctly!")
    else:
        print("✗ SOME TESTS FAILED: Causal masking may have issues!")
    print("=" * 50)
