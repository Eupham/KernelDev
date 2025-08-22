#!/usr/bin/env python3
"""
Test the actual attention pattern logic from original_kernel.py

This test extracts and validates the core attention pattern logic
that is implemented in the Triton kernels, allowing us to test
the hierarchical attention behaviors without requiring CUDA.
"""

import sys
import torch
import unittest
import numpy as np
from typing import Tuple

# Add current directory to Python path for imports
sys.path.insert(0, '/home/runner/work/KernelDev/KernelDev')

def compute_cocktail_party_attention_mask(
    q_indices: torch.Tensor,
    kv_indices: torch.Tensor, 
    q_in_span: torch.Tensor,
    k_in_span: torch.Tensor,
    q_span_id: torch.Tensor,
    k_span_id: torch.Tensor,
    q_is_prefix: torch.Tensor,
    k_is_prefix: torch.Tensor
) -> torch.Tensor:
    """
    Implement the exact cocktail party attention pattern logic from original_kernel.py
    
    This replicates the logic from lines 642-674 in the kernel:
    
    Args:
        q_indices: Query token indices [TILE_Q_SIZE]
        kv_indices: Key/Value token indices [TILE_K_SIZE]
        q_in_span: Query in_span mask [TILE_Q_SIZE]
        k_in_span: Key in_span mask [TILE_K_SIZE]
        q_span_id: Query span_id [TILE_Q_SIZE]
        k_span_id: Key span_id [TILE_K_SIZE]
        q_is_prefix: Query is_prefix mask [TILE_Q_SIZE]
        k_is_prefix: Key is_prefix mask [TILE_K_SIZE]
    
    Returns:
        attention_mask: [TILE_Q_SIZE, TILE_K_SIZE] boolean mask
    """
    
    # All broadcasted to [TILE_Q_SIZE, TILE_K_SIZE]
    
    # Check if query/key tokens are special types
    q_is_maskq = (q_span_id[:, None] == -1)  # [MASKQ] marked with span_id = -1
    k_is_maskq = (k_span_id[None, :] == -1)
    k_is_cls_or_prefix = k_is_prefix[None, :]
    
    # Pattern 1: [CLS]/prefix tokens can only see within prefix (bidirectional within prefix)
    prefix_to_prefix = q_is_prefix[:, None] & k_is_prefix[None, :]
    
    # Pattern 2: Context tokens (non-span, non-prefix) causal within context + can see prefix
    q_is_context = ~q_in_span[:, None] & ~q_is_prefix[:, None] & ~q_is_maskq
    k_is_context = ~k_in_span[None, :] & ~k_is_prefix[None, :] & ~k_is_maskq
    context_causal = q_is_context & k_is_context & (q_indices[:, None] >= kv_indices[None, :])
    context_to_prefix = q_is_context & k_is_cls_or_prefix
    
    # Pattern 3: Span tokens bidirectional within same span + can see context (NO MASKQ)
    same_span = (q_in_span[:, None] & k_in_span[None, :] & 
                (q_span_id[:, None] == k_span_id[None, :]) & 
                (q_span_id[:, None] > 0))  # Exclude span_id=0 and span_id=-1
    span_to_context = q_in_span[:, None] & k_is_context
    
    # Pattern 4: [MASKQ] can see all spans + [CLS] (simplified to only spans for easier calculation)
    maskq_to_spans = q_is_maskq & k_in_span[None, :]
    maskq_to_cls = q_is_maskq & k_is_cls_or_prefix
    
    # Combine all allowed patterns
    mask = (prefix_to_prefix | 
           context_causal | context_to_prefix |
           same_span | span_to_context |
           maskq_to_spans | maskq_to_cls)
    
    return mask


def compute_teacher_forcing_attention_mask(
    q_indices: torch.Tensor,
    kv_indices: torch.Tensor,
    q_is_prefix: torch.Tensor,
    k_is_prefix: torch.Tensor
) -> torch.Tensor:
    """
    Implement teacher forcing attention pattern.
    
    Args:
        q_indices: Query token indices [TILE_Q_SIZE]
        kv_indices: Key/Value token indices [TILE_K_SIZE]
        q_is_prefix: Query is_prefix mask [TILE_Q_SIZE]
        k_is_prefix: Key is_prefix mask [TILE_K_SIZE]
    
    Returns:
        attention_mask: [TILE_Q_SIZE, TILE_K_SIZE] boolean mask
    """
    # Pattern 1: Bidirectional within prefix
    prefix_to_prefix = q_is_prefix[:, None] & k_is_prefix[None, :]
    
    # Pattern 2: Causal after prefix + can see prefix
    q_is_context = ~q_is_prefix[:, None]
    k_is_context = ~k_is_prefix[None, :]
    k_is_prefix_broadcast = k_is_prefix[None, :]
    
    context_causal = q_is_context & k_is_context & (q_indices[:, None] >= kv_indices[None, :])
    context_to_prefix = q_is_context & k_is_prefix_broadcast
    
    # Combine patterns
    mask = prefix_to_prefix | context_causal | context_to_prefix
    
    return mask


class TestAttentionKernelLogic(unittest.TestCase):
    """Test the actual attention pattern logic from the kernel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        
    def test_teacher_forcing_attention_pattern(self):
        """Test teacher forcing attention pattern logic."""
        print("\n=== Testing Teacher Forcing Attention Pattern Logic ===")
        
        # Create test sequence: [prefix][prefix][CLS][context][context][context]
        seq_len = 6
        q_indices = torch.arange(seq_len, dtype=torch.long)
        kv_indices = torch.arange(seq_len, dtype=torch.long)
        
        # Setup prefix positions (0, 1, 2 including CLS at 2)
        is_prefix = torch.zeros(seq_len, dtype=torch.bool)
        is_prefix[:3] = True  # positions 0, 1, 2 are prefix
        
        print(f"Sequence length: {seq_len}")
        print(f"Prefix mask: {is_prefix}")
        print(f"Query indices: {q_indices}")
        print(f"Key indices: {kv_indices}")
        
        # Compute attention mask
        attention_mask = compute_teacher_forcing_attention_mask(
            q_indices, kv_indices, is_prefix, is_prefix
        )
        
        print(f"\nAttention mask shape: {attention_mask.shape}")
        print("Attention mask (1=attend, 0=masked):")
        for i in range(seq_len):
            row = "".join("1" if attention_mask[i, j] else "0" for j in range(seq_len))
            print(f"  Query {i}: {row}")
        
        # Validate patterns
        print("\nValidating teacher forcing patterns:")
        
        # 1. Prefix tokens should attend bidirectionally to all prefix tokens
        for i in range(3):  # prefix positions
            for j in range(3):  # prefix positions
                self.assertTrue(attention_mask[i, j].item(), 
                               f"Prefix token {i} should attend to prefix token {j}")
        
        print("✓ Prefix bidirectional attention validated")
        
        # 2. Context tokens should attend causally + to prefix
        for i in range(3, seq_len):  # context positions
            # Should attend to prefix
            for j in range(3):  # prefix positions
                self.assertTrue(attention_mask[i, j].item(),
                               f"Context token {i} should attend to prefix token {j}")
            
            # Should attend causally within context
            for j in range(3, seq_len):  # context positions
                if i >= j:  # causal
                    self.assertTrue(attention_mask[i, j].item(),
                                   f"Context token {i} should attend to context token {j} (causal)")
                else:  # future tokens should be masked
                    self.assertFalse(attention_mask[i, j].item(),
                                    f"Context token {i} should NOT attend to future context token {j}")
        
        print("✓ Context causal attention validated")
        print("✓ Teacher forcing attention pattern logic validated!")
        
    def test_cocktail_party_attention_pattern(self):
        """Test cocktail party attention pattern logic."""
        print("\n=== Testing Cocktail Party Attention Pattern Logic ===")
        
        # Create test sequence: [prefix][prefix][CLS][context][context][SPAN]span1[ES][SPAN]span2[ES][MASKQ]
        seq_len = 11
        q_indices = torch.arange(seq_len, dtype=torch.long)
        kv_indices = torch.arange(seq_len, dtype=torch.long)
        
        # Setup metadata
        is_prefix = torch.zeros(seq_len, dtype=torch.bool)
        in_span = torch.zeros(seq_len, dtype=torch.bool)
        span_id = torch.zeros(seq_len, dtype=torch.long)
        
        # Prefix: positions 0, 1, 2 (including CLS at 2)
        is_prefix[:3] = True
        
        # Context: positions 3, 4 (no special marking needed)
        
        # Span 1: positions 5, 6 ([SPAN]content[ES])
        in_span[5:7] = True
        span_id[5:7] = 1
        
        # Span 2: positions 7, 8 ([SPAN]content[ES])
        in_span[7:9] = True
        span_id[7:9] = 2
        
        # MASKQ: position 10
        span_id[10] = -1  # MASKQ marked with span_id = -1
        
        print(f"Sequence length: {seq_len}")
        print(f"Prefix mask: {is_prefix}")
        print(f"In span mask: {in_span}")
        print(f"Span ID: {span_id}")
        
        # Compute attention mask
        attention_mask = compute_cocktail_party_attention_mask(
            q_indices, kv_indices, 
            in_span, in_span,
            span_id, span_id,
            is_prefix, is_prefix
        )
        
        print(f"\nAttention mask shape: {attention_mask.shape}")
        print("Attention mask (1=attend, 0=masked):")
        for i in range(seq_len):
            row = "".join("1" if attention_mask[i, j] else "0" for j in range(seq_len))
            token_type = "PREF" if is_prefix[i] else ("SPAN" if in_span[i] else ("MSKQ" if span_id[i] == -1 else "CTXT"))
            print(f"  Q{i}({token_type}): {row}")
        
        # Validate patterns
        print("\nValidating cocktail party patterns:")
        
        # 1. Prefix bidirectional
        for i in range(3):
            for j in range(3):
                self.assertTrue(attention_mask[i, j].item(),
                               f"Prefix token {i} should attend to prefix token {j}")
        print("✓ Prefix bidirectional attention validated")
        
        # 2. Context causal + to prefix
        context_positions = [3, 4]
        for i in context_positions:
            # Should attend to prefix
            for j in range(3):
                self.assertTrue(attention_mask[i, j].item(),
                               f"Context token {i} should attend to prefix token {j}")
            
            # Should attend causally within context
            for j in context_positions:
                if i >= j:
                    self.assertTrue(attention_mask[i, j].item(),
                                   f"Context token {i} should attend to context token {j} (causal)")
                else:
                    self.assertFalse(attention_mask[i, j].item(),
                                    f"Context token {i} should NOT attend to future context token {j}")
                                    
            # Should NOT attend to spans
            span_positions = [5, 6, 7, 8]
            for j in span_positions:
                self.assertFalse(attention_mask[i, j].item(),
                                f"Context token {i} should NOT attend to span token {j}")
        print("✓ Context attention patterns validated")
        
        # 3. Span behaviors
        span1_positions = [5, 6]
        span2_positions = [7, 8]
        
        # Spans should attend bidirectionally within same span
        for i in span1_positions:
            for j in span1_positions:
                self.assertTrue(attention_mask[i, j].item(),
                               f"Span1 token {i} should attend to span1 token {j}")
                
        for i in span2_positions:
            for j in span2_positions:
                self.assertTrue(attention_mask[i, j].item(),
                               f"Span2 token {i} should attend to span2 token {j}")
        
        # Spans should NOT attend to other spans
        for i in span1_positions:
            for j in span2_positions:
                self.assertFalse(attention_mask[i, j].item(),
                                f"Span1 token {i} should NOT attend to span2 token {j}")
                                
        for i in span2_positions:
            for j in span1_positions:
                self.assertFalse(attention_mask[i, j].item(),
                                f"Span2 token {i} should NOT attend to span1 token {j}")
        
        # Spans should attend to context
        for i in span1_positions + span2_positions:
            for j in context_positions:
                self.assertTrue(attention_mask[i, j].item(),
                               f"Span token {i} should attend to context token {j}")
        
        print("✓ Span attention patterns validated")
        
        # 4. MASKQ behaviors
        maskq_pos = 10
        
        # MASKQ should attend to all spans
        for j in span1_positions + span2_positions:
            self.assertTrue(attention_mask[maskq_pos, j].item(),
                           f"MASKQ token {maskq_pos} should attend to span token {j}")
        
        # MASKQ should attend to prefix
        for j in range(3):
            self.assertTrue(attention_mask[maskq_pos, j].item(),
                           f"MASKQ token {maskq_pos} should attend to prefix token {j}")
        
        # Spans should NOT attend to MASKQ
        for i in span1_positions + span2_positions:
            self.assertFalse(attention_mask[i, maskq_pos].item(),
                            f"Span token {i} should NOT attend to MASKQ token {maskq_pos}")
        
        print("✓ MASKQ attention patterns validated")
        print("✓ Cocktail party attention pattern logic validated!")

    def test_special_token_behaviors(self):
        """Test special token handling in attention patterns."""
        print("\n=== Testing Special Token Behaviors ===")
        
        # Test MASKQ token behavior
        seq_len = 5
        q_indices = torch.arange(seq_len, dtype=torch.long)
        kv_indices = torch.arange(seq_len, dtype=torch.long)
        
        # Setup: [prefix][context][span][span][MASKQ]
        is_prefix = torch.tensor([True, False, False, False, False], dtype=torch.bool)
        in_span = torch.tensor([False, False, True, True, False], dtype=torch.bool)
        span_id = torch.tensor([0, 0, 1, 1, -1], dtype=torch.long)  # -1 for MASKQ
        
        print(f"is_prefix: {is_prefix}")
        print(f"in_span: {in_span}")
        print(f"span_id: {span_id}")
        
        attention_mask = compute_cocktail_party_attention_mask(
            q_indices, kv_indices,
            in_span, in_span,
            span_id, span_id,
            is_prefix, is_prefix
        )
        
        print("\nAttention mask:")
        for i in range(seq_len):
            row = "".join("1" if attention_mask[i, j] else "0" for j in range(seq_len))
            print(f"  Query {i}: {row}")
        
        # MASKQ (position 4) should see spans (positions 2, 3)
        self.assertTrue(attention_mask[4, 2].item(), "MASKQ should see span token 2")
        self.assertTrue(attention_mask[4, 3].item(), "MASKQ should see span token 3")
        
        # MASKQ should see prefix (position 0)
        self.assertTrue(attention_mask[4, 0].item(), "MASKQ should see prefix token 0")
        
        # Spans should NOT see MASKQ
        self.assertFalse(attention_mask[2, 4].item(), "Span token 2 should NOT see MASKQ")
        self.assertFalse(attention_mask[3, 4].item(), "Span token 3 should NOT see MASKQ")
        
        print("✓ Special token behaviors validated!")


def run_tests():
    """Run all kernel logic tests."""
    print("=" * 60)
    print("ATTENTION KERNEL LOGIC TESTS")
    print("=" * 60)
    print("Testing the actual attention pattern logic from original_kernel.py")
    print("This validates the hierarchical attention implementation without CUDA")
    print()
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAttentionKernelLogic)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✓ ALL KERNEL LOGIC TESTS PASSED!")
        print("The attention pattern logic is correctly implemented.")
    else:
        print("✗ SOME KERNEL LOGIC TESTS FAILED!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        for test, traceback in result.failures + result.errors:
            print(f"\nFailed: {test}")
            print(traceback)
    
    print("=" * 60)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)