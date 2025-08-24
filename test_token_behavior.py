"""
Token Behavior Testing for Hierarchical Attention Patterns

This module tests the attention behaviors across both Teacher Forcing and 
Cocktail Party tasks without requiring the Triton kernel to be runnable.
It validates the attention masking logic and ensures correct token interactions.

Expected behaviors:
1. Teacher Forcing:
   - Any token before CLS and including CLS are bidirectional
   - All tokens after CLS should be causal and should see the CLS token
   - PAD tokens are ignored

2. Cocktail Party (4 parts):
   - Prefix up to CLS: bidirectional
   - Context: causal and may include [MASK] token
   - Span islands: [SPAN]candidate text[ES] - they see context, 
     context doesn't see them, inside spans they are causal,
     each island cannot see another island
   - MASKQ: sees all islands, islands don't see it
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
import unittest

# Define special tokens (extracted from data_builder.py to avoid dependencies)
SPECIAL_TOKENS = {
    '[PAD]': 0,
    '[CLS]': 1,
    '[MASK]': 2,
    '[SPAN]': 3,
    '[ES]': 4,
    '[MASKQ]': 5,
}


def create_reference_attention_mask(
    tokens: torch.Tensor,
    in_span: torch.Tensor,
    span_id: torch.Tensor,
    is_prefix: torch.Tensor,
    task_type: str = "teacher_forcing"
) -> torch.Tensor:
    """
    Create reference attention mask based on expected token behaviors.
    
    Args:
        tokens: Input token sequence [seq_len]
        in_span: Boolean tensor marking tokens within span boundaries [seq_len]
        span_id: Integer tensor assigning unique IDs to each span [seq_len]
        is_prefix: Boolean tensor marking prefix tokens [seq_len]
        task_type: Either "teacher_forcing" or "cocktail_party"
    
    Returns:
        attention_mask: Boolean mask [seq_len, seq_len] where True means allowed attention
    """
    seq_len = tokens.shape[0]
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    
    # Get special token IDs
    pad_token = SPECIAL_TOKENS['[PAD]']
    cls_token = SPECIAL_TOKENS['[CLS]']
    mask_token = SPECIAL_TOKENS['[MASK]']
    span_token = SPECIAL_TOKENS['[SPAN]']
    es_token = SPECIAL_TOKENS['[ES]']
    maskq_token = SPECIAL_TOKENS['[MASKQ]']
    
    # Find valid (non-PAD) positions
    valid_positions = tokens != pad_token
    
    if task_type == "teacher_forcing":
        # Teacher Forcing Rules:
        # 1. Prefix tokens (before and including CLS) are bidirectional
        # 2. Tokens after CLS are causal and can see CLS
        # 3. PAD tokens are ignored
        
        for i in range(seq_len):
            if not valid_positions[i]:
                continue
                
            for j in range(seq_len):
                if not valid_positions[j]:
                    continue
                    
                # Rule 1: Prefix tokens can see all other prefix tokens (bidirectional)
                if is_prefix[i] and is_prefix[j]:
                    mask[i, j] = True
                
                # Rule 2: Non-prefix tokens can see prefix tokens and previous non-prefix tokens (causal)
                elif not is_prefix[i]:
                    if is_prefix[j] or j <= i:  # Can see prefix or causal
                        mask[i, j] = True
    
    elif task_type == "cocktail_party":
        # Cocktail Party Rules:
        # 1. Prefix tokens (before and including CLS) are bidirectional within prefix
        # 2. Context tokens are causal within context + can see prefix
        # 3. Span tokens are bidirectional within same span + can see context
        # 4. MASKQ can see all spans + prefix
        
        for i in range(seq_len):
            if not valid_positions[i]:
                continue
                
            # Determine query token type
            q_is_prefix = is_prefix[i]
            q_in_span = in_span[i]
            q_span_id = span_id[i]
            q_is_maskq = (q_span_id == -1)  # MASKQ marked with span_id = -1
            q_is_context = not q_in_span and not q_is_prefix and not q_is_maskq
            
            for j in range(seq_len):
                if not valid_positions[j]:
                    continue
                    
                # Determine key token type
                k_is_prefix = is_prefix[j]
                k_in_span = in_span[j]
                k_span_id = span_id[j]
                k_is_maskq = (k_span_id == -1)
                k_is_context = not k_in_span and not k_is_prefix and not k_is_maskq
                
                # Rule 1: Prefix tokens can see within prefix (bidirectional)
                if q_is_prefix and k_is_prefix:
                    mask[i, j] = True
                
                # Rule 2: Context tokens causal within context + can see prefix
                elif q_is_context:
                    if k_is_prefix:  # Context can see prefix
                        mask[i, j] = True
                    elif k_is_context and j <= i:  # Causal within context
                        mask[i, j] = True
                
                # Rule 3: Span tokens bidirectional within same span + can see context
                elif q_in_span and q_span_id > 0:  # Valid span (not MASKQ)
                    if k_is_context:  # Span can see context
                        mask[i, j] = True
                    elif k_in_span and q_span_id == k_span_id and k_span_id > 0:  # Same span
                        mask[i, j] = True
                
                # Rule 4: MASKQ can see all spans + prefix
                elif q_is_maskq:
                    if k_is_prefix or (k_in_span and k_span_id > 0):  # MASKQ sees prefix and spans
                        mask[i, j] = True
    
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    return mask


def create_teacher_forcing_sequence(
    prefix_text: str = "Answer the question:",
    context_text: str = "What is the capital of France? Paris is the capital.",
    seq_len: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a Teacher Forcing sequence with metadata.
    
    Returns:
        tokens, in_span, span_id, is_prefix
    """
    # Create token sequence: {prefix}[CLS]{context}[PAD]...
    tokens = []
    
    # Add prefix tokens (simplified as single tokens)
    prefix_tokens = [ord(c) + len(SPECIAL_TOKENS) for c in prefix_text[:10]]  # Simplified
    tokens.extend(prefix_tokens)
    
    # Add CLS token
    cls_pos = len(tokens)
    tokens.append(SPECIAL_TOKENS['[CLS]'])
    
    # Add context tokens
    context_tokens = [ord(c) + len(SPECIAL_TOKENS) for c in context_text[:30]]  # Simplified
    tokens.extend(context_tokens)
    
    # Pad to seq_len
    while len(tokens) < seq_len:
        tokens.append(SPECIAL_TOKENS['[PAD]'])
    
    tokens = torch.tensor(tokens[:seq_len], dtype=torch.long)
    
    # Create metadata
    in_span = torch.zeros(seq_len, dtype=torch.bool)
    span_id = torch.zeros(seq_len, dtype=torch.long)
    is_prefix = torch.zeros(seq_len, dtype=torch.bool)
    
    # Mark prefix tokens (including CLS)
    is_prefix[:cls_pos + 1] = True
    
    return tokens, in_span, span_id, is_prefix


def create_cocktail_party_sequence(
    prefix_text: str = "Answer:",
    context_text: str = "What is 2+2? The answer is",
    spans: List[str] = ["four", "five", "six"],
    seq_len: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a Cocktail Party sequence with metadata.
    Format: {prefix}[CLS]{context with [MASK]}[SPAN]option1[ES][SPAN]option2[ES]...[MASKQ]
    
    Returns:
        tokens, in_span, span_id, is_prefix
    """
    tokens = []
    
    # Add prefix tokens
    prefix_tokens = [ord(c) + len(SPECIAL_TOKENS) for c in prefix_text[:8]]
    tokens.extend(prefix_tokens)
    
    # Add CLS token
    cls_pos = len(tokens)
    tokens.append(SPECIAL_TOKENS['[CLS]'])
    
    # Add context with MASK
    context_tokens = [ord(c) + len(SPECIAL_TOKENS) for c in context_text[:15]]
    tokens.extend(context_tokens)
    tokens.append(SPECIAL_TOKENS['[MASK]'])  # Mask token in context
    
    context_end = len(tokens)
    
    # Add span islands
    span_positions = []
    for span_idx, span_text in enumerate(spans):
        span_start = len(tokens)
        tokens.append(SPECIAL_TOKENS['[SPAN]'])
        
        span_content_start = len(tokens)
        span_tokens = [ord(c) + len(SPECIAL_TOKENS) for c in span_text[:8]]
        tokens.extend(span_tokens)
        span_content_end = len(tokens)
        
        tokens.append(SPECIAL_TOKENS['[ES]'])
        span_end = len(tokens)
        
        span_positions.append((span_start, span_end, span_idx + 1))
    
    # Add MASKQ token
    maskq_pos = len(tokens)
    tokens.append(SPECIAL_TOKENS['[MASKQ]'])
    
    # Pad to seq_len
    while len(tokens) < seq_len:
        tokens.append(SPECIAL_TOKENS['[PAD]'])
    
    tokens = torch.tensor(tokens[:seq_len], dtype=torch.long)
    
    # Create metadata
    in_span = torch.zeros(seq_len, dtype=torch.bool)
    span_id = torch.zeros(seq_len, dtype=torch.long)
    is_prefix = torch.zeros(seq_len, dtype=torch.bool)
    
    # Mark prefix tokens (including CLS)
    is_prefix[:cls_pos + 1] = True
    
    # Mark span tokens and assign span IDs
    for span_start, span_end, span_idx in span_positions:
        if span_end <= seq_len:
            in_span[span_start:span_end] = True
            span_id[span_start:span_end] = span_idx
    
    # Mark MASKQ specially
    if maskq_pos < seq_len:
        span_id[maskq_pos] = -1  # Special marker for MASKQ
    
    return tokens, in_span, span_id, is_prefix


class TestTokenBehavior(unittest.TestCase):
    """Test suite for token behavior validation."""
    
    def test_teacher_forcing_basic(self):
        """Test basic Teacher Forcing attention patterns."""
        tokens, in_span, span_id, is_prefix = create_teacher_forcing_sequence()
        mask = create_reference_attention_mask(tokens, in_span, span_id, is_prefix, "teacher_forcing")
        
        # Find CLS position
        cls_token = SPECIAL_TOKENS['[CLS]']
        cls_pos = (tokens == cls_token).nonzero(as_tuple=True)[0]
        self.assertEqual(len(cls_pos), 1, "Should have exactly one CLS token")
        cls_pos = cls_pos[0].item()
        
        # Test 1: Prefix tokens (including CLS) should be bidirectional among themselves
        prefix_positions = torch.where(is_prefix)[0]
        for i in prefix_positions:
            for j in prefix_positions:
                if tokens[i] != SPECIAL_TOKENS['[PAD]'] and tokens[j] != SPECIAL_TOKENS['[PAD]']:
                    self.assertTrue(mask[i, j], f"Prefix token {i} should see prefix token {j}")
        
        # Test 2: Tokens after CLS should be causal
        valid_positions = torch.where(tokens != SPECIAL_TOKENS['[PAD]'])[0]
        non_prefix_positions = [pos for pos in valid_positions if not is_prefix[pos]]
        
        for i in non_prefix_positions:
            for j in non_prefix_positions:
                if j <= i:
                    self.assertTrue(mask[i, j], f"Token {i} should see earlier token {j} (causal)")
                else:
                    self.assertFalse(mask[i, j], f"Token {i} should not see future token {j} (causal)")
        
        # Test 3: Tokens after CLS should see CLS token
        for i in non_prefix_positions:
            self.assertTrue(mask[i, cls_pos], f"Token {i} should see CLS token at {cls_pos}")
        
        print("✓ Teacher Forcing basic patterns validated")
    
    def test_cocktail_party_basic(self):
        """Test basic Cocktail Party attention patterns."""
        tokens, in_span, span_id, is_prefix = create_cocktail_party_sequence()
        mask = create_reference_attention_mask(tokens, in_span, span_id, is_prefix, "cocktail_party")
        
        # Find special positions
        cls_pos = (tokens == SPECIAL_TOKENS['[CLS]']).nonzero(as_tuple=True)[0][0].item()
        maskq_positions = torch.where(span_id == -1)[0]
        
        # Test 1: Prefix tokens should be bidirectional among themselves
        prefix_positions = torch.where(is_prefix)[0]
        for i in prefix_positions:
            for j in prefix_positions:
                if tokens[i] != SPECIAL_TOKENS['[PAD]'] and tokens[j] != SPECIAL_TOKENS['[PAD]']:
                    self.assertTrue(mask[i, j], f"Prefix token {i} should see prefix token {j}")
        
        # Test 2: Context tokens should be causal within context and see prefix
        context_positions = []
        valid_positions = torch.where(tokens != SPECIAL_TOKENS['[PAD]'])[0]
        for pos in valid_positions:
            if not in_span[pos] and not is_prefix[pos] and span_id[pos] != -1:
                context_positions.append(pos)
        
        for i in context_positions:
            # Should see prefix
            for j in prefix_positions:
                if tokens[j] != SPECIAL_TOKENS['[PAD]']:
                    self.assertTrue(mask[i, j], f"Context token {i} should see prefix token {j}")
            
            # Should be causal within context
            for j in context_positions:
                if j <= i:
                    self.assertTrue(mask[i, j], f"Context token {i} should see earlier context token {j}")
                else:
                    self.assertFalse(mask[i, j], f"Context token {i} should not see future context token {j}")
        
        print("✓ Cocktail Party basic patterns validated")
    
    def test_cocktail_party_span_isolation(self):
        """Test span isolation in Cocktail Party attention."""
        tokens, in_span, span_id, is_prefix = create_cocktail_party_sequence()
        mask = create_reference_attention_mask(tokens, in_span, span_id, is_prefix, "cocktail_party")
        
        # Find span positions by span_id
        span_positions = {}
        valid_positions = torch.where(tokens != SPECIAL_TOKENS['[PAD]'])[0]
        
        for pos in valid_positions:
            sid = span_id[pos].item()
            if sid > 0:  # Valid span (not MASKQ or context)
                if sid not in span_positions:
                    span_positions[sid] = []
                span_positions[sid].append(pos)
        
        # Test 3: Span tokens should only see same span + context (not other spans)
        for span1_id, span1_positions in span_positions.items():
            for span2_id, span2_positions in span_positions.items():
                if span1_id != span2_id:
                    # Different spans should not see each other
                    for i in span1_positions:
                        for j in span2_positions:
                            self.assertFalse(mask[i, j], 
                                f"Span {span1_id} token {i} should not see span {span2_id} token {j}")
        
        # Test 4: Span tokens should see context
        context_positions = []
        for pos in valid_positions:
            if not in_span[pos] and not is_prefix[pos] and span_id[pos] != -1:
                context_positions.append(pos)
        
        for span_id_val, span_positions_list in span_positions.items():
            for span_pos in span_positions_list:
                for context_pos in context_positions:
                    self.assertTrue(mask[span_pos, context_pos], 
                        f"Span token {span_pos} should see context token {context_pos}")
        
        print("✓ Cocktail Party span isolation validated")
    
    def test_cocktail_party_maskq_behavior(self):
        """Test MASKQ token behavior in Cocktail Party attention."""
        tokens, in_span, span_id, is_prefix = create_cocktail_party_sequence()
        mask = create_reference_attention_mask(tokens, in_span, span_id, is_prefix, "cocktail_party")
        
        # Find MASKQ and span positions
        maskq_positions = torch.where(span_id == -1)[0]
        
        if len(maskq_positions) > 0:
            maskq_pos = maskq_positions[0].item()
            
            # Find span positions
            span_positions = []
            valid_positions = torch.where(tokens != SPECIAL_TOKENS['[PAD]'])[0]
            for pos in valid_positions:
                if in_span[pos] and span_id[pos] > 0:
                    span_positions.append(pos)
            
            # Find prefix positions
            prefix_positions = torch.where(is_prefix)[0]
            
            # Test 5: MASKQ should see all spans and prefix
            for span_pos in span_positions:
                self.assertTrue(mask[maskq_pos, span_pos], 
                    f"MASKQ token {maskq_pos} should see span token {span_pos}")
            
            for prefix_pos in prefix_positions:
                if tokens[prefix_pos] != SPECIAL_TOKENS['[PAD]']:
                    self.assertTrue(mask[maskq_pos, prefix_pos], 
                        f"MASKQ token {maskq_pos} should see prefix token {prefix_pos}")
            
            # Test 6: Span tokens should NOT see MASKQ
            for span_pos in span_positions:
                self.assertFalse(mask[span_pos, maskq_pos], 
                    f"Span token {span_pos} should not see MASKQ token {maskq_pos}")
        
        print("✓ Cocktail Party MASKQ behavior validated")
    
    def test_pad_token_handling(self):
        """Test that PAD tokens are properly ignored."""
        for task_type in ["teacher_forcing", "cocktail_party"]:
            if task_type == "teacher_forcing":
                tokens, in_span, span_id, is_prefix = create_teacher_forcing_sequence()
            else:
                tokens, in_span, span_id, is_prefix = create_cocktail_party_sequence()
            
            mask = create_reference_attention_mask(tokens, in_span, span_id, is_prefix, task_type)
            
            # Find PAD positions
            pad_positions = torch.where(tokens == SPECIAL_TOKENS['[PAD]'])[0]
            
            # PAD tokens should not attend to anything and nothing should attend to PAD
            for pad_pos in pad_positions:
                # PAD should not attend to anything
                self.assertTrue(torch.all(~mask[pad_pos, :]), 
                    f"PAD token {pad_pos} should not attend to anything in {task_type}")
                
                # Nothing should attend to PAD
                self.assertTrue(torch.all(~mask[:, pad_pos]), 
                    f"No token should attend to PAD token {pad_pos} in {task_type}")
        
        print("✓ PAD token handling validated")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test sequence with only prefix and CLS
        tokens = torch.tensor([
            ord('H') + len(SPECIAL_TOKENS), 
            ord('i') + len(SPECIAL_TOKENS),
            SPECIAL_TOKENS['[CLS]'],
            SPECIAL_TOKENS['[PAD]'],
            SPECIAL_TOKENS['[PAD]']
        ])
        
        in_span = torch.zeros(5, dtype=torch.bool)
        span_id = torch.zeros(5, dtype=torch.long)
        is_prefix = torch.tensor([True, True, True, False, False])
        
        mask = create_reference_attention_mask(tokens, in_span, span_id, is_prefix, "teacher_forcing")
        
        # All prefix tokens should see each other
        for i in range(3):
            for j in range(3):
                self.assertTrue(mask[i, j], f"Prefix tokens should be bidirectional")
        
        print("✓ Edge cases validated")


def visualize_attention_pattern(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    in_span: torch.Tensor,
    span_id: torch.Tensor,
    is_prefix: torch.Tensor,
    title: str = "Attention Pattern"
):
    """
    Visualize attention pattern for debugging and understanding.
    """
    print(f"\n=== {title} ===")
    
    # Create reverse mapping for special tokens
    special_token_names = {v: k for k, v in SPECIAL_TOKENS.items()}
    
    # Print token sequence with metadata
    print("Tokens and metadata:")
    print("Pos | Token | Prefix | InSpan | SpanID")
    print("-" * 40)
    
    for i in range(len(tokens)):
        if tokens[i] == SPECIAL_TOKENS['[PAD]']:
            continue
            
        token_val = tokens[i].item()
        if token_val in special_token_names:
            token_name = special_token_names[token_val]
        else:
            # Try to convert back to character
            char_val = token_val - len(SPECIAL_TOKENS)
            if 0 <= char_val <= 127:  # ASCII range
                token_name = f"'{chr(char_val)}'"
            else:
                token_name = f"T{token_val}"
        
        prefix_mark = "✓" if is_prefix[i] else " "
        span_mark = "✓" if in_span[i] else " "
        span_id_str = f"{span_id[i].item():2d}" if span_id[i] != 0 else "  "
        
        print(f"{i:3d} | {token_name:8s} | {prefix_mark:6s} | {span_mark:6s} | {span_id_str}")
    
    # Print attention matrix (only non-PAD positions)
    valid_positions = [i for i in range(len(tokens)) if tokens[i] != SPECIAL_TOKENS['[PAD]']]
    
    if len(valid_positions) <= 20:  # Only print for small sequences
        print(f"\nAttention Matrix ({len(valid_positions)}x{len(valid_positions)}):")
        print("     ", end="")
        for j in valid_positions:
            print(f"{j:3d}", end="")
        print()
        
        for i in valid_positions:
            print(f"{i:3d}: ", end="")
            for j in valid_positions:
                symbol = "█" if mask[i, j] else "·"
                print(f"  {symbol}", end="")
            print()
    
    print(f"Total attention connections: {mask.sum().item()}")


def create_small_demo_sequences():
    """Create small demo sequences for clear visualization."""
    
    # Teacher Forcing example: "Hi[CLS]OK"
    tf_tokens = torch.tensor([
        ord('H') + len(SPECIAL_TOKENS),  # 0: H (prefix)
        ord('i') + len(SPECIAL_TOKENS),  # 1: i (prefix)
        SPECIAL_TOKENS['[CLS]'],         # 2: [CLS] (prefix)
        ord('O') + len(SPECIAL_TOKENS),  # 3: O (context)
        ord('K') + len(SPECIAL_TOKENS),  # 4: K (context)
    ])
    
    tf_in_span = torch.zeros(5, dtype=torch.bool)
    tf_span_id = torch.zeros(5, dtype=torch.long)
    tf_is_prefix = torch.tensor([True, True, True, False, False])
    
    # Cocktail Party example: "Q[CLS]What[SPAN]A[ES][SPAN]B[ES][MASKQ]"
    cp_tokens = torch.tensor([
        ord('Q') + len(SPECIAL_TOKENS),  # 0: Q (prefix)
        SPECIAL_TOKENS['[CLS]'],         # 1: [CLS] (prefix)
        ord('W') + len(SPECIAL_TOKENS),  # 2: W (context)
        ord('h') + len(SPECIAL_TOKENS),  # 3: h (context)
        SPECIAL_TOKENS['[SPAN]'],        # 4: [SPAN] (span 1)
        ord('A') + len(SPECIAL_TOKENS),  # 5: A (span 1)
        SPECIAL_TOKENS['[ES]'],          # 6: [ES] (span 1)
        SPECIAL_TOKENS['[SPAN]'],        # 7: [SPAN] (span 2)
        ord('B') + len(SPECIAL_TOKENS),  # 8: B (span 2)
        SPECIAL_TOKENS['[ES]'],          # 9: [ES] (span 2)
        SPECIAL_TOKENS['[MASKQ]'],       # 10: [MASKQ]
    ])
    
    cp_in_span = torch.tensor([False, False, False, False, True, True, True, True, True, True, False])
    cp_span_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, -1])
    cp_is_prefix = torch.tensor([True, True, False, False, False, False, False, False, False, False, False])
    
    return (tf_tokens, tf_in_span, tf_span_id, tf_is_prefix), (cp_tokens, cp_in_span, cp_span_id, cp_is_prefix)


def demo_attention_patterns():
    """Demonstrate attention patterns with small, clear examples."""
    print("\n" + "="*60)
    print("DETAILED ATTENTION PATTERN DEMONSTRATION")
    print("="*60)
    
    (tf_tokens, tf_in_span, tf_span_id, tf_is_prefix), (cp_tokens, cp_in_span, cp_span_id, cp_is_prefix) = create_small_demo_sequences()
    
    # Teacher Forcing Demo
    print("\n--- Teacher Forcing Demo: 'Hi[CLS]OK' ---")
    tf_mask = create_reference_attention_mask(tf_tokens, tf_in_span, tf_span_id, tf_is_prefix, "teacher_forcing")
    visualize_attention_pattern(tf_tokens, tf_mask, tf_in_span, tf_span_id, tf_is_prefix, "Teacher Forcing Demo")
    
    # Cocktail Party Demo
    print("\n--- Cocktail Party Demo: 'Q[CLS]Wh[SPAN]A[ES][SPAN]B[ES][MASKQ]' ---")
    cp_mask = create_reference_attention_mask(cp_tokens, cp_in_span, cp_span_id, cp_is_prefix, "cocktail_party")
    visualize_attention_pattern(cp_tokens, cp_mask, cp_in_span, cp_span_id, cp_is_prefix, "Cocktail Party Demo")
def run_comprehensive_tests():
    """Run all tests and provide detailed output."""
    print("Starting comprehensive token behavior tests...")
    
    # First show detailed demos
    demo_attention_patterns()
    
    # Test Teacher Forcing
    print("\n" + "="*60)
    print("TESTING TEACHER FORCING")
    print("="*60)
    
    tokens, in_span, span_id, is_prefix = create_teacher_forcing_sequence()
    mask = create_reference_attention_mask(tokens, in_span, span_id, is_prefix, "teacher_forcing")
    visualize_attention_pattern(tokens, mask, in_span, span_id, is_prefix, "Teacher Forcing Pattern")
    
    # Test Cocktail Party
    print("\n" + "="*60)
    print("TESTING COCKTAIL PARTY")
    print("="*60)
    
    tokens, in_span, span_id, is_prefix = create_cocktail_party_sequence()
    mask = create_reference_attention_mask(tokens, in_span, span_id, is_prefix, "cocktail_party")
    visualize_attention_pattern(tokens, mask, in_span, span_id, is_prefix, "Cocktail Party Pattern")
    
    # Run unit tests
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_comprehensive_tests()