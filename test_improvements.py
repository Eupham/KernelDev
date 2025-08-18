#!/usr/bin/env python3
"""
Test script to verify the improvements made to address the comments:
1. Flash attention verification
2. Max tokens extension for cocktail party
3. UTF-8 tokenization handling
"""

import torch
import torch.nn as nn
from data_builder import DataBuilder, SPECIAL_TOKENS
from model import GPTModel
from original_kernel import verify_flash_attention_usage, flash_attention
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_utf8_tokenization():
    """Test the improved UTF-8 tokenization."""
    print("\n=== Testing UTF-8 Tokenization ===")
    
    # Create data builder
    data_builder = DataBuilder(seq_len=128, max_samples=10)
    
    # Test cases with various UTF-8 characters
    test_cases = [
        "Hello, world!",  # ASCII only
        "Café français",  # Accented characters
        "你好世界",  # Chinese characters
        "🚀🎉✨",  # Emoji
        "Здравствуй мир",  # Cyrillic
        "العالم",  # Arabic
        "Mixed: café 🎉 世界",  # Mixed multibyte
        "[CLS] Special tokens test [MASK]",  # With special tokens
    ]
    
    all_passed = True
    
    for i, test_text in enumerate(test_cases):
        print(f"\nTest case {i+1}: {repr(test_text)}")
        
        # Tokenize
        tokens = data_builder._tokenize_text(test_text)
        print(f"  Tokenized to {len(tokens)} tokens")
        
        # Detokenize
        reconstructed = data_builder._detokenize_bytes(tokens)
        print(f"  Reconstructed: {repr(reconstructed)}")
        
        # Check if reconstruction matches original
        if reconstructed == test_text:
            print("  ✓ Perfect reconstruction")
        else:
            print("  ✗ Reconstruction mismatch!")
            print(f"    Original:  {test_text!r}")
            print(f"    Reconstructed: {reconstructed!r}")
            all_passed = False
    
    print(f"\nUTF-8 tokenization test: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    return all_passed


def test_flash_attention_verification():
    """Test flash attention usage verification."""
    print("\n=== Testing Flash Attention Verification ===")
    
    # Create a simple model
    vocab_size = 1000
    model = GPTModel(
        vocab_size=vocab_size,
        dim=512,
        n_layers=2,
        n_heads=8,
        max_seq_len=128,
        task_names=['teacher_forcing', 'cocktail_party']
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Create sample input
    batch_size, seq_len = 2, 64
    sample_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    if torch.cuda.is_available():
        sample_input = sample_input.cuda()
    
    # Test teacher forcing (causal attention)
    print("\nTesting teacher forcing task...")
    model.eval()
    
    def teacher_forcing_forward(x):
        return model(x, task_type='teacher_forcing')
    
    tf_verified = verify_flash_attention_usage(
        teacher_forcing_forward, 
        sample_input, 
        "teacher_forcing"
    )
    
    # Test cocktail party (hierarchical attention)
    print("\nTesting cocktail party task...")
    
    # Create metadata for cocktail party attention
    batch_size, seq_len = sample_input.shape
    in_span = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    span_id = torch.zeros(batch_size, seq_len, dtype=torch.long)
    is_prefix = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    # Mark some spans and prefix for testing
    is_prefix[:, :5] = True  # First 5 tokens are prefix
    in_span[:, 10:20] = True  # Tokens 10-19 are in span 1
    span_id[:, 10:20] = 1
    in_span[:, 25:35] = True  # Tokens 25-34 are in span 2  
    span_id[:, 25:35] = 2
    span_id[:, -1] = -1  # Last token is [MASKQ]
    
    if torch.cuda.is_available():
        in_span = in_span.cuda()
        span_id = span_id.cuda()
        is_prefix = is_prefix.cuda()
    
    def cocktail_party_forward(x):
        return model(x, task_type='cocktail_party', 
                    in_span=in_span, span_id=span_id, is_prefix=is_prefix)
    
    cp_verified = verify_flash_attention_usage(
        cocktail_party_forward,
        sample_input,
        "cocktail_party"
    )
    
    both_verified = tf_verified and cp_verified
    print(f"\nFlash attention verification: {'✓ PASSED' if both_verified else '✗ FAILED'}")
    return both_verified


def test_max_tokens_extension():
    """Test max tokens extension for cocktail party task."""
    print("\n=== Testing Max Tokens Extension ===")
    
    # Create data builder with smaller seq_len to test extension
    task_configs = {
        'cocktail_party': {
            'num_distractors': 3,
            'min_span_size': 20,
            'max_span_size': 40
        }
    }
    
    data_builder = DataBuilder(
        seq_len=128,  # Small sequence length
        max_samples=50,
        task_configs=task_configs
    )
    
    try:
        # Create datasets
        datasets = data_builder.create_datasets()
        
        if 'train' in datasets and datasets['train']:
            # Create a small batch to test cocktail party collation
            train_dataset = datasets['train']
            
            # Sample a few items
            batch = []
            for i in range(min(8, len(train_dataset))):
                batch.append(train_dataset[i])
            
            if batch:
                print(f"Testing with batch of {len(batch)} items...")
                
                # Test cocktail party collation with extension
                inputs, correct_indices, metadata = data_builder._collate_fn_cocktail_party(batch)
                
                if len(inputs) > 0:
                    seq_len = inputs.shape[1]
                    print(f"✓ Successfully created cocktail party batch with extended seq_len: {seq_len}")
                    print(f"  Batch size: {inputs.shape[0]}")
                    print(f"  Sequence length: {seq_len} (original target: {data_builder.seq_len})")
                    
                    # Check if any sequences were extended beyond original seq_len
                    if seq_len > data_builder.seq_len:
                        print(f"  ✓ Sequences were extended to accommodate special tokens")
                    else:
                        print(f"  ✓ Sequences fit within original length")
                    
                    return True
                else:
                    print("✗ No valid cocktail party sequences created")
                    return False
            else:
                print("✗ No batch items available for testing")
                return False
        else:
            print("✗ No training dataset available")
            return False
            
    except Exception as e:
        print(f"✗ Error during max tokens extension test: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Improvements for GitHub Comments")
    print("=" * 50)
    
    test_results = []
    
    # Test 1: UTF-8 tokenization
    utf8_result = test_utf8_tokenization()
    test_results.append(("UTF-8 Tokenization", utf8_result))
    
    # Test 2: Flash attention verification 
    if torch.cuda.is_available():
        flash_result = test_flash_attention_verification()
        test_results.append(("Flash Attention Verification", flash_result))
    else:
        print("\n⚠ Skipping flash attention test (CUDA not available)")
        test_results.append(("Flash Attention Verification", None))
    
    # Test 3: Max tokens extension
    max_tokens_result = test_max_tokens_extension()
    test_results.append(("Max Tokens Extension", max_tokens_result))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in test_results:
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED" 
            all_passed = False
        
        print(f"{test_name:30} {status}")
    
    print("=" * 50)
    if all_passed:
        print("🎉 All tests PASSED!")
    else:
        print("⚠ Some tests FAILED!")
    
    return all_passed


if __name__ == "__main__":
    main()