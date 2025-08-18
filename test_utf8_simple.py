#!/usr/bin/env python3
"""
Simple test script to verify UTF-8 tokenization improvements.
"""

import sys
import os
sys.path.append('.')

from data_builder import DataBuilder, SPECIAL_TOKENS

def test_utf8_tokenization():
    """Test the improved UTF-8 tokenization."""
    print("=== Testing UTF-8 Tokenization ===")
    
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
        try:
            tokens = data_builder._tokenize_text(test_text)
            print(f"  Tokenized to {len(tokens)} tokens")
            
            # Show token details for debugging
            print(f"  First few tokens: {tokens[:10] if len(tokens) > 10 else tokens}")
            
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
                
                # Show character-by-character comparison
                for j, (orig_char, recon_char) in enumerate(zip(test_text, reconstructed)):
                    if orig_char != recon_char:
                        print(f"    Mismatch at position {j}: {orig_char!r} != {recon_char!r}")
                        break
                
                all_passed = False
        
        except Exception as e:
            print(f"  ✗ Error during tokenization: {e}")
            all_passed = False
    
    print(f"\nUTF-8 tokenization test: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    return all_passed


def test_special_tokens():
    """Test that special tokens are handled correctly."""
    print("\n=== Testing Special Token Handling ===")
    
    data_builder = DataBuilder(seq_len=128, max_samples=10)
    
    # Test special token detection
    test_text = "[CLS] This is a test [MASK] with special tokens [SPAN] content [ES] and [MASKQ]"
    
    print(f"Testing: {repr(test_text)}")
    
    tokens = data_builder._tokenize_text(test_text)
    print(f"Tokenized to {len(tokens)} tokens")
    
    # Check that special tokens are properly identified
    special_found = []
    for token in tokens:
        if token in SPECIAL_TOKENS.values():
            token_name = [k for k, v in SPECIAL_TOKENS.items() if v == token][0]
            special_found.append(token_name)
    
    print(f"Special tokens found: {special_found}")
    
    reconstructed = data_builder._detokenize_bytes(tokens)
    print(f"Reconstructed: {repr(reconstructed)}")
    
    success = reconstructed == test_text
    print(f"Special tokens test: {'✓ PASSED' if success else '✗ FAILED'}")
    return success


def test_byte_boundary_handling():
    """Test handling of bytes at UTF-8 character boundaries."""
    print("\n=== Testing Byte Boundary Handling ===")
    
    data_builder = DataBuilder(seq_len=128, max_samples=10)
    
    # Test text with various multibyte characters
    test_text = "🔥 Test with emoji 🎉 and accénted cháracters"
    
    print(f"Testing: {repr(test_text)}")
    
    # Get UTF-8 bytes to understand what we're dealing with
    utf8_bytes = test_text.encode('utf-8')
    print(f"UTF-8 byte length: {len(utf8_bytes)}")
    print(f"Character length: {len(test_text)}")
    
    tokens = data_builder._tokenize_text(test_text)
    print(f"Token count: {len(tokens)}")
    
    reconstructed = data_builder._detokenize_bytes(tokens)
    print(f"Reconstructed: {repr(reconstructed)}")
    
    success = reconstructed == test_text
    print(f"Byte boundary test: {'✓ PASSED' if success else '✗ FAILED'}")
    
    if not success:
        # Show detailed comparison
        print("Detailed comparison:")
        print(f"  Original bytes:      {test_text.encode('utf-8')}")
        print(f"  Reconstructed bytes: {reconstructed.encode('utf-8')}")
    
    return success


def main():
    """Run all UTF-8 tests."""
    print("Testing UTF-8 Tokenization Improvements")
    print("=" * 50)
    
    test_results = []
    
    # Test 1: Basic UTF-8 tokenization
    utf8_result = test_utf8_tokenization()
    test_results.append(("UTF-8 Tokenization", utf8_result))
    
    # Test 2: Special token handling
    special_result = test_special_tokens()
    test_results.append(("Special Token Handling", special_result))
    
    # Test 3: Byte boundary handling
    boundary_result = test_byte_boundary_handling()
    test_results.append(("Byte Boundary Handling", boundary_result))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in test_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        if not result:
            all_passed = False
        
        print(f"{test_name:30} {status}")
    
    print("=" * 50)
    if all_passed:
        print("🎉 All UTF-8 tests PASSED!")
    else:
        print("⚠ Some tests FAILED!")
    
    return all_passed


if __name__ == "__main__":
    main()