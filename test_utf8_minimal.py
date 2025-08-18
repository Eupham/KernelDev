#!/usr/bin/env python3
"""
Minimal test to verify UTF-8 tokenization improvements without dependencies.
"""

# Define special tokens (copied from data_builder.py)
SPECIAL_TOKENS = {
    '[PAD]': 0,
    '[CLS]': 1,
    '[MASK]': 2,
    '[SPAN]': 3,
    '[ES]': 4,
    '[MASKQ]': 5,
}
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)


class MinimalTokenizer:
    """Minimal tokenizer to test UTF-8 improvements."""
    
    def _tokenize_text(self, text: str) -> list:
        """
        Tokenize text using UTF-8 byte-level tokenization with proper multibyte support.
        
        This function properly handles multibyte UTF-8 characters by encoding the entire
        character and including all bytes in the token sequence.
        """
        tokens = []
        i = 0
        while i < len(text):
            found = False
            # First check for special tokens
            for token_str, token_id in SPECIAL_TOKENS.items():
                if text[i:].startswith(token_str):
                    tokens.append(token_id)
                    i += len(token_str)
                    found = True
                    break
            
            if not found:
                # Handle UTF-8 characters properly by encoding the entire character
                char = text[i]
                utf8_bytes = char.encode('utf-8')
                
                # Add all bytes of the UTF-8 character to tokens
                for byte_val in utf8_bytes:
                    tokens.append(byte_val + NUM_SPECIAL_TOKENS)
                
                i += 1
        return tokens

    def _detokenize_bytes(self, tokens: list, skip_special_tokens=False) -> str:
        """
        Detokenize UTF-8 byte tokens back to text with proper multibyte reconstruction.
        
        This function reconstructs multibyte UTF-8 characters by collecting bytes
        and decoding them properly.
        """
        special_token_map = {v: k for k, v in SPECIAL_TOKENS.items()}
        byte_sequence = []
        text_parts = []
        
        for t in tokens:
            if t in special_token_map:
                # Process any pending byte sequence first
                if byte_sequence:
                    try:
                        decoded_text = bytes(byte_sequence).decode('utf-8', errors='replace')
                        text_parts.append(decoded_text)
                        byte_sequence = []
                    except UnicodeDecodeError:
                        # Handle corrupted sequences gracefully
                        text_parts.append('�' * len(byte_sequence))
                        byte_sequence = []
                
                # Add special token if not skipping
                if not skip_special_tokens:
                    text_parts.append(special_token_map[t])
            else:
                # Collect bytes for UTF-8 reconstruction
                byte_val = t - NUM_SPECIAL_TOKENS
                if 0 <= byte_val <= 255:  # Valid byte range
                    byte_sequence.append(byte_val)
        
        # Process any remaining byte sequence
        if byte_sequence:
            try:
                decoded_text = bytes(byte_sequence).decode('utf-8', errors='replace')
                text_parts.append(decoded_text)
            except UnicodeDecodeError:
                # Handle corrupted sequences gracefully
                text_parts.append('�' * len(byte_sequence))
        
        return "".join(text_parts)


def test_utf8_tokenization():
    """Test the improved UTF-8 tokenization."""
    print("=== Testing UTF-8 Tokenization ===")
    
    tokenizer = MinimalTokenizer()
    
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
        
        # Show UTF-8 byte information
        utf8_bytes = test_text.encode('utf-8')
        print(f"  UTF-8 bytes: {len(utf8_bytes)} bytes, {len(test_text)} characters")
        
        # Tokenize
        try:
            tokens = tokenizer._tokenize_text(test_text)
            print(f"  Tokenized to {len(tokens)} tokens")
            
            # Show first few tokens for debugging
            print(f"  First few tokens: {tokens[:10] if len(tokens) > 10 else tokens}")
            
            # Detokenize
            reconstructed = tokenizer._detokenize_bytes(tokens)
            print(f"  Reconstructed: {repr(reconstructed)}")
            
            # Check if reconstruction matches original
            if reconstructed == test_text:
                print("  ✓ Perfect reconstruction")
            else:
                print("  ✗ Reconstruction mismatch!")
                print(f"    Original:      {test_text!r}")
                print(f"    Reconstructed: {reconstructed!r}")
                
                # Show byte-level comparison
                orig_bytes = test_text.encode('utf-8')
                recon_bytes = reconstructed.encode('utf-8')
                print(f"    Original bytes:      {list(orig_bytes)}")
                print(f"    Reconstructed bytes: {list(recon_bytes)}")
                
                all_passed = False
        
        except Exception as e:
            print(f"  ✗ Error during tokenization: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print(f"\nUTF-8 tokenization test: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    return all_passed


def test_old_vs_new_tokenization():
    """Compare old (broken) vs new (fixed) tokenization."""
    print("\n=== Comparing Old vs New Tokenization ===")
    
    def old_tokenize_text(text: str) -> list:
        """Old tokenization method that only captures first byte."""
        tokens = []
        i = 0
        while i < len(text):
            found = False
            for token_str, token_id in SPECIAL_TOKENS.items():
                if text[i:].startswith(token_str):
                    tokens.append(token_id)
                    i += len(token_str)
                    found = True
                    break
            if not found:
                # This is the buggy line - only captures first byte
                tokens.append(text[i].encode('utf-8')[0] + NUM_SPECIAL_TOKENS)
                i += 1
        return tokens
    
    def old_detokenize_bytes(tokens: list) -> str:
        """Old detokenization method."""
        special_token_map = {v: k for k, v in SPECIAL_TOKENS.items()}
        decoded_tokens = []
        for t in tokens:
            if t in special_token_map:
                decoded_tokens.append(special_token_map[t])
            else:
                decoded_tokens.append(chr(t - NUM_SPECIAL_TOKENS))
        return "".join(decoded_tokens)
    
    tokenizer = MinimalTokenizer()
    
    # Test with a problematic case
    test_text = "Café 🎉"
    print(f"Testing: {repr(test_text)}")
    print(f"UTF-8 bytes: {list(test_text.encode('utf-8'))}")
    
    # Old method
    print("\nOld method:")
    try:
        old_tokens = old_tokenize_text(test_text)
        old_reconstructed = old_detokenize_bytes(old_tokens)
        print(f"  Tokens: {old_tokens}")
        print(f"  Reconstructed: {repr(old_reconstructed)}")
        print(f"  Correct: {old_reconstructed == test_text}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # New method
    print("\nNew method:")
    try:
        new_tokens = tokenizer._tokenize_text(test_text)
        new_reconstructed = tokenizer._detokenize_bytes(new_tokens)
        print(f"  Tokens: {new_tokens}")
        print(f"  Reconstructed: {repr(new_reconstructed)}")
        print(f"  Correct: {new_reconstructed == test_text}")
        return new_reconstructed == test_text
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing UTF-8 Tokenization Improvements")
    print("=" * 50)
    
    test_results = []
    
    # Test 1: UTF-8 tokenization
    utf8_result = test_utf8_tokenization()
    test_results.append(("UTF-8 Tokenization", utf8_result))
    
    # Test 2: Compare old vs new
    comparison_result = test_old_vs_new_tokenization()
    test_results.append(("Old vs New Comparison", comparison_result))
    
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
        print("🎉 All tests PASSED!")
        print("\nThe UTF-8 tokenization fix properly handles multibyte characters!")
    else:
        print("⚠ Some tests FAILED!")
    
    return all_passed


if __name__ == "__main__":
    main()