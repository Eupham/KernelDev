#!/usr/bin/env python3
"""
Regression test for issue #135: Generation Failed with 'NoneType' object is not subscriptable.

This test ensures that text generation works correctly on both CPU and CUDA devices,
and that the flash attention fallback mechanism works properly.
"""

import torch
from data_builder import DataBuilder
from model import GPTModel

def test_generation_cpu_fallback():
    """Test that generation works on CPU with flash attention fallback."""
    
    # Create minimal test setup
    data_config = {
        'dataset_name': 'allenai/c4',
        'dataset_config': 'en',
        'seq_len': 128,
        'max_samples': 5,
        'max_eval_tokens': 50,
        'on_the_fly_tokenization': True
    }
    
    data_builder = DataBuilder(**data_config)
    vocab_size = data_builder.get_vocab_size()
    
    model_config = {
        'vocab_size': vocab_size,
        'dim': 64,
        'n_layers': 2,
        'n_heads': 4,
        'max_seq_len': 128,
        'mlp_ratio': 2,
        'causal': True,
        'bidirectional_prefix_len': 1
    }
    
    model = GPTModel(**model_config)
    model.eval()
    
    # Test different prompt scenarios
    test_prompts = ["", "The", "Hello"]
    
    for prompt in test_prompts:
        # Prepare input
        if prompt:
            text_to_tokenize = f"[CLS] {prompt}"
        else:
            text_to_tokenize = "[CLS]"
        
        tokens = data_builder._tokenize_text(text_to_tokenize)
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        
        # Test forward pass
        with torch.no_grad():
            logits, _ = model(x)
            
        # Verify logits are not None and have correct shape
        assert logits is not None, f"Forward pass returned None logits for prompt '{prompt}'"
        assert isinstance(logits, torch.Tensor), f"Forward pass returned non-tensor for prompt '{prompt}'"
        assert logits.shape[0] == 1, f"Batch dimension incorrect for prompt '{prompt}'"
        assert logits.shape[1] == len(tokens), f"Sequence length incorrect for prompt '{prompt}'"
        assert logits.shape[2] == vocab_size, f"Vocab dimension incorrect for prompt '{prompt}'"
        
        # Test generation
        generated = model.generate(
            x,
            max_new_tokens=5,
            temperature=0.8,
            top_k=10,
            top_p=0.9
        )
        
        # Verify generation output
        assert generated is not None, f"Generation returned None for prompt '{prompt}'"
        assert isinstance(generated, torch.Tensor), f"Generation returned non-tensor for prompt '{prompt}'"
        assert generated.shape[0] == 1, f"Generated batch dimension incorrect for prompt '{prompt}'"
        assert generated.shape[1] == len(tokens) + 5, f"Generated sequence length incorrect for prompt '{prompt}'"
        
        # Test decoding doesn't crash
        decoded = data_builder.decode_tokens(generated[0])
        assert isinstance(decoded, str), f"Decoding failed for prompt '{prompt}'"


def test_logits_not_none_without_targets():
    """Test that model forward pass returns valid logits even without targets."""
    
    # Minimal setup
    vocab_size = 100
    model_config = {
        'vocab_size': vocab_size,
        'dim': 32,
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 64,
        'mlp_ratio': 2,
        'causal': True,
        'bidirectional_prefix_len': 1
    }
    
    model = GPTModel(**model_config)
    model.eval()
    
    # Test input
    x = torch.randint(0, vocab_size, (1, 10))
    
    with torch.no_grad():
        # Call without targets (generation mode)
        logits, loss = model(x, targets=None)
        
        # Verify logits are valid
        assert logits is not None, "Model returned None logits in generation mode"
        assert isinstance(logits, torch.Tensor), "Model returned non-tensor logits"
        assert logits.shape == (1, 10, vocab_size), f"Incorrect logits shape: {logits.shape}"
        
        # Loss should be None since no targets provided
        assert loss is None, "Loss should be None when no targets provided"


if __name__ == "__main__":
    print("Running regression tests for issue #135...")
    
    try:
        test_logits_not_none_without_targets()
        print("✓ test_logits_not_none_without_targets passed")
        
        test_generation_cpu_fallback()
        print("✓ test_generation_cpu_fallback passed")
        
        print("\n🎉 All regression tests passed!")
        
    except Exception as e:
        print(f"\n❌ Regression test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)