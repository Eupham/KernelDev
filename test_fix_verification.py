#!/usr/bin/env python3
"""
Test script to verify that the generation issue is fixed.
This mimics the exact scenario from the issue report.
"""

import torch
import torch.nn.functional as F
from data_builder import DataBuilder
from model import GPTModel
from train_loop import Trainer, TrainingConfig

def test_inference_sample_generation():
    """Test the exact scenario from the issue: generate_inference_sample."""
    
    # Create minimal config
    config = TrainingConfig(
        device='cpu',  # Use CPU to test our fix
        use_amp=False,
        inference_prompts=["", "The", "In", "Once upon a time"],
        inference_max_length=50,
        inference_temperature=0.8,
        inference_top_k=50,
        inference_top_p=0.9
    )
    
    # Create data builder
    data_config = {
        'dataset_name': 'allenai/c4',
        'dataset_config': 'en',
        'seq_len': 512,
        'max_samples': 10,  # Very small for quick test
        'max_eval_tokens': 100,
        'on_the_fly_tokenization': True
    }
    
    print("Creating data builder...")
    data_builder = DataBuilder(**data_config)
    vocab_size = data_builder.get_vocab_size()
    
    # Create model
    model_config = {
        'vocab_size': vocab_size,
        'dim': 256,
        'n_layers': 4,
        'n_heads': 8,
        'max_seq_len': 512,
        'mlp_ratio': 4,
        'causal': True,
        'bidirectional_prefix_len': 1
    }
    
    print("Creating model...")
    model = GPTModel(**model_config)
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(model, config, data_builder)
    
    # Test the exact method from the issue
    print("\nTesting generate_inference_sample...")
    try:
        generated_texts = trainer.generate_inference_sample(
            prompts=config.inference_prompts,
            max_length=config.inference_max_length,
            temperature=config.inference_temperature,
            top_k=config.inference_top_k,
            top_p=config.inference_top_p
        )
        
        print("✓ Generation succeeded!")
        for i, (prompt, generated_text) in enumerate(zip(config.inference_prompts, generated_texts)):
            if prompt:
                print(f"Prompt: '{prompt}' → '{generated_text}'")
            else:
                print(f"No prompt → '{generated_text}'")
                
        # Verify no failures
        failures = [text for text in generated_texts if text.startswith("Generation failed:")]
        if failures:
            print(f"\n❌ Found {len(failures)} generation failures:")
            for failure in failures:
                print(f"  {failure}")
            return False
        else:
            print(f"\n✓ All {len(generated_texts)} generations succeeded!")
            return True
            
    except Exception as e:
        print(f"❌ Generation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_generation_methods():
    """Test individual generation methods to ensure robustness."""
    
    # Create minimal setup
    data_config = {
        'dataset_name': 'allenai/c4',
        'dataset_config': 'en', 
        'seq_len': 256,
        'max_samples': 5,
        'max_eval_tokens': 50,
        'on_the_fly_tokenization': True
    }
    
    data_builder = DataBuilder(**data_config)
    vocab_size = data_builder.get_vocab_size()
    
    model_config = {
        'vocab_size': vocab_size,
        'dim': 128,
        'n_layers': 2,
        'n_heads': 4,
        'max_seq_len': 256,
        'mlp_ratio': 4,
        'causal': True,
        'bidirectional_prefix_len': 1
    }
    
    model = GPTModel(**model_config)
    model.eval()
    
    print("\nTesting direct model.generate()...")
    try:
        # Test direct generation
        test_input = torch.tensor([[1]], dtype=torch.long)  # Just [CLS] token
        
        with torch.no_grad():
            generated = model.generate(
                test_input,
                max_new_tokens=10,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            
        print(f"✓ Direct generation succeeded: shape {generated.shape}")
        print(f"  Generated tokens: {generated[0].tolist()}")
        
        # Test decoding
        decoded = data_builder.decode_tokens(generated[0])
        print(f"  Decoded text: '{decoded}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing Fix for Generation Issue #135 ===\n")
    
    # Test 1: Individual methods
    success1 = test_individual_generation_methods()
    
    # Test 2: Full inference sample generation (the original failing scenario)
    success2 = test_inference_sample_generation()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The generation issue has been fixed.")
    else:
        print("\n❌ Some tests failed. The issue may not be fully resolved.")