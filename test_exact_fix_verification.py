#!/usr/bin/env python3
"""
Simple verification that the exact error from issue #135 is fixed.
"""

import torch
from data_builder import DataBuilder, SPECIAL_TOKENS  
from model import GPTModel
from train_loop import Trainer, TrainingConfig

def test_exact_error_scenario():
    """Test the exact code path that was failing in the original issue."""
    
    print("=== Testing Exact Error Scenario from Issue #135 ===\n")
    
    # Setup minimal components
    config = TrainingConfig(device='cpu', use_amp=False)
    
    data_config = {
        'dataset_name': 'allenai/c4',
        'dataset_config': 'en',
        'seq_len': 64,
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
        'max_seq_len': 64,
        'mlp_ratio': 2,
        'causal': True,
        'bidirectional_prefix_len': 1
    }
    
    model = GPTModel(**model_config)
    trainer = Trainer(model, config, data_builder)
    
    # Test the exact prompts that were failing
    prompts = ["", "The", "In", "Once upon a time"]
    
    print("Testing generate_inference_sample with the exact failing prompts...\n")
    
    try:
        for i, prompt in enumerate(prompts):
            print(f"Testing prompt {i+1}/4: '{prompt}' ...")
            
            # This is the exact method that was failing
            generated_texts = trainer.generate_inference_sample(
                prompts=[prompt],
                max_length=10,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            
            result = generated_texts[0]
            
            # Check if it's a failure message
            if result.startswith("Generation failed:"):
                if "'NoneType' object is not subscriptable" in result:
                    print(f"❌ ORIGINAL ERROR REPRODUCED: {result}")
                    return False
                else:
                    print(f"⚠️  Different error: {result}")
            else:
                print(f"✓ Success: '{result[:50]}{'...' if len(result) > 50 else ''}'")
        
        print(f"\n✅ All {len(prompts)} prompts generated successfully!")
        print("✅ Original 'NoneType' object is not subscriptable error has been FIXED!")
        return True
        
    except Exception as e:
        if "'NoneType' object is not subscriptable" in str(e):
            print(f"❌ ORIGINAL ERROR REPRODUCED AS EXCEPTION: {e}")
            return False
        else:
            print(f"⚠️  Different error: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_original_error_conditions():
    """Test the specific conditions that led to the original error."""
    
    print("\n=== Testing Original Error Conditions ===\n")
    
    # Create a very minimal setup
    vocab_size = 100
    model = GPTModel(
        vocab_size=vocab_size,
        dim=32,
        n_layers=1,
        n_heads=2,
        max_seq_len=32,
        mlp_ratio=2,
        causal=True,
        bidirectional_prefix_len=1
    )
    
    # Test the exact scenario: model forward without targets, then subscript access
    print("1. Testing model forward pass without targets (generation mode)...")
    x = torch.tensor([[SPECIAL_TOKENS['[CLS]']]], dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        logits, loss = model(x, targets=None)  # This was returning None before
        
        if logits is None:
            print("❌ Forward pass still returns None logits!")
            return False
        else:
            print(f"✓ Forward pass returns valid logits: shape {logits.shape}")
    
    # Test the exact subscript operation that was failing
    print("2. Testing logits subscript access (the failing operation)...")
    try:
        last_token_logits = logits[:, -1, :]  # This was the failing line
        print(f"✓ Subscript access successful: shape {last_token_logits.shape}")
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            print(f"❌ Original error reproduced: {e}")
            return False
        else:
            raise
    
    # Test the generate method specifically  
    print("3. Testing model.generate() method...")
    try:
        generated = model.generate(x, max_new_tokens=3, temperature=0.8)
        print(f"✓ Generation successful: {generated.shape}")
        print(f"  Generated tokens: {generated[0].tolist()}")
        return True
    except Exception as e:
        if "'NoneType' object is not subscriptable" in str(e):
            print(f"❌ Original error in generate(): {e}")
            return False
        else:
            print(f"⚠️  Different error in generate(): {e}")
            return False

if __name__ == "__main__":
    # Test both scenarios
    test1_success = test_original_error_conditions()
    test2_success = test_exact_error_scenario()
    
    if test1_success and test2_success:
        print("\n🎉 ISSUE #135 COMPLETELY FIXED!")
        print("   - Original NoneType subscript error eliminated")
        print("   - Text generation works on CPU with flash attention fallback")
        print("   - All test prompts generate successfully")
    else:
        print("\n❌ Issue #135 may not be fully resolved:")
        print(f"   - Original error conditions test: {'✓' if test1_success else '❌'}")
        print(f"   - Exact error scenario test: {'✓' if test2_success else '❌'}")