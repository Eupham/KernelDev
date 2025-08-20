#!/usr/bin/env python3
"""
Test script to reproduce the generation issue.
"""

import torch
import torch.nn.functional as F
from data_builder import DataBuilder
from model import GPTModel
from train_loop import TrainingConfig

def test_generation_issue():
    """Test the generation functionality to reproduce the NoneType error."""
    
    # Create a simple data builder
    data_config = {
        'dataset_name': 'allenai/c4',
        'dataset_config': 'en',
        'seq_len': 512,
        'max_samples': 100,
        'max_eval_tokens': 1000,
        'on_the_fly_tokenization': True
    }
    
    print("Creating data builder...")
    data_builder = DataBuilder(**data_config)
    vocab_size = data_builder.get_vocab_size()
    print(f"Vocab size: {vocab_size}")
    
    # Create a simple model
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
    model.eval()
    
    # Test tokenization
    print("\nTesting tokenization...")
    test_prompts = ["", "The", "In", "Once upon a time"]
    
    for prompt in test_prompts:
        try:
            if prompt:
                text_to_tokenize = f"[CLS] {prompt}"
            else:
                text_to_tokenize = "[CLS]"
            
            print(f"Tokenizing: '{text_to_tokenize}'")
            tokens = data_builder._tokenize_text(text_to_tokenize)
            print(f"Tokens: {tokens[:10]}...")  # Show first 10 tokens
            
            # Convert to tensor
            x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            print(f"Tensor shape: {x.shape}")
            
            # Test model forward pass
            print("Testing model forward pass...")
            with torch.no_grad():
                logits, _ = model(x)
                print(f"Logits shape: {logits.shape}")
                print(f"Logits type: {type(logits)}")
                
                # Test generation
                print("Testing generation...")
                generated = model.generate(
                    x,
                    max_new_tokens=5,  # Generate just a few tokens
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9
                )
                print(f"Generated shape: {generated.shape}")
                print(f"Generated type: {type(generated)}")
                print(f"Generated tensor: {generated}")
                
                # Test decoding
                print("Testing decoding...")
                if generated is not None:
                    generated_tokens = generated[0].cpu().tolist()
                    print(f"Generated tokens: {generated_tokens}")
                    decoded_text = data_builder.decode_tokens(generated_tokens)
                    print(f"Decoded text: '{decoded_text}'")
                else:
                    print("ERROR: Generated tensor is None!")
                    
        except Exception as e:
            print(f"ERROR for prompt '{prompt}': {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_generation_issue()