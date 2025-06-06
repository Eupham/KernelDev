#!/usr/bin/env python3
"""
Test script to verify inference sampling functionality.
"""

import torch
from data_builder import create_data_builder
from model import GPTModel
from train_loop import TrainingConfig, Trainer

def test_inference_sampling():
    """Test inference sampling functionality."""
    print("=== Testing Inference Sampling ===")
    
    # Create a small test configuration
    data_config = {
        'dataset_name': 'allenai/c4',
        'dataset_config': 'en',
        'seq_len': 128,
        'max_samples': 100,  # Small number for quick test
        'max_eval_tokens': 5000
    }
    
    # Create data builder
    print("Creating data builder...")
    data_builder = create_data_builder(**data_config)
    
    # Create a small model
    print("Creating model...")
    model_config = {
        'vocab_size': data_builder.get_vocab_size(),
        'dim': 128,
        'n_layers': 2,
        'n_heads': 4,
        'max_seq_len': 256,
        'mlp_ratio': 4,
        'causal': True
    }
    model = GPTModel(**model_config)
    
    # Create training config with inference parameters
    training_config = TrainingConfig(
        num_epochs=1,
        learning_rate=1e-4,
        batch_size=2,
        save_every=50,
        eval_every=25,
        log_every=10,
        checkpoint_dir="test_checkpoints",
        device="auto",
        use_amp=False,
        scaler=None,
        # Inference sampling parameters
        inference_prompts=["", "The", "AI"],
        inference_max_length=30,
        inference_temperature=0.8,
        inference_top_k=20,
        inference_top_p=0.9
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(model, training_config, data_builder)
    
    # Test individual methods
    print("\n=== Testing Individual Methods ===")
    
    # Test tokenization
    print("Testing tokenization...")
    test_text = "Hello world"
    tokens = data_builder._tokenize_text(test_text)
    decoded = data_builder.decode_tokens(tokens)
    print(f"Original: '{test_text}'")
    print(f"Tokens: {tokens[:10]}...")
    print(f"Decoded: '{decoded}'")
    
    # Test perplexity calculation (with minimal data)
    print("\nTesting perplexity calculation...")
    try:
        datasets = data_builder.create_datasets()
        if 'validation' in datasets:
            dataloaders = data_builder.create_dataloaders(datasets, batch_size=2, num_workers=0)
            val_loader = dataloaders['validation']
            perplexity = trainer.calculate_perplexity(val_loader, max_batches=2)
            print(f"Perplexity: {perplexity:.2f}")
        else:
            print("No validation data available")
    except Exception as e:
        print(f"Perplexity test failed: {e}")
    
    # Test inference sampling
    print("\nTesting inference sampling...")
    try:
        generated_texts = trainer.generate_inference_sample(
            prompts=["", "The", "AI"],
            max_length=20,
            temperature=0.8,
            top_k=20,
            top_p=0.9
        )
        
        for i, (prompt, text) in enumerate(zip(["", "The", "AI"], generated_texts)):
            print(f"Sample {i+1}:")
            if prompt:
                print(f"  Prompt: '{prompt}' → '{text}'")
            else:
                print(f"  No prompt → '{text}'")
    except Exception as e:
        print(f"Inference sampling test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test JSON saving
    print("\nTesting JSON saving...")
    try:
        trainer.save_inference_sample(
            step=100,
            val_loss=2.5,
            perplexity=12.0,
            generated_texts=["sample text 1", "sample text 2", "sample text 3"],
            prompts=["", "The", "AI"]
        )
        print("✓ JSON saving successful")
    except Exception as e:
        print(f"JSON saving test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Inference Sampling Test Complete ===")

if __name__ == "__main__":
    test_inference_sampling()
