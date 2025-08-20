#!/usr/bin/env python3
"""
Test to simulate the exact training scenario that was failing in issue #135.
"""

import torch
from data_builder import DataBuilder  
from model import GPTModel
from train_loop import Trainer, TrainingConfig

def test_training_with_inference():
    """Simulate training loop with inference generation at step intervals."""
    
    # Minimal training config
    config = TrainingConfig(
        device='cpu',
        use_amp=False,
        num_epochs=1,
        learning_rate=0.001,
        eval_every=5,  # Generate inference every 5 steps
        log_every=5,
        inference_prompts=["", "The", "In", "Once upon a time"],
        inference_max_length=20,
        inference_temperature=0.8,
        inference_top_k=50,
        inference_top_p=0.9
    )
    
    # Data setup
    data_config = {
        'dataset_name': 'allenai/c4',
        'dataset_config': 'en',
        'seq_len': 128,
        'max_samples': 20,
        'max_eval_tokens': 100,
        'on_the_fly_tokenization': True
    }
    
    print("Setting up data and model...")
    data_builder = DataBuilder(**data_config)
    vocab_size = data_builder.get_vocab_size()
    
    model_config = {
        'vocab_size': vocab_size,
        'dim': 128,
        'n_layers': 2,
        'n_heads': 4,
        'max_seq_len': 128,
        'mlp_ratio': 2,
        'causal': True,
        'bidirectional_prefix_len': 1
    }
    
    model = GPTModel(**model_config)
    trainer = Trainer(model, config, data_builder)
    
    # Create minimal dataloaders
    try:
        datasets = data_builder.create_datasets()
        dataloaders = data_builder.create_dataloaders(
            datasets=datasets,
            batch_size=2,
            num_workers=0,
            shuffle_train=True
        )
    except Exception as e:
        print(f"Could not create dataloaders: {e}")
        # Create fake data for testing
        fake_data = [(torch.randint(0, vocab_size, (128,)), torch.randint(0, vocab_size, (128,))) for _ in range(10)]
        from torch.utils.data import DataLoader
        dataloaders = {
            'train': {'teacher_forcing': DataLoader(fake_data, batch_size=2, shuffle=True)},
            'validation': {'teacher_forcing': DataLoader(fake_data[:3], batch_size=2)}
        }
    
    print("Testing inference generation during 'training'...")
    
    # Simulate a few training steps with inference generation
    step_count = 0
    for epoch in range(1):
        for batch_idx, batch in enumerate(dataloaders['train']['teacher_forcing']):
            step_count += 1
            
            # Simulate training step (just a forward pass)
            inputs, targets = batch
            model.train()
            logits, loss = model(inputs, targets=targets, task_name='teacher_forcing')
            
            print(f"Step {step_count}: Training loss = {loss:.4f}")
            
            # Test inference generation at eval intervals (like the original issue)
            if step_count % config.eval_every == 0:
                print(f"\n=== Generating Inference Sample at Step {step_count} ===")
                
                try:
                    # This is the exact call that was failing in the issue
                    generated_texts = trainer.generate_inference_sample(
                        prompts=config.inference_prompts,
                        max_length=config.inference_max_length,
                        temperature=config.inference_temperature,
                        top_k=config.inference_top_k,
                        top_p=config.inference_top_p
                    )
                    
                    # Print results like in the original issue
                    print(f"=== Inference Sample at Step {step_count} ===")
                    print(f"Validation Loss: {loss:.4f}")
                    for prompt, generated_text in zip(config.inference_prompts, generated_texts):
                        if prompt:
                            print(f"Prompt: '{prompt}' → '{generated_text}'")
                        else:
                            print(f"No prompt → '{generated_text}'")
                    print("=" * 50)
                    
                    # Check for failures
                    failures = [text for text in generated_texts if text.startswith("Generation failed:")]
                    if failures:
                        print(f"❌ Found generation failures: {failures}")
                        return False
                        
                except Exception as e:
                    print(f"❌ Inference generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            
            # Stop after a few steps for testing
            if step_count >= 10:
                break
        break
    
    print("\n✓ Training simulation with inference generation completed successfully!")
    return True

if __name__ == "__main__":
    print("=== Testing Training Loop with Inference Generation ===")
    
    success = test_training_with_inference()
    
    if success:
        print("\n🎉 Test passed! The exact scenario from issue #135 now works correctly.")
    else:
        print("\n❌ Test failed! Issue #135 may not be fully resolved.")