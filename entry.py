import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our custom modules
from model import GPTModel
from data_builder import DataBuilder, create_data_builder
from train_loop import Trainer, TrainingConfig, create_trainer


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    print("=== GPT Model Training with Flash Attention ===")
    print("Setting up configuration...")
    
    # Data configuration (T4-optimized)
    data_config = {
        'dataset_name': 'allenai/c4',
        'dataset_config': 'en',
        'seq_len': 1024,  # Reduced for T4
        'max_samples': 2000,  # Default for C4
        'max_eval_tokens': 25000  # Reduced for faster evaluation
    }
    
    # Model configuration (T4-optimized)
    model_config = {
        'vocab_size': 256,  # UTF-8 byte vocabulary size
        'dim': 1024,  # Reduced for T4
        'n_layers': 8,  # Reduced for T4
        'n_heads': 16,  # Reduced for T4
        'max_seq_len': 1024,  # Reduced for T4
        'mlp_ratio': 4,  # Reduced for T4
        'causal': True  # Using causal attention
    }
    
    # Training configuration (T4-optimized)
    training_config = TrainingConfig(
        num_epochs=2,  # Reduced for T4
        learning_rate=5e-4,  # Slightly higher for smaller model
        weight_decay=0.01,
        warmup_steps=50,  # Reduced for T4
        max_grad_norm=1.0,
        save_every=300,  # Reduced for T4
        eval_every=100,  # Reduced for T4
        log_every=25,  # Reduced for T4
        checkpoint_dir="checkpoints"
    )
    
    batch_size = 2  # Small batch size for T4
    
    print(f"Device: {training_config.device}")
    print(f"Model config: {model_config}")
    print(f"Data config: {data_config}")
    print(f"Training config: batch_size={batch_size}, epochs={training_config.num_epochs}")
    
    # Create data builder
    print("\n=== Loading and Processing Data ===")
    data_builder = create_data_builder(**data_config)
    
    # Create dataloaders
    try:
        dataloaders = data_builder.create_dataloaders(
            batch_size=batch_size,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            shuffle_train=True
        )
        
        # Update vocab size based on actual tokenizer
        actual_vocab_size = data_builder.get_vocab_size()
        model_config['vocab_size'] = actual_vocab_size
        print(f"Confirmed vocab_size: {actual_vocab_size} (UTF-8 bytes)")
        
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        print("This might be due to missing datasets library or network issues.")
        print("Please install with: pip install datasets")
        return
    
    # Show data info
    for split_name, dataloader in dataloaders.items():
        print(f"{split_name}: {len(dataloader)} batches of size {batch_size}")
    
    # Test a batch
    if 'train' in dataloaders:
        print("\n=== Data Sample ===")
        for x, y in dataloaders['train']:
            print(f"Batch shape: {x.shape}")
            print(f"Sample tokens: {x[0][:20].tolist()}")
            
            # Decode sample text
            sample_text = data_builder.decode_tokens(x[0][:50])
            print(f"Sample text: '{sample_text[:100]}...'")
            break
    
    # Initialize model
    print(f"\n=== Initializing Model ===")
    model = GPTModel(**model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    print(f"\n=== Setting up Trainer ===")
    trainer = create_trainer(
        model=model,
        config=training_config,
        data_builder=data_builder
    )
    
    # Initial evaluation
    print(f"\n=== Initial Evaluation ===")
    if 'train' in dataloaders and 'validation' in dataloaders:
        # Limit initial evaluation to 500 batches for speed
        initial_train_loss = trainer.evaluate(dataloaders['train'], max_batches=500)
        initial_val_loss = trainer.evaluate(dataloaders['validation'])
        print(f"Initial training loss: {initial_train_loss:.4f}")
        print(f"Initial validation loss: {initial_val_loss:.4f}")
    
    # Test causal vs non-causal attention
    print(f"\n=== Testing Causal vs Non-Causal Attention ===")
    test_causal_attention(model, dataloaders, training_config.device, data_builder)
    
    # Start training
    print(f"\n=== Starting Training ===")
    try:
        trainer.train(
            train_loader=dataloaders.get('train'),
            val_loader=dataloaders.get('validation')
        )
        
        print(f"\n=== Training Completed ===")
        
        # Final evaluation
        if 'train' in dataloaders and 'validation' in dataloaders:
            # Limit final evaluation to 500 batches for speed
            final_train_loss = trainer.evaluate(dataloaders['train'], max_batches=500)
            final_val_loss = trainer.evaluate(dataloaders['validation'])
            print(f"Final training loss: {final_train_loss:.4f}")
            print(f"Final validation loss: {final_val_loss:.4f}")
            
            # Show improvement
            if 'initial_train_loss' in locals():
                train_improvement = initial_train_loss - final_train_loss
                val_improvement = initial_val_loss - final_val_loss
                print(f"Training loss improvement: {train_improvement:.4f}")
                print(f"Validation loss improvement: {val_improvement:.4f}")
        
        # Plot training curves
        print(f"\n=== Plotting Results ===")
        curves_path = Path(training_config.checkpoint_dir) / "training_curves.png"
        trainer.plot_training_curves(save_path=str(curves_path))
        
        # Test text generation
        print(f"\n=== Testing Text Generation ===")
        test_generation(trainer, data_builder)
        
        # Show best metrics
        print(f"\n=== Best Results ===")
        print(f"Best validation loss: {trainer.metrics.best_val_loss:.4f} at step {trainer.metrics.best_step}")
        print(f"Total training steps: {trainer.metrics.total_steps}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== Training Session Complete ===")


def test_causal_attention(model, dataloaders, device, data_builder):
    """Test the difference between causal and non-causal attention."""
    if 'train' not in dataloaders:
        return
    
    # Get a sample batch
    for x, y in dataloaders['train']:
        x = x.to(device)
        break
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Test with causal=True (default)
        print("Testing with causal=True...")
        logits_causal, _ = model(x)
        
        # Test with causal=False by modifying the attention layers
        print("Testing with causal=False...")
        # Temporarily change causal setting
        original_causal = []
        for block in model.blocks:
            original_causal.append(block.attn.causal)
            block.attn.causal = False
        
        logits_non_causal, _ = model(x)
        
        # Restore original causal setting
        for i, block in enumerate(model.blocks):
            block.attn.causal = original_causal[i]
        
        # Compare outputs
        diff = torch.abs(logits_causal - logits_non_causal).mean()
        print(f"Mean absolute difference between causal and non-causal: {diff:.6f}")
        
        if diff > 1e-6:
            print("✓ Causal masking is working correctly (outputs differ)")
        else:
            print("⚠ Causal masking might not be working (outputs are identical)")
    
    model.train()


def test_generation(trainer, data_builder):
    """Test text generation with the trained model."""
    try:
        print("Generating sample text...")
        
        # Test with empty prompt
        generated_text = trainer.generate_sample(
            prompt="",
            max_length=50,
            temperature=0.8,
            top_k=50
        )
        print(f"Generated text (no prompt):\n'{generated_text}'\n")
        
        # Test with a prompt
        generated_text = trainer.generate_sample(
            prompt="The",
            max_length=50,
            temperature=0.8,
            top_k=50
        )
        print(f"Generated text (with prompt 'The'):\n'{generated_text}'\n")
        
    except Exception as e:
        print(f"Text generation failed: {e}")


if __name__ == "__main__":
    main()
