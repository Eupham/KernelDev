import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our custom modules
from model import GPTModel
from data_builder import DataBuilder, create_data_builder
from train_loop import Trainer, TrainingConfig, create_trainer


def print_gpu_info():
    """Print comprehensive GPU information and optimization status."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"=== GPU Information ===")
        print(f"Device: {torch.cuda.get_device_name(device)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(device)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        print(f"Current Memory Usage: {torch.cuda.memory_allocated(device) / 1024**3:.1f} GB")
        print(f"Current Memory Cached: {torch.cuda.memory_reserved(device) / 1024**3:.1f} GB")
        
        # Check if T4 optimizations will be applied
        cap = torch.cuda.get_device_capability(device)
        if cap >= (7, 5) and cap < (8, 0):
            print("✓ T4-optimized flash attention kernels will be used")
        elif cap >= (8, 0) and cap < (9, 0):
            print("✓ A100-optimized flash attention kernels will be used")
        elif cap >= (9, 0):
            print("✓ H100-optimized flash attention kernels will be used")
        else:
            print("⚠ Using fallback flash attention kernels")
        print()
    else:
        print("CUDA not available!")


def main():
    # Print GPU information first
    print_gpu_info()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    print("=== GPT Model Training with Flash Attention ===")
    print("Setting up configuration...")
    
    # Data configuration (T4-optimized for better utilization)
    data_config = {
        'dataset_name': 'allenai/c4',
        'dataset_config': 'en',
        'seq_len': 1024,  # Increased for better GPU utilization
        'max_samples': 5000,  # Increased for more data
        'max_eval_tokens': 50000  # Increased for better evaluation
    }
    
    # Model configuration (T4-optimized for better utilization)
    model_config = {
        'vocab_size': 256,  # UTF-8 byte vocabulary size
        'dim': 512,  # Increased for better GPU utilization (T4 can handle this)
        'n_layers': 12,  # Increased for better GPU utilization
        'n_heads': 16,  # Increased (dim must be divisible by n_heads: 512/16=32)
        'max_seq_len': 2048,  # Increased sequence length
        'mlp_ratio': 4,  # Keep standard ratio
        'causal': True  # Using causal attention
    }
    
    # Training configuration (T4-optimized for better utilization)
    # Training configuration (T4-optimized for better utilization)
    training_config = TrainingConfig(
        num_epochs=3,  # Increased for better training
        learning_rate=3e-4,  # Standard learning rate
        weight_decay=0.01,
        warmup_steps=100,  # Increased warmup
        max_grad_norm=1.0,
        save_every=500,  # Reasonable checkpoint frequency
        eval_every=200,  # Regular evaluation
        log_every=50,  # Regular logging
        checkpoint_dir="checkpoints"
    )
    
    # Estimate optimal batch size for T4
    estimated_batch_size, memory_info = estimate_optimal_batch_size(model_config, available_memory_gb=15)
    print(f"\n=== Memory Estimation ===")
    print(memory_info)
    
    # Use a conservative batch size (slightly lower than estimated)
    batch_size = min(estimated_batch_size, 16)  # Cap at 16 for safety
    print(f"Using batch_size: {batch_size}")
    
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
        # Limit initial evaluation to 10 batches for speed
        initial_train_loss = trainer.evaluate(dataloaders['train'], max_batches=10)
        initial_val_loss = trainer.evaluate(dataloaders['validation'], max_batches=10)
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
            # Limit final evaluation to 10 batches for speed
            final_train_loss = trainer.evaluate(dataloaders['train'], max_batches=10)
            final_val_loss = trainer.evaluate(dataloaders['validation'], max_batches=10)
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


def estimate_optimal_batch_size(model_config, available_memory_gb=15):
    """Estimate optimal batch size for T4 GPU based on model parameters and sequence length."""
    # Estimate memory usage per sample (very rough estimate)
    # Memory = (parameters * 4 bytes) + (activations memory)
    
    dim = model_config['dim']
    n_layers = model_config['n_layers']
    seq_len = model_config['max_seq_len']
    vocab_size = model_config['vocab_size']
    
    # Rough parameter count estimation
    param_count = (
        vocab_size * dim +  # embedding
        n_layers * (
            4 * dim * dim +  # attention weights (Q, K, V, O)
            2 * dim +        # attention layer norms
            8 * dim * dim +  # MLP weights (assuming 4x expansion)
            2 * dim          # MLP layer norms
        ) +
        dim + vocab_size * dim  # final layer norm + output projection
    )
    
    # Memory estimates (in GB)
    model_memory = param_count * 4 / (1024**3)  # 4 bytes per parameter
    activation_memory_per_sample = (seq_len * dim * n_layers * 4) / (1024**3)  # rough estimate
    
    # Reserve memory for gradients (same as model) and optimizer state (2x model for Adam)
    total_model_memory = model_memory * 4  # model + gradients + optimizer state
    
    # Available memory for activations
    available_for_activations = available_memory_gb - total_model_memory - 2  # 2GB buffer
    
    if available_for_activations <= 0:
        return 1, f"Model too large! Estimated model memory: {total_model_memory:.1f}GB"
    
    # Estimate batch size
    estimated_batch_size = max(1, int(available_for_activations / activation_memory_per_sample))
    
    info = (
        f"Estimated memory usage:\n"
        f"  Model + gradients + optimizer: {total_model_memory:.1f}GB\n"
        f"  Activation memory per sample: {activation_memory_per_sample*1000:.1f}MB\n"
        f"  Available for activations: {available_for_activations:.1f}GB\n"
        f"  Recommended batch size: {estimated_batch_size}"
    )
    
    return estimated_batch_size, info


if __name__ == "__main__":
    main()
