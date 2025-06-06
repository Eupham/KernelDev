import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

# Import our custom modules
from model import GPTModel
from data_builder import DataBuilder, create_data_builder
from train_loop import Trainer, TrainingConfig, create_trainer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        return {}


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge YAML config with command-line arguments, with CLI args taking precedence."""
    # If no config file was loaded, create default structure
    if not config:
        config = {
            'training': {},
            'data': {},
            'model': {},
            'hardware': {},
            'evaluation': {},
            'generation': {},
            'logging': {}
        }
    
    # Command-line arguments override config file values
    if hasattr(args, 'precision') and args.precision is not None:
        config['training']['precision'] = args.precision
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if hasattr(args, 'seq_len') and args.seq_len is not None:
        config['data']['seq_len'] = args.seq_len
    if hasattr(args, 'epochs') and args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    
    return config


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train GPT model with configurable precision')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to YAML configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--precision', 
        type=int, 
        choices=[16, 32], 
        default=None,
        help='Floating point precision: 16 for fp16/mixed precision, 32 for fp32 (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size (overrides config and auto-estimation)'
    )
    
    parser.add_argument(
        '--seq-len',
        type=int,
        default=None,
        help='Sequence length for training (overrides config)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    return parser.parse_args()


def setup_precision(model, precision):
    """Setup model precision and return appropriate dtype and scaler."""
    if precision == 16:
        print(f"Setting up mixed precision training (fp16)...")
        # Convert model to half precision
        model.half()
        dtype = torch.float16
        
        # Setup gradient scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler()
        use_amp = True
        
        print("✓ Model converted to fp16")
        print("✓ Gradient scaler initialized for mixed precision")
        
    else:  # precision == 32
        print(f"Using full precision training (fp32)...")
        model.float()
        dtype = torch.float32
        scaler = None
        use_amp = False
        
        print("✓ Model using fp32 precision")
    
    return dtype, scaler, use_amp


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
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Merge config with command-line arguments (CLI takes precedence)
    config = merge_config_with_args(config, args)
    
    # Extract configuration values with defaults
    training_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    hardware_cfg = config.get('hardware', {})
    eval_cfg = config.get('evaluation', {})
    gen_cfg = config.get('generation', {})
    logging_cfg = config.get('logging', {})
    
    # Set random seed for reproducibility
    seed = config.get('random_seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Print GPU information if enabled
    if logging_cfg.get('show_gpu_info', True):
        print_gpu_info()
    
    # Configuration summary
    precision = training_cfg.get('precision', 32)
    print("=== GPT Model Training with Flash Attention ===")
    print(f"Precision: fp{precision}")
    print(f"Mixed Precision Training: {'Enabled' if precision == 16 else 'Disabled'}")
    print("Setting up configuration...")
    
    # Data configuration
    data_config = {
        'dataset_name': data_cfg.get('dataset_name', 'allenai/c4'),
        'dataset_config': data_cfg.get('dataset_config', 'en'),
        'seq_len': data_cfg.get('seq_len', 1024),
        'max_samples': data_cfg.get('max_samples', 5000),
        'max_eval_tokens': data_cfg.get('max_eval_tokens', 50000)
    }
    
    # Model configuration
    model_config = {
        'vocab_size': model_cfg.get('vocab_size', 256),
        'dim': model_cfg.get('dim', 512),
        'n_layers': model_cfg.get('n_layers', 12),
        'n_heads': model_cfg.get('n_heads', 16),
        'max_seq_len': model_cfg.get('max_seq_len', 2048),
        'mlp_ratio': model_cfg.get('mlp_ratio', 4),
        'causal': model_cfg.get('causal', True)
    }
    
    # Initialize model
    print(f"\n=== Initializing Model ===")
    model = GPTModel(**model_config)
    
    # Setup precision and mixed precision training
    print(f"\n=== Setting up Precision ===")
    dtype, scaler, use_amp = setup_precision(model, precision)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter dtype: {next(model.parameters()).dtype}")
    
    # Training configuration
    training_config = TrainingConfig(
        num_epochs=training_cfg.get('epochs', 3),
        learning_rate=training_cfg.get('learning_rate', 3e-4),
        weight_decay=training_cfg.get('weight_decay', 0.01),
        warmup_steps=training_cfg.get('warmup_steps', 100),
        max_grad_norm=training_cfg.get('max_grad_norm', 1.0),
        save_every=training_cfg.get('save_every', 500),
        eval_every=training_cfg.get('eval_every', 200),
        log_every=training_cfg.get('log_every', 50),
        checkpoint_dir=training_cfg.get('checkpoint_dir', "checkpoints"),
        device=hardware_cfg.get('device', 'auto'),
        use_amp=use_amp,
        scaler=scaler
    )
    
    # Estimate optimal batch size with precision consideration
    if logging_cfg.get('show_memory_estimation', True):
        estimated_batch_size, memory_info = estimate_optimal_batch_size(
            model_config, 
            available_memory_gb=hardware_cfg.get('available_memory_gb', 15), 
            precision=precision
        )
        print(f"\n=== Memory Estimation ===")
        print(memory_info)
    else:
        estimated_batch_size = 8  # Fallback default
    
    # Determine batch size
    config_batch_size = training_cfg.get('batch_size')
    if config_batch_size is not None:
        batch_size = config_batch_size
        print(f"Using configured batch_size: {batch_size}")
    else:
        # Use a conservative batch size (slightly lower than estimated)
        batch_size = min(estimated_batch_size, 16)  # Cap at 16 for safety
        print(f"Using estimated batch_size: {batch_size}")
    
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
            num_workers=data_cfg.get('num_workers', 0),
            shuffle_train=data_cfg.get('shuffle_train', True)
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
        max_eval_batches = eval_cfg.get('max_eval_batches', 10)
        initial_train_loss = trainer.evaluate(dataloaders['train'], max_batches=max_eval_batches)
        initial_val_loss = trainer.evaluate(dataloaders['validation'], max_batches=max_eval_batches)
        print(f"Initial training loss: {initial_train_loss:.4f}")
        print(f"Initial validation loss: {initial_val_loss:.4f}")
    
    # Test causal vs non-causal attention
    if logging_cfg.get('test_attention_modes', True):
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
            max_eval_batches = eval_cfg.get('max_eval_batches', 10)
            final_train_loss = trainer.evaluate(dataloaders['train'], max_batches=max_eval_batches)
            final_val_loss = trainer.evaluate(dataloaders['validation'], max_batches=max_eval_batches)
            print(f"Final training loss: {final_train_loss:.4f}")
            print(f"Final validation loss: {final_val_loss:.4f}")
            
            # Show improvement
            if 'initial_train_loss' in locals():
                train_improvement = initial_train_loss - final_train_loss
                val_improvement = initial_val_loss - final_val_loss
                print(f"Training loss improvement: {train_improvement:.4f}")
                print(f"Validation loss improvement: {val_improvement:.4f}")
        
        # Plot training curves
        if logging_cfg.get('save_training_plots', True):
            print(f"\n=== Plotting Results ===")
            curves_path = Path(training_config.checkpoint_dir) / "training_curves.png"
            trainer.plot_training_curves(save_path=str(curves_path))
        
        # Test text generation
        if logging_cfg.get('test_generation', True):
            print(f"\n=== Testing Text Generation ===")
            test_generation(trainer, data_builder, gen_cfg)
        
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


def test_generation(trainer, data_builder, gen_cfg=None):
    """Test text generation with the trained model."""
    if gen_cfg is None:
        gen_cfg = {}
    
    try:
        print("Generating sample text...")
        
        max_length = gen_cfg.get('max_length', 50)
        temperature = gen_cfg.get('temperature', 0.8)
        top_k = gen_cfg.get('top_k', 50)
        test_prompts = gen_cfg.get('test_prompts', ["", "The"])
        
        for prompt in test_prompts:
            generated_text = trainer.generate_sample(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k
            )
            
            if prompt:
                print(f"Generated text (prompt: '{prompt}'):\n'{generated_text}'\n")
            else:
                print(f"Generated text (no prompt):\n'{generated_text}'\n")
        
    except Exception as e:
        print(f"Text generation failed: {e}")


def estimate_optimal_batch_size(model_config, available_memory_gb=15, precision=32):
    """Estimate optimal batch size for T4 GPU based on model parameters, sequence length, and precision."""
    # Estimate memory usage per sample
    # Memory = (parameters * bytes_per_param) + (activations memory)
    
    dim = model_config['dim']
    n_layers = model_config['n_layers']
    seq_len = model_config['max_seq_len']
    vocab_size = model_config['vocab_size']
    
    # Bytes per parameter based on precision
    bytes_per_param = 4 if precision == 32 else 2  # fp32 = 4 bytes, fp16 = 2 bytes
    bytes_per_activation = 4 if precision == 32 else 2  # activation precision
    
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
    model_memory = param_count * bytes_per_param / (1024**3)
    activation_memory_per_sample = (seq_len * dim * n_layers * bytes_per_activation) / (1024**3)
    
    # Reserve memory for gradients (same as model) and optimizer state (2x model for Adam)
    # Note: Gradients and optimizer states typically remain in fp32 even with mixed precision
    gradient_memory = param_count * 4 / (1024**3)  # gradients in fp32
    optimizer_memory = param_count * 8 / (1024**3)  # Adam: 2x fp32 states (momentum + variance)
    total_model_memory = model_memory + gradient_memory + optimizer_memory
    
    # Available memory for activations
    available_for_activations = available_memory_gb - total_model_memory - 2  # 2GB buffer
    
    if available_for_activations <= 0:
        return 1, f"Model too large! Estimated model memory: {total_model_memory:.1f}GB"
    
    # Estimate batch size
    estimated_batch_size = max(1, int(available_for_activations / activation_memory_per_sample))
    
    precision_str = f"fp{precision}"
    info = (
        f"Estimated memory usage ({precision_str}):\n"
        f"  Model parameters: {model_memory:.1f}GB\n"
        f"  Gradients: {gradient_memory:.1f}GB\n"
        f"  Optimizer states: {optimizer_memory:.1f}GB\n"
        f"  Total model memory: {total_model_memory:.1f}GB\n"
        f"  Activation memory per sample: {activation_memory_per_sample*1000:.1f}MB\n"
        f"  Available for activations: {available_for_activations:.1f}GB\n"
        f"  Recommended batch size: {estimated_batch_size}"
    )
    
    return estimated_batch_size, info


if __name__ == "__main__":
    main()
