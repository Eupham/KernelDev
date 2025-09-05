"""
GPT Model Training Entry Point with Hierarchical Attention Support

This is the main entry script for training GPT models with support for both standard
teacher forcing and specialized cocktail party tasks. Handles configuration management,
distributed training setup, precision optimization, and comprehensive training coordination.

Key Features:
- Multi-GPU distributed training with automatic process spawning
- Mixed precision training (fp16, bf16, fp32) with automatic GPU detection
- Memory-aware batch size estimation for optimal resource utilization
- YAML-based configuration with command-line override support
- Comprehensive GPU profiling and performance monitoring
- Task-specific training and evaluation for teacher forcing and cocktail party

Usage:
    python entry.py                                    # Default configuration
    python entry.py --config config_fast.yaml         # Custom config file
    python entry.py --precision bf16 --batch-size 8   # Override specific settings
    python entry.py --nproc_per_node 4                # 4-GPU distributed training
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse # Keep this, but ArgumentParser might be used from new imports
import yaml
from pathlib import Path
from typing import Dict, Any

import sys
import subprocess
import socket
import os
# Ensure ArgumentParser and REMAINDER are available if argparse is re-imported or used directly
from argparse import ArgumentParser, REMAINDER

# Import our custom modules
from model import GPTModel
from data_builder import DataBuilder, create_data_builder
from train_loop import Trainer, TrainingConfig, create_trainer, find_latest_checkpoint_path

# =============================================================================
# Utility Functions
# =============================================================================


def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)

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
    if hasattr(args, 'output_dir') and args.output_dir is not None:
        config['training']['checkpoint_dir'] = args.output_dir
    
    return config


# Removed redundant parse_args() function.
# All parsing is now handled in the if __name__ == "__main__": block.

def setup_precision(model, precision):
    """Setup model precision and return appropriate dtype and scaler."""
    if precision == 16 or precision == '16':
        print(f"Setting up mixed precision training (fp16)...")
        # Keep model in fp32 for mixed precision training
        # The model will be automatically cast to fp16 during forward pass
        model.float()  # Don't convert to half, let autocast handle it
        dtype = torch.float16
        
        # Setup gradient scaler for mixed precision (using new API)
        scaler = torch.amp.GradScaler('cuda')
        use_amp = True
        
        print("✓ Model prepared for mixed precision training")
        print("✓ Gradient scaler initialized for mixed precision")
        
    elif precision == 'bf16':
        print(f"Setting up mixed precision training (bf16)...")
        # Keep model in fp32 for mixed precision training
        # The model will be automatically cast to bf16 during forward pass
        model.float()  # Don't convert to bfloat16, let autocast handle it
        dtype = torch.bfloat16
        
        # Setup gradient scaler for mixed precision (using new API)
        # Note: bf16 typically doesn't need gradient scaling due to wider dynamic range
        # but we'll keep it for consistency and safety
        scaler = torch.amp.GradScaler('cuda')
        use_amp = True
        
        print("✓ Model prepared for bf16 mixed precision training")
        print("✓ Gradient scaler initialized for mixed precision")
        
    else:  # precision == 32 or precision == '32'
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


def start_actual_training(cli_args):
    """
    Encapsulates the actual training setup and execution.
    `cli_args` can be an argparse.Namespace object or a compatible dict/object.
    """
    # Load configuration from YAML file
    config_file_path = cli_args.config if hasattr(cli_args, 'config') else 'config.yaml'
    config = load_config(config_file_path)
    
    # Merge config with command-line arguments (CLI takes precedence)
    if not isinstance(cli_args, argparse.Namespace):
        pass # Assumes compatible attributes
    config = merge_config_with_args(config, cli_args)
    
    # Extract configuration values
    training_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    hardware_cfg = config.get('hardware', {})
    eval_cfg = config.get('evaluation', {})
    gen_cfg = config.get('generation', {})
    logging_cfg = config.get('logging', {})
    
    # Set random seed
    torch.manual_seed(config.get('random_seed', 42))
    np.random.seed(config.get('random_seed', 42))
    
    # Print GPU info
    if logging_cfg.get('show_gpu_info', True):
        print_gpu_info()
    
    # Configuration summary
    precision = training_cfg.get('precision', 32)
    print("=== GPT Model Training with Flash Attention ===")
    # ... (precision summary print statements)
    
    # Model configuration
    model_config = {
        'vocab_size': model_cfg.get('vocab_size', 256),
        'dim': model_cfg.get('dim', 512),
        'n_layers': model_cfg.get('n_layers', 12),
        'n_heads': model_cfg.get('n_heads', 16),
        'max_seq_len': model_cfg.get('max_seq_len', 2048),
        'mlp_ratio': model_cfg.get('mlp_ratio', 4),
        'causal': model_cfg.get('causal', True),
        'task_names': list(config.get('tasks', {}).keys())
    }
    
    # Initialize model
    print(f"\n=== Initializing Model ===")
    model = GPTModel(**model_config)
    
    # Setup precision
    print(f"\n=== Setting up Precision ===")
    dtype, scaler, use_amp = setup_precision(model, precision)
    
    # Parameter count
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Estimate and determine batch size
    if logging_cfg.get('show_memory_estimation', True):
        estimated_batch_size, memory_info = estimate_optimal_batch_size(
            model_config,
            available_memory_gb=hardware_cfg.get('available_memory_gb', 15),
            precision=precision
        )
        print(f"\n=== Memory Estimation ===\n{memory_info}")
    else:
        estimated_batch_size = 8

    config_batch_size = training_cfg.get('batch_size')
    batch_size = config_batch_size if config_batch_size is not None else min(estimated_batch_size, 16)
    print(f"Using batch_size: {batch_size}")

    # Create TrainingConfig
    training_config = TrainingConfig(
        batch_size=batch_size,
        **training_cfg,
        **hardware_cfg,
        **config.get('inference', {}),
        use_amp=use_amp,
        scaler=scaler
    )
    
    # Create data builder and update vocab size
    print("\n=== Creating Data Builder ===")
    data_builder = create_data_builder(**data_cfg, task_configs=config.get('tasks', {}))
    model.vocab_size = data_builder.get_vocab_size()
    print(f"Confirmed vocab_size: {model.vocab_size} (UTF-8 bytes)")

    # Create trainer
    print(f"\n=== Setting up Trainer ===")
    trainer = create_trainer(model=model, config=training_config, data_builder=data_builder)

    # Check for checkpoints and determine samples to skip
    samples_to_skip = 0
    if training_config.auto_resume:
        latest_checkpoint = trainer.find_latest_checkpoint()
        if latest_checkpoint:
            print(f"Found existing checkpoint: {latest_checkpoint}")
            try:
                resume_state = trainer.load_checkpoint(latest_checkpoint)
                samples_to_skip = resume_state.get('processed_samples', 0)
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")

    # Create dataloaders
    print("\n=== Loading and Processing Data ===")
    dataloaders = data_builder.create_dataloaders(
        batch_size=batch_size,
        num_workers=data_cfg.get('num_workers', 0),
        shuffle_train=data_cfg.get('shuffle_train', True),
        samples_to_skip=samples_to_skip
    )
    
    # Initial evaluation if not resuming
    if samples_to_skip == 0:
        print("\n=== Initial Evaluation ===")
        if 'validation' in dataloaders:
            trainer.evaluate(dataloaders['validation'], config.get('tasks', {}), max_batches=10)

    # Start training
    print(f"\n=== Starting Training ===")
    trainer.train(
        train_loaders=dataloaders.get('train'),
        val_loaders=dataloaders.get('validation'),
        task_configs=config.get('tasks', {})
    )
    
    # Post-training actions
    print("\n=== Training Completed ===")
    if logging_cfg.get('save_training_plots', True):
        trainer.plot_training_curves(save_path=Path(training_config.checkpoint_dir) / "training_curves.png")
    if logging_cfg.get('test_generation', True):
        test_generation(trainer, data_builder, gen_cfg)

# --- End of original main logic, now in start_actual_training ---

def test_generation(trainer, data_builder, gen_cfg=None):
    """Test text generation with the trained model."""
    if gen_cfg is None:
        gen_cfg = {}
    
    try:
        print("Generating sample text...")
        
        max_length = gen_cfg.get('max_length', 50)
        temperature = gen_cfg.get('temperature', 0.8)
        top_k = gen_cfg.get('top_k', 50)
        top_p = gen_cfg.get('top_p', 0.9)
        test_prompts = gen_cfg.get('test_prompts', ["", "The"])
        
        for prompt in test_prompts:
            generated_text = trainer.generate_sample(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
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
    if precision == 32 or precision == '32':
        bytes_per_param = 4  # fp32 = 4 bytes
        bytes_per_activation = 4
        precision_str = "fp32"
    elif precision == 'bf16':
        bytes_per_param = 2  # bf16 = 2 bytes (same as fp16)
        bytes_per_activation = 2
        precision_str = "bf16"
    else:  # precision == 16 or precision == '16'
        bytes_per_param = 2  # fp16 = 2 bytes
        bytes_per_activation = 2
        precision_str = "fp16"
    
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
    # Main argument parser for the entry script, including distributed launch args
    parser = ArgumentParser(description="GPT Model Training Entry Script")
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="Number of processes to launch for distributed training on this node."
    )
    # Add other existing arguments from the original parse_args()
    # These are arguments that the training script itself needs, not just the launcher.
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to YAML configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--precision',
        type=str,
        choices=['16', '32', 'bf16'],
        default=None, # Default to None, so config file is source of truth unless overridden
        help='Floating point precision: 16 for fp16/mixed precision, 32 for fp32, bf16 for bfloat16/mixed precision (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None, # Default to None
        help='Override batch size (overrides config and auto-estimation)'
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=None, # Default to None
        help='Sequence length for training (overrides config)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None, # Default to None
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None, # Default to None
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save checkpoints and logs (overrides config)'
    )
    # Use parse_args() which will capture all defined args.
    # REMAINDER is not needed here as we explicitly define training args.
    args = parser.parse_args()

    if "IS_WORKER_PROCESS" in os.environ:
        print(f"Worker process RANK: {os.environ.get('RANK', 'N/A')}, LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'N/A')} starting.")
        # Worker processes receive all arguments and proceed to training
        start_actual_training(args)
    elif args.nproc_per_node > 1:
        print(f"Main process launching {args.nproc_per_node} worker processes.")
        master_addr = "127.0.0.1"
        master_port = find_free_port()
        world_size = args.nproc_per_node

        processes = []

        # Construct the base command for worker processes
        # We need to pass all arguments *except* --nproc_per_node to the workers
        worker_cmd_args = [sys.executable, sys.argv[0]] # script itself

        # Iterate over sys.argv to rebuild arguments, skipping --nproc_per_node
        skip_next_arg = False
        for i, arg_val in enumerate(sys.argv[1:]):
            if skip_next_arg:
                skip_next_arg = False
                continue
            if arg_val == "--nproc_per_node":
                skip_next_arg = True # Skip the value of nproc_per_node
                continue
            worker_cmd_args.append(arg_val)

        for rank in range(world_size):
            env = os.environ.copy()
            env["MASTER_ADDR"] = master_addr
            env["MASTER_PORT"] = master_port
            env["WORLD_SIZE"] = str(world_size)
            env["RANK"] = str(rank)
            env["LOCAL_RANK"] = str(rank) # Assuming single-node, local_rank == rank
            env["IS_WORKER_PROCESS"] = "1"
            env["PYTHONUNBUFFERED"] = "1"

            print(f"Launching worker RANK {rank} with command: {' '.join(worker_cmd_args)}")
            try:
                process = subprocess.Popen(worker_cmd_args, env=env)
                processes.append(process)
            except Exception as e:
                print(f"Error launching process for RANK {rank}: {e}")
                for p_term in processes:
                    try: p_term.terminate()
                    except: pass # best effort
                sys.exit(1)


        for rank, process in enumerate(processes):
            process.wait()
            if process.returncode != 0:
                print(f"Worker process RANK {rank} (PID {process.pid}) exited with error code {process.returncode}.")

        print("All worker processes finished.")
        sys.exit(0) # Main launcher process exits after workers are done
    else:
        print("Running in single process mode (nproc_per_node = 1).")
        # In single process mode, RANK and WORLD_SIZE might not be set by an external launcher.
        # For consistency with how init_distributed in train_loop might expect these for non-DDP single GPU:
        if "RANK" not in os.environ: os.environ["RANK"] = "0"
        if "WORLD_SIZE" not in os.environ: os.environ["WORLD_SIZE"] = "1"
        if "LOCAL_RANK" not in os.environ: os.environ["LOCAL_RANK"] = "0"
        start_actual_training(args)
