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
from train_loop import Trainer, TrainingConfig, create_trainer


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
    
    # NSP and CPU attention fallback arguments
    # training_config = config.setdefault('training', {}) # Ensure 'training' key exists
    if hasattr(args, 'nsp_task') and args.nsp_task is not None: # args.nsp_task can be True or False if flag is used
        config.setdefault('training', {})['nsp_task'] = args.nsp_task
    # If args.nsp_task is None (flag not used), we don't set it here.
    # The default of True will be handled by config.get('training', {}).get('nsp_task', True)
    # in start_actual_training or by TrainingConfig default.

    if hasattr(args, 'nsp_loss_weight') and args.nsp_loss_weight is not None:
        config.setdefault('training', {})['nsp_loss_weight'] = args.nsp_loss_weight
    if hasattr(args, 'cpu_test_attention') and args.cpu_test_attention is not None:
        config.setdefault('hardware', {})['cpu_test_attention'] = args.cpu_test_attention # Ensure hardware key exists

    # Handle use_cls_prefix_attention
    if hasattr(args, 'use_cls_prefix_attention') and args.use_cls_prefix_attention is not None:
        config.setdefault('model', {})['use_cls_prefix_attention'] = args.use_cls_prefix_attention

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
    # If cli_args is a namespace from parse_args in the new main, it should have 'config' attribute
    config_file_path = cli_args.config if hasattr(cli_args, 'config') else 'config.yaml'
    config = load_config(config_file_path)
    
    # Merge config with command-line arguments (CLI takes precedence)
    # Ensure cli_args is a Namespace for merge_config_with_args if it expects one
    # If cli_args might not be a full Namespace, adjust merge_config_with_args or pass parameters carefully
    if not isinstance(cli_args, argparse.Namespace):
        # If cli_args is not a namespace (e.g. from worker process re-parsing with limited args)
        # we might need to be careful here. For now, assume it has compatible attributes.
        # A cleaner way might be to pass a dictionary of overrides.
        pass # Assuming cli_args has the necessary attributes like precision, batch_size etc.

    config = merge_config_with_args(config, cli_args)
    
    # Extract configuration values with defaults
    training_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    hardware_cfg = config.get('hardware', {})
    eval_cfg = config.get('evaluation', {})
    gen_cfg = config.get('generation', {})
    logging_cfg = config.get('logging', {})

    # NSP and CPU attention fallback settings from config
    # Default nsp_task to True if not specified in YAML or by CLI
    nsp_task_enabled = training_cfg.get('nsp_task', True)
    nsp_lw = training_cfg.get('nsp_loss_weight', 0.5)
    cpu_test_mode = hardware_cfg.get('cpu_test_attention', False)
    
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
    if precision == 'bf16':
        print(f"Precision: bf16 (bfloat16)")
        print(f"Mixed Precision Training: Enabled (bf16)")
    elif precision == 16 or precision == '16':
        print(f"Precision: fp16")
        print(f"Mixed Precision Training: Enabled (fp16)")
    else:
        print(f"Precision: fp32")
        print(f"Mixed Precision Training: Disabled")
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
    
    # Model configuration is prepared (model_cfg)
    # DataBuilder is needed next to finalize vocab_size and special tokens for model_config

    # Create data builder
    print("\n=== Loading and Processing Data ===")
    # Pass nsp_task to create_data_builder
    data_config_with_nsp = {**data_config, 'nsp_task': nsp_task_enabled}
    data_builder = create_data_builder(**data_config_with_nsp)

    # Update model_config with actual vocab_size and cls_token_id if NSP
    actual_vocab_size = data_builder.get_vocab_size()
    model_config['vocab_size'] = actual_vocab_size # This must be set for the model
    model_config['nsp_task'] = nsp_task_enabled
    if nsp_task_enabled:
        model_config['cls_token_id'] = data_builder.cls_token_id
        # Determine use_cls_prefix_attention based on config or default to True if NSP is on
        use_cls_prefix_attention_from_config = model_cfg.get('use_cls_prefix_attention', None) # model_cfg is from loaded config
        if use_cls_prefix_attention_from_config is None:
            model_config['use_cls_prefix_attention'] = True # Default to True if NSP is on and not specified
        else:
            model_config['use_cls_prefix_attention'] = use_cls_prefix_attention_from_config
    else:
        model_config['cls_token_id'] = None # Explicitly None if NSP is off
        model_config['use_cls_prefix_attention'] = False # Not relevant if NSP is off


    # Now instantiate the model with the fully defined model_config
    print(f"\n=== Initializing Model ===")
    # print(f"Instantiating GPTModel with final model_config: {model_config}") # Removed debug print
    model = GPTModel(**model_config)
    
    # Setup precision and mixed precision training now that model is instantiated
    print(f"\n=== Setting up Precision ===")
    # print(f"Setting up precision for the model (Precision: {precision})...") # Removed debug print
    dtype, scaler, use_amp = setup_precision(model, precision) # Now model exists

    # Determine effective device *before* TrainingConfig instantiation
    device_from_cfg = hardware_cfg.get('device', 'auto')
    if cpu_test_mode: # cpu_test_mode is defined earlier
        effective_device = 'cpu'
        print(f"CPU test mode active. Overriding device to CPU.")
    elif device_from_cfg == 'auto':
        effective_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        effective_device = device_from_cfg
    print(f"Effective device determined as: {effective_device}")

    # Instantiate TrainingConfig *after* model, setup_precision, and device determination
    print(f"\n=== Initializing Training Configuration ===")
    training_config_params = {
        'num_epochs': training_cfg.get('epochs', 1),
        'learning_rate': training_cfg.get('learning_rate', 3e-4),
        'weight_decay': training_cfg.get('weight_decay', 0.01),
        'warmup_steps': training_cfg.get('warmup_steps', 100),
        'max_grad_norm': training_cfg.get('max_grad_norm', 1.0),
        'save_every': training_cfg.get('save_every', 500),
        'eval_every': training_cfg.get('eval_every', 200),
        'log_every': training_cfg.get('log_every', 50),
        'checkpoint_dir': training_cfg.get('checkpoint_dir', "checkpoints"),
        'device': effective_device,  # Pass the determined effective_device
        'nsp_task': nsp_task_enabled, # nsp_task_enabled defined earlier
        'nsp_loss_weight': nsp_lw,    # nsp_lw defined earlier
        'scaler': scaler,             # Pass scaler from setup_precision
        'use_amp': use_amp,           # Pass use_amp from setup_precision
        # Inference params from gen_cfg (defined earlier)
        'inference_prompts': gen_cfg.get('prompts', ["", "The", "In", "Once upon a time"]),
        'inference_max_length': gen_cfg.get('max_length', 100),
        'inference_temperature': gen_cfg.get('temperature', 0.8),
        'inference_top_k': gen_cfg.get('top_k', 50),
        'inference_top_p': gen_cfg.get('top_p', 0.9),
        # Plateau params from training_cfg
        'plateau_monitor_metric': training_cfg.get('plateau_monitor_metric', 'val_loss'),
        'plateau_patience': training_cfg.get('plateau_patience', 10),
        'plateau_threshold': training_cfg.get('plateau_threshold', 1e-4),
        'plateau_mode': training_cfg.get('plateau_mode', 'min'),
        # Scheduler params from training_cfg
        'scheduler_T0_epoch_fraction': training_cfg.get('scheduler_T0_epoch_fraction', 0.1),
        'scheduler_T_mult': training_cfg.get('scheduler_T_mult', 1)
    }
    training_config = TrainingConfig(**training_config_params)
    print(f"TrainingConfig initialized with device: {training_config.device}")

    # Ensure model is on the correct device (especially after CPU fallback or re-initialization)
    # This should happen AFTER setup_precision and TrainingConfig instantiation (which sets the target device)
    model.to(training_config.device) # training_config.device is now the single source of truth for device
    print(f"Model explicitly moved to device: {training_config.device} after TrainingConfig initialization.")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter dtype: {next(model.parameters()).dtype}")
    
    # Training configuration
    inference_cfg = config.get('inference', {})
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
        scaler=scaler,
        # Inference sampling parameters
        inference_prompts=inference_cfg.get('prompts', ["", "The", "In", "Once upon a time"]),
        inference_max_length=inference_cfg.get('max_length', 100),
        inference_temperature=inference_cfg.get('temperature', 0.8),
        inference_top_k=inference_cfg.get('top_k', 50),
        inference_top_p=inference_cfg.get('top_p', 0.9),
        nsp_task=nsp_task_enabled, # Pass to TrainingConfig
        nsp_loss_weight=nsp_lw    # Pass to TrainingConfig
    )
    
    # CPU Attention Fallback Logic
    if cpu_test_mode:
        print("CPU Attention Fallback Mode ENABLED")
        import original_kernel # Ensure it's imported

        def cpu_flash_attention_fallback(q, k, v, lens, sm_scale, causal, autotune, return_lse, prescale_qk, precision, is_prefix_token_mask=None):
            B, H, T, D = q.shape
            # Ensure all inputs are on the same device as q. device for new tensors.
            target_device = q.device
            k = k.to(target_device)
            v = v.to(target_device)
            if is_prefix_token_mask is not None:
                is_prefix_token_mask = is_prefix_token_mask.to(target_device)

            q_scaled = q * sm_scale

            attn_bias = torch.zeros(B, H, T, T, dtype=q.dtype, device=target_device)

            if causal:
                causal_mask_values = torch.triu(torch.ones(T, T, device=target_device, dtype=torch.bool), diagonal=1)
                expanded_causal_mask = causal_mask_values.unsqueeze(0).unsqueeze(0)

                if is_prefix_token_mask is not None:
                    prefix_q_mask = is_prefix_token_mask.view(1, 1, T, 1).expand(B, H, T, T)
                    final_causal_mask = torch.where(prefix_q_mask, torch.zeros_like(expanded_causal_mask), expanded_causal_mask)
                else:
                    final_causal_mask = expanded_causal_mask

                attn_bias.masked_fill_(final_causal_mask, float("-inf"))

            attn_weights = torch.matmul(q_scaled, k.transpose(-2, -1)) + attn_bias
            attn_weights = F.softmax(attn_weights, dim=-1)
            output = torch.matmul(attn_weights, v)

            if return_lse:
                scores_for_lse = torch.matmul(q_scaled, k.transpose(-2, -1)) + attn_bias
                scores_for_lse = torch.where(attn_bias == float("-inf"), torch.full_like(scores_for_lse, -float("inf")), scores_for_lse)
                lse = torch.logsumexp(scores_for_lse, dim=-1)
            else:
                lse = torch.empty(0, device=target_device, dtype=q.dtype)

            return output, lse

        original_kernel.flash_attention = cpu_flash_attention_fallback
        # effective_device will be 'cpu' due to cpu_test_mode=True,
        # training_config will get this device, and model will be moved to it.
        # No need for model.to('cpu') here.
        print(f"Flash attention overridden with CPU fallback.")

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
    # Pass nsp_task to create_data_builder
    data_config_with_nsp = {**data_config, 'nsp_task': nsp_task_enabled}
    data_builder = create_data_builder(**data_config_with_nsp)

    # Create dataloaders
    try:
        dataloaders = data_builder.create_dataloaders(
            batch_size=batch_size,
            num_workers=data_cfg.get('num_workers', 0),
            shuffle_train=data_cfg.get('shuffle_train', True)
        )
        print(f"Vocab size from data_builder: {actual_vocab_size} (UTF-8 bytes potentially extended for NSP)")
        
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
        # Adjust for NSPDataset output (input_ids, lm_targets, nsp_label)
        if nsp_task_enabled:
            for input_ids_sample, _, nsp_label_sample in dataloaders['train']:
                print(f"NSP Batch shapes: IDs-{input_ids_sample.shape}, NSP Labels-{nsp_label_sample.shape}")
                print(f"Sample NSP input tokens: {input_ids_sample[0][:20].tolist()}")
                sample_text = data_builder.decode_tokens(input_ids_sample[0][:50])
                print(f"Sample NSP text: '{sample_text[:100]}...'")
                break
        else:
            for x, y in dataloaders['train']:
                print(f"Batch shape: {x.shape}")
                print(f"Sample tokens: {x[0][:20].tolist()}")
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
        # Initial evaluation returns combined loss. NSP metrics are updated in trainer.metrics
        initial_train_loss_combined = trainer.evaluate(dataloaders['train'], max_batches=max_eval_batches)
        initial_val_loss_combined = trainer.evaluate(dataloaders['validation'], max_batches=max_eval_batches)

        print(f"Initial training loss (combined): {initial_train_loss_combined:.4f}")
        if nsp_task_enabled and trainer.metrics.val_nsp_losses: # Check if metrics got populated
             print(f"  Initial train NSP loss: {trainer.metrics.val_nsp_losses[-2]:.4f}, NSP Acc: {trainer.metrics.val_nsp_accuracies[-2]:.4f}") # Eval on train populates val lists

        print(f"Initial validation loss (combined): {initial_val_loss_combined:.4f}")
        if nsp_task_enabled and trainer.metrics.val_nsp_losses:
             print(f"  Initial validation NSP loss: {trainer.metrics.val_nsp_losses[-1]:.4f}, NSP Acc: {trainer.metrics.val_nsp_accuracies[-1]:.4f}")
    
    # Test causal vs non-causal attention
    if logging_cfg.get('test_attention_modes', True) and not cpu_test_mode : # Skip if CPU mode as it might be slow or not the point
        print(f"\n=== Testing Causal vs Non-Causal Attention ===")
        test_causal_attention(model, dataloaders, training_config.device, data_builder, nsp_task_enabled)
    
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
            final_train_loss_combined = trainer.evaluate(dataloaders['train'], max_batches=max_eval_batches)
            final_val_loss_combined = trainer.evaluate(dataloaders['validation'], max_batches=max_eval_batches)

            print(f"Final training loss (combined): {final_train_loss_combined:.4f}")
            if nsp_task_enabled and len(trainer.metrics.val_nsp_losses) >= 4: # Assuming at least two evals on train and two on val
                 print(f"  Final train NSP loss: {trainer.metrics.val_nsp_losses[-2]:.4f}, NSP Acc: {trainer.metrics.val_nsp_accuracies[-2]:.4f}")

            print(f"Final validation loss (combined): {final_val_loss_combined:.4f}")
            if nsp_task_enabled and len(trainer.metrics.val_nsp_losses) >= 2:
                 print(f"  Final validation NSP loss: {trainer.metrics.val_nsp_losses[-1]:.4f}, NSP Acc: {trainer.metrics.val_nsp_accuracies[-1]:.4f}")

            # Show improvement
            if 'initial_train_loss_combined' in locals(): # Check if initial eval was done
                train_improvement = initial_train_loss_combined - final_train_loss_combined
                val_improvement = initial_val_loss_combined - final_val_loss_combined
                print(f"Training loss improvement (combined): {train_improvement:.4f}")
                print(f"Validation loss improvement (combined): {val_improvement:.4f}")
        
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

# --- End of original main logic, now in start_actual_training ---

def test_causal_attention(model, dataloaders, device, data_builder, nsp_task_enabled: bool):
    """Test the difference between causal and non-causal attention."""
    if 'train' not in dataloaders or not dataloaders['train']:
        print("Warning: Train dataloader is empty or not found in test_causal_attention. Skipping test.")
        return

    batch_iter = iter(dataloaders['train'])
    try:
        first_batch = next(batch_iter)
    except StopIteration:
        print("Warning: Train dataloader is empty in test_causal_attention. Skipping test.")
        return

    if nsp_task_enabled:
        # Expecting (input_ids, lm_targets, nsp_label)
        if len(first_batch) == 3:
            x, _, _ = first_batch
        else:
            print(f"Warning: Expected 3 items in NSP batch, got {len(first_batch)}. Check dataloader. Skipping test_causal_attention.")
            return
    else:
        # Expecting (input_ids, lm_targets)
        if len(first_batch) == 2:
            x, _ = first_batch # y (lm_targets) is not used in this function
        else:
            print(f"Warning: Expected 2 items in non-NSP batch, got {len(first_batch)}. Check dataloader. Skipping test_causal_attention.")
            return

    x = x.to(device)
    # The rest of the function uses 'x'

    model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Test with causal=True (default)
        print("Testing with causal=True...")
        # model.forward now returns logits, loss, nsp_logits
        logits_causal, _, _ = model(x)
        
        # Test with causal=False by modifying the attention layers
        print("Testing with causal=False...")
        # Temporarily change causal setting
        original_causal = []
        for block in model.blocks:
            original_causal.append(block.attn.causal)
            block.attn.causal = False
        
        # model.forward now returns logits, loss, nsp_logits
        logits_non_causal, _, _ = model(x)
        
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
    # Changed --nsp-task to allow True/False, defaulting to None if not specified by user
    parser.add_argument(
        '--nsp-task',
        type=lambda x: (str(x).lower() == 'true'),
        default=None,  # If not used, arg is None. Config/code default will be True.
        help='Set NSP task status (True/False). If not specified, defaults to True.'
    )
    parser.add_argument('--nsp-loss-weight', type=float, default=None, help='Weight for NSP loss component (overrides config, e.g., 0.5).')
    parser.add_argument(
        '--use-cls-prefix-attention',
        type=lambda x: (str(x).lower() == 'true'),
        default=None,
        help='Enable/disable special prefix attention for CLS token when NSP task is active. (e.g., True or False, overrides config)'
    )
    parser.add_argument('--cpu-test-attention', action='store_true', help='Use CPU fallback for attention mechanism (for testing).')

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
