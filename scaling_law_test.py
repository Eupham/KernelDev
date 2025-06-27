#!/usr/bin/env python3
"""
Scaling Law Test Script for Learning Rate Optimization

This script tests different learning rates with varying batch sizes to find
the optimal learning rate for a given model size based on the configuration
in config.yaml.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# Import our custom modules
from model import GPTModel
from data_builder import create_data_builder
from train_loop import Trainer, TrainingConfig

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

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

def evaluate_learning_rate(
    model: GPTModel,
    data_builder,
    batch_size: int,
    num_batches: int,
    learning_rate: float,
    device: str,
    config: Dict[str, Any]
) -> float:
    """
    Train a model with specified learning rate and evaluate its performance.
    Returns the average loss over the last 10% of training.
    """
    # Create training config with specified learning rate
    training_config = TrainingConfig(
        num_epochs=1,  # Just one epoch for quick testing
        learning_rate=learning_rate,
        weight_decay=config['training']['weight_decay'],
        warmup_steps=min(10, num_batches // 10),  # 10% of batches or 10 steps, whichever is smaller
        max_grad_norm=config['training']['max_grad_norm'],
        save_every=num_batches + 1,  # Don't save checkpoints during scaling tests
        eval_every=num_batches + 1,  # Don't evaluate during scaling tests
        log_every=num_batches // 5,  # Log 5 times during the run
        checkpoint_dir=None,  # Don't save checkpoints
        device=device,
    )
    
    # Reset model parameters
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
            
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    model.train()
    losses = []
    
    for step in range(num_batches):
        # Get a batch of data
        batch = next(iter(data_builder.get_train_dataloader(batch_size)))
        input_ids = batch['input_ids'].to(device)
        
        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Compute loss (shift logits and labels for next-token prediction)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Backward pass and optimize
        loss.backward()
        if config['training']['max_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        
        if (step+1) % training_config.log_every == 0:
            print(f"Batch: {step+1}/{num_batches}, LR: {learning_rate}, Loss: {loss.item():.4f}")
    
    # Return average loss over the last 10% of training
    return np.mean(losses[-max(1, num_batches // 10):])

def run_scaling_law_test(
    config_path: str = "KernelDev/config.yaml",
    learning_rates: List[float] = None,
    batch_sizes: List[int] = None,
    num_batches_list: List[int] = None,
    output_dir: str = "scaling_results"
):
    """
    Run scaling law tests to find optimal learning rate for different batch sizes.
    
    Args:
        config_path: Path to the configuration YAML file
        learning_rates: List of learning rates to test
        batch_sizes: List of batch sizes to test
        num_batches_list: List of number of batches to run for each experiment
        output_dir: Directory to save results
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set default values if not provided
    if learning_rates is None:
        # Test a reasonable range of learning rates (log scale)
        learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    
    if batch_sizes is None:
        batch_sizes = [16]  # Default batch size
    
    if num_batches_list is None:
        num_batches_list = [10, 100, 1000]  # Default number of batches to test
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create data builder
    data_config = config['data']
    data_builder = create_data_builder(
        dataset_name=data_config['dataset_name'],
        dataset_config=data_config['dataset_config'],
        seq_len=data_config['seq_len'],
        max_samples=data_config['max_samples'],
        vocab_size=config['model']['vocab_size']
    )
    
    # Create model
    model_config = config['model']
    model = GPTModel(
        vocab_size=data_builder.vocab_size,  # Use actual vocab size from data builder
        dim=model_config['dim'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        max_seq_len=model_config['max_seq_len'],
        mlp_ratio=model_config['mlp_ratio'],
        causal=model_config['causal']
    ).to(device)
    
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run tests for each combination of batch size and number of batches
    results = {}
    best_configs = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'='*80}\nTesting with batch size: {batch_size}\n{'='*80}")
        
        for num_batches in num_batches_list:
            print(f"\n{'-'*50}\nRunning {num_batches} batches\n{'-'*50}")
            
            experiment_losses = []
            
            for lr in learning_rates:
                print(f"\nTesting learning rate: {lr}")
                start_time = time.time()
                
                # Evaluate this learning rate
                avg_loss = evaluate_learning_rate(
                    model=model,
                    data_builder=data_builder,
                    batch_size=batch_size,
                    num_batches=num_batches,
                    learning_rate=lr,
                    device=device,
                    config=config
                )
                
                duration = time.time() - start_time
                experiment_losses.append((lr, avg_loss))
                
                print(f"Learning rate: {lr}, Avg Loss: {avg_loss:.6f}, Time: {duration:.2f}s")
            
            # Find best learning rate for this configuration
            best_lr, best_loss = min(experiment_losses, key=lambda x: x[1])
            print(f"\nBest learning rate for {num_batches} batches of size {batch_size}: {best_lr} (loss: {best_loss:.6f})")
            
            # Store results
            experiment_key = f"batch_size_{batch_size}_num_batches_{num_batches}"
            results[experiment_key] = experiment_losses
            best_configs[experiment_key] = (best_lr, best_loss)
    
    # Save results
    results_file = os.path.join(output_dir, "scaling_law_results.npz")
    np.savez(results_file, 
             learning_rates=np.array(learning_rates),
             batch_sizes=np.array(batch_sizes),
             num_batches_list=np.array(num_batches_list),
             results=results,
             best_configs=best_configs)
    
    # Create plots
    create_scaling_plots(results, learning_rates, batch_sizes, num_batches_list, output_dir)
    
    # Print summary of optimal learning rates
    print("\n\n" + "="*50)
    print("SCALING LAW TEST RESULTS SUMMARY")
    print("="*50)
    
    for batch_size in batch_sizes:
        for num_batches in num_batches_list:
            key = f"batch_size_{batch_size}_num_batches_{num_batches}"
            best_lr, best_loss = best_configs[key]
            print(f"Batch Size: {batch_size}, Batches: {num_batches}, Best LR: {best_lr}, Loss: {best_loss:.6f}")
    
    # Final recommendation based on largest experiment
    final_key = f"batch_size_{batch_sizes[0]}_num_batches_{max(num_batches_list)}"
    final_best_lr, _ = best_configs[final_key]
    
    print("\n" + "="*50)
    print(f"RECOMMENDED LEARNING RATE: {final_best_lr}")
    print("="*50)
    
    return results, best_configs

def create_scaling_plots(results, learning_rates, batch_sizes, num_batches_list, output_dir):
    """Create plots visualizing the scaling law test results."""
    # Create figure for each batch size
    for batch_size in batch_sizes:
        plt.figure(figsize=(12, 8))
        
        for num_batches in num_batches_list:
            key = f"batch_size_{batch_size}_num_batches_{num_batches}"
            if key in results:
                lr_values = [item[0] for item in results[key]]
                loss_values = [item[1] for item in results[key]]
                
                plt.plot(lr_values, loss_values, 'o-', label=f"{num_batches} batches")
                
        plt.xscale('log')  # Learning rates on log scale
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title(f'Learning Rate vs. Loss (Batch Size: {batch_size})')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        # Find optimal point and mark it
        best_lr = None
        best_loss = float('inf')
        best_num_batches = None
        
        for num_batches in num_batches_list:
            key = f"batch_size_{batch_size}_num_batches_{num_batches}"
            if key in results:
                min_loss_idx = np.argmin([item[1] for item in results[key]])
                lr, loss = results[key][min_loss_idx]
                
                if loss < best_loss:
                    best_loss = loss
                    best_lr = lr
                    best_num_batches = num_batches
                
                plt.scatter([lr], [loss], marker='*', s=200, 
                           label=f'Best LR for {num_batches}: {lr}', zorder=5)
        
        plt.axvline(x=best_lr, color='r', linestyle='--', alpha=0.3,
                   label=f'Overall Best LR: {best_lr}')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'scaling_law_batch_{batch_size}.png')
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    
    # Create 3D surface plot if we have multiple batch sizes
    if len(batch_sizes) > 1 and len(num_batches_list) > 1:
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Prepare data for 3D plotting
            X, Y, Z = [], [], []
            
            for batch_size in batch_sizes:
                for num_batches in num_batches_list:
                    key = f"batch_size_{batch_size}_num_batches_{num_batches}"
                    if key in results:
                        for lr, loss in results[key]:
                            X.append(np.log10(lr))  # log10 of learning rate
                            Y.append(np.log10(batch_size * num_batches))  # log10 of total tokens
                            Z.append(loss)
            
            # Create the 3D scatter plot
            ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=50)
            
            ax.set_xlabel('Log10 Learning Rate')
            ax.set_ylabel('Log10 Total Tokens')
            ax.set_zlabel('Loss')
            ax.set_title('Scaling Law: Loss vs Learning Rate and Training Size')
            
            plot_path = os.path.join(output_dir, 'scaling_law_3d.png')
            plt.savefig(plot_path)
            print(f"3D plot saved to {plot_path}")
            
        except ImportError:
            print("3D plotting requires mpl_toolkits.mplot3d")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scaling law tests for learning rate optimization")
    parser.add_argument('--config', default='KernelDev/config.yaml', help='Path to configuration YAML file')
    parser.add_argument('--output-dir', default='scaling_results', help='Directory to save results')
    parser.add_argument('--batch-size', default=16, type=int, help='Batch size to use for testing')
    args = parser.parse_args()
    
    # Set default parameters
    learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    batch_sizes = [args.batch_size]  # Use the specified batch size
    num_batches_list = [10, 100, 1000]  # As requested in the task
    
    run_scaling_law_test(
        config_path=args.config,
        learning_rates=learning_rates,
        batch_sizes=batch_sizes,
        num_batches_list=num_batches_list,
        output_dir=args.output_dir
    )
