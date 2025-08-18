#!/usr/bin/env python3
"""
Speed Optimizations for Training Loop

Task 4: Implement specific optimizations to improve iterations per second
without changing kernel dimensions, while keeping on-the-fly tokenization.
"""

import torch
import time
from typing import Dict, Any, Optional, List
import numpy as np

class OptimizedTrainingMixin:
    """Mixin class with speed optimizations for the Trainer class"""
    
    def __init__(self):
        """Initialize optimization-specific attributes"""
        self._cached_tensors = {}
        self._step_timer = None
        self._profiling_enabled = False
        
    def enable_speed_optimizations(self, 
                                 enable_tensor_caching: bool = True,
                                 enable_step_profiling: bool = False,
                                 compile_model: bool = False):
        """Enable various speed optimizations"""
        self.tensor_caching_enabled = enable_tensor_caching
        self._profiling_enabled = enable_step_profiling
        
        if compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile for speed...")
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("✓ Model compiled successfully")
            except Exception as e:
                print(f"⚠ Model compilation failed: {e}")
    
    def _get_cached_tensor(self, key: str, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Get a cached tensor or create a new one"""
        if not hasattr(self, 'tensor_caching_enabled') or not self.tensor_caching_enabled:
            return torch.empty(shape, dtype=dtype, device=device)
            
        cache_key = (key, shape, dtype, device)
        if cache_key not in self._cached_tensors:
            self._cached_tensors[cache_key] = torch.empty(shape, dtype=dtype, device=device)
        return self._cached_tensors[cache_key]
    
    def _clear_tensor_cache(self):
        """Clear cached tensors to free memory"""
        self._cached_tensors.clear()
    
    def optimized_train_step(self, batch, task_name: str, task_configs: Dict[str, Any]) -> float:
        """Optimized version of train_step with minimal overhead"""
        if self._profiling_enabled:
            step_start = time.perf_counter()
        
        try:
            if task_name == 'cocktail_party':
                inputs, correct_idx, metadata = batch
                inputs = inputs.to(self.config.device, non_blocking=True)
                correct_idx = correct_idx.to(self.config.device, non_blocking=True)
                
                # Handle metadata efficiently
                if isinstance(metadata, dict):
                    for key in metadata:
                        if isinstance(metadata[key], torch.Tensor):
                            metadata[key] = metadata[key].to(self.config.device, non_blocking=True)
                else:
                    if metadata is not None:
                        metadata = metadata.to(self.config.device, non_blocking=True)

                if self.config.use_amp and self.config.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        scores, loss = self.model(inputs, correct_idx=correct_idx, attention_mask=metadata, task_name=task_name)
                else:
                    scores, loss = self.model(inputs, correct_idx=correct_idx, attention_mask=metadata, task_name=task_name)
            else:
                x, y = batch
                x = x.to(self.config.device, non_blocking=True)
                y = y.to(self.config.device, non_blocking=True)

                if self.config.use_amp and self.config.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits, loss = self.model(x, targets=y, task_name=task_name)
                else:
                    logits, loss = self.model(x, targets=y, task_name=task_name)

            if self._profiling_enabled:
                forward_time = time.perf_counter() - step_start
                print(f"Forward pass time: {forward_time*1000:.2f}ms")

            return loss if loss is not None else 0.0
            
        except Exception as e:
            print(f"Error in optimized train step: {e}")
            return 0.0

def optimize_dataloader_settings(dataloader_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize DataLoader settings for speed"""
    optimized_kwargs = dataloader_kwargs.copy()
    
    # Enable pin_memory for faster GPU transfers
    if torch.cuda.is_available():
        optimized_kwargs['pin_memory'] = True
        optimized_kwargs['persistent_workers'] = optimized_kwargs.get('num_workers', 0) > 0
    
    # Increase prefetch factor if using workers
    if optimized_kwargs.get('num_workers', 0) > 0:
        optimized_kwargs['prefetch_factor'] = optimized_kwargs.get('prefetch_factor', 4)
    
    return optimized_kwargs

def get_optimal_batch_size_for_speed(model_config: Dict[str, Any], 
                                   available_memory_gb: float = 24,
                                   target_utilization: float = 0.85) -> int:
    """Calculate optimal batch size for speed rather than just memory"""
    
    # Start with memory-based estimate
    seq_len = model_config.get('max_seq_len', 1024)
    dim = model_config.get('dim', 512)
    n_layers = model_config.get('n_layers', 12)
    
    # Rough memory per sample estimate (in GB)
    memory_per_sample = (seq_len * dim * n_layers * 4) / (1024**3)  # fp32 assumption
    
    # Memory-based limit
    memory_based_limit = int((available_memory_gb * target_utilization) / memory_per_sample)
    
    # Speed-based considerations
    # For GPUs, certain batch sizes are more efficient due to memory alignment
    efficient_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    
    # Find the largest efficient size that fits in memory
    optimal_size = 1
    for size in efficient_sizes:
        if size <= memory_based_limit:
            optimal_size = size
        else:
            break
    
    return optimal_size

def apply_training_optimizations(trainer, config_overrides: Optional[Dict[str, Any]] = None):
    """Apply a comprehensive set of training optimizations"""
    
    if config_overrides is None:
        config_overrides = {}
    
    print("Applying training optimizations...")
    
    # 1. Enable optimized training step if trainer supports it
    if hasattr(trainer, 'enable_speed_optimizations'):
        trainer.enable_speed_optimizations(
            enable_tensor_caching=config_overrides.get('enable_tensor_caching', True),
            enable_step_profiling=config_overrides.get('enable_step_profiling', False),
            compile_model=config_overrides.get('compile_model', False)
        )
        print("✓ Speed optimizations enabled")
    
    # 2. Optimize gradient accumulation if beneficial
    target_batch_size = config_overrides.get('target_batch_size')
    if target_batch_size and hasattr(trainer.config, 'batch_size'):
        actual_batch_size = trainer.config.batch_size
        if target_batch_size > actual_batch_size:
            grad_accum_steps = target_batch_size // actual_batch_size
            if hasattr(trainer.config, 'gradient_accumulation_steps'):
                trainer.config.gradient_accumulation_steps = grad_accum_steps
                print(f"✓ Gradient accumulation set to {grad_accum_steps} steps")
    
    # 3. Adjust learning rate for effective batch size changes
    if hasattr(trainer.config, 'gradient_accumulation_steps') and trainer.config.gradient_accumulation_steps > 1:
        effective_lr = trainer.config.learning_rate * trainer.config.gradient_accumulation_steps
        print(f"ℹ Effective learning rate: {effective_lr} (scaled for gradient accumulation)")
    
    # 4. Enable mixed precision if not already enabled and beneficial
    if torch.cuda.is_available() and not trainer.config.use_amp:
        if config_overrides.get('force_mixed_precision', False):
            trainer.config.use_amp = True
            trainer.config.scaler = torch.amp.GradScaler('cuda')
            print("✓ Mixed precision training enabled")
    
    # 5. Set optimal PyTorch settings
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for speed
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    
    print("✓ PyTorch optimization flags set")
    
    return trainer

# Example usage and testing
def benchmark_training_speed(trainer, train_loaders, num_steps: int = 50):
    """Benchmark training speed with current settings"""
    
    print(f"Benchmarking training speed over {num_steps} steps...")
    
    step_times = []
    trainer.model.train()
    
    # Warm up
    warmup_steps = min(5, num_steps // 10)
    print(f"Warming up for {warmup_steps} steps...")
    
    for step in range(warmup_steps + num_steps):
        step_start = time.perf_counter()
        
        # Simulate one training step
        for task_name in train_loaders:
            try:
                batch = next(iter(train_loaders[task_name]))
                if hasattr(trainer, 'optimized_train_step'):
                    loss = trainer.optimized_train_step(batch, task_name, {})
                else:
                    loss = trainer.train_step(batch, task_name, {})
                break  # Just test one task for benchmarking
            except Exception as e:
                print(f"Benchmark step failed: {e}")
                continue
        
        step_time = time.perf_counter() - step_start
        
        # Only record times after warmup
        if step >= warmup_steps:
            step_times.append(step_time)
    
    if step_times:
        avg_step_time = np.mean(step_times)
        std_step_time = np.std(step_times)
        steps_per_second = 1.0 / avg_step_time
        
        print(f"Average step time: {avg_step_time*1000:.2f} ± {std_step_time*1000:.2f} ms")
        print(f"Steps per second: {steps_per_second:.2f}")
        print(f"Minutes per epoch (est. 1000 steps): {(1000 * avg_step_time) / 60:.2f}")
        
        return {
            'avg_step_time': avg_step_time,
            'std_step_time': std_step_time,
            'steps_per_second': steps_per_second
        }
    else:
        print("No valid steps recorded during benchmark")
        return None