"""
Training Infrastructure for Hierarchical Attention Models

This module provides comprehensive training and evaluation framework for GPT models
with support for both teacher forcing and cocktail party tasks. Includes distributed
training, mixed precision, and task-specific evaluation metrics.

Key Components:
- TrainingConfig: Configuration management for training parameters
- TrainingMetrics: Performance tracking and logging
- Trainer: Main training loop with multi-task support
- Evaluation utilities for both task types
- Text generation with advanced sampling controls
- Distributed training coordination
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_builder import BIO_TAGS
from torch.distributions import Bernoulli
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import time
import os
import json
import math
from pathlib import Path

# =============================================================================
# Configuration Classes
# =============================================================================


# =============================================================================
# Configuration Classes
# =============================================================================

class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(
        self,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        save_every: int = 1000,
        eval_every: int = 500,
        log_every: int = 100,
        moving_avg_window: int = 100,
        inference_every: int = 500,
        save_logs_json_every: int = 500,
        checkpoint_dir: str = "checkpoints",
        device: str = "auto",
        use_amp: bool = False,
        scaler: Optional[Any] = None,
        # Checkpoint configuration
        auto_resume: bool = True,
        max_checkpoints: int = 2,
        # Inference sampling parameters
        inference_prompts: List[str] = None,
        inference_max_length: int = 100,
        inference_temperature: float = 0.8,
        inference_top_k: int = 50,
        inference_top_p: float = 0.9,
        # Plateau detection parameters
        plateau_monitor_metric: str = 'val_loss',
        plateau_patience: int = 10,
        plateau_threshold: float = 1e-4,
        plateau_mode: str = 'min',
        # Scheduler parameters
        scheduler_T0_epoch_fraction: float = 0.1,
        scheduler_T_mult: int = 1
    ):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.save_every = save_every
        self.eval_every = eval_every
        self.log_every = log_every
        self.moving_avg_window = moving_avg_window
        self.inference_every = inference_every
        self.save_logs_json_every = save_logs_json_every
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp
        self.scaler = scaler
        
        # Checkpoint configuration
        self.auto_resume = auto_resume
        self.max_checkpoints = max_checkpoints
        
        # Inference sampling configuration
        self.inference_prompts = inference_prompts or ["", "The", "In", "Once upon a time"]
        self.inference_max_length = inference_max_length
        self.inference_temperature = inference_temperature
        self.inference_top_k = inference_top_k
        self.inference_top_p = inference_top_p

        # Plateau detection parameters
        self.plateau_monitor_metric = plateau_monitor_metric
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold
        self.plateau_mode = plateau_mode

        # Scheduler parameters
        self.scheduler_T0_epoch_fraction = scheduler_T0_epoch_fraction
        self.scheduler_T_mult = scheduler_T_mult
        
        # Auto-detect device
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Create checkpoint directory
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.local_rank = -1 # For DDP
        self.is_distributed = False # Will be set by init_distributed

# Note: This function is called in Trainer.__init__ and sets Trainer's attributes.
def init_distributed(trainer_instance: 'Trainer'):
    """Initializes the distributed training environment."""
    if dist.is_available() and dist.is_initialized():
        trainer_instance.is_distributed = True
        # Ensure local_rank is set if already initialized
        if hasattr(trainer_instance.config, 'local_rank') and trainer_instance.config.local_rank == -1:
             trainer_instance.config.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        return

    rank_env = os.environ.get('RANK')
    world_size_env = os.environ.get('WORLD_SIZE')
    local_rank_env = os.environ.get('LOCAL_RANK')

    if rank_env is not None and world_size_env is not None:
        try:
            rank = int(rank_env)
            world_size = int(world_size_env)

            if local_rank_env is not None:
                local_rank = int(local_rank_env)
            else: # Fallback if LOCAL_RANK is not set (e.g. older torch versions or different launcher)
                local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0

            trainer_instance.config.local_rank = local_rank

            if torch.cuda.is_available():
                backend = 'nccl'
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
                trainer_instance.is_distributed = True
                trainer_instance.config.device = torch.device(f"cuda:{local_rank}")
                print(f"Distributed training initialized (RANK {rank}/{world_size}, LOCAL_RANK {local_rank}) with backend: {backend} on device {trainer_instance.config.device}")
            else:
                print("CUDA not available. Distributed training with NCCL backend not possible.")
                trainer_instance.is_distributed = False
        except ValueError:
            print("RANK, WORLD_SIZE, or LOCAL_RANK environment variables are not valid integers.")
            trainer_instance.is_distributed = False
        except Exception as e:
            print(f"Error initializing distributed group: {e}")
            trainer_instance.is_distributed = False
    else:
        print("RANK and/or WORLD_SIZE env variables not set. Running in non-distributed mode.")
        trainer_instance.is_distributed = False

    if not trainer_instance.is_distributed:
        if trainer_instance.config.device == "auto" or not isinstance(trainer_instance.config.device, torch.device):
             trainer_instance.config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer_instance.config.local_rank = 0 # Default for non-distributed
        print(f"Running in non-distributed mode on device: {trainer_instance.config.device}")

# =============================================================================
# Metrics Tracking
# =============================================================================

class TrainingMetrics:
    """Class to track and manage training metrics."""
    
    def __init__(self, moving_avg_window: int = 100):
        self.train_losses = []
        self.val_losses = []
        self.cocktail_party_metrics = []
        self.learning_rates = []
        self.step_times = []
        self.total_steps = 0
        self.best_val_loss = float('inf')
        self.best_step = 0
        self.moving_avg_window = moving_avg_window
        self.recent_train_losses = []
    
    def update(
        self,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        cocktail_party_metrics: Optional[Dict[str, float]] = None,
        learning_rate: Optional[float] = None,
        step_time: Optional[float] = None
    ):
        """Update metrics with new values."""
        if train_loss is not None:
            self.train_losses.append(train_loss)
            self.recent_train_losses.append(train_loss)
            if len(self.recent_train_losses) > self.moving_avg_window:
                self.recent_train_losses.pop(0)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_step = self.total_steps
        
        if cocktail_party_metrics is not None:
            self.cocktail_party_metrics.append(cocktail_party_metrics)

        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        
        if step_time is not None:
            self.step_times.append(step_time)
        
        self.total_steps += 1

    def get_loss_moving_average(self) -> float:
        if not self.recent_train_losses:
            return 0.0
        return np.mean(self.recent_train_losses)

    def get_loss_variance(self) -> float:
        if len(self.recent_train_losses) < 2:
            return 0.0
        return np.var(self.recent_train_losses)
    
    def get_avg_step_time(self, last_n: int = 100) -> float:
        """Get average step time for the last N steps."""
        if not self.step_times:
            return 0.0
        recent_times = self.step_times[-last_n:]
        return np.mean(recent_times)
    
    def save_metrics(self, filepath: str):
        """Save metrics to a file."""
        metrics_dict = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'cocktail_party_metrics': self.cocktail_party_metrics,
            'learning_rates': self.learning_rates,
            'step_times': self.step_times,
            'total_steps': self.total_steps,
            'best_val_loss': self.best_val_loss,
            'best_step': self.best_step
        }
        torch.save(metrics_dict, filepath)
    
    def save_metrics_json(self, filepath: str):
        """Save metrics to a JSON file."""
        metrics_dict = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'cocktail_party_metrics': self.cocktail_party_metrics,
            'learning_rates': self.learning_rates,
            'step_times': self.step_times,
            'total_steps': self.total_steps,
            'best_val_loss': self.best_val_loss,
            'best_step': self.best_step,
            'timestamp': time.time()
        }
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

# =============================================================================
# Main Trainer Class
# =============================================================================

class Trainer:
    """Main training class that handles the training loop."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
        data_builder: Any = None
    ):
        self.model = model
        self.config = config
        self.data_builder = data_builder
        self.metrics = TrainingMetrics(moving_avg_window=self.config.log_every)
        self.is_distributed = False # Will be set by init_distributed

        # Initialize plateau tracking attributes
        self.plateau_patience_counter: int = 0
        self.plateau_best_metric_val: float = float('inf') if self.config.plateau_mode == 'min' else float('-inf')
        self.steps_per_epoch: Optional[int] = None

        # Initialize distributed training
        init_distributed(self)
        
        # Apply speed optimizations
        self._apply_pytorch_optimizations()

    def _apply_pytorch_optimizations(self):
        """Apply PyTorch optimization flags for speed"""
        try:
            # Enable cuDNN benchmark for consistent input sizes
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                
                # Allow TF32 for speed on modern GPUs
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                if hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    
            if not self.is_distributed or dist.get_rank() == 0:
                print("✓ PyTorch optimization flags enabled for speed")
        except Exception as e:
            if not self.is_distributed or dist.get_rank() == 0:
                print(f"⚠ Could not apply all PyTorch optimizations: {e}")

        if self.is_distributed and dist.get_world_size() > 1:
            # self.config.device is set in init_distributed
            self.model.to(self.config.device)
            # find_unused_parameters can be true if some outputs of the model are not used in loss calculation
            self.model = DDP(self.model, device_ids=[self.config.local_rank], output_device=self.config.local_rank, find_unused_parameters=False)
            print(f"Model moved to device: {self.config.device} and wrapped with DDP (world size: {dist.get_world_size()}).")
        else:
            # self.config.device is set in init_distributed or by original logic
            self.model.to(self.config.device)
            print(f"Model moved to device: {self.config.device} (non-distributed or world_size=1).")

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            fused=True
        )
        
        # Initialize learning rate scheduler
        self.initial_lr = self.config.learning_rate
        self.scheduler = None # Will be initialized in train()
        
        print(f"Trainer initialized on device: {self.config.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def warmup_lr(self, step: int) -> float:
        """Calculate learning rate with warmup."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * step / self.config.warmup_steps
        return self.config.learning_rate
    
    def train_step(self, batch: Tuple, task_name: str, task_configs: Dict[str, Any]) -> float:
        """Perform a single training step with speed optimizations."""
        if task_name == 'cocktail_party':
            inputs, correct_idx, metadata = batch
            if inputs.numel() == 0:
                return 0.0
            
            # Use non_blocking=True for faster GPU transfers
            inputs = inputs.to(self.config.device, non_blocking=True)
            correct_idx = correct_idx.to(self.config.device, non_blocking=True)
            
            # Move metadata tensors to device efficiently
            if isinstance(metadata, dict):
                metadata = {k: v.to(self.config.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                          for k, v in metadata.items()}
            else:
                if metadata is not None:
                    metadata = metadata.to(self.config.device, non_blocking=True)

            if self.config.use_amp and self.config.scaler is not None:
                with torch.amp.autocast('cuda'):
                    scores, loss = self.model(inputs, correct_idx=correct_idx, attention_mask=metadata, task_name=task_name)
            else:
                scores, loss = self.model(inputs, correct_idx=correct_idx, attention_mask=metadata, task_name=task_name)

        else:
            # Teacher forcing and other tasks
            x, y = batch
            # Use non_blocking=True for faster GPU transfers
            x = x.to(self.config.device, non_blocking=True)
            y = y.to(self.config.device, non_blocking=True)

            # Generate metadata for teacher forcing to ensure consistent attention patterns
            metadata = self._generate_teacher_forcing_metadata(x)

            if self.config.use_amp and self.config.scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits, loss = self.model(x, targets=y, attention_mask=metadata, task_name=task_name)
            else:
                logits, loss = self.model(x, targets=y, attention_mask=metadata, task_name=task_name)

        if loss is None:
            return 0.0

        return loss

    def _generate_teacher_forcing_metadata(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate metadata tensors for teacher forcing tasks to ensure consistent attention patterns."""
        from data_builder import SPECIAL_TOKENS
        
        batch_size, seq_len = x.shape
        device = x.device
        
        # Initialize metadata tensors
        in_span = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        span_id = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        is_prefix = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        
        cls_token_id = SPECIAL_TOKENS['[CLS]']
        
        # For teacher forcing: Mark everything up to and including [CLS] as prefix
        # This ensures bidirectional attention within prefix, causal after
        for batch_idx in range(batch_size):
            cls_positions = (x[batch_idx] == cls_token_id).nonzero(as_tuple=True)[0]
            if len(cls_positions) > 0:
                # Mark everything up to and including the first [CLS] as prefix
                cls_pos = cls_positions[0].item()
                is_prefix[batch_idx, :cls_pos + 1] = True
        
        # For teacher forcing: in_span and span_id remain zero (no spans)
        
        return {
            'in_span': in_span,
            'span_id': span_id,
            'is_prefix': is_prefix
        }

    def _calculate_accuracy(self, scores: torch.Tensor, correct_idx: torch.Tensor) -> Dict[str, float]:
        """Calculates accuracy for the contrastive task."""
        predicted_idx = torch.argmax(scores, dim=1)
        accuracy = (predicted_idx == correct_idx).float().mean().item()
        return {'accuracy': accuracy}

    def evaluate(self, dataloaders: Dict[str, DataLoader], task_configs: Dict[str, Any], max_batches: Optional[int] = 50, task_to_evaluate: Optional[str] = None) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model on a dataset."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        cocktail_party_metrics = {}
        
        with torch.no_grad():
            for task_name, dataloader in dataloaders.items():
                if task_to_evaluate and task_name != task_to_evaluate:
                    continue
                for batch_idx, batch in enumerate(dataloader):
                    if max_batches is not None and batch_idx >= max_batches:
                        if not task_to_evaluate:
                            print(f"Evaluation for task {task_name} limited to {max_batches} batches for speed")
                        break

                    loss = None
                    if task_name == 'cocktail_party':
                        inputs, correct_idx, metadata = batch
                        if inputs.numel() == 0:
                            continue
                        inputs, correct_idx = inputs.to(self.config.device), correct_idx.to(self.config.device)
                        
                        # Move metadata tensors to device
                        if isinstance(metadata, dict):
                            metadata = {k: v.to(self.config.device) for k, v in metadata.items()}
                        else:
                            metadata = metadata.to(self.config.device)

                        if self.config.use_amp:
                            with torch.amp.autocast('cuda'):
                                scores, loss = self.model(inputs, correct_idx=correct_idx, attention_mask=metadata, task_name=task_name)
                        else:
                            scores, loss = self.model(inputs, correct_idx=correct_idx, attention_mask=metadata, task_name=task_name)

                        if loss is not None and scores.numel() > 0:
                            metrics = self._calculate_accuracy(scores, correct_idx)
                            for k, v in metrics.items():
                                if k not in cocktail_party_metrics:
                                    cocktail_party_metrics[k] = []
                                cocktail_party_metrics[k].append(v)
                    else:
                        # Teacher forcing and other tasks
                        x, y = batch
                        x, y = x.to(self.config.device), y.to(self.config.device)

                        # Generate metadata for teacher forcing to ensure consistent attention patterns
                        metadata = self._generate_teacher_forcing_metadata(x)

                        if self.config.use_amp:
                            with torch.amp.autocast('cuda'):
                                logits, loss = self.model(x, targets=y, attention_mask=metadata, task_name=task_name)
                        else:
                            logits, loss = self.model(x, targets=y, attention_mask=metadata, task_name=task_name)

                    if loss is not None:
                        if isinstance(loss, dict):
                            batch_loss = 0
                            for loss_name, loss_value in loss.items():
                                weight = task_configs.get(task_name, {}).get(f"{loss_name}_weight", 1.0)
                                batch_loss += weight * loss_value
                            total_loss += batch_loss.item()
                        else:
                            total_loss += loss.item()
                        num_batches += 1

        self.model.train()

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        avg_cocktail_party_metrics = {k: np.mean(v) for k, v in cocktail_party_metrics.items()}

        return avg_loss, avg_cocktail_party_metrics

    
    def save_checkpoint(self, step: int, train_loaders: Dict[str, DataLoader], is_best: bool = False):
        """Save model checkpoint with rotation to keep only the 2 most recent."""
        if not self.is_distributed or dist.get_rank() == 0:
            model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
            
            sampler_states = {
                name: loader.sampler.state_dict()
                for name, loader in train_loaders.items()
                if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'state_dict')
            }
            
            dataset_state = getattr(self, 'dataset_state', {})
            dataset_state['sampler_states'] = sampler_states

            checkpoint = {
                'step': step,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'metrics': self.metrics.__dict__,
                'config': self.config.__dict__,
                'dataset_state': dataset_state,
            }

            checkpoint_path = os.path.join(
                self.config.checkpoint_dir,
                f'checkpoint_step_{step}.pt'
            )
            torch.save(checkpoint, checkpoint_path)

            # Manage checkpoint rotation - keep only 2 most recent regular checkpoints
            self._cleanup_old_checkpoints()

            if is_best:
                best_path = os.path.join(
                    self.config.checkpoint_dir,
                    'best_checkpoint.pt'
                )
                torch.save(checkpoint, best_path)

            print(f"Checkpoint saved by Rank 0: {checkpoint_path}")
        else:
            # Ensure all processes are synchronized before rank 0 might save a new checkpoint
            # or other processes might proceed with a new model state if loading occurs.
            if self.is_distributed:
                dist.barrier()
        
    def _cleanup_old_checkpoints(self):
        """Keep only the max_checkpoints most recent regular checkpoints."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        # Find all regular checkpoint files (not best_checkpoint.pt)
        checkpoint_files = []
        for file_path in checkpoint_dir.glob('checkpoint_step_*.pt'):
            try:
                # Extract step number from filename
                step_num = int(file_path.stem.split('_')[-1])
                checkpoint_files.append((step_num, file_path))
            except (ValueError, IndexError):
                continue
        
        # Sort by step number, newest first
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        
        # Remove all but the max_checkpoints most recent
        max_checkpoints = getattr(self.config, 'max_checkpoints', 2)
        for _, file_path in checkpoint_files[max_checkpoints:]:
            try:
                file_path.unlink()
                print(f"Removed old checkpoint: {file_path}")
            except FileNotFoundError:
                pass  # File already removed
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint file."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        if not checkpoint_dir.exists():
            return None
        
        # Find all regular checkpoint files
        checkpoint_files = []
        for file_path in checkpoint_dir.glob('checkpoint_step_*.pt'):
            try:
                step_num = int(file_path.stem.split('_')[-1])
                checkpoint_files.append((step_num, file_path))
            except (ValueError, IndexError):
                continue
        
        if not checkpoint_files:
            return None
        
        # Return the most recent checkpoint
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        return str(checkpoint_files[0][1])
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        # Ensure all processes load the same checkpoint
        # In DDP, it's common to load checkpoint on all ranks,
        # or load on rank 0 and then broadcast. Loading on all ranks is simpler.
        map_location = self.config.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        state_dict = checkpoint['model_state_dict']
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model

        # Handle 'module.' prefix differences
        current_keys_have_module = all(k.startswith('module.') for k in model_to_load.state_dict().keys())
        checkpoint_keys_have_module = all(k.startswith('module.') for k in state_dict.keys())

        if isinstance(self.model, DDP): # Current model is DDP
            if not checkpoint_keys_have_module: # Checkpoint from non-DDP
                # self.model.module.load_state_dict(state_dict) # Already handled by model_to_load
                pass
            else: # Checkpoint from DDP (has module. prefix)
                new_state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                state_dict = new_state_dict
        else: # Current model is NOT DDP
            if checkpoint_keys_have_module: # Checkpoint from DDP
                new_state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                state_dict = new_state_dict
            # else: non-DDP to non-DDP, no change needed

        model_to_load.load_state_dict(state_dict)

        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded.")

        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded.")

        for key, value in checkpoint['metrics'].items():
            setattr(self.metrics, key, value)
        
        if 'dataset_state' in checkpoint:
            self.dataset_state = checkpoint['dataset_state']
            print(f"Dataset state loaded: epoch {self.dataset_state.get('current_epoch', 'N/A')}, batch {self.dataset_state.get('current_batch', 'N/A')}")
        
        print(f"Checkpoint loaded from {checkpoint_path} at step {checkpoint['step']}")
        return checkpoint['step']
    
    def _fast_forward_dataloader(self, dataloader: DataLoader, batch_to_resume: int):
        """Fast-forwards a dataloader iterator to a specific batch index."""
        if batch_to_resume == 0:
            return iter(dataloader)

        print(f"Fast-forwarding dataloader to batch {batch_to_resume}...")
        it = iter(dataloader)
        for _ in range(batch_to_resume):
            try:
                next(it)
            except StopIteration:
                print(f"Warning: Reached end of dataloader while fast-forwarding.")
                # Return a new iterator from the beginning
                return iter(dataloader)
        print("Fast-forward complete.")
        return it

    def train_epoch(
        self,
        train_loaders: Dict[str, DataLoader],
        val_loaders: Optional[Dict[str, DataLoader]] = None,
        epoch: int = 0,
        task_configs: Dict[str, Any] = None,
        batch_to_resume: int = 0,
    ):
        """Train for one epoch."""
        self.model.train()
        start_time = time.time()
        epoch_losses = []

        if batch_to_resume > 0:
            train_iters = {
                task: self._fast_forward_dataloader(loader, batch_to_resume)
                for task, loader in train_loaders.items()
            }
        else:
            train_iters = {task: iter(loader) for task, loader in train_loaders.items()}

        if not hasattr(self, 'dataset_state'):
            self.dataset_state = {}
        
        self.dataset_state.update({
            'current_epoch': epoch,
            'steps_per_epoch': self.steps_per_epoch,
            'total_epochs': self.config.num_epochs,
        })
        
        for batch_idx in range(batch_to_resume, self.steps_per_epoch):
            step_start = time.time()
            self.dataset_state['current_batch'] = batch_idx
            
            total_loss = 0
            individual_losses = {}

            for task_name, task_iter in train_iters.items():
                try:
                    batch = next(task_iter)
                except StopIteration:
                    train_iters[task_name] = iter(train_loaders[task_name])
                    batch = next(train_iters[task_name])

                loss = self.train_step(batch, task_name, task_configs)
                if loss is not None and not (isinstance(loss, float) and loss == 0.0):
                    individual_losses[task_name] = loss.item()
                    total_loss += loss

            if not isinstance(total_loss, torch.Tensor):
                continue

            epoch_losses.append(total_loss.item())

            self.optimizer.zero_grad()
            if self.config.use_amp and self.config.scaler:
                self.config.scaler.scale(total_loss).backward()
                if self.config.max_grad_norm > 0:
                    self.config.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.config.scaler.step(self.optimizer)
                self.config.scaler.update()
            else:
                total_loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                self.scheduler.step()
            
            step_time = time.time() - step_start
            self.metrics.update(
                train_loss=total_loss.item(),
                learning_rate=current_lr,
                step_time=step_time
            )

            if (not self.is_distributed or dist.get_rank() == 0) and self.metrics.total_steps % self.config.log_every == 0:
                avg_step_time = self.metrics.get_avg_step_time()
                loss_ma = self.metrics.get_loss_moving_average()
                log_str = f"Epoch {epoch+1}, Step {self.metrics.total_steps}, Total Loss: {total_loss.item():.4f} (MA: {loss_ma:.4f}), LR: {current_lr:.6f}, Step Time: {avg_step_time:.3f}s"
                print(log_str)

            if (not self.is_distributed or dist.get_rank() == 0):
                if val_loaders and self.metrics.total_steps % self.config.eval_every == 0:
                    val_loss, cocktail_party_metrics = self.evaluate(val_loaders, task_configs)
                    self.metrics.update(val_loss=val_loss, cocktail_party_metrics=cocktail_party_metrics)
                    is_best = val_loss < self.metrics.best_val_loss
                    print(f"Validation Loss: {val_loss:.4f} {'(Best!)' if is_best else ''}")
                    if is_best:
                        self.save_checkpoint(self.metrics.total_steps, train_loaders, is_best=True)

                if self.metrics.total_steps > 0 and self.metrics.total_steps % self.config.save_every == 0:
                    self.save_checkpoint(self.metrics.total_steps, train_loaders)

                if self.metrics.total_steps > 0 and self.metrics.total_steps % self.config.save_logs_json_every == 0:
                    logs_dir = Path(self.config.checkpoint_dir) / "training_logs"
                    logs_dir.mkdir(exist_ok=True)
                    json_path = logs_dir / "training_logs.json"
                    self.metrics.save_metrics_json(str(json_path))
                    print(f"Training logs saved to JSON: {json_path}")

                # Periodic inference
                if val_loaders is not None and self.metrics.total_steps > 0 and self.metrics.total_steps % self.config.inference_every == 0:
                    print(f"\n=== Generating Inference Sample at Step {self.metrics.total_steps} ===")
                    perplexity = self.calculate_perplexity(val_loaders, max_batches=20)
                    current_val_loss = self.metrics.val_losses[-1] if self.metrics.val_losses else float('inf')

                    prompts = self.config.inference_prompts
                    generated_texts = self.generate_inference_sample(
                        prompts=prompts,
                        max_length=self.config.inference_max_length,
                        temperature=self.config.inference_temperature,
                        top_k=self.config.inference_top_k,
                        top_p=self.config.inference_top_p
                    )
                    self.save_inference_sample(
                        step=self.metrics.total_steps,
                        val_loss=current_val_loss,
                        perplexity=perplexity,
                        generated_texts=generated_texts,
                        prompts=prompts
                    )
        
        # Epoch summary (only on rank 0 if distributed)
        if not self.is_distributed or dist.get_rank() == 0:
            # This epoch_losses is from rank 0 only if distributed.
            # For a true average, losses would need to be gathered.
            avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
            epoch_time = time.time() - start_time
            print(
                f"Epoch {epoch+1} completed (Rank {dist.get_rank() if self.is_distributed else 0}): "
                f"Avg Loss: {avg_loss:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
        
        return avg_loss
    
    def train(
        self,
        train_loaders: Dict[str, DataLoader],
        val_loaders: Optional[Dict[str, DataLoader]] = None,
        task_configs: Dict[str, Any] = None,
        resume_from_checkpoint: Optional[bool] = None
    ):
        """Main training loop."""
        start_epoch = 0
        batch_to_resume = 0
        
        if resume_from_checkpoint is None:
            resume_from_checkpoint = self.config.auto_resume

        # Initialize scheduler here so it can be loaded from checkpoint
        self.steps_per_epoch = max(len(loader) for loader in train_loaders.values())
        total_steps = self.steps_per_epoch * self.config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=0)
        
        if resume_from_checkpoint:
            latest_checkpoint = self.find_latest_checkpoint()
            if latest_checkpoint:
                try:
                    loaded_step = self.load_checkpoint(latest_checkpoint)
                    if hasattr(self, 'dataset_state'):
                        start_epoch = self.dataset_state.get('current_epoch', 0)
                        batch_to_resume = self.dataset_state.get('current_batch', 0) + 1

                        if 'sampler_states' in self.dataset_state:
                            for name, state in self.dataset_state['sampler_states'].items():
                                if name in train_loaders and hasattr(train_loaders[name].sampler, 'load_state_dict'):
                                    train_loaders[name].sampler.load_state_dict(state)
                                    print(f"Loaded sampler state for '{name}'.")

                        print(f"Resuming training from step {loaded_step}, epoch {start_epoch + 1}, batch {batch_to_resume}")
                except Exception as e:
                    print(f"Failed to load checkpoint {latest_checkpoint}: {e}")
            else:
                print("No existing checkpoints found. Starting fresh.")

        try:
            for epoch in range(start_epoch, self.config.num_epochs):
                if self.is_distributed:
                    for loader in train_loaders.values():
                        if hasattr(loader.sampler, 'set_epoch'):
                            loader.sampler.set_epoch(epoch)
                
                self.train_epoch(train_loaders, val_loaders, epoch, task_configs, batch_to_resume)
                batch_to_resume = 0  # Reset for subsequent epochs

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving final checkpoint...")
        
        finally:
            if not self.is_distributed or dist.get_rank() == 0:
                print("Saving final checkpoint and metrics...")
                self.save_checkpoint(self.metrics.total_steps, train_loaders)
                metrics_path = Path(self.config.checkpoint_dir) / 'training_metrics.json'
                self.metrics.save_metrics_json(str(metrics_path))
                print(f"Final metrics saved to {metrics_path}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        if self.metrics.train_losses:
            axes[0, 0].plot(self.metrics.train_losses, 'b-', alpha=0.7, label='Train Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        # Validation loss
        if self.metrics.val_losses:
            val_steps = np.linspace(0, len(self.metrics.train_losses), len(self.metrics.val_losses))
            axes[0, 1].plot(val_steps, self.metrics.val_losses, 'r-', alpha=0.7, label='Val Loss')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # Learning rate
        if self.metrics.learning_rates:
            axes[1, 0].plot(self.metrics.learning_rates, 'g-', alpha=0.7)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Step times
        if self.metrics.step_times:
            axes[1, 1].plot(self.metrics.step_times, 'orange', alpha=0.7)
            axes[1, 1].set_title('Step Time')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Time (s)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved: {save_path}")
        
        plt.show()
    
    def generate_sample(
        self,
        prompt: str = "",
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = None
    ) -> str:
        """Generate a text sample from the model."""
        if not self.data_builder:
            print("No data_builder provided for text generation.")
            return ""
        
        self.model.eval()
        model_to_generate_from = self.model.module if isinstance(self.model, DDP) else self.model
        
        with torch.no_grad():
            if prompt:
                # Tokenize prompt using DataBuilder's method
                tokens = self.data_builder._tokenize_text(f"[CLS] {prompt}")
                x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.config.device)
            else:
                # Start with a CLS token
                tokens = self.data_builder._tokenize_text("[CLS]")
                x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.config.device)
            
            if self.config.use_amp:
                # Use mixed precision for generation
                with torch.amp.autocast('cuda'):
                    generated = model_to_generate_from.generate(
                        x,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
            else:
                # Standard precision generation
                generated = model_to_generate_from.generate(
                    x,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
            
            # Decode to text
            generated_text = self.data_builder.decode_tokens(generated[0])
            
        self.model.train()
        return generated_text
    
    def calculate_perplexity(self, dataloaders: Dict[str, DataLoader], max_batches: Optional[int] = 50) -> float:
        """Calculate perplexity on a dataset."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for task_name, dataloader in dataloaders.items():
                # Perplexity is only meaningful for the generative, teacher_forcing task
                if task_name != 'teacher_forcing':
                    continue
                for batch_idx, batch in enumerate(dataloader):
                    if max_batches is not None and batch_idx >= max_batches:
                        break

                    x, y = batch
                    x, y = x.to(self.config.device), y.to(self.config.device)
                    
                    if self.config.use_amp and self.config.scaler is not None:
                        with torch.amp.autocast('cuda'):
                            logits, loss = self.model(x, targets=y, task_name=task_name)
                    else:
                        logits, loss = self.model(x, targets=y, task_name=task_name)

                    if loss is not None:
                        # Direct loss accumulation without uncertainty weighting
                        total_loss += loss.item()
                        num_batches += 1
        
        self.model.train()
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss != float('inf') else float('inf')
        return perplexity
    
    def generate_inference_sample(
        self,
        prompts: List[str] = None,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> List[str]:
        """Generate inference samples with top-k, top-p, and temperature sampling."""
        if not self.data_builder:
            return ["No data_builder provided for text generation."]
        
        if prompts is None:
            prompts = ["", "The", "In", "Once upon a time"]
        
        self.model.eval()
        model_to_generate_from = self.model.module if isinstance(self.model, DDP) else self.model
        generated_texts = []
        
        with torch.no_grad():
            for prompt in prompts:
                try:
                    if prompt:
                        # Tokenize prompt using DataBuilder's method
                        tokens = self.data_builder._tokenize_text(f"[CLS] {prompt}")
                        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.config.device)
                    else:
                        # Start with a CLS token
                        tokens = self.data_builder._tokenize_text("[CLS]")
                        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.config.device)
                    
                    # Generate tokens with top-k, top-p sampling
                    generated = model_to_generate_from.generate(
                        x,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                    
                    # Decode to text
                    generated_text = self.data_builder.decode_tokens(generated[0])
                    generated_texts.append(generated_text)
                    
                except Exception as e:
                    generated_texts.append(f"Generation failed: {str(e)}")
        
        self.model.train()
        return generated_texts
    
    def save_inference_sample(
        self, 
        step: int, 
        val_loss: float, 
        perplexity: float,
        generated_texts: List[str],
        prompts: List[str] = None
    ):
        """Save inference sample to JSON file with metadata."""
        if prompts is None:
            prompts = ["", "The", "In", "Once upon a time"]
        
        # Create inference samples directory
        inference_dir = Path(self.config.checkpoint_dir) / "inference_samples"
        inference_dir.mkdir(exist_ok=True)
        
        # Create sample entry
        sample_entry = {
            "step": step,
            "validation_loss": val_loss,
            "perplexity": perplexity,
            "timestamp": time.time(),
            "samples": []
        }
        
        for prompt, generated_text in zip(prompts, generated_texts):
            sample_entry["samples"].append({
                "prompt": prompt,
                "generated_text": generated_text
            })
        
        # Load existing samples or create new list
        samples_file = inference_dir / "inference_samples.json"
        if samples_file.exists():
            try:
                with open(samples_file, 'r') as f:
                    all_samples = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                all_samples = []
        else:
            all_samples = []
        
        # Add new sample
        all_samples.append(sample_entry)
        
        # Save updated samples
        with open(samples_file, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        # Print the generated samples
        print(f"\n=== Inference Sample at Step {step} ===")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        for prompt, generated_text in zip(prompts, generated_texts):
            if prompt:
                print(f"Prompt: '{prompt}' → '{generated_text}'")
            else:
                print(f"No prompt → '{generated_text}'")
        print("=" * 50)


def find_latest_checkpoint_path(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint file in the given directory."""
    checkpoint_dir_path = Path(checkpoint_dir)
    
    if not checkpoint_dir_path.exists():
        return None
    
    # Find all regular checkpoint files
    checkpoint_files = []
    for file_path in checkpoint_dir_path.glob('checkpoint_step_*.pt'):
        try:
            step_num = int(file_path.stem.split('_')[-1])
            checkpoint_files.append((step_num, file_path))
        except (ValueError, IndexError):
            continue
    
    if not checkpoint_files:
        return None
    
    # Return the most recent checkpoint
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    return str(checkpoint_files[0][1])


def create_trainer(
    model: torch.nn.Module,
    config: TrainingConfig,
    data_builder: Any = None
) -> Trainer:
    """Factory function to create a Trainer instance."""
    return Trainer(model, config, data_builder)


if __name__ == "__main__":
    # Test the trainer (requires a model to be passed)
    print("Trainer module loaded successfully!")
    print("Use create_trainer() to create a trainer instance.")
