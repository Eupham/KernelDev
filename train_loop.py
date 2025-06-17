import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
        checkpoint_dir: str = "checkpoints",
        device: str = "auto",
        use_amp: bool = False,
        scaler: Optional[Any] = None,
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
        scheduler_T_mult: int = 1,
        nsp_loss_weight: float = 0.5 # Added NSP loss weight
    ):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.save_every = save_every
        self.eval_every = eval_every
        self.log_every = log_every
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp
        self.scaler = scaler
        
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
        self.nsp_loss_weight = nsp_loss_weight # Store NSP loss weight
        
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


class TrainingMetrics:
    """Class to track and manage training metrics."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.step_times = []
        self.total_steps = 0
        self.best_val_loss = float('inf')
        self.best_step = 0
    
    def update(
        self,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        step_time: Optional[float] = None
    ):
        """Update metrics with new values."""
        if train_loss is not None:
            self.train_losses.append(train_loss)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_step = self.total_steps
        
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        
        if step_time is not None:
            self.step_times.append(step_time)
        
        self.total_steps += 1
    
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
            'learning_rates': self.learning_rates,
            'step_times': self.step_times,
            'total_steps': self.total_steps,
            'best_val_loss': self.best_val_loss,
            'best_step': self.best_step
        }
        torch.save(metrics_dict, filepath)


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
        self.metrics = TrainingMetrics()
        self.is_distributed = False # Will be set by init_distributed

        # Initialize plateau tracking attributes
        self.plateau_patience_counter: int = 0
        self.plateau_best_metric_val: float = float('inf') if self.config.plateau_mode == 'min' else float('-inf')
        self.steps_per_epoch: Optional[int] = None

        # Initialize distributed training
        init_distributed(self)

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
            weight_decay=self.config.weight_decay
        )
        
        # Initialize learning rate scheduler
        self.initial_lr = self.config.learning_rate
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.warmup_steps, # Placeholder T_0, will be updated
            T_mult=self.config.scheduler_T_mult,
            eta_min=self.initial_lr * 0.1 # eta_min based on initial_lr
        )
        
        print(f"Trainer initialized on device: {self.config.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def warmup_lr(self, step: int) -> float:
        """Calculate learning rate with warmup."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * step / self.config.warmup_steps
        return self.config.learning_rate
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform a single training step."""
        input_ids = batch['input_ids'].to(self.config.device)
        lm_labels = batch['lm_labels'].to(self.config.device)
        token_type_ids = batch.get('token_type_ids', None)
        nsp_label = batch.get('nsp_label', None)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.config.device)
        if nsp_label is not None:
            nsp_label = nsp_label.to(self.config.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        total_loss = None
        if self.config.use_amp and self.config.scaler is not None:
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                # model.forward(self, x, targets=None, nsp_labels=None)
                lm_logits, nsp_logits, lm_loss, nsp_loss = self.model(
                    input_ids, targets=lm_labels, nsp_labels=nsp_label
                )

            if lm_loss is None and nsp_loss is None:
                print("Warning: Both LM and NSP loss are None in train_step (AMP).")
                return 0.0
            
            current_batch_loss = 0
            if lm_loss is not None:
                current_batch_loss += lm_loss
            if nsp_loss is not None and self.config.nsp_loss_weight > 0:
                current_batch_loss += self.config.nsp_loss_weight * nsp_loss

            if isinstance(current_batch_loss, int) and current_batch_loss == 0:
                if lm_loss is not None: total_loss = lm_loss
                elif nsp_loss is not None: total_loss = nsp_loss # nsp_loss_weight might be 0
                else: return 0.0 # Should be caught by the None check
            else:
                total_loss = current_batch_loss

            if total_loss is None or (isinstance(total_loss, torch.Tensor) and total_loss.numel() == 0) : # handle cases where total_loss might be an empty tensor or still None
                print("Warning: total_loss is None or empty before backward pass in AMP mode.")
                return 0.0

            # Mixed precision backward pass
            self.config.scaler.scale(total_loss).backward()
            
            # Gradient clipping with mixed precision
            if self.config.max_grad_norm > 0:
                self.config.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            
            # Optimizer step with mixed precision
            self.config.scaler.step(self.optimizer)
            self.config.scaler.update()
            
        else:
            # Standard precision forward pass
            # model.forward(self, x, targets=None, nsp_labels=None)
            lm_logits, nsp_logits, lm_loss, nsp_loss = self.model(
                input_ids, targets=lm_labels, nsp_labels=nsp_label
            )

            if lm_loss is None and nsp_loss is None:
                print("Warning: Both LM and NSP loss are None in train_step.")
                return 0.0

            current_batch_loss = 0
            if lm_loss is not None:
                current_batch_loss += lm_loss
            if nsp_loss is not None and self.config.nsp_loss_weight > 0:
                current_batch_loss += self.config.nsp_loss_weight * nsp_loss

            if isinstance(current_batch_loss, int) and current_batch_loss == 0:
                if lm_loss is not None: total_loss = lm_loss
                elif nsp_loss is not None: total_loss = nsp_loss
                else: return 0.0
            else:
                total_loss = current_batch_loss
            
            if total_loss is None or (isinstance(total_loss, torch.Tensor) and total_loss.numel() == 0) :
                print("Warning: total_loss is None or empty before backward pass.")
                return 0.0
            
            # Standard precision backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
        
        return total_loss.item()
    
    def evaluate(self, dataloader: DataLoader, max_batches: Optional[int] = 50) -> float:
        """Evaluate the model on a dataset."""
        self.model.eval()
        accumulated_eval_loss = 0.0 # Renamed for clarity
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    print(f"Evaluation limited to {max_batches} batches for speed")
                    break
                
                input_ids = batch['input_ids'].to(self.config.device)
                lm_labels = batch['lm_labels'].to(self.config.device)
                token_type_ids = batch.get('token_type_ids', None)
                nsp_label = batch.get('nsp_label', None)

                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.config.device)
                if nsp_label is not None:
                    nsp_label = nsp_label.to(self.config.device)

                lm_loss, nsp_loss = None, None
                if self.config.use_amp:
                    with torch.amp.autocast('cuda'):
                        _, _, lm_loss, nsp_loss = self.model(input_ids, targets=lm_labels, nsp_labels=nsp_label)
                else:
                    _, _, lm_loss, nsp_loss = self.model(input_ids, targets=lm_labels, nsp_labels=nsp_label)

                batch_combined_loss = 0
                has_loss_term = False
                if lm_loss is not None:
                    batch_combined_loss += lm_loss.item()
                    has_loss_term = True
                if nsp_loss is not None and self.config.nsp_loss_weight > 0:
                    batch_combined_loss += self.config.nsp_loss_weight * nsp_loss.item()
                    has_loss_term = True
                
                if has_loss_term:
                    accumulated_eval_loss += batch_combined_loss
                    num_batches += 1
        
        self.model.train()
        return accumulated_eval_loss / num_batches if num_batches > 0 else float('inf')
    
    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        if not self.is_distributed or dist.get_rank() == 0:
            model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
            checkpoint = {
                'step': step,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'metrics': self.metrics.__dict__, # Note: metrics are rank-local
                'config': self.config.__dict__
            }

            checkpoint_path = os.path.join(
                self.config.checkpoint_dir,
                f'checkpoint_step_{step}.pt'
            )
            torch.save(checkpoint, checkpoint_path)

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
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore metrics (rank-local)
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_step_{step}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir,
                'best_checkpoint.pt'
            )
            torch.save(checkpoint, best_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore metrics
        for key, value in checkpoint['metrics'].items():
            setattr(self.metrics, key, value)
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['step']
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epoch: int = 0
    ):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        start_time = time.time()

        if self.is_distributed and hasattr(train_loader.sampler, 'set_epoch') and dist.get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()
            
            # Training step
            loss = self.train_step(batch)
            epoch_losses.append(loss)
            
            # Update learning rate scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Update metrics
            step_time = time.time() - step_start
            self.metrics.update(
                train_loss=loss,
                learning_rate=current_lr,
                step_time=step_time
            )
            

            # Logging (only on rank 0 if distributed)
            if (not self.is_distributed or dist.get_rank() == 0) and \
               self.metrics.total_steps % self.config.log_every == 0:
                avg_step_time = self.metrics.get_avg_step_time()
                print(
                    f"Epoch {epoch+1}, Step {self.metrics.total_steps}, Rank {dist.get_rank() if self.is_distributed else 0}, "
                    f"Loss: {loss:.4f}, LR: {current_lr:.6f}, "
                    f"Step Time: {avg_step_time:.3f}s"
                )
            
            # Evaluation (only on rank 0 if distributed)
            if (not self.is_distributed or dist.get_rank() == 0) and \
               val_loader is not None and \
               self.metrics.total_steps % self.config.eval_every == 0:
                val_loss = self.evaluate(val_loader)
                self.metrics.update(val_loss=val_loss) # rank-local metric
                
                is_best = val_loss < self.metrics.best_val_loss # rank-local best
                print(f"Validation Loss (Rank {dist.get_rank() if self.is_distributed else 0}): {val_loss:.4f} {'(Best!)' if is_best else ''}")
                
                if is_best: # save_checkpoint itself is rank 0 guarded
                    self.save_checkpoint(self.metrics.total_steps, is_best=True)

                # Plateau detection logic
                current_metric_val = val_loss # Assuming val_loss is the metric for now

                improved = False
                if self.config.plateau_mode == 'min':
                    if current_metric_val < self.plateau_best_metric_val - self.config.plateau_threshold:
                        improved = True
                elif self.config.plateau_mode == 'max':
                    if current_metric_val > self.plateau_best_metric_val + self.config.plateau_threshold:
                        improved = True

                if improved:
                    self.plateau_best_metric_val = current_metric_val
                    self.plateau_patience_counter = 0
                    print(f"Metric improved to {current_metric_val:.4f}. Resetting plateau patience.")
                else:
                    self.plateau_patience_counter += 1
                    print(f"Metric did not improve significantly. Plateau patience: {self.plateau_patience_counter}/{self.config.plateau_patience}")

                if self.plateau_patience_counter >= self.config.plateau_patience:
                    print(f"Plateau detected! Metric did not improve for {self.config.plateau_patience} evaluations.")
                    self.plateau_patience_counter = 0 # Reset counter

                    if self.steps_per_epoch is None:
                        print("Error: self.steps_per_epoch is not set. Cannot re-initialize scheduler correctly.")
                    else:
                        new_T0 = math.ceil(self.steps_per_epoch * self.config.scheduler_T0_epoch_fraction)
                        if new_T0 < 1: new_T0 = 1

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.initial_lr

                        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                            self.optimizer,
                            T_0=new_T0,
                            T_mult=self.config.scheduler_T_mult,
                            eta_min=self.config.learning_rate * 0.1 # Or a configured value
                        )
                        print(f"Scheduler re-initialized due to plateau. New T_0 = {new_T0}. LR reset to {self.initial_lr:.6f}")
            
            # Regular checkpoint saving & inference (only on rank 0 if distributed)
            if (not self.is_distributed or dist.get_rank() == 0) and \
               self.metrics.total_steps > 0 and \
               self.metrics.total_steps % self.config.save_every == 0:
                self.save_checkpoint(self.metrics.total_steps) # rank 0 guarded
                
                if val_loader is not None:
                    print(f"\n=== Generating Inference Sample at Step {self.metrics.total_steps} (Rank 0) ===")
                    perplexity = self.calculate_perplexity(val_loader, max_batches=20)
                    # Use last val_loss from metrics if available, else current val_loss if eval was just run
                    current_val_loss_for_sample = self.metrics.val_losses[-1] if self.metrics.val_losses else (val_loss if 'val_loss' in locals() and self.metrics.total_steps % self.config.eval_every == 0 else float('inf'))

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
                        val_loss=current_val_loss_for_sample,
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
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """Main training loop."""
        print(f"Starting training for {self.config.num_epochs} epochs...")

        train_sampler = None
        val_sampler = None
        if self.is_distributed and dist.get_world_size() > 1:
            train_sampler = DistributedSampler(train_loader.dataset, shuffle=True, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            # Keep existing train_loader settings like num_workers, pin_memory
            train_loader = DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size, # Use existing batch_size
                sampler=train_sampler,
                num_workers=getattr(train_loader, 'num_workers', 0),
                pin_memory=getattr(train_loader, 'pin_memory', False)
            )
            if val_loader:
                val_sampler = DistributedSampler(val_loader.dataset, shuffle=False, num_replicas=dist.get_world_size(), rank=dist.get_rank())
                val_loader = DataLoader(
                    val_loader.dataset,
                    batch_size=getattr(val_loader, 'batch_size', 1), # Use existing or default
                    sampler=val_sampler,
                    num_workers=getattr(val_loader, 'num_workers', 0),
                    pin_memory=getattr(val_loader, 'pin_memory', False)
                )

        if not self.is_distributed or dist.get_rank() == 0:
            print(f"Training batches per epoch (Rank 0 view): {len(train_loader)}")
            if val_loader:
                print(f"Validation batches (Rank 0 view): {len(val_loader)}")

        # Calculate steps_per_epoch and re-initialize scheduler
        self.steps_per_epoch = len(train_loader)
        if not self.is_distributed or dist.get_rank() == 0:
            print(f"Calculated steps_per_epoch: {self.steps_per_epoch}")

        new_T0 = math.ceil(self.steps_per_epoch * self.config.scheduler_T0_epoch_fraction)
        if new_T0 < 1: # Ensure T_0 is at least 1
            print(f"Warning: Calculated T_0 ({new_T0}) is less than 1. Setting to 1.")
            new_T0 = 1

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=new_T0,
            T_mult=self.config.scheduler_T_mult,
            eta_min=self.config.learning_rate * 0.1 # Assuming 0.1 factor for eta_min
        )
        if not self.is_distributed or dist.get_rank() == 0:
            print(f"Scheduler re-initialized with T_0 = {new_T0} based on steps_per_epoch.")
        
        # Initial evaluation (only on rank 0)
        if val_loader and (not self.is_distributed or dist.get_rank() == 0):
            initial_val_loss = self.evaluate(val_loader)
            self.metrics.update(val_loss=initial_val_loss) # rank-local metric
            print(f"Initial validation loss (Rank 0): {initial_val_loss:.4f}")
        
        try:
            for epoch in range(self.config.num_epochs):
                if self.is_distributed and train_sampler is not None and dist.get_world_size() > 1:
                    train_sampler.set_epoch(epoch)

                avg_loss = self.train_epoch(train_loader, val_loader, epoch) # avg_loss is rank-local
                
                # Save final checkpoint for epoch (rank 0 guarded in save_checkpoint)
                self.save_checkpoint(self.metrics.total_steps)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            raise
        
        finally:
            # Save final metrics (rank 0 guarded)
            if not self.is_distributed or dist.get_rank() == 0:
                metrics_path = os.path.join(
                    self.config.checkpoint_dir,
                    'training_metrics.pt' # rank 0's metrics
                )
                self.metrics.save_metrics(metrics_path)
                print(f"Training metrics saved (Rank 0): {metrics_path}")
    
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
                tokens = self.data_builder._tokenize_text(prompt)
                x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.config.device)
            else:
                # Start with a random token
                x = torch.randint(0, self.data_builder.vocab_size, (1, 1)).to(self.config.device)
            
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
    
    def calculate_perplexity(self, dataloader: DataLoader, max_batches: Optional[int] = 50) -> float:
        """Calculate perplexity on a dataset."""
        self.model.eval()
        total_lm_loss = 0 # Perplexity should be based on LM loss only
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                # Assuming batch is a dictionary now
                input_ids = batch['input_ids'].to(self.config.device)
                lm_labels = batch['lm_labels'].to(self.config.device)
                # NSP labels are not needed for perplexity of LM

                lm_loss = None # Initialize lm_loss for the batch
                if self.config.use_amp and self.config.scaler is not None: # scaler might be None even if use_amp is true
                    with torch.amp.autocast('cuda'):
                        # model returns: lm_logits, nsp_logits, lm_loss, nsp_loss
                        _, _, lm_loss, _ = self.model(input_ids, targets=lm_labels, nsp_labels=None)
                else:
                    _, _, lm_loss, _ = self.model(input_ids, targets=lm_labels, nsp_labels=None)
                
                if lm_loss is not None:
                    total_lm_loss += lm_loss.item()
                    num_batches += 1
        
        self.model.train()
        avg_lm_loss = total_lm_loss / num_batches if num_batches > 0 else float('inf')
        perplexity = math.exp(avg_lm_loss) if avg_lm_loss != float('inf') else float('inf')
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
                        tokens = self.data_builder._tokenize_text(prompt)
                        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.config.device)
                    else:
                        # Start with a random token
                        x = torch.randint(0, self.data_builder.vocab_size, (1, 1)).to(self.config.device)
                    
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
