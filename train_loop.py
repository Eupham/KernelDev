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
    
    def __init__(self, moving_avg_window: int = 100):
        self.train_losses = []
        self.val_losses = []
        self.cocktail_party_metrics = []
        self.distractor_loc_metrics = []
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
        distractor_loc_metrics: Optional[Dict[str, float]] = None,
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

        if distractor_loc_metrics is not None:
            self.distractor_loc_metrics.append(distractor_loc_metrics)

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
            'distractor_loc_metrics': self.distractor_loc_metrics,
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
        self.metrics = TrainingMetrics(moving_avg_window=self.config.log_every)
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
        self.scheduler = None # Will be initialized in train()
        
        print(f"Trainer initialized on device: {self.config.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def warmup_lr(self, step: int) -> float:
        """Calculate learning rate with warmup."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * step / self.config.warmup_steps
        return self.config.learning_rate
    
    def train_step(self, batch: Tuple, task_name: str, task_configs: Dict[str, Any]) -> float:
        """Perform a single training step."""
        if task_name == 'cocktail_party':
            inputs, correct_idx, attn_mask = batch
            if inputs.numel() == 0:
                return 0.0
            inputs, correct_idx, attn_mask = inputs.to(self.config.device), correct_idx.to(self.config.device), attn_mask.to(self.config.device)

            if self.config.use_amp and self.config.scaler is not None:
                with torch.amp.autocast('cuda'):
                    scores, loss = self.model(inputs, correct_idx=correct_idx, task_name=task_name)
            else:
                scores, loss = self.model(inputs, correct_idx=correct_idx, task_name=task_name)

        elif task_name == 'soft_jigsaw':
            inputs, p_star, attn_mask = batch
            if inputs is None: return 0.0 # Handle empty batches
            inputs, p_star, attn_mask = inputs.to(self.config.device), p_star.to(self.config.device), attn_mask.to(self.config.device)

            task_cfg = task_configs.get('soft_jigsaw', {})
            tau = task_cfg.get('tau', 0.1)

            if self.config.use_amp and self.config.scaler is not None:
                with torch.amp.autocast('cuda'):
                    P_hat, loss = self.model(inputs, p_star=p_star, task_name=task_name, tau=tau)
            else:
                P_hat, loss = self.model(inputs, p_star=p_star, task_name=task_name, tau=tau)
        elif task_name == 'distractor_loc':
            x_prime, m_star, c_true, l_true = batch
            if x_prime is None: return 0.0

            x_prime, m_star, c_true, l_true = (
                x_prime.to(self.config.device),
                m_star.to(self.config.device),
                c_true.to(self.config.device),
                l_true.to(self.config.device),
            )

            if self.config.use_amp and self.config.scaler is not None:
                with torch.amp.autocast('cuda'):
                    predictions, loss = self.model(x_prime, task_name=task_name, m_star=m_star, c_true=c_true, l_true=l_true)
            else:
                predictions, loss = self.model(x_prime, task_name=task_name, m_star=m_star, c_true=c_true, l_true=l_true)
        else:
            x, y = batch
            x, y = x.to(self.config.device), y.to(self.config.device)

            if self.config.use_amp and self.config.scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits, loss = self.model(x, targets=y, task_name=task_name)
            else:
                logits, loss = self.model(x, targets=y, task_name=task_name)

        if loss is None:
            return 0.0

        return loss

    def _calculate_accuracy(self, scores: torch.Tensor, correct_idx: torch.Tensor) -> Dict[str, float]:
        """Calculates accuracy for the contrastive task."""
        predicted_idx = torch.argmax(scores, dim=1)
        accuracy = (predicted_idx == correct_idx).float().mean().item()
        return {'accuracy': accuracy}

    def evaluate(self, dataloaders: Dict[str, DataLoader], task_configs: Dict[str, Any], max_batches: Optional[int] = 50, task_to_evaluate: Optional[str] = None) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Evaluate the model on a dataset."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        cocktail_party_metrics = {}
        distractor_loc_metrics = {}
        
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
                        inputs, correct_idx, attn_mask = batch
                        if inputs.numel() == 0:
                            continue
                        inputs, correct_idx, attn_mask = inputs.to(self.config.device), correct_idx.to(self.config.device), attn_mask.to(self.config.device)

                        if self.config.use_amp:
                            with torch.amp.autocast('cuda'):
                                scores, loss = self.model(inputs, correct_idx=correct_idx, task_name=task_name)
                        else:
                            scores, loss = self.model(inputs, correct_idx=correct_idx, task_name=task_name)

                        if loss is not None and scores.numel() > 0:
                            metrics = self._calculate_accuracy(scores, correct_idx)
                            for k, v in metrics.items():
                                if k not in cocktail_party_metrics:
                                    cocktail_party_metrics[k] = []
                                cocktail_party_metrics[k].append(v)
                    elif task_name == 'soft_jigsaw':
                        inputs, p_star, attn_mask = batch
                        if inputs is None or inputs.size(0) == 0:
                            continue
                        inputs, p_star, attn_mask = inputs.to(self.config.device), p_star.to(self.config.device), attn_mask.to(self.config.device)

                        task_cfg = task_configs.get('soft_jigsaw', {})
                        tau = task_cfg.get('tau', 0.1)

                        if self.config.use_amp:
                            with torch.amp.autocast('cuda'):
                                P_hat, loss = self.model(inputs, p_star=p_star, task_name=task_name, tau=tau)
                        else:
                            P_hat, loss = self.model(inputs, p_star=p_star, task_name=task_name, tau=tau)
                    elif task_name == 'distractor_loc':
                        x_prime, m_star, c_true, l_true = batch
                        if x_prime is None: continue
                        x_prime, m_star, c_true, l_true = x_prime.to(self.config.device), m_star.to(self.config.device), c_true.to(self.config.device), l_true.to(self.config.device)

                        if self.config.use_amp:
                            with torch.amp.autocast('cuda'):
                                predictions, loss = self.model(x_prime, task_name=task_name, m_star=m_star, c_true=c_true, l_true=l_true)
                        else:
                            predictions, loss = self.model(x_prime, task_name=task_name, m_star=m_star, c_true=c_true, l_true=l_true)

                        if loss is not None:
                            m_hat, (c_hat, l_hat) = predictions
                            intersection = (m_hat * m_star).sum(dim=1)
                            union = m_hat.sum(dim=1) + m_star.sum(dim=1) - intersection
                            iou = (intersection / (union + 1e-6)).mean().item()
                            c_mae = F.l1_loss(c_hat, c_true).item()
                            l_mae = F.l1_loss(l_hat, l_true).item()

                            metrics = {'mask_iou': iou, 'center_mae': c_mae, 'length_mae': l_mae, 'total_loss': loss.item()}
                            for k, v in metrics.items():
                                if k not in distractor_loc_metrics:
                                    distractor_loc_metrics[k] = []
                                distractor_loc_metrics[k].append(v)
                    else:
                        x, y = batch
                        x, y = x.to(self.config.device), y.to(self.config.device)

                        if self.config.use_amp:
                            with torch.amp.autocast('cuda'):
                                logits, loss = self.model(x, targets=y, task_name=task_name)
                        else:
                            logits, loss = self.model(x, targets=y, task_name=task_name)

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
        avg_distractor_loc_metrics = {k: np.mean(v) for k, v in distractor_loc_metrics.items()}

        return avg_loss, avg_cocktail_party_metrics, avg_distractor_loc_metrics

    
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
        for key, value in checkpoint['metrics'].items():
            setattr(self.metrics, key, value)
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['step']
    
    def train_epoch(
        self,
        train_loaders: Dict[str, DataLoader],
        val_loaders: Optional[Dict[str, DataLoader]] = None,
        epoch: int = 0,
        task_configs: Dict[str, Any] = None
    ):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        start_time = time.time()

        # Create iterators for all training dataloaders
        train_iters = {task: iter(loader) for task, loader in train_loaders.items()}
        
        for batch_idx in range(self.steps_per_epoch):
            step_start = time.time()
            
            total_loss = 0
            individual_losses = {}

            for task_name, task_iter in train_iters.items():
                try:
                    batch = next(task_iter)
                except StopIteration:
                    train_iters[task_name] = iter(train_loaders[task_name])
                    batch = next(train_iters[task_name])

                loss = self.train_step(batch, task_name, task_configs)

                if loss is None or (isinstance(loss, float) and loss == 0.0):
                    continue

                individual_losses[task_name] = loss.item()

                # Uncertainty-based weighting
                log_sigma = self.model.module.log_sigmas[task_name] if isinstance(self.model, DDP) else self.model.log_sigmas[task_name]
                weighted_loss = 0.5 * torch.exp(-2 * log_sigma) * loss + log_sigma

                total_loss += weighted_loss.squeeze()

            epoch_losses.append(total_loss.item())

            # Backpropagate combined loss
            self.optimizer.zero_grad()
            if self.config.use_amp and self.config.scaler is not None:
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

            # Get current LR *before* scheduler steps to log the correct value for the current step
            current_lr = self.optimizer.param_groups[0]['lr']

            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            step_time = time.time() - step_start
            self.metrics.update(
                train_loss=total_loss.item(),
                learning_rate=current_lr,
                step_time=step_time
            )
            

            # Logging (only on rank 0 if distributed)
            if (not self.is_distributed or dist.get_rank() == 0) and \
               self.metrics.total_steps % self.config.log_every == 0:
                avg_step_time = self.metrics.get_avg_step_time()
                loss_ma = self.metrics.get_loss_moving_average()
                loss_var = self.metrics.get_loss_variance()
                log_str = f"Epoch {epoch+1}, Step {self.metrics.total_steps}, Rank {dist.get_rank() if self.is_distributed else 0}, "
                log_str += f"Total Loss: {total_loss.item():.4f} (MA: {loss_ma:.4f}, Var: {loss_var:.4f}), "
                for task_name, loss_value in individual_losses.items():
                    log_sigma = self.model.module.log_sigmas[task_name].item() if isinstance(self.model, DDP) else self.model.log_sigmas[task_name].item()
                    sigma = math.exp(log_sigma)
                    log_str += f"{task_name}: {loss_value:.4f} (σ: {sigma:.4f}), "

                log_str += f"LR: {current_lr:.6f}, Step Time: {avg_step_time:.3f}s"
                print(log_str)

            # Evaluation, saving, and inference block (runs on rank 0)
            if (not self.is_distributed or dist.get_rank() == 0):
                # Evaluation
                if val_loaders is not None and self.metrics.total_steps % self.config.eval_every == 0:
                    val_loss, cocktail_party_metrics, distractor_loc_metrics = self.evaluate(val_loaders, task_configs)
                    self.metrics.update(val_loss=val_loss, cocktail_party_metrics=cocktail_party_metrics, distractor_loc_metrics=distractor_loc_metrics)

                    is_best = val_loss < self.metrics.best_val_loss
                    print(f"Validation Loss: {val_loss:.4f} {'(Best!)' if is_best else ''}")
                    if cocktail_party_metrics:
                        log_str = "Overall Cocktail Party Metrics: "
                        for k, v in cocktail_party_metrics.items():
                            log_str += f"{k}: {v:.4f}, "
                        print(log_str)

                    if distractor_loc_metrics:
                        log_str = "Overall Distractor Loc Metrics: "
                        for k, v in distractor_loc_metrics.items():
                            log_str += f"{k}: {v:.4f}, "
                        print(log_str)

                    if is_best:
                        self.save_checkpoint(self.metrics.total_steps, is_best=True)

                    # Plateau detection logic
                    improved = False
                    if self.config.plateau_mode == 'min':
                        if val_loss < self.plateau_best_metric_val - self.config.plateau_threshold:
                            improved = True
                    else: # max mode
                        if val_loss > self.plateau_best_metric_val + self.config.plateau_threshold:
                            improved = True

                    if improved:
                        self.plateau_best_metric_val = val_loss
                        self.plateau_patience_counter = 0
                        print(f"Metric improved to {val_loss:.4f}. Resetting plateau patience.")
                    else:
                        self.plateau_patience_counter += 1
                        print(f"Metric did not improve. Plateau patience: {self.plateau_patience_counter}/{self.config.plateau_patience}")

                    if self.plateau_patience_counter >= self.config.plateau_patience:
                        print("Plateau detected! Consider stopping training or adjusting learning rate manually.")
                        self.plateau_patience_counter = 0

                # Regular checkpoint saving
                if self.metrics.total_steps > 0 and self.metrics.total_steps % self.config.save_every == 0:
                    self.save_checkpoint(self.metrics.total_steps)

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
        task_configs: Dict[str, Any] = None
    ):
        """Main training loop."""
        print(f"Starting training for {self.config.num_epochs} epochs...")

        if self.is_distributed and dist.get_world_size() > 1:
            # Re-wrap dataloaders with DistributedSampler if in DDP mode
            for task, loader in train_loaders.items():
                sampler = DistributedSampler(loader.dataset, shuffle=True, num_replicas=dist.get_world_size(), rank=dist.get_rank())
                train_loaders[task] = DataLoader(
                    loader.dataset, batch_size=loader.batch_size, sampler=sampler,
                    num_workers=getattr(loader, 'num_workers', 0), pin_memory=getattr(loader, 'pin_memory', False),
                    collate_fn=loader.collate_fn
                )
            if val_loaders:
                for task, loader in val_loaders.items():
                    sampler = DistributedSampler(loader.dataset, shuffle=False, num_replicas=dist.get_world_size(), rank=dist.get_rank())
                    val_loaders[task] = DataLoader(
                        loader.dataset, batch_size=loader.batch_size, sampler=sampler,
                        num_workers=getattr(loader, 'num_workers', 0), pin_memory=getattr(loader, 'pin_memory', False),
                        collate_fn=loader.collate_fn
                    )

        # Calculate steps_per_epoch based on the largest dataloader
        self.steps_per_epoch = max(len(loader) for loader in train_loaders.values())
        if not self.is_distributed or dist.get_rank() == 0:
            print(f"Calculated steps_per_epoch: {self.steps_per_epoch}")

        total_steps = self.steps_per_epoch * self.config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=0
        )
        if not self.is_distributed or dist.get_rank() == 0:
            print(f"Scheduler initialized with T_max = {total_steps} for a single decay.")

        # Initial evaluation
        if val_loaders and (not self.is_distributed or dist.get_rank() == 0):
            initial_val_loss, initial_cocktail_metrics, initial_distractor_metrics = self.evaluate(val_loaders, task_configs)
            self.metrics.update(val_loss=initial_val_loss, cocktail_party_metrics=initial_cocktail_metrics, distractor_loc_metrics=initial_distractor_metrics)
            print(f"Initial validation loss: {initial_val_loss:.4f}")
            if initial_cocktail_metrics:
                log_str = "Initial cocktail party metrics: "
                for k, v in initial_cocktail_metrics.items():
                    log_str += f"{k}: {v:.4f}, "
                print(log_str)
            if initial_distractor_metrics:
                log_str = "Initial distractor loc metrics: "
                for k, v in initial_distractor_metrics.items():
                    log_str += f"{k}: {v:.4f}, "
                print(log_str)

        try:
            for epoch in range(self.config.num_epochs):
                if self.is_distributed and dist.get_world_size() > 1:
                    for loader in train_loaders.values():
                        loader.sampler.set_epoch(epoch)

                self.train_epoch(train_loaders, val_loaders, epoch, task_configs)
                
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
