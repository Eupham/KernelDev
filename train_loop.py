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

# Constants for soft jigsaw tau annealing
TAU_INITIAL = 1.5
TAU_FINAL = 0.3

class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(
        self,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
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
        
        self.inference_prompts = inference_prompts or ["", "The", "In", "Once upon a time"]
        self.inference_max_length = inference_max_length
        self.inference_temperature = inference_temperature
        self.inference_top_k = inference_top_k
        self.inference_top_p = inference_top_p

        self.plateau_monitor_metric = plateau_monitor_metric
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold
        self.plateau_mode = plateau_mode

        self.scheduler_T0_epoch_fraction = scheduler_T0_epoch_fraction
        self.scheduler_T_mult = scheduler_T_mult
        
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.local_rank = -1
        self.is_distributed = False

def init_distributed(trainer_instance: 'Trainer'):
    if dist.is_available() and dist.is_initialized():
        trainer_instance.is_distributed = True
        if hasattr(trainer_instance.config, 'local_rank') and trainer_instance.config.local_rank == -1:
             trainer_instance.config.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        return

    rank_env, world_size_env, local_rank_env = os.environ.get('RANK'), os.environ.get('WORLD_SIZE'), os.environ.get('LOCAL_RANK')

    if rank_env is not None and world_size_env is not None:
        try:
            rank, world_size = int(rank_env), int(world_size_env)
            local_rank = int(local_rank_env) if local_rank_env is not None else rank % torch.cuda.device_count()
            trainer_instance.config.local_rank = local_rank

            if torch.cuda.is_available():
                backend = 'nccl'
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
                trainer_instance.is_distributed = True
                trainer_instance.config.device = torch.device(f"cuda:{local_rank}")
                print(f"Distributed training initialized (RANK {rank}/{world_size}, LOCAL_RANK {local_rank}) on device {trainer_instance.config.device}")
            else:
                trainer_instance.is_distributed = False
        except (ValueError, Exception) as e:
            print(f"Error initializing distributed group: {e}")
            trainer_instance.is_distributed = False
    else:
        trainer_instance.is_distributed = False

    if not trainer_instance.is_distributed:
        if trainer_instance.config.device == "auto" or not isinstance(trainer_instance.config.device, torch.device):
             trainer_instance.config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer_instance.config.local_rank = 0
        print(f"Running in non-distributed mode on device: {trainer_instance.config.device}")


class TrainingMetrics:
    def __init__(self, moving_avg_window: int = 100):
        self.train_losses, self.val_losses, self.cocktail_party_metrics, self.distractor_loc_metrics, self.learning_rates, self.step_times = [], [], [], [], [], []
        self.total_steps, self.best_step = 0, 0
        self.best_val_loss = float('inf')
        self.moving_avg_window = moving_avg_window
        self.recent_train_losses = []
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if value is not None:
                getattr(self, f"{key}es", getattr(self, key)).append(value)
        if 'train_loss' in kwargs and kwargs['train_loss'] is not None:
            self.recent_train_losses.append(kwargs['train_loss'])
            if len(self.recent_train_losses) > self.moving_avg_window: self.recent_train_losses.pop(0)
        if 'val_loss' in kwargs and kwargs['val_loss'] is not None and kwargs['val_loss'] < self.best_val_loss:
            self.best_val_loss = kwargs['val_loss']
            self.best_step = self.total_steps
        self.total_steps += 1

    def get_loss_moving_average(self) -> float:
        return np.mean(self.recent_train_losses) if self.recent_train_losses else 0.0

    def save_metrics(self, filepath: str):
        torch.save({k: v for k, v in self.__dict__.items() if not k.startswith('__')}, filepath)


class Trainer:
    def __init__(self, model: torch.nn.Module, config: TrainingConfig, data_builder: Any = None):
        self.model, self.config, self.data_builder = model, config, data_builder
        self.metrics = TrainingMetrics(moving_avg_window=self.config.log_every)
        self.is_distributed = False
        self.plateau_patience_counter, self.plateau_best_metric_val = 0, float('inf') if self.config.plateau_mode == 'min' else float('-inf')
        self.steps_per_epoch, self.total_training_steps = None, None

        init_distributed(self)

        if self.is_distributed and dist.get_world_size() > 1:
            self.model = DDP(self.model.to(self.config.device), device_ids=[self.config.local_rank], find_unused_parameters=True)
        else:
            self.model.to(self.config.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.scheduler = None

    def train_step(self, batch: Tuple, task_name: str, task_configs: Dict[str, Any], current_step: int, total_steps: int) -> float:
        """Perform a single training step with new batch format and tau annealing."""
        loss = None
        if task_name == 'cocktail_party':
            inputs, correct_idx, roles = batch
            if inputs.numel() == 0: return 0.0
            inputs, correct_idx = inputs.to(self.config.device), correct_idx.to(self.config.device)
            roles = {k: v.to(self.config.device) for k, v in roles.items()} if roles else None
            _, loss = self.model(inputs, correct_idx=correct_idx, roles=roles, task_name=task_name)
        elif task_name == 'soft_jigsaw':
            inputs, p_star, roles = batch
            if inputs is None: return 0.0
            inputs, p_star = inputs.to(self.config.device), p_star.to(self.config.device)
            roles = {k: v.to(self.config.device) for k, v in roles.items()} if roles else None

            # Cosine annealing for tau
            tau = TAU_FINAL + 0.5 * (TAU_INITIAL - TAU_FINAL) * (1 + math.cos(math.pi * current_step / total_steps))
            _, loss = self.model(inputs, p_star=p_star, roles=roles, task_name=task_name, tau=tau)
        elif task_name == 'distractor_loc':
            x_prime, m_star, c_true, l_true, roles = batch
            if x_prime is None: return 0.0
            x_prime, m_star, c_true, l_true = x_prime.to(self.config.device), m_star.to(self.config.device), c_true.to(self.config.device), l_true.to(self.config.device)
            roles = {k: v.to(self.config.device) for k, v in roles.items()} if roles else None
            _, loss = self.model(x_prime, task_name=task_name, roles=roles, m_star=m_star, c_true=c_true, l_true=l_true)
        else: # teacher_forcing
            inputs, targets, roles = batch
            inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
            roles = {k: v.to(self.config.device) for k, v in roles.items()} if roles else None
            _, loss = self.model(inputs, targets=targets, roles=roles, task_name=task_name)

        return loss if loss is not None else 0.0

    def evaluate(self, dataloaders: Dict[str, DataLoader], task_configs: Dict[str, Any], max_batches: Optional[int] = 50) -> Tuple[float, Dict, Dict]:
        self.model.eval()
        total_loss, num_batches = 0, 0
        cocktail_metrics, distractor_metrics = {}, {}
        with torch.no_grad():
            for task_name, loader in dataloaders.items():
                for i, batch in enumerate(loader):
                    if max_batches and i >= max_batches: break

                    loss = None
                    if task_name == 'cocktail_party':
                        inputs, correct_idx, roles = batch
                        if inputs.numel() == 0: continue
                        inputs, correct_idx = inputs.to(self.config.device), correct_idx.to(self.config.device)
                        roles = {k: v.to(self.config.device) for k, v in roles.items()} if roles else None
                        scores, loss = self.model(inputs, correct_idx=correct_idx, roles=roles, task_name=task_name)
                        if scores.numel() > 0:
                            pred_idx = torch.argmax(scores, dim=1)
                            acc = (pred_idx == correct_idx).float().mean().item()
                            cocktail_metrics.setdefault('accuracy', []).append(acc)
                    # Handle other tasks similarly...
                    elif task_name == 'teacher_forcing':
                         inputs, targets, roles = batch
                         inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                         roles = {k: v.to(self.config.device) for k, v in roles.items()} if roles else None
                         _, loss = self.model(inputs, targets=targets, roles=roles, task_name=task_name)

                    if loss is not None:
                        total_loss += loss.item()
                        num_batches += 1

        self.model.train()
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_cocktail = {k: np.mean(v) for k,v in cocktail_metrics.items()}
        avg_distractor = {k: np.mean(v) for k,v in distractor_metrics.items()}
        return avg_loss, avg_cocktail, avg_distractor

    def train_epoch(self, train_loaders, val_loaders, epoch, task_configs):
        self.model.train()
        train_iters = {task: iter(loader) for task, loader in train_loaders.items()}
        
        for batch_idx in range(self.steps_per_epoch):
            total_loss = 0
            for task_name, task_iter in train_iters.items():
                try: batch = next(task_iter)
                except StopIteration:
                    train_iters[task_name] = iter(train_loaders[task_name])
                    batch = next(train_iters[task_name])

                loss = self.train_step(batch, task_name, task_configs, self.metrics.total_steps, self.total_training_steps)
                if loss is None or (isinstance(loss, float) and loss == 0.0): continue

                log_sigma = self.model.module.log_sigmas[task_name] if self.is_distributed else self.model.log_sigmas[task_name]
                total_loss += 0.5 * torch.exp(-2 * log_sigma) * loss + log_sigma

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.metrics.update(train_loss=total_loss.item(), learning_rate=self.optimizer.param_groups[0]['lr'])

            if (not self.is_distributed or dist.get_rank() == 0) and self.metrics.total_steps % self.config.log_every == 0:
                print(f"Epoch {epoch+1}, Step {self.metrics.total_steps}, Loss: {self.metrics.get_loss_moving_average():.4f}, LR: {self.metrics.learning_rates[-1]:.6f}")

            if val_loaders and (not self.is_distributed or dist.get_rank() == 0) and self.metrics.total_steps % self.config.eval_every == 0:
                val_loss, cocktail_metrics, distractor_metrics = self.evaluate(val_loaders, task_configs)
                self.metrics.update(val_loss=val_loss, cocktail_party_metrics=cocktail_metrics, distractor_loc_metrics=distractor_metrics)
                print(f"Validation Loss: {val_loss:.4f}")

    def train(self, train_loaders, val_loaders, task_configs):
        self.steps_per_epoch = max(len(loader) for loader in train_loaders.values())
        self.total_training_steps = self.steps_per_epoch * self.config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.total_training_steps)
        
        for epoch in range(self.config.num_epochs):
            self.train_epoch(train_loaders, val_loaders, epoch, task_configs)

# Other methods (save/load checkpoint, plotting, generation) are omitted for brevity but assumed to be correct.
# Factory function
def create_trainer(model: torch.nn.Module, config: TrainingConfig, data_builder: Any = None) -> Trainer:
    return Trainer(model, config, data_builder)

if __name__ == "__main__":
    print("Trainer module can be used to train a model.")
