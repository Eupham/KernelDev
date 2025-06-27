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
import contextlib

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
        inference_prompts: List[str] = None,
        inference_max_length: int = 100,
        inference_temperature: float = 0.8,
        inference_top_k: int = 50,
        inference_top_p: float = 0.9,
        plateau_monitor_metric: str = 'val_loss',
        plateau_patience: int = 10,
        plateau_threshold: float = 1e-4,
        plateau_mode: str = 'min',
        scheduler_T0_epoch_fraction: float = 0.1,
        scheduler_T_mult: int = 1,
        use_levenshtein_task: bool = False,
        levenshtein_loss_weight: float = 0.1,
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
        self.use_levenshtein_task = use_levenshtein_task
        self.levenshtein_loss_weight = levenshtein_loss_weight
        
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
    rank_env = os.environ.get('RANK')
    world_size_env = os.environ.get('WORLD_SIZE')
    local_rank_env = os.environ.get('LOCAL_RANK')
    if rank_env is not None and world_size_env is not None:
        try:
            rank = int(rank_env)
            world_size = int(world_size_env)
            local_rank = int(local_rank_env) if local_rank_env is not None else rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
            trainer_instance.config.local_rank = local_rank
            if torch.cuda.is_available():
                backend = 'nccl'
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
                trainer_instance.is_distributed = True
                trainer_instance.config.device = torch.device(f"cuda:{local_rank}")
            else:
                trainer_instance.is_distributed = False
        except Exception as e:
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
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.step_times = []
        self.total_steps = 0
        self.best_val_loss = float('inf')
        self.best_step = 0

        self.lm_losses = []
        self.lev_aux_losses = []
        self.pred_dist_orig_means = []

        self.val_lm_losses = []
        self.val_lev_aux_losses = []
        self.val_pred_dist_orig_means = []

    def update(self, train_loss=None, val_loss=None, learning_rate=None, step_time=None,
                 lm_loss_component=None, lev_aux_loss=None, pred_dist_orig_mean=None,
                 val_lm_loss_component=None, val_lev_aux_loss=None, val_pred_dist_orig_mean=None):
        if train_loss is not None: self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_step = self.total_steps
        if learning_rate is not None: self.learning_rates.append(learning_rate)
        if step_time is not None: self.step_times.append(step_time)
        if lm_loss_component is not None: self.lm_losses.append(lm_loss_component)
        if lev_aux_loss is not None: self.lev_aux_losses.append(lev_aux_loss)
        if pred_dist_orig_mean is not None: self.pred_dist_orig_means.append(pred_dist_orig_mean)

        if val_lm_loss_component is not None: self.val_lm_losses.append(val_lm_loss_component)
        if val_lev_aux_loss is not None: self.val_lev_aux_losses.append(val_lev_aux_loss)
        if val_pred_dist_orig_mean is not None: self.val_pred_dist_orig_means.append(val_pred_dist_orig_mean)
        self.total_steps += 1
    
    def get_avg_step_time(self, last_n=100):
        return np.mean(self.step_times[-last_n:]) if self.step_times else 0.0
    
    def save_metrics(self, filepath):
        metrics_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        torch.save(metrics_dict, filepath)

class Trainer:
    def __init__(self, model, config, data_builder=None):
        self.model = model
        self.config = config
        self.data_builder = data_builder
        self.metrics = TrainingMetrics()
        self.is_distributed = False
        self.plateau_patience_counter = 0
        self.plateau_best_metric_val = float('inf') if self.config.plateau_mode == 'min' else float('-inf')
        self.steps_per_epoch = None
        init_distributed(self)

        if self.is_distributed and dist.get_world_size() > 1:
            self.model.to(self.config.device)
            self.model = DDP(self.model, device_ids=[self.config.local_rank], output_device=self.config.local_rank, find_unused_parameters=True)
        else:
            self.model.to(self.config.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.initial_lr = self.config.learning_rate
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=self.config.warmup_steps, T_mult=self.config.scheduler_T_mult, eta_min=self.initial_lr * 0.1
        )
        print(f"Trainer initialized on device: {self.config.device}. Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[float, Optional[float], Optional[float], Optional[float]]:
        # Metrics initialization
        mean_lm_loss_component_item = None
        mean_lev_aux_loss_item = None
        mean_pred_dist_orig_item = None # For monitoring pred_dist on original items

        # 1. Batch Unpacking (LevenshteinDataset format)
        if self.config.use_levenshtein_task:
            # input_tokens, lm_targets, true_lev_distances, is_shuffled_flags
            # (B, T), (B, T), (B,), (B,)
            input_tokens, lm_targets, true_lev_distances, is_shuffled_flags = batch
            input_tokens = input_tokens.to(self.config.device)
            lm_targets = lm_targets.to(self.config.device)
            true_lev_distances = true_lev_distances.to(self.config.device)
            is_shuffled_flags = is_shuffled_flags.to(self.config.device)
        else: # Standard LM task
            input_tokens, lm_targets = batch
            input_tokens = input_tokens.to(self.config.device)
            lm_targets = lm_targets.to(self.config.device)
            # For non-Levenshtein tasks, these will remain None or 0.0
            true_lev_distances = None
            is_shuffled_flags = None

        self.optimizer.zero_grad()
        
        combined_loss = torch.tensor(0.0, device=self.config.device, dtype=torch.float32)
        final_batch_lm_loss_component = torch.tensor(0.0, device=self.config.device, dtype=torch.float32)

        autocast_context = torch.amp.autocast('cuda') if self.config.use_amp and self.config.scaler is not None else contextlib.suppress()

        with autocast_context:
            # 2. Single Main Forward Pass (Pass 1)
            # lm_targets for shuffled items are all ignore_idx, so per_item_lm_loss_all will be effectively 0 for them.
            lm_logits_all, per_item_lm_loss_all, predicted_lev_distances_all = self.model(
                input_tokens,
                lm_targets, # lm_targets are valid for original, all-ignore for shuffled
                force_disable_prefix_attention=False
            )

            # 3. Levenshtein Auxiliary Loss Calculation
            mean_lev_aux_loss_tensor = torch.tensor(0.0, device=self.config.device, dtype=torch.float32)
            if self.config.use_levenshtein_task and predicted_lev_distances_all is not None and true_lev_distances is not None:
                loss_fn_dist = torch.nn.MSELoss()
                # This MSE is calculated over all items.
                # For original items, true_lev_distances is 0.0.
                # For shuffled items, true_lev_distances is normalized Levenshtein distance.
                lev_aux_loss_per_item = loss_fn_dist(predicted_lev_distances_all.float(), true_lev_distances.float())
                mean_lev_aux_loss_tensor = lev_aux_loss_per_item # MSELoss with reduction='mean' is already a scalar
                mean_lev_aux_loss_item = mean_lev_aux_loss_tensor.item()

                # For monitoring: mean predicted distance on original items
                if is_shuffled_flags is not None:
                    original_item_mask_for_dist_pred = (is_shuffled_flags == 0.0)
                    if original_item_mask_for_dist_pred.any():
                        mean_pred_dist_orig_item = predicted_lev_distances_all[original_item_mask_for_dist_pred].mean().item()

            # 4. Isolate Original Items' Data for LM Loss (Simplified - No Self-Critique)
            if self.config.use_levenshtein_task and is_shuffled_flags is not None:
                original_item_mask = (is_shuffled_flags == 0.0)
                if original_item_mask.any():
                    # per_item_lm_loss_all is already correctly calculated by model for valid targets
                    # and should be effectively zero or ignorable for shuffled items due to masked lm_targets.
                    # We select only the losses corresponding to original items.
                    per_item_lm_loss_orig = per_item_lm_loss_all[original_item_mask]

                    # 5. Simplified LM Loss Calculation (No Self-Critique Forward Pass)
                    if per_item_lm_loss_orig is not None and per_item_lm_loss_orig.numel() > 0:
                        # Use the original LM loss directly without critique-based scaling
                        final_batch_lm_loss_component = per_item_lm_loss_orig.float().mean()
                    # else: final_batch_lm_loss_component remains 0 if no original items or no loss.
                # else (no original items in batch): final_batch_lm_loss_component remains 0.

            else: # Not using Levenshtein task (standard LM)
                if per_item_lm_loss_all is not None:
                     final_batch_lm_loss_component = per_item_lm_loss_all.float().mean()

            mean_lm_loss_component_item = final_batch_lm_loss_component.item() if final_batch_lm_loss_component.requires_grad else None

            # 7. Total Loss
            combined_loss = final_batch_lm_loss_component
            if self.config.use_levenshtein_task:
                combined_loss = combined_loss + (self.config.levenshtein_loss_weight * mean_lev_aux_loss_tensor)

        # --- End of autocast_context for AMP ---

        # Handle cases where loss might not require grad (e.g. all items were shuffled, no LM loss)
        if not combined_loss.requires_grad and combined_loss.abs().item() < 1e-9 : # If loss is effectively zero and has no grad
             if not self.config.use_levenshtein_task or (self.config.use_levenshtein_task and not mean_lev_aux_loss_tensor.requires_grad):
                # This can happen if batch had only shuffled items, or if aux loss also has no grad (e.g. model not changing outputs)
                # Return 0s or Nones to indicate no actual backpropagation happened for this step.
                return 0.0, None, mean_lev_aux_loss_item, mean_pred_dist_orig_item


        if self.config.use_amp and self.config.scaler is not None:
            self.config.scaler.scale(combined_loss).backward()
            if self.config.max_grad_norm > 0:
                self.config.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.config.scaler.step(self.optimizer)
            self.config.scaler.update()
        else:
            combined_loss.backward()
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        # 8. Return values for metrics
        return combined_loss_val, mean_lm_loss_component_item, mean_lev_aux_loss_item, mean_pred_dist_orig_item
    
    def evaluate(self, dataloader: DataLoader, max_batches: Optional[int] = 50) -> float:
        self.model.eval()
        total_combined_loss_epoch = 0
        accum_lm_loss_component = 0.0
        accum_lev_aux_loss = 0.0
        accum_pred_dist_orig_mean = 0.0
        num_batches_processed = 0
        num_lm_batches = 0
        num_lev_batches = 0
        num_pred_dist_orig_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    print(f"Evaluation limited to {max_batches} batches for speed")
                    break
                
                # Unpack batch according to LevenshteinDataset (4 items) or standard (2 items)
                if self.config.use_levenshtein_task:
                    input_tokens, lm_targets, true_lev_distances, is_shuffled_flags = batch
                    input_tokens = input_tokens.to(self.config.device)
                    lm_targets = lm_targets.to(self.config.device)
                    true_lev_distances = true_lev_distances.to(self.config.device)
                    is_shuffled_flags = is_shuffled_flags.to(self.config.device) # Used for metrics
                else: # Standard LM task
                    input_tokens, lm_targets = batch
                    input_tokens = input_tokens.to(self.config.device)
                    lm_targets = lm_targets.to(self.config.device)
                    true_lev_distances, is_shuffled_flags = None, None # Not applicable

                current_batch_lm_loss_tensor = torch.tensor(0.0, device=self.config.device, dtype=torch.float32)
                current_batch_aux_loss_tensor = torch.tensor(0.0, device=self.config.device, dtype=torch.float32)

                per_item_lm_loss, predicted_lev_distances = None, None # Model returns predicted_lev_distances for the input_tokens

                autocast_context_eval = torch.amp.autocast('cuda') if self.config.use_amp else contextlib.suppress()
                with autocast_context_eval:
                    # Model's forward pass; lm_targets are already masked for shuffled items by the dataset
                    _, per_item_lm_loss, predicted_lev_distances = self.model(
                        input_tokens,
                        lm_targets,
                        force_disable_prefix_attention=False
                    )

                if per_item_lm_loss is not None:
                    current_batch_lm_loss_tensor = per_item_lm_loss.mean().float() # per_item_lm_loss can be empty if batch has only fully masked items
                    if not torch.isnan(current_batch_lm_loss_tensor) and not torch.isinf(current_batch_lm_loss_tensor):
                         accum_lm_loss_component += current_batch_lm_loss_tensor.item()
                         num_lm_batches += 1


                if self.config.use_levenshtein_task and predicted_lev_distances is not None and true_lev_distances is not None:
                    loss_fn_dist = torch.nn.MSELoss()
                    # Compare model's predicted distances with the true distances from the batch
                    aux_loss_for_batch = loss_fn_dist(predicted_lev_distances.float(), true_lev_distances.float())
                    current_batch_aux_loss_tensor = aux_loss_for_batch
                    if not torch.isnan(current_batch_aux_loss_tensor) and not torch.isinf(current_batch_aux_loss_tensor):
                        accum_lev_aux_loss += current_batch_aux_loss_tensor.item()
                        num_lev_batches +=1

                    # For monitoring: mean predicted distance on original items
                    if is_shuffled_flags is not None:
                         original_item_mask_eval = (is_shuffled_flags == 0.0)
                         if original_item_mask_eval.any() and predicted_lev_distances[original_item_mask_eval].numel() > 0:
                             mean_pred_dist_orig_batch = predicted_lev_distances[original_item_mask_eval].mean().item()
                             if not math.isnan(mean_pred_dist_orig_batch) and not math.isinf(mean_pred_dist_orig_batch):
                                 accum_pred_dist_orig_mean += mean_pred_dist_orig_batch
                                 num_pred_dist_orig_batches +=1

                batch_total_loss = current_batch_lm_loss_tensor
                if self.config.use_levenshtein_task:
                    batch_total_loss = batch_total_loss + (self.config.levenshtein_loss_weight * current_batch_aux_loss_tensor)
                total_combined_loss_epoch += batch_total_loss.item()
                num_batches_processed +=1

        self.model.train()
        avg_combined_loss = total_combined_loss_epoch / num_batches_processed if num_batches_processed > 0 else float('inf')
        avg_lm_loss_component = accum_lm_loss_component / num_lm_batches if num_lm_batches > 0 else 0.0
        avg_lev_aux_loss = accum_lev_aux_loss / num_lev_batches if num_lev_batches > 0 else 0.0
        avg_pred_dist_orig_mean = accum_pred_dist_orig_mean / num_pred_dist_orig_batches if num_pred_dist_orig_batches > 0 else 0.0
        self.metrics.update(
            val_loss=avg_combined_loss,
            val_lm_loss_component=avg_lm_loss_component,
            val_lev_aux_loss=avg_lev_aux_loss,
            val_pred_dist_orig_mean=avg_pred_dist_orig_mean
        )
        return avg_combined_loss
    
    def save_checkpoint(self, step: int, is_best: bool = False):
        if not self.is_distributed or dist.get_rank() == 0:
            model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
            checkpoint = {
                'step': step,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'metrics': self.metrics.__dict__,
                'config': self.config.__dict__
            }
            checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_step_{step}.pt')
            torch.save(checkpoint, checkpoint_path)
            if is_best:
                best_path = os.path.join(self.config.checkpoint_dir, 'best_checkpoint.pt')
                torch.save(checkpoint, best_path)
            print(f"Checkpoint saved by Rank 0: {checkpoint_path}")
        else:
            if self.is_distributed: dist.barrier()
        
    def load_checkpoint(self, checkpoint_path: str):
        map_location = self.config.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint['model_state_dict']
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        current_keys_have_module = all(k.startswith('module.') for k in model_to_load.state_dict().keys())
        checkpoint_keys_have_module = all(k.startswith('module.') for k in state_dict.keys())
        if isinstance(self.model, DDP) and not checkpoint_keys_have_module: pass
        elif not isinstance(self.model, DDP) and checkpoint_keys_have_module:
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        model_to_load.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        for key, value in checkpoint['metrics'].items():
            if hasattr(self.metrics, key): setattr(self.metrics, key, value)
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint.get('step', 0)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epoch: int = 0
    ):
        self.model.train()
        epoch_losses = []
        start_time = time.time()
        if self.is_distributed and hasattr(train_loader.sampler, 'set_epoch') and dist.get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()
            combined_loss_item, current_lm_loss_item, current_lev_loss_item, \
                current_pred_dist_orig_item = self.train_step(batch)
            epoch_losses.append(combined_loss_item)
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            self.metrics.update(
                train_loss=combined_loss_item,
                learning_rate=current_lr,
                step_time=time.time() - step_start,
                lm_loss_component=current_lm_loss_item,
                lev_aux_loss=current_lev_loss_item,
                pred_dist_orig_mean=current_pred_dist_orig_item
            )
            
            if (not self.is_distributed or dist.get_rank() == 0) and \
               self.metrics.total_steps % self.config.log_every == 0:
                avg_step_time = self.metrics.get_avg_step_time()
                log_msg = f"Epoch {epoch+1}, Step {self.metrics.total_steps}, Rank {dist.get_rank() if self.is_distributed else 0}, Loss: {combined_loss_item:.4f}"
                if self.config.use_levenshtein_task:
                    if current_lm_loss_item is not None: log_msg += f", LM Comp: {current_lm_loss_item:.4f}"
                    if current_lev_loss_item is not None: log_msg += f", Lev Aux (shuf): {current_lev_loss_item:.4f}"
                    if current_pred_dist_orig_item is not None: log_msg += f", Pred Dist (orig): {current_pred_dist_orig_item:.4f}"
                log_msg += f", LR: {current_lr:.6f}, Step Time: {avg_step_time:.3f}s"
                print(log_msg)
            
            if (not self.is_distributed or dist.get_rank() == 0) and \
               val_loader is not None and \
               self.metrics.total_steps % self.config.eval_every == 0:
                val_loss = self.evaluate(val_loader)
                is_best = val_loss < self.metrics.best_val_loss
                print(f"Validation Loss (Rank {dist.get_rank() if self.is_distributed else 0}): {val_loss:.4f} {'(Best!)' if is_best else ''}")
                if self.config.use_levenshtein_task:
                    if self.metrics.val_lm_losses: print(f"  Val LM Comp: {self.metrics.val_lm_losses[-1]:.4f}")
                    if self.metrics.val_lev_aux_losses: print(f"  Val Lev Aux (shuf): {self.metrics.val_lev_aux_losses[-1]:.4f}")
                    if self.metrics.val_pred_dist_orig_means: print(f"  Val Pred Dist (orig): {self.metrics.val_pred_dist_orig_means[-1]:.4f}")
                if is_best: self.save_checkpoint(self.metrics.total_steps, is_best=True)
                current_metric_val = val_loss
                improved = False
                if self.config.plateau_mode == 'min':
                    if current_metric_val < self.plateau_best_metric_val - self.config.plateau_threshold: improved = True
                elif self.config.plateau_mode == 'max':
                    if current_metric_val > self.plateau_best_metric_val + self.config.plateau_threshold: improved = True
                if improved:
                    self.plateau_best_metric_val = current_metric_val
                    self.plateau_patience_counter = 0
                else:
                    self.plateau_patience_counter += 1
                if self.plateau_patience_counter >= self.config.plateau_patience:
                    print(f"Plateau detected! Metric did not improve for {self.config.plateau_patience} evaluations.")
                    self.plateau_patience_counter = 0
                    if self.steps_per_epoch is None:
                        print("Error: self.steps_per_epoch is not set.")
                    else:
                        new_T0 = math.ceil(self.steps_per_epoch * self.config.scheduler_T0_epoch_fraction); new_T0 = max(1,new_T0)
                        for param_group in self.optimizer.param_groups: param_group['lr'] = self.initial_lr
                        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                            self.optimizer, T_0=new_T0, T_mult=self.config.scheduler_T_mult, eta_min=self.config.learning_rate * 0.1
                        )
                        print(f"Scheduler re-initialized. New T_0 = {new_T0}. LR reset to {self.initial_lr:.6f}")
            
            if (not self.is_distributed or dist.get_rank() == 0) and \
               self.metrics.total_steps > 0 and \
               self.metrics.total_steps % self.config.save_every == 0:
                self.save_checkpoint(self.metrics.total_steps)
                if val_loader is not None:
                    perplexity = self.calculate_perplexity(val_loader, max_batches=20)
                    current_val_loss_for_sample = self.metrics.val_losses[-1] if self.metrics.val_losses else float('inf')
                    generated_outputs = self.generate_inference_sample(
                        prompts=self.config.inference_prompts,
                        max_length=self.config.inference_max_length,
                        temperature=self.config.inference_temperature,
                        top_k=self.config.inference_top_k,
                        top_p=self.config.inference_top_p
                    )
                    self.save_inference_sample(
                        step=self.metrics.total_steps,
                        val_loss=current_val_loss_for_sample,
                        perplexity=perplexity,
                        generated_outputs=generated_outputs
                    )
        
        if not self.is_distributed or dist.get_rank() == 0:
            avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1} completed (Rank {dist.get_rank() if self.is_distributed else 0}): Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        return avg_loss
    
    def train(self, train_loader, val_loader=None):
        print(f"Starting training for {self.config.num_epochs} epochs...")
        train_sampler, val_sampler = None, None
        if self.is_distributed and dist.get_world_size() > 1:
            train_sampler = DistributedSampler(train_loader.dataset, shuffle=True, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            train_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, sampler=train_sampler, num_workers=getattr(train_loader, 'num_workers', 0), pin_memory=getattr(train_loader, 'pin_memory', False))
            if val_loader:
                val_sampler = DistributedSampler(val_loader.dataset, shuffle=False, num_replicas=dist.get_world_size(), rank=dist.get_rank())
                val_loader = DataLoader(val_loader.dataset, batch_size=getattr(val_loader, 'batch_size', 1), sampler=val_sampler, num_workers=getattr(val_loader, 'num_workers', 0), pin_memory=getattr(val_loader, 'pin_memory', False))

        if not self.is_distributed or dist.get_rank() == 0:
            print(f"Training batches per epoch (Rank 0 view): {len(train_loader)}")
            if val_loader: print(f"Validation batches (Rank 0 view): {len(val_loader)}")

        self.steps_per_epoch = len(train_loader)
        if not self.is_distributed or dist.get_rank() == 0: print(f"Calculated steps_per_epoch: {self.steps_per_epoch}")
        new_T0 = math.ceil(self.steps_per_epoch * self.config.scheduler_T0_epoch_fraction); new_T0 = max(1,new_T0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=new_T0, T_mult=self.config.scheduler_T_mult, eta_min=self.config.learning_rate * 0.1)
        if not self.is_distributed or dist.get_rank() == 0: print(f"Scheduler re-initialized with T_0 = {new_T0} based on steps_per_epoch.")
        
        if val_loader and (not self.is_distributed or dist.get_rank() == 0):
            initial_val_loss = self.evaluate(val_loader)
            self.metrics.update(val_loss=initial_val_loss)
            print(f"Initial validation loss (Rank 0): {initial_val_loss:.4f}")

        try:
            for epoch in range(self.config.num_epochs):
                if self.is_distributed and train_sampler is not None and dist.get_world_size() > 1: train_sampler.set_epoch(epoch)
                self.train_epoch(train_loader, val_loader, epoch)
                self.save_checkpoint(self.metrics.total_steps)
        except KeyboardInterrupt: print("\nTraining interrupted by user.")
        except Exception as e: print(f"\nTraining failed with error: {e}"); raise
        finally:
            if not self.is_distributed or dist.get_rank() == 0:
                metrics_path = os.path.join(self.config.checkpoint_dir, 'training_metrics.pt')
                self.metrics.save_metrics(metrics_path)
                print(f"Training metrics saved (Rank 0): {metrics_path}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        import contextlib
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        ax_flat = axes.flatten()
        plot_idx = 0

        if self.metrics.train_losses:
            ax_flat[plot_idx].plot(self.metrics.train_losses, 'b-', alpha=0.7, label='Train Loss (Combined)')
            ax_flat[plot_idx].set_title('Training Loss (Combined)')
            ax_flat[plot_idx].set_xlabel('Step'); ax_flat[plot_idx].set_ylabel('Loss'); ax_flat[plot_idx].grid(True, alpha=0.3); ax_flat[plot_idx].legend()
        plot_idx += 1

        if self.metrics.val_losses:
            val_loss_steps = np.arange(len(self.metrics.val_losses)) * self.config.eval_every
            ax_flat[plot_idx].plot(val_loss_steps, self.metrics.val_losses, 'r-', alpha=0.7, label='Val Loss (Combined)')
            ax_flat[plot_idx].set_title('Validation Loss (Combined)')
            ax_flat[plot_idx].set_xlabel('Step'); ax_flat[plot_idx].set_ylabel('Loss'); ax_flat[plot_idx].grid(True, alpha=0.3); ax_flat[plot_idx].legend()
        plot_idx += 1

        if self.metrics.lm_losses:
            ax_flat[plot_idx].plot(self.metrics.lm_losses, 'g-', alpha=0.5, label='LM Loss Comp (Train)')
            ax_flat[plot_idx].set_title('LM Loss Component (Training)')
            ax_flat[plot_idx].set_xlabel('Step'); ax_flat[plot_idx].set_ylabel('Loss'); ax_flat[plot_idx].grid(True, alpha=0.3); ax_flat[plot_idx].legend()
        plot_idx += 1

        if self.config.use_levenshtein_task:
            if self.metrics.lev_aux_losses:
                 ax_flat[plot_idx].plot(self.metrics.lev_aux_losses, 'c-', alpha=0.7, label='Lev Aux Loss (Shuf, Train)')
            ax_flat[plot_idx].set_title('Aux Losses (Train)')
            ax_flat[plot_idx].set_xlabel('Step'); ax_flat[plot_idx].set_ylabel('Loss'); ax_flat[plot_idx].grid(True, alpha=0.3); ax_flat[plot_idx].legend()
        plot_idx += 1

        if self.config.use_levenshtein_task and \
           (self.metrics.val_lm_losses or self.metrics.val_lev_aux_losses or self.metrics.val_pred_dist_orig_means):
            val_steps = np.arange(len(self.metrics.val_losses)) * self.config.eval_every
            ax_val_main = ax_flat[plot_idx]
            if self.metrics.val_lm_losses:
                 ax_val_main.plot(val_steps[:len(self.metrics.val_lm_losses)], self.metrics.val_lm_losses, 'r--', alpha=0.7, label='LM Loss Comp (Val)')
            if self.metrics.val_lev_aux_losses:
                ax_val_main.plot(val_steps[:len(self.metrics.val_lev_aux_losses)], self.metrics.val_lev_aux_losses, 'm--', alpha=0.7, label='Lev Aux Loss (Shuf, Val)')
            ax_val_main.set_title('Validation Loss Components')
            ax_val_main.set_xlabel('Step'); ax_val_main.set_ylabel('Loss'); ax_val_main.grid(True, alpha=0.3)

            if self.metrics.val_pred_dist_orig_means:
                ax_val_sec = ax_val_main.twinx()
                ax_val_sec.plot(val_steps[:len(self.metrics.val_pred_dist_orig_means)], self.metrics.val_pred_dist_orig_means, 'k:', alpha=0.6, label='Mean Pred Dist (Orig, Val)')
                ax_val_sec.set_ylabel('Mean Pred Dist (Val)', color='k')
                ax_val_sec.tick_params(axis='y', labelcolor='k')
                lines, labels = ax_val_main.get_legend_handles_labels()
                lines2, labels2 = ax_val_sec.get_legend_handles_labels()
                ax_val_main.legend(lines + lines2, labels + labels2, loc='best')
            else:
                ax_val_main.legend(loc='best')
        plot_idx += 1
        
        if self.metrics.learning_rates:
            ax_flat[plot_idx].plot(self.metrics.learning_rates, 'darkorange', alpha=0.7, label='Learning Rate')
            ax_flat[plot_idx].set_title('Learning Rate')
            ax_flat[plot_idx].set_xlabel('Step'); ax_flat[plot_idx].set_ylabel('LR'); ax_flat[plot_idx].grid(True, alpha=0.3); ax_flat[plot_idx].legend()
        plot_idx += 1
        
        for i in range(plot_idx, len(ax_flat)): fig.delaxes(ax_flat[i])
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_sample(self, prompt="", max_length=100, temperature=0.8, top_k=50, top_p=None):
        if not self.data_builder: return ""
        self.model.eval()
        model_to_generate_from = self.model.module if isinstance(self.model, DDP) else self.model
        with torch.no_grad():
            tokens = self.data_builder._tokenize_text(prompt) if prompt else []
            x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.config.device) if tokens else \
                torch.randint(0, self.data_builder.vocab_size, (1, 1)).to(self.config.device)
            autocast_context = torch.amp.autocast('cuda') if self.config.use_amp else contextlib.suppress()
            with autocast_context:
                generated = model_to_generate_from.generate(x, max_new_tokens=max_length, temperature=temperature, top_k=top_k, top_p=top_p)
            generated_text = self.data_builder.decode_tokens(generated[0])
        self.model.train()
        return generated_text
    
    def calculate_perplexity(self, dataloader, max_batches=None):
        self.model.eval()
        total_loss, num_batches = 0, 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches: break

                # Updated batch unpacking for perplexity
                if self.config.use_levenshtein_task :
                    input_ids, lm_targets, _, _ = batch # Original text and its LM targets
                elif hasattr(self.config, 'nsp_task') and self.config.nsp_task: # old check, for safety
                    input_ids, lm_targets, _ = batch
                else:
                    input_ids, lm_targets = batch
                input_ids, lm_targets = input_ids.to(self.config.device), lm_targets.to(self.config.device)
                
                lm_loss = None
                autocast_context = torch.amp.autocast('cuda') if self.config.use_amp and self.config.scaler is not None else contextlib.suppress()
                with autocast_context:
                    # Model's forward for perplexity should focus on LM loss from original_tokens_cls
                    # The third output (aux_score) is not used for perplexity.
                    _, lm_loss, _ = self.model(input_ids, lm_targets, force_disable_prefix_attention=True) # Force disable prefix for pure LM perplexity
                if lm_loss is not None:
                    total_loss += lm_loss.mean().item()
                    num_batches += 1
        self.model.train()
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return math.exp(avg_loss) if avg_loss != float('inf') else float('inf')
    
    def generate_inference_sample(self, prompts=None, max_length=100, temperature=0.8, top_k=50, top_p=0.9):
        if not self.data_builder: return [{"type": "error", "prompt": "N/A", "text": "No data_builder provided."}]
        standard_prompts = prompts or ["", "The", "In", "Once upon a time"]
        cls_test_prompts_text = ["What is the capital of France?", "Tell me a short story about a robot."]
        self.model.eval()
        model_to_generate_from = self.model.module if isinstance(self.model, DDP) else self.model
        all_generated_outputs = []
        autocast_context = torch.amp.autocast('cuda') if self.config.use_amp else contextlib.suppress()
        with torch.no_grad():
            for prompt_text in standard_prompts:
                try:
                    tokens = self.data_builder._tokenize_text(prompt_text) if prompt_text else []
                    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.config.device) if tokens else \
                        torch.randint(0, self.data_builder.vocab_size, (1,1)).to(self.config.device)
                    with autocast_context:
                        generated_ids = model_to_generate_from.generate(x, max_new_tokens=max_length, temperature=temperature, top_k=top_k, top_p=top_p, use_prefix_attention_in_prompt=False)
                    decoded_text = self.data_builder.decode_tokens(generated_ids[0])
                    all_generated_outputs.append({'type': 'standard', 'prompt': prompt_text, 'text': decoded_text})
                except Exception as e:
                    all_generated_outputs.append({'type': 'error', 'prompt': prompt_text, 'text': f"Generation failed: {str(e)}"})

            model_is_lev_task_and_configured_for_cls_prefix = \
                self.config.use_levenshtein_task and \
                hasattr(model_to_generate_from, 'use_cls_prefix_attention') and \
                model_to_generate_from.use_cls_prefix_attention and \
                hasattr(self.data_builder, 'cls_token_id') and \
                self.data_builder.cls_token_id is not None

            if model_is_lev_task_and_configured_for_cls_prefix: # Check if model is configured for this
                print(f"  Running CLS prefix attention inference tests (cls_id: {self.data_builder.cls_token_id})...")
                for prompt_text in cls_test_prompts_text:
                    try:
                        text_tokens = self.data_builder._tokenize_text(prompt_text)
                        input_tokens_with_cls = [self.data_builder.cls_token_id] + text_tokens
                        x_cls = torch.tensor(input_tokens_with_cls, dtype=torch.long).unsqueeze(0).to(self.config.device)
                        with autocast_context:
                            generated_on_ids = model_to_generate_from.generate(x_cls.clone(), max_new_tokens=max_length, temperature=temperature,top_k=top_k, top_p=top_p, use_prefix_attention_in_prompt=True)
                            generated_off_ids = model_to_generate_from.generate(x_cls.clone(), max_new_tokens=max_length, temperature=temperature,top_k=top_k, top_p=top_p, use_prefix_attention_in_prompt=False)
                        decoded_text_on = self.data_builder.decode_tokens(generated_on_ids[0])
                        all_generated_outputs.append({'type': 'CLS_Prefix_ON', 'prompt': f"[CLS] {prompt_text}", 'text': decoded_text_on})
                        decoded_text_off = self.data_builder.decode_tokens(generated_off_ids[0])
                        all_generated_outputs.append({'type': 'CLS_Prefix_OFF', 'prompt': f"[CLS] {prompt_text}", 'text': decoded_text_off})
                    except Exception as e:
                        all_generated_outputs.append({'type': 'error', 'prompt': f"[CLS] {prompt_text}", 'text': f"CLS Generation failed: {str(e)}"})
            else:
                 print(f"  Skipping CLS prefix attention inference tests based on model/data_builder configuration for Levenshtein task.")
        self.model.train()
        return all_generated_outputs
    
    def save_inference_sample(self, step, val_loss, perplexity, generated_outputs, prompts=None ):
        inference_dir = Path(self.config.checkpoint_dir) / "inference_samples"; inference_dir.mkdir(exist_ok=True)
        sample_entry = {"step": step, "validation_loss": val_loss, "perplexity": perplexity, "timestamp": time.time(), "samples": generated_outputs}
        samples_file = inference_dir / "inference_samples.json"
        all_samples = []
        if samples_file.exists():
            try: all_samples = json.load(open(samples_file, 'r'))
            except: pass
        all_samples.append(sample_entry)
        with open(samples_file, 'w') as f: json.dump(all_samples, f, indent=2)
        print(f"\n=== Inference Sample at Step {step} ==="); print(f"Validation Loss: {val_loss:.4f}"); print(f"Perplexity: {perplexity:.2f}")
        for item in generated_outputs: print(f"Type: {item.get('type', 'N/A')}, Prompt: '{item.get('prompt', 'N/A')}' → '{item.get('text', 'Error')}'")
        print("=" * 50)

def create_trainer(model, config, data_builder=None): return Trainer(model, config, data_builder)
if __name__ == "__main__": print("Trainer module loaded successfully!")
