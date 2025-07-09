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
    # Check if already initialized (e.g., by DeepSpeed or another launcher)
    if dist.is_available() and dist.is_initialized():
        trainer_instance.is_distributed = True
        # Attempt to get local_rank if not already set in config, common for some launchers
        if not hasattr(trainer_instance.config, 'local_rank') or trainer_instance.config.local_rank == -1:
            trainer_instance.config.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        # Assume device is already set correctly by the external launcher or will be handled
        # For example, if DDP is used, device is often set based on local_rank
        if torch.cuda.is_available():
            if hasattr(trainer_instance.config, 'local_rank') and trainer_instance.config.local_rank != -1:
                 trainer_instance.config.device = torch.device(f"cuda:{trainer_instance.config.local_rank}")
            else: # Fallback if local_rank couldn't be determined but dist is initialized
                 trainer_instance.config.device = torch.device('cuda')
        else:
            trainer_instance.config.device = torch.device('cpu')
        print(f"Distributed training already initialized. Using rank: {dist.get_rank()}, world_size: {dist.get_world_size()}, device: {trainer_instance.config.device}")
        return

    # Standard environment variable check for torch.distributed.launch or similar
    rank_env = os.environ.get('RANK')
    world_size_env = os.environ.get('WORLD_SIZE')
    local_rank_env = os.environ.get('LOCAL_RANK')

    if rank_env is not None and world_size_env is not None:
        try:
            rank = int(rank_env)
            world_size = int(world_size_env)

            if world_size > 1 and torch.cuda.is_available(): # Only init if world_size > 1 and CUDA is present
                local_rank = int(local_rank_env) if local_rank_env is not None else rank % torch.cuda.device_count()
                trainer_instance.config.local_rank = local_rank

                backend = 'nccl'
                torch.cuda.set_device(local_rank)
                # MASTER_ADDR and MASTER_PORT must be set in the environment for this to succeed
                dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
                trainer_instance.is_distributed = True
                trainer_instance.config.device = torch.device(f"cuda:{local_rank}")
                print(f"Distributed training initialized by train_loop. Rank: {rank}, World Size: {world_size}, Device: {trainer_instance.config.device}")
            elif world_size > 1 and not torch.cuda.is_available():
                print("Warning: Distributed training requested (world_size > 1) but CUDA is not available. Falling back to non-distributed CPU mode.")
                trainer_instance.is_distributed = False
            else: # world_size is 1 or less
                trainer_instance.is_distributed = False
        except Exception as e:
            print(f"Error initializing distributed group: {e}. Falling back to non-distributed mode.")
            trainer_instance.is_distributed = False
    else:
        trainer_instance.is_distributed = False

    if not trainer_instance.is_distributed:
        # Set device for non-distributed mode
        if torch.cuda.is_available():
            trainer_instance.config.device = torch.device('cuda')
            trainer_instance.config.local_rank = 0 # Default local_rank for single GPU
        else:
            trainer_instance.config.device = torch.device('cpu')
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
        self.nsp_losses = []
        self.pred_dist_orig_means = []

        self.val_lm_losses = []
        self.val_lev_aux_losses = []
        self.val_nsp_losses = []
        self.val_pred_dist_orig_means = []

    def update(self, train_loss=None, val_loss=None, learning_rate=None, step_time=None,
                 lm_loss_component=None, lev_aux_loss=None, nsp_loss=None, pred_dist_orig_mean=None,
                 val_lm_loss_component=None, val_lev_aux_loss=None, val_nsp_loss=None, val_pred_dist_orig_mean=None):
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
        if nsp_loss is not None: self.nsp_losses.append(nsp_loss)
        if pred_dist_orig_mean is not None: self.pred_dist_orig_means.append(pred_dist_orig_mean)

        if val_lm_loss_component is not None: self.val_lm_losses.append(val_lm_loss_component)
        if val_lev_aux_loss is not None: self.val_lev_aux_losses.append(val_lev_aux_loss)
        if val_nsp_loss is not None: self.val_nsp_losses.append(val_nsp_loss)
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
        """
        Train step with multi-task support.
        Returns: (combined_loss_val, lm_loss_item, unshuffle_loss_item, nsp_loss_item)
        """
        # Metrics initialization
        mean_lm_loss_component_item = None
        unshuffle_loss_item = None
        mean_nsp_loss_item = None
        # mean_pred_dist_orig_item removed

        # 1. Batch Unpacking (now 5 items)
        if self.config.use_levenshtein_task: # Indicates multi-task mode and presence of all 5 items
            input_tokens, next_token_lm_targets, unshuffle_seq_targets, auxiliary_values, task_type_flags = batch
            input_tokens = input_tokens.to(self.config.device)
            next_token_lm_targets = next_token_lm_targets.to(self.config.device)
            unshuffle_seq_targets = unshuffle_seq_targets.to(self.config.device)
            auxiliary_values = auxiliary_values.to(self.config.device) # NSP labels for type 2
            task_type_flags = task_type_flags.to(self.config.device)
        else: # Standard LM task (single task mode)
            if len(batch) == 2: # Original single LM task format
                input_tokens, next_token_lm_targets = batch
            elif len(batch) == 5: # If data pipeline provides 5 items even for single task
                 input_tokens, next_token_lm_targets, _, _, _ = batch # unshuffle_seq_targets and aux_values are ignored
            else:
                raise ValueError(f"Unexpected batch structure with {len(batch)} items in single-task mode.")
            # Create dummy/default values for other components for consistent code paths
            unshuffle_seq_targets = None
            auxiliary_values = None
            task_type_flags = None # Handled as pure LM task by logic below

            input_tokens = input_tokens.to(self.config.device)
            next_token_lm_targets = next_token_lm_targets.to(self.config.device)

        self.optimizer.zero_grad()
        
        # Initialize loss components
        final_batch_lm_loss_component = torch.tensor(0.0, device=self.config.device, dtype=torch.float32)
        unshuffle_loss_tensor = torch.tensor(0.0, device=self.config.device, dtype=torch.float32)
        mean_nsp_loss_tensor = torch.tensor(0.0, device=self.config.device, dtype=torch.float32)

        autocast_context = torch.amp.autocast('cuda') if self.config.use_amp and self.config.scaler is not None else contextlib.suppress()

        with autocast_context:
            # 2. Model Call - model now returns (lm_logits, per_item_lm_loss, nsp_logits)
            # per_item_lm_loss is calculated by the model using next_token_lm_targets
            lm_logits_all, per_item_lm_loss_all, nsp_logits_all = self.model(
                input_tokens,
                next_token_lm_targets,
                force_disable_prefix_attention=False # Keep prefix attention enabled by default
            )

            # 3. Loss Calculation
            if self.config.use_levenshtein_task and task_type_flags is not None: # Multi-task mode
                lm_task_mask = (task_type_flags == 0.0)  # Pure LM task
                lev_task_mask = (task_type_flags == 1.0) # Levenshtein/Unshuffle task
                nsp_task_mask = (task_type_flags == 2.0)  # NSP task
                
                # LM Loss Component (Task Type 0)
                if lm_task_mask.any() and per_item_lm_loss_all is not None:
                    # per_item_lm_loss_all is already calculated based on next_token_lm_targets
                    # and should be valid only for non-ignored indices.
                    valid_lm_losses = per_item_lm_loss_all[lm_task_mask]
                    if valid_lm_losses.numel() > 0: # Ensure there are actual values after masking
                        final_batch_lm_loss_component = valid_lm_losses.float().mean()
                
                # Unshuffle Seq2Seq Loss (Task Type 1)
                if lev_task_mask.any() and unshuffle_seq_targets is not None:
                    predictions_for_unshuffle = lm_logits_all[lev_task_mask]
                    targets_for_unshuffle = unshuffle_seq_targets[lev_task_mask]

                    if predictions_for_unshuffle.numel() > 0 and targets_for_unshuffle.numel() > 0:
                        vocab_size = predictions_for_unshuffle.size(-1)
                        # lm_ignore_idx is typically -1, consistent with dataset implementations
                        unshuffle_loss_tensor = F.cross_entropy(
                            predictions_for_unshuffle.reshape(-1, vocab_size),
                            targets_for_unshuffle.reshape(-1),
                            ignore_index= -1
                        )
                        unshuffle_loss_item = unshuffle_loss_tensor.item()
                
                # NSP Loss (Task Type 2)
                if nsp_task_mask.any() and nsp_logits_all is not None:
                    nsp_predicted = nsp_logits_all[nsp_task_mask]
                    nsp_targets = auxiliary_values[nsp_task_mask].long() # NSP labels are in auxiliary_values
                    if nsp_predicted.numel() > 0 and nsp_targets.numel() > 0:
                        loss_fn_nsp = torch.nn.CrossEntropyLoss()
                        mean_nsp_loss_tensor = loss_fn_nsp(nsp_predicted, nsp_targets)
                        mean_nsp_loss_item = mean_nsp_loss_tensor.item()
                
                # Combined Loss for multi-task
                combined_loss = final_batch_lm_loss_component
                combined_loss = combined_loss + (self.config.levenshtein_loss_weight * unshuffle_loss_tensor)
                nsp_loss_weight = getattr(self.config, 'nsp_loss_weight', 0.1)
                combined_loss = combined_loss + (nsp_loss_weight * mean_nsp_loss_tensor)
                
            else: # Standard LM task (single task mode)
                if per_item_lm_loss_all is not None:
                     final_batch_lm_loss_component = per_item_lm_loss_all.float().mean()
                combined_loss = final_batch_lm_loss_component

            # For logging the LM component if it's a valid tensor
            if isinstance(final_batch_lm_loss_component, torch.Tensor) and final_batch_lm_loss_component.numel() > 0 :
                 mean_lm_loss_component_item = final_batch_lm_loss_component.item()


        # --- End of autocast_context for AMP ---

        combined_loss_val = combined_loss.item()

        # Handle cases where loss might not require grad (e.g., batch with no valid labels for any active task)
        if not combined_loss.requires_grad and combined_loss.abs().item() < 1e-9:
             return 0.0, mean_lm_loss_component_item, unshuffle_loss_item, mean_nsp_loss_item

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

        return combined_loss_val, mean_lm_loss_component_item, unshuffle_loss_item, mean_nsp_loss_item

    def evaluate(self, dataloader: DataLoader, max_batches: Optional[int] = 50) -> float:
        self.model.eval()
        total_combined_loss_epoch = 0.0
        accum_lm_loss_component = 0.0
        accum_unshuffle_loss = 0.0 # Renamed from accum_lev_aux_loss
        accum_nsp_loss = 0.0       # For NSP specific metric

        num_batches_processed = 0
        num_lm_batches = 0
        num_unshuffle_batches = 0  # Renamed from num_lev_batches
        num_nsp_batches = 0        # For NSP specific metric

        # Default lm_ignore_idx to -1 if not in config, for consistency
        lm_ignore_idx = getattr(self.config, 'lm_ignore_idx', -1)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    print(f"Evaluation limited to {max_batches} batches for speed")
                    break
                
                # 1. Batch Unpacking (5-tuple)
                if self.config.use_levenshtein_task: # Multi-task mode
                    input_tokens, next_token_lm_targets, unshuffle_seq_targets, auxiliary_values, task_type_flags = batch
                    input_tokens = input_tokens.to(self.config.device)
                    next_token_lm_targets = next_token_lm_targets.to(self.config.device)
                    unshuffle_seq_targets = unshuffle_seq_targets.to(self.config.device)
                    auxiliary_values = auxiliary_values.to(self.config.device)
                    task_type_flags = task_type_flags.to(self.config.device)
                else: # Standard LM task
                    if len(batch) == 2:
                        input_tokens, next_token_lm_targets = batch
                    elif len(batch) == 5: # If dataset still provides 5 items
                        input_tokens, next_token_lm_targets, _, _, _ = batch
                    else:
                        raise ValueError("Unexpected batch structure in single-task mode during eval.")

                    input_tokens = input_tokens.to(self.config.device)
                    next_token_lm_targets = next_token_lm_targets.to(self.config.device)
                    # Placeholders for multi-task components
                    batch_dim_size = input_tokens.size(0)
                    unshuffle_seq_targets = torch.full_like(next_token_lm_targets, lm_ignore_idx)
                    auxiliary_values = torch.zeros(batch_dim_size, device=self.config.device, dtype=torch.float)
                    task_type_flags = torch.zeros(batch_dim_size, device=self.config.device, dtype=torch.float) # All LM type

                # Initialize per-batch losses for combining
                current_batch_lm_loss = torch.tensor(0.0, device=self.config.device, dtype=torch.float32)
                current_batch_unshuffle_loss = torch.tensor(0.0, device=self.config.device, dtype=torch.float32)
                current_batch_nsp_loss = torch.tensor(0.0, device=self.config.device, dtype=torch.float32)

                autocast_context_eval = torch.amp.autocast('cuda') if self.config.use_amp else contextlib.suppress()
                with autocast_context_eval:
                    # 2. Model Call (model uses next_token_lm_targets for its internal per_item_lm_loss)
                    lm_logits_all, per_item_lm_loss_all, nsp_logits_all = self.model(
                        input_tokens,
                        next_token_lm_targets, # Model calculates LM loss based on this
                        force_disable_prefix_attention=False # Allow prefix attention based on model config
                    )

                # 3. Metric Calculation
                lm_task_mask_eval = (task_type_flags == 0.0)
                lev_task_mask_eval = (task_type_flags == 1.0) # Unshuffle task
                nsp_task_mask_eval = (task_type_flags == 2.0)

                # LM Loss Component (Val LM Comp) - for Type 0
                if lm_task_mask_eval.any() and per_item_lm_loss_all is not None:
                    lm_loss_for_lm_items = per_item_lm_loss_all[lm_task_mask_eval]
                    if lm_loss_for_lm_items.numel() > 0:
                        valid_lm_items = lm_loss_for_lm_items[lm_loss_for_lm_items != float('inf')]
                        if valid_lm_items.numel() > 0:
                            current_lm_loss_value = valid_lm_items.float().mean().item()
                            if not math.isnan(current_lm_loss_value) and not math.isinf(current_lm_loss_value):
                                accum_lm_loss_component += current_lm_loss_value
                                num_lm_batches += 1
                                current_batch_lm_loss = torch.tensor(current_lm_loss_value, device=self.config.device)


                # Unshuffle Seq2Seq Loss Component (Val Unshuffle Aux) - for Type 1
                if self.config.use_levenshtein_task and lev_task_mask_eval.any() and lm_logits_all is not None and unshuffle_seq_targets is not None:
                    predictions_for_unshuffle_eval = lm_logits_all[lev_task_mask_eval]
                    targets_for_unshuffle_eval = unshuffle_seq_targets[lev_task_mask_eval]

                    if predictions_for_unshuffle_eval.numel() > 0 and targets_for_unshuffle_eval.numel() > 0:
                        vocab_size = predictions_for_unshuffle_eval.size(-1)
                        current_unshuffle_loss_tensor_calc = F.cross_entropy(
                            predictions_for_unshuffle_eval.reshape(-1, vocab_size),
                            targets_for_unshuffle_eval.reshape(-1),
                            ignore_index=lm_ignore_idx
                        )
                        current_batch_unshuffle_loss = current_unshuffle_loss_tensor_calc # Store for combined loss
                        if not torch.isnan(current_unshuffle_loss_tensor_calc) and not torch.isinf(current_unshuffle_loss_tensor_calc):
                            accum_unshuffle_loss += current_unshuffle_loss_tensor_calc.item()
                            num_unshuffle_batches += 1

                # NSP Loss Component (Val NSP) - for Type 2
                if self.config.use_levenshtein_task and nsp_task_mask_eval.any() and nsp_logits_all is not None:
                    nsp_predicted_eval = nsp_logits_all[nsp_task_mask_eval]
                    nsp_targets_eval = auxiliary_values[nsp_task_mask_eval].long()
                    if nsp_predicted_eval.numel() > 0 and nsp_targets_eval.numel() > 0:
                        current_nsp_loss_tensor_calc = F.cross_entropy(nsp_predicted_eval, nsp_targets_eval)
                        current_batch_nsp_loss = current_nsp_loss_tensor_calc # Store for combined loss
                        if not torch.isnan(current_nsp_loss_tensor_calc) and not torch.isinf(current_nsp_loss_tensor_calc):
                            accum_nsp_loss += current_nsp_loss_tensor_calc.item()
                            num_nsp_batches +=1

                # Combined Validation Loss for the batch
                # This logic needs to be robust to batches containing only one type of task.
                # The combined loss should reflect the average of losses present in the batch, weighted.
                # For evaluation, it's often just the primary task loss (LM), or a sum if others are comparable.
                # Given train_step combines them, evaluate should too for consistency of "val_loss".

                # A simple sum of components present in the batch, then average over num_batches_processed.
                # If a component is not present, its tensor (e.g. current_batch_lm_loss) remains 0.
                batch_total_loss_val = current_batch_lm_loss # Start with LM loss (if any)
                if self.config.use_levenshtein_task:
                    batch_total_loss_val = batch_total_loss_val + \
                                           (self.config.levenshtein_loss_weight * current_batch_unshuffle_loss) + \
                                           (getattr(self.config, 'nsp_loss_weight', 0.1) * current_batch_nsp_loss)

                total_combined_loss_epoch += batch_total_loss_val.item()
                num_batches_processed +=1

        self.model.train()
        avg_combined_loss = total_combined_loss_epoch / num_batches_processed if num_batches_processed > 0 else float('inf')
        avg_lm_loss_component = accum_lm_loss_component / num_lm_batches if num_lm_batches > 0 else 0.0
        avg_unshuffle_loss = accum_unshuffle_loss / num_unshuffle_batches if num_unshuffle_batches > 0 else 0.0 # New avg metric
        avg_nsp_loss = accum_nsp_loss / num_nsp_batches if num_nsp_batches > 0 else 0.0 # New avg metric

        self.metrics.update(
            val_loss=avg_combined_loss,
            val_lm_loss_component=avg_lm_loss_component,
            val_lev_aux_loss=avg_unshuffle_loss, # Store avg_unshuffle_loss in val_lev_aux_loss slot
            val_nsp_loss=avg_nsp_loss
            # val_pred_dist_orig_mean removed
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
            # train_step now returns 4 items: combined_loss, lm_loss, unshuffle_loss, nsp_loss
            combined_loss_item, current_lm_loss_item, current_unshuffle_loss_item, \
                current_nsp_loss_item = self.train_step(batch)
            epoch_losses.append(combined_loss_item)
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            self.metrics.update(
                train_loss=combined_loss_item,
                learning_rate=current_lr,
                step_time=time.time() - step_start,
                lm_loss_component=current_lm_loss_item,
                lev_aux_loss=current_unshuffle_loss_item, # Use lev_aux_loss field for unshuffle_loss_item
                nsp_loss=current_nsp_loss_item
                # pred_dist_orig_mean is removed
            )
            
            if (not self.is_distributed or dist.get_rank() == 0) and \
               self.metrics.total_steps % self.config.log_every == 0:
                avg_step_time = self.metrics.get_avg_step_time()
                log_msg = f"Epoch {epoch+1}, Step {self.metrics.total_steps}, Rank {dist.get_rank() if self.is_distributed else 0}, Loss: {combined_loss_item:.4f}"
                if self.config.use_levenshtein_task: # This means multi-task is active
                    if current_lm_loss_item is not None: log_msg += f", LM Comp: {current_lm_loss_item:.4f}"
                    if current_unshuffle_loss_item is not None: log_msg += f", Unshuffle Aux: {current_unshuffle_loss_item:.4f}"
                    if current_nsp_loss_item is not None: log_msg += f", NSP: {current_nsp_loss_item:.4f}"
                log_msg += f", LR: {current_lr:.6f}, Step Time: {avg_step_time:.3f}s"
                print(log_msg)
            
            if (not self.is_distributed or dist.get_rank() == 0) and \
               val_loader is not None and \
               self.metrics.total_steps % self.config.eval_every == 0:
                val_loss = self.evaluate(val_loader) # Calls the updated evaluate method
                is_best = val_loss < self.metrics.best_val_loss
                print(f"Validation Loss (Rank {dist.get_rank() if self.is_distributed else 0}): {val_loss:.4f} {'(Best!)' if is_best else ''}")
                if self.config.use_levenshtein_task: # This means multi-task
                    if self.metrics.val_lm_losses: print(f"  Val LM Comp: {self.metrics.val_lm_losses[-1]:.4f}")
                    if self.metrics.val_lev_aux_losses: print(f"  Val Unshuffle Aux: {self.metrics.val_lev_aux_losses[-1]:.4f}") # Updated log
                    if self.metrics.val_nsp_losses: print(f"  Val NSP: {self.metrics.val_nsp_losses[-1]:.4f}") # Log NSP
                    # val_pred_dist_orig_means is removed from metrics and logging
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
                
                per_item_lm_loss_for_ppl = None # Initialize to None
                autocast_context = torch.amp.autocast('cuda') if self.config.use_amp and self.config.scaler is not None else contextlib.suppress()
                with autocast_context:
                    # Model's forward for perplexity should focus on LM loss from original_tokens_cls
                    # It returns: lm_logits, per_item_lm_loss, predicted_lev_distances, nsp_logits
                    _, per_item_lm_loss_for_ppl, _, _ = self.model(input_ids, lm_targets, force_disable_prefix_attention=True) # Force disable prefix for pure LM perplexity

                if per_item_lm_loss_for_ppl is not None:
                    total_loss += per_item_lm_loss_for_ppl.mean().item()
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
