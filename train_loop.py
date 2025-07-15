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

# Task ID Bytes (using ord() for clarity on origin)
TASK_ID_LM = ord('0')           # For Language Modeling tasks
TASK_ID_UNSHUFFLE = ord('1')    # For the Unshuffle Seq2Seq task (formerly Levenshtein)
TASK_ID_NSP = ord('2')          # For Next Sentence Prediction tasks
TASK_ID_SPAN_SELECT = ord('3')  # New task for Span Selection

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
        nsp_loss_weight: float = 0.1,
        rl_loss_weight: float = 0.1,
        span_selection_loss_weight: float = 0.1, # New
        lm_ignore_idx: int = -1,
        long_term_loss_window: int = 1000,
        short_term_loss_window: int = 100, # New
        dynamic_loss_adjustment_factor: float = 0.1 # New
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
        self.nsp_loss_weight = nsp_loss_weight
        self.rl_loss_weight = rl_loss_weight
        self.span_selection_loss_weight = span_selection_loss_weight # New
        self.lm_ignore_idx = lm_ignore_idx
        self.long_term_loss_window = long_term_loss_window
        self.short_term_loss_window = short_term_loss_window
        self.dynamic_loss_adjustment_factor = dynamic_loss_adjustment_factor
        
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
        if not hasattr(trainer_instance.config, 'local_rank') or trainer_instance.config.local_rank == -1:
            trainer_instance.config.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if torch.cuda.is_available():
            if hasattr(trainer_instance.config, 'local_rank') and trainer_instance.config.local_rank != -1:
                 trainer_instance.config.device = torch.device(f"cuda:{trainer_instance.config.local_rank}")
            else:
                 trainer_instance.config.device = torch.device('cuda')
        else:
            trainer_instance.config.device = torch.device('cpu')
        print(f"Distributed training already initialized. Using rank: {dist.get_rank()}, world_size: {dist.get_world_size()}, device: {trainer_instance.config.device}")
        return

    rank_env = os.environ.get('RANK')
    world_size_env = os.environ.get('WORLD_SIZE')
    local_rank_env = os.environ.get('LOCAL_RANK')

    if rank_env is not None and world_size_env is not None:
        try:
            rank = int(rank_env)
            world_size = int(world_size_env)

            if world_size > 1 and torch.cuda.is_available():
                local_rank = int(local_rank_env) if local_rank_env is not None else rank % torch.cuda.device_count()
                trainer_instance.config.local_rank = local_rank
                backend = 'nccl'
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
                trainer_instance.is_distributed = True
                trainer_instance.config.device = torch.device(f"cuda:{local_rank}")
                print(f"Distributed training initialized by train_loop. Rank: {rank}, World Size: {world_size}, Device: {trainer_instance.config.device}")
            elif world_size > 1 and not torch.cuda.is_available():
                print("Warning: Distributed training requested (world_size > 1) but CUDA is not available. Falling back to non-distributed CPU mode.")
                trainer_instance.is_distributed = False
            else:
                trainer_instance.is_distributed = False
        except Exception as e:
            print(f"Error initializing distributed group: {e}. Falling back to non-distributed mode.")
            trainer_instance.is_distributed = False
    else:
        trainer_instance.is_distributed = False

    if not trainer_instance.is_distributed:
        if torch.cuda.is_available():
            trainer_instance.config.device = torch.device('cuda')
            trainer_instance.config.local_rank = 0
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
        self.rank_aux_losses = []
        self.nsp_losses = []
        self.span_selection_losses = [] # New

        self.val_lm_losses = []
        self.val_rank_aux_losses = []
        self.val_nsp_losses = []
        self.val_span_selection_losses = [] # New
        self.penalty_rewards = []
        self.rl_rewards = []

    def update(self, train_loss=None, val_loss=None, learning_rate=None, step_time=None,
                 lm_loss_component=None, rank_aux_loss=None, nsp_loss=None, span_selection_loss=None, # New
                 val_lm_loss_component=None, val_rank_aux_loss=None, val_nsp_loss=None, val_span_selection_loss=None, # New
                 penalty_reward=None, rl_reward=None):
        if train_loss is not None: self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_step = self.total_steps
        if learning_rate is not None: self.learning_rates.append(learning_rate)
        if step_time is not None: self.step_times.append(step_time)
        if lm_loss_component is not None: self.lm_losses.append(lm_loss_component)
        if rank_aux_loss is not None: self.rank_aux_losses.append(rank_aux_loss)
        if nsp_loss is not None: self.nsp_losses.append(nsp_loss)
        if span_selection_loss is not None: self.span_selection_losses.append(span_selection_loss)
        if rl_reward is not None: self.rl_rewards.append(rl_reward)

        if val_lm_loss_component is not None: self.val_lm_losses.append(val_lm_loss_component)
        if val_rank_aux_loss is not None: self.val_rank_aux_losses.append(val_rank_aux_loss)
        if val_nsp_loss is not None: self.val_nsp_losses.append(val_nsp_loss)
        if val_span_selection_loss is not None: self.val_span_selection_losses.append(val_span_selection_loss)
        if penalty_reward is not None:
            self.penalty_rewards.append(penalty_reward)
        self.total_steps += 1
    
    def get_avg_step_time(self, last_n=100):
        return np.mean(self.step_times[-last_n:]) if self.step_times else 0.0

    def get_avg_combined_loss(self, last_n: int) -> Optional[float]:
        if not self.train_losses:
            return None
        relevant_losses = self.train_losses[-last_n:]
        if not relevant_losses: # Should not happen if self.train_losses is not empty
            return None
        return np.mean(relevant_losses)

    def get_avg_lm_loss(self, last_n: int) -> Optional[float]:
        if not self.lm_losses:
            return None
        relevant_losses = self.lm_losses[-last_n:]
        if not relevant_losses:
            return None
        # Filter out None values that might have been appended if loss wasn't computed
        relevant_losses = [l for l in relevant_losses if l is not None]
        if not relevant_losses:
            return None
        return np.mean(relevant_losses)

    def get_avg_rank_loss(self, last_n: int) -> Optional[float]: # Renamed, uses self.rank_aux_losses
        if not self.rank_aux_losses:
            return None
        relevant_losses = self.rank_aux_losses[-last_n:]
        relevant_losses = [l for l in relevant_losses if l is not None]
        if not relevant_losses:
            return None
        return np.mean(relevant_losses)

    def get_avg_span_selection_loss(self, last_n: int) -> Optional[float]:
        if not self.span_selection_losses:
            return None
        relevant_losses = self.span_selection_losses[-last_n:]
        relevant_losses = [l for l in relevant_losses if l is not None]
        if not relevant_losses:
            return None
        return np.mean(relevant_losses)

    def get_avg_nsp_loss(self, last_n: int) -> Optional[float]:
        if not self.nsp_losses:
            return None
        relevant_losses = self.nsp_losses[-last_n:]
        relevant_losses = [l for l in relevant_losses if l is not None]
        if not relevant_losses:
            return None
        return np.mean(relevant_losses)

    def get_avg_penalty_reward(self, last_n: int) -> Optional[float]:
        if not self.penalty_rewards:
            return None
        relevant_values = self.penalty_rewards[-last_n:]
        # Filter out None values, though penalty_reward should always be float (even 0.0)
        relevant_values = [v for v in relevant_values if v is not None]
        if not relevant_values:
            return None
        return np.mean(relevant_values)

    def get_avg_rl_reward(self, last_n: int) -> Optional[float]:
        if not self.rl_rewards:
            return None
        relevant_rewards = [r.item() if isinstance(r, torch.Tensor) else r for r in self.rl_rewards[-last_n:]]
        relevant_rewards = [r for r in relevant_rewards if r is not None]
        if not relevant_rewards:
            return None
        return np.mean(relevant_rewards)
    
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
        Returns: (combined_loss_val, lm_loss_item, rank_loss_item, nsp_loss_item, penalty_reward_item)
        """
        mean_lm_loss_component_item = 0.0
        rank_loss_item = 0.0
        mean_nsp_loss_item = 0.0
        span_selection_loss_item = 0.0
        rl_reward_item = 0.0

        # The model's forward method now returns a dictionary of outputs.
        # The batch structure for multi-task is:
        # input_tokens, next_token_lm_targets, rank_regression_targets (formerly unshuffle_seq_targets), auxiliary_values, task_type_flags

        if self.config.use_levenshtein_task: # This flag now means "use multi-task including rank regression"
            # Unpack the new 6-item tuple
            input_tokens, next_token_lm_targets, rank_targets, auxiliary_values, task_type_flags, true_original_ranks = batch
            input_tokens = input_tokens.to(self.config.device)
            next_token_lm_targets = next_token_lm_targets.to(self.config.device)
            rank_targets = rank_targets.to(self.config.device) # These are float targets for ranks
            auxiliary_values = auxiliary_values.to(self.config.device)
            task_type_flags = task_type_flags.to(self.config.device)
            true_original_ranks = true_original_ranks.to(self.config.device) # New item for RL reward
        else: # Single-task LM mode
            if len(batch) == 2:
                input_tokens, next_token_lm_targets = batch
            elif len(batch) == 6: # If data loader still yields 6 items even in single task mode
                 input_tokens, next_token_lm_targets, _, _, _, _ = batch
            else:
                raise ValueError(f"Unexpected batch structure with {len(batch)} items in single-task mode.")
            rank_targets = None
            auxiliary_values = None
            task_type_flags = None # Indicates pure LM task if None or all 0.0
            true_original_ranks = None

            input_tokens = input_tokens.to(self.config.device)
            next_token_lm_targets = next_token_lm_targets.to(self.config.device)

        self.optimizer.zero_grad()
        
        final_batch_lm_loss_component = torch.tensor(0.0, device=self.config.device, dtype=torch.float32)
        rank_loss_tensor = torch.tensor(0.0, device=self.config.device, dtype=torch.float32)
        mean_nsp_loss_tensor = torch.tensor(0.0, device=self.config.device, dtype=torch.float32)
        span_selection_loss_tensor = torch.tensor(0.0, device=self.config.device, dtype=torch.float32) # New

        autocast_context = torch.amp.autocast('cuda') if self.config.use_amp and self.config.scaler is not None else contextlib.suppress()

        with autocast_context:
            # Model now returns a dictionary of outputs
            model_outputs = self.model(
                input_tokens,
                next_token_lm_targets,
                force_disable_prefix_attention=False
            )

            per_item_lm_loss_all = model_outputs.get('lm_loss')

            if self.config.use_levenshtein_task and task_type_flags is not None: # Multi-task logic
                lm_task_mask = (task_type_flags == 0.0)
                rank_task_mask = (task_type_flags == 1.0)
                nsp_task_mask = (task_type_flags == 2.0)
                span_task_mask = (task_type_flags == 3.0) # New
                
                # LM Loss component (from items marked as LM task)
                pg_loss = torch.tensor(0.0, device=self.config.device, dtype=torch.float32)
                if lm_task_mask.any():
                    # Standard CE Loss
                    if per_item_lm_loss_all is not None:
                        valid_lm_losses = per_item_lm_loss_all[lm_task_mask]
                        if valid_lm_losses.numel() > 0:
                            final_batch_lm_loss_component = valid_lm_losses.float().mean()

                    # --- RL Self-Critique Step ---
                    # 1. Greedy decode to get generated sequence (action)
                    lm_logits_for_rl = model_outputs['lm_logits'][lm_task_mask]
                    generated_tokens = torch.argmax(lm_logits_for_rl.detach(), dim=-1)

                    # 2. Get reward by passing generated sequence back through model
                    with torch.no_grad():
                        critique_outputs = self.model(generated_tokens)
                        predicted_ranks_for_gen = critique_outputs['rank_outputs']

                    # 3. Calculate reward = -MSE(predicted_ranks, true_ranks)
                    true_ranks_for_rl = true_original_ranks[lm_task_mask]
                    predicted_ranks_for_gen = predicted_ranks_for_gen.squeeze(-1)

                    rank_ignore_val = float(getattr(self.config, 'lm_ignore_idx', -1.0))
                    reward_mask = (true_ranks_for_rl != rank_ignore_val)

                    # Ensure shapes match for masking
                    if predicted_ranks_for_gen.shape != true_ranks_for_rl.shape:
                         # This can happen if generation is shorter than seq_len, pad preds
                         pad_shape = (predicted_ranks_for_gen.shape[0], true_ranks_for_rl.shape[1] - predicted_ranks_for_gen.shape[1])
                         padding = torch.full(pad_shape, rank_ignore_val, device=self.config.device)
                         predicted_ranks_for_gen = torch.cat([predicted_ranks_for_gen, padding], dim=1)

                    if reward_mask.any():
                        reward = -F.mse_loss(
                            predicted_ranks_for_gen[reward_mask],
                            true_ranks_for_rl[reward_mask]
                        )
                    else:
                        reward = torch.tensor(0.0, device=self.config.device)

                    # 4. Calculate Policy Gradient loss
                    log_probs = F.log_softmax(lm_logits_for_rl, dim=-1)
                    # Gather the log_probs of the generated tokens
                    action_log_probs = log_probs.gather(dim=-1, index=generated_tokens.unsqueeze(-1)).squeeze(-1)

                    # Apply mask from lm_targets to only consider loss on valid (non-padded) tokens
                    lm_targets_for_rl = next_token_lm_targets[lm_task_mask]
                    policy_loss_mask = (lm_targets_for_rl != self.config.lm_ignore_idx)

                    # REINFORCE loss: -reward * log_prob(action)
                    # We want to minimize this, so we use -reward.
                    # The reward is already negative MSE, so we are minimizing -(-MSE) * log_prob = MSE * log_prob
                    # This seems off. Let's make reward positive: higher reward = lower MSE
                    reward = -reward # Now reward is positive (or zero)

                    # PG loss for minimization is -E[R * log(pi)]. So -reward * action_log_probs
                    # Clip reward to prevent excessively large gradients
                    clipped_reward = torch.clamp(reward, -1.0, 1.0)
                    if torch.isfinite(clipped_reward):
                        pg_loss_unmasked = -clipped_reward.detach() * action_log_probs

                        # Sum loss only for valid tokens and average over batch, adding epsilon to prevent division by zero
                        pg_loss = (pg_loss_unmasked * policy_loss_mask).sum() / (policy_loss_mask.sum() + 1e-9)
                    else:
                        pg_loss = torch.tensor(0.0, device=self.config.device, dtype=torch.float32)

                    # The reward is now positive, so we can log it directly.
                    # It will be passed to metrics.update later.

                # Rank Regression Loss component (from items marked as Rank task)
                if rank_task_mask.any() and rank_targets is not None:
                    rank_outputs = model_outputs['rank_outputs']
                    if rank_outputs is not None:
                        print(f"Rank outputs: {rank_outputs}")
                        predictions_for_rank = rank_outputs[rank_task_mask].squeeze(-1)
                        targets_for_rank = rank_targets[rank_task_mask]
                        if predictions_for_rank.numel() > 0 and targets_for_rank.numel() > 0:
                            rank_ignore_val = float(getattr(self.config, 'lm_ignore_idx', -1.0))
                            valid_rank_mask = (targets_for_rank != rank_ignore_val)
                            if valid_rank_mask.any():
                                if torch.isfinite(targets_for_rank[valid_rank_mask]).all():
                                    rank_loss_tensor = F.l1_loss(
                                        predictions_for_rank[valid_rank_mask],
                                        targets_for_rank[valid_rank_mask]
                                    )
                                    print(f"Rank loss tensor: {rank_loss_tensor}")
                                    rank_loss_item = rank_loss_tensor.item()
                                else:
                                    print("Warning: rank_targets contains non-finite values. Skipping batch.")
                
                # NSP Loss component (from items marked as NSP task)
                if nsp_task_mask.any():
                    nsp_logits = model_outputs['nsp_logits']
                    if nsp_logits is not None:
                        nsp_predicted = nsp_logits[nsp_task_mask]
                        nsp_targets = auxiliary_values[nsp_task_mask].long()
                        if nsp_predicted.numel() > 0 and nsp_targets.numel() > 0:
                            loss_fn_nsp = torch.nn.CrossEntropyLoss()
                            mean_nsp_loss_tensor = loss_fn_nsp(nsp_predicted, nsp_targets)
                            mean_nsp_loss_item = mean_nsp_loss_tensor.item()

                # Span Selection Loss component (from items marked as Span task)
                if span_task_mask.any():
                    span_outputs = model_outputs['span_selection_logits']
                    if span_outputs is not None:
                        # Output is now a single scalar (regression), target is the index
                        span_predicted_index = span_outputs[span_task_mask]
                        span_target_index = auxiliary_values[span_task_mask].long()
                        if span_predicted_index.numel() > 0 and span_target_index.numel() > 0:
                            loss_fn_span = torch.nn.CrossEntropyLoss()
                            span_selection_loss_tensor = loss_fn_span(span_predicted_index, span_target_index)
                            span_selection_loss_item = span_selection_loss_tensor.item()
                
                # Combine losses for multi-task items
                # Start with LM loss component (which could be 0 if no LM items in batch or all ignored)
                print(f"lm_loss: {final_batch_lm_loss_component.item()}, pg_loss: {pg_loss.item()}, rank_loss: {rank_loss_tensor.item()}, nsp_loss: {mean_nsp_loss_tensor.item()}, span_loss: {span_selection_loss_tensor.item()}")
                combined_loss = final_batch_lm_loss_component + (self.config.rl_loss_weight * pg_loss)
                # Add weighted Rank Regression loss
                combined_loss = combined_loss + (self.config.levenshtein_loss_weight * rank_loss_tensor)
                # Add weighted NSP loss
                combined_loss = combined_loss + (self.config.nsp_loss_weight * mean_nsp_loss_tensor)
                # Add weighted Span Selection loss
                combined_loss = combined_loss + (self.config.span_selection_loss_weight * span_selection_loss_tensor)
                
            else: # Single-task LM mode
                if per_item_lm_loss_all is not None: # Should contain losses for all items
                     final_batch_lm_loss_component = per_item_lm_loss_all.float().mean()
                combined_loss = final_batch_lm_loss_component # Only LM loss

            # Ensure lm_loss_component_item is a float for metrics
            if isinstance(final_batch_lm_loss_component, torch.Tensor) and final_batch_lm_loss_component.numel() > 0 :
                 mean_lm_loss_component_item = final_batch_lm_loss_component.item()
            elif isinstance(final_batch_lm_loss_component, float): # If it was already a float (e.g. from .mean().item())
                 mean_lm_loss_component_item = final_batch_lm_loss_component


        # Store the original combined_loss value for metrics and logging
        raw_combined_loss_val = combined_loss.item() if isinstance(combined_loss, torch.Tensor) else float(combined_loss)


        # --- Dynamic Loss Adjustment (remains the same) ---
        final_loss_for_backward = combined_loss
        current_penalty_reward_val = 0.0

        use_dynamic_adjustment = hasattr(self.config, 'long_term_loss_window') and \
                                 hasattr(self.config, 'short_term_loss_window') and \
                                 hasattr(self.config, 'dynamic_loss_adjustment_factor')

        if use_dynamic_adjustment and self.config.long_term_loss_window > 0 and self.config.short_term_loss_window > 0:
            avg_loss_long = self.metrics.get_avg_combined_loss(last_n=self.config.long_term_loss_window)
            avg_loss_short = self.metrics.get_avg_combined_loss(last_n=self.config.short_term_loss_window)

            if avg_loss_long is not None and avg_loss_short is not None and \
               self.metrics.total_steps >= self.config.long_term_loss_window:
                signal_val = avg_loss_short - avg_loss_long
                current_penalty_reward_val = signal_val * self.config.dynamic_loss_adjustment_factor
                penalty_reward_tensor = torch.tensor(current_penalty_reward_val,
                                                     device=combined_loss.device,
                                                     dtype=combined_loss.dtype).detach()
                final_loss_for_backward = combined_loss + penalty_reward_tensor
        # --- End of Dynamic Loss Adjustment ---

        # Handle cases where loss might be effectively zero or non-finite before backward pass
        if not isinstance(final_loss_for_backward, torch.Tensor) or \
           not final_loss_for_backward.requires_grad or \
           not torch.isfinite(final_loss_for_backward) or \
           final_loss_for_backward.abs().item() < 1e-9 : # If loss is ~0 and no grad
             # Return raw loss values; penalty/reward also effectively 0 or not meaningful.
             # The reward from the RL step is now also returned.
             rl_reward_item = reward.item() if 'reward' in locals() and isinstance(reward, torch.Tensor) else 0.0
             return raw_combined_loss_val, mean_lm_loss_component_item, rank_loss_item, mean_nsp_loss_item, 0.0, rl_reward_item

        if self.config.use_amp and self.config.scaler is not None:
            self.config.scaler.scale(final_loss_for_backward).backward() # final_loss_for_backward is used for gradient calculation
            if self.config.max_grad_norm > 0:
                self.config.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.config.scaler.step(self.optimizer)
            self.config.scaler.update()
        else:
            final_loss_for_backward.backward()
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        # The reward from the RL step is now also returned.
        rl_reward_item = reward.item() if 'reward' in locals() and isinstance(reward, torch.Tensor) else 0.0
        # Return raw_combined_loss_val for metrics, and current_penalty_reward_val for new metric tracking
        return raw_combined_loss_val, mean_lm_loss_component_item, rank_loss_item, mean_nsp_loss_item, current_penalty_reward_val, rl_reward_item

    def evaluate(self, dataloader: DataLoader, max_batches: Optional[int] = 50) -> float:
        self.model.eval()
        total_combined_loss_epoch = 0.0
        accum_lm_loss_component = 0.0
        accum_rank_loss = 0.0
        accum_nsp_loss = 0.0
        accum_span_selection_loss = 0.0 # New

        num_batches_processed = 0
        num_lm_batches = 0
        num_rank_batches = 0
        num_nsp_batches = 0
        num_span_selection_batches = 0 # New

        # Use lm_ignore_idx for LM loss and as basis for rank_ignore_idx_float
        lm_ignore_idx = getattr(self.config, 'lm_ignore_idx', -1)
        rank_ignore_val = float(lm_ignore_idx) # For rank targets

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    print(f"Evaluation limited to {max_batches} batches for speed")
                    break
                
                # Batch unpacking, similar to train_step
                if self.config.use_levenshtein_task: # Multi-task mode
                    # Unpack 6 items, ignoring the 6th (true_original_ranks) as it's not used in evaluation
                    input_tokens, next_token_lm_targets, rank_targets, auxiliary_values, task_type_flags, _ = batch
                    input_tokens = input_tokens.to(self.config.device)
                    next_token_lm_targets = next_token_lm_targets.to(self.config.device)
                    rank_targets = rank_targets.to(self.config.device) # Float tensor for ranks
                    auxiliary_values = auxiliary_values.to(self.config.device)
                    task_type_flags = task_type_flags.to(self.config.device)
                else: # Single-task LM mode
                    if len(batch) == 2: input_tokens, next_token_lm_targets = batch
                    elif len(batch) == 6: input_tokens, next_token_lm_targets, _, _, _, _ = batch
                    else: raise ValueError("Unexpected batch structure in single-task mode during eval.")

                    input_tokens = input_tokens.to(self.config.device)
                    next_token_lm_targets = next_token_lm_targets.to(self.config.device)
                    # Create dummy tensors for other components if in single-task mode
                    batch_dim_size = input_tokens.size(0)
                    rank_targets = torch.full_like(next_token_lm_targets, rank_ignore_val, dtype=torch.float32)
                    auxiliary_values = torch.zeros(batch_dim_size, device=self.config.device, dtype=torch.float)
                    task_type_flags = torch.zeros(batch_dim_size, device=self.config.device, dtype=torch.float) # All LM task

                current_batch_lm_component_loss_val = 0.0
                current_batch_rank_component_loss_val = 0.0
                current_batch_nsp_component_loss_val = 0.0
                current_batch_span_selection_loss_val = 0.0 # New

                autocast_context_eval = torch.amp.autocast('cuda') if self.config.use_amp else contextlib.suppress()
                with autocast_context_eval:
                    # Model returns a dictionary of outputs
                    model_outputs = self.model(
                        input_tokens,
                        next_token_lm_targets,
                        force_disable_prefix_attention=False
                    )
                    per_item_lm_loss_all = model_outputs.get('lm_loss')
                    rank_regression_outputs_all = model_outputs.get('rank_outputs')
                    nsp_logits_all = model_outputs.get('nsp_logits')
                    span_selection_logits_all = model_outputs.get('span_selection_logits')


                lm_task_mask_eval = (task_type_flags == 0.0)
                rank_task_mask_eval = (task_type_flags == 1.0)
                nsp_task_mask_eval = (task_type_flags == 2.0)
                span_task_mask_eval = (task_type_flags == 3.0) # New

                # LM Loss
                if lm_task_mask_eval.any() and per_item_lm_loss_all is not None:
                    lm_losses_for_lm_items = per_item_lm_loss_all[lm_task_mask_eval]
                    if lm_losses_for_lm_items.numel() > 0:
                        valid_lm_losses = lm_losses_for_lm_items[torch.isfinite(lm_losses_for_lm_items)]
                        if valid_lm_losses.numel() > 0:
                            current_batch_lm_component_loss_val = valid_lm_losses.float().mean().item()
                            accum_lm_loss_component += current_batch_lm_component_loss_val
                            num_lm_batches += 1

                # Rank Regression Loss
                if self.config.use_levenshtein_task and rank_task_mask_eval.any() and \
                   rank_regression_outputs_all is not None and rank_targets is not None:

                    preds_for_rank_eval = rank_regression_outputs_all[rank_task_mask_eval].squeeze(-1)
                    targets_for_rank_eval = rank_targets[rank_task_mask_eval]

                    if preds_for_rank_eval.numel() > 0 and targets_for_rank_eval.numel() > 0:
                        valid_rank_mask_eval = (targets_for_rank_eval != rank_ignore_val)
                        if valid_rank_mask_eval.any():
                            rank_loss_val = F.mse_loss(
                                preds_for_rank_eval[valid_rank_mask_eval],
                                targets_for_rank_eval[valid_rank_mask_eval]
                            ).item()
                            if not math.isnan(rank_loss_val) and not math.isinf(rank_loss_val):
                                current_batch_rank_component_loss_val = rank_loss_val
                                accum_rank_loss += rank_loss_val
                                num_rank_batches +=1

                # NSP Loss
                if self.config.use_levenshtein_task and nsp_task_mask_eval.any() and nsp_logits_all is not None:
                    nsp_predicted_eval = nsp_logits_all[nsp_task_mask_eval]
                    nsp_targets_eval = auxiliary_values[nsp_task_mask_eval].long()
                    if nsp_predicted_eval.numel() > 0 and nsp_targets_eval.numel() > 0:
                        nsp_loss_val = F.cross_entropy(nsp_predicted_eval, nsp_targets_eval).item()
                        if not math.isnan(nsp_loss_val) and not math.isinf(nsp_loss_val):
                            current_batch_nsp_component_loss_val = nsp_loss_val
                            accum_nsp_loss += nsp_loss_val
                            num_nsp_batches +=1

                # Span Selection Loss
                if self.config.use_levenshtein_task and span_task_mask_eval.any() and span_selection_logits_all is not None:
                    span_predicted_eval = span_selection_logits_all[span_task_mask_eval]
                    span_targets_eval = auxiliary_values[span_task_mask_eval].long()
                    if span_predicted_eval.numel() > 0 and span_targets_eval.numel() > 0:
                        span_loss_val = F.cross_entropy(span_predicted_eval, span_targets_eval).item()
                        if not math.isnan(span_loss_val) and not math.isinf(span_loss_val):
                            current_batch_span_selection_loss_val = span_loss_val
                            accum_span_selection_loss += span_loss_val
                            num_span_selection_batches += 1

                # Combine component losses for the batch (as float values)
                batch_combined_loss_val = current_batch_lm_component_loss_val
                if self.config.use_levenshtein_task: # If multi-tasking
                    if num_rank_batches > 0:
                        batch_combined_loss_val += (self.config.levenshtein_loss_weight * current_batch_rank_component_loss_val)
                    if num_nsp_batches > 0:
                        batch_combined_loss_val += (self.config.nsp_loss_weight * current_batch_nsp_component_loss_val)
                    if num_span_selection_batches > 0:
                        batch_combined_loss_val += (self.config.span_selection_loss_weight * current_batch_span_selection_loss_val)

                total_combined_loss_epoch += batch_combined_loss_val
                num_batches_processed +=1

        self.model.train()
        avg_combined_loss = total_combined_loss_epoch / num_batches_processed if num_batches_processed > 0 else float('inf')
        avg_lm_loss_component = accum_lm_loss_component / num_batches_processed if num_batches_processed > 0 else 0.0
        avg_rank_loss_component = accum_rank_loss / num_batches_processed if num_batches_processed > 0 else 0.0
        avg_nsp_loss_component = accum_nsp_loss / num_batches_processed if num_batches_processed > 0 else 0.0
        avg_span_selection_loss_component = accum_span_selection_loss / num_batches_processed if num_batches_processed > 0 else 0.0

        self.metrics.update(
            val_loss=avg_combined_loss,
            val_lm_loss_component=avg_lm_loss_component,
            val_rank_aux_loss=avg_rank_loss_component,
            val_nsp_loss=avg_nsp_loss_component,
            val_span_selection_loss=avg_span_selection_loss_component # New
        )
        print(f"  Val LM Comp: {avg_lm_loss_component:.4f}")
        print(f"  Val RankReg Aux: {avg_rank_loss_component:.4f}")
        print(f"  Val NSP: {avg_nsp_loss_component:.4f}")
        print(f"  Val SpanSelect: {avg_span_selection_loss_component:.4f}")
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

        # Curriculum learning for shuffle probability
        if self.config.use_levenshtein_task and hasattr(train_loader, 'dataset'):
            dataset_instance = train_loader.dataset # This is CombinedMultiTaskDataset

            if hasattr(dataset_instance, 'update_lev_shuffle_parameters') and \
               callable(getattr(dataset_instance, 'update_lev_shuffle_parameters')):

                total_epochs = self.config.num_epochs
                # Calculate progress: 0.0 at epoch 0, towards 1.0 at final epoch
                # Ensure no division by zero if total_epochs is 1
                progress = epoch / max(1, total_epochs - 1) if total_epochs > 1 else 0.0
                progress = min(max(progress, 0.0), 1.0) # Clamp progress to [0,1]

                start_min_p = 0.05  # Minimum shuffle probability always
                initial_max_p = 0.05 # Max shuffle probability at the start of training
                end_max_p = 0.5     # Max shuffle probability at the end of training curriculum

                # Interpolate current_max_p for the epoch
                current_max_p_for_epoch = initial_max_p + (end_max_p - initial_max_p) * progress

                # Clamp the calculated max_p to be within [start_min_p, end_max_p]
                # and also ensure it's not less than start_min_p itself.
                current_max_p_for_epoch = max(start_min_p, min(current_max_p_for_epoch, end_max_p))

                min_p_for_epoch = start_min_p

                # print(f"Epoch {epoch+1}/{total_epochs}: Updating shuffle range to [{min_p_for_epoch:.4f}, {current_max_p_for_epoch:.4f}]")
                dataset_instance.update_lev_shuffle_parameters(min_p_for_epoch, current_max_p_for_epoch)
            else:
                if not self.is_distributed or dist.get_rank() == 0:
                    print("Warning: Training dataset does not support shuffle parameter updates for curriculum.")

        # Existing train_epoch logic starts here
        epoch_losses = []
        start_time = time.time()
        if self.is_distributed and hasattr(train_loader.sampler, 'set_epoch') and dist.get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(train_loader):
            print(f"Processing batch {batch_idx}")
            step_start = time.time()
            # Unpack the new 6-item tuple from train_step, which now includes rl_reward_item
            combined_loss_item, current_lm_loss_item, current_rank_loss_item, \
                current_nsp_loss_item, current_penalty_reward, rl_reward_item = self.train_step(batch)
            epoch_losses.append(combined_loss_item)
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            self.metrics.update(
                train_loss=combined_loss_item,
                learning_rate=current_lr,
                step_time=time.time() - step_start,
                lm_loss_component=current_lm_loss_item,
                rank_aux_loss=current_rank_loss_item,
                nsp_loss=current_nsp_loss_item,
                penalty_reward=current_penalty_reward,
                rl_reward=rl_reward_item # Log the new reward
            )
            
            if (not self.is_distributed or dist.get_rank() == 0) and \
               self.metrics.total_steps % self.config.log_every == 0:

                long_window = self.config.long_term_loss_window
                log_interval_window = self.config.log_every

                avg_combined_loss = self.metrics.get_avg_combined_loss(last_n=long_window)
                avg_lm_comp = self.metrics.get_avg_lm_loss(last_n=long_window)
                avg_rank_aux = self.metrics.get_avg_rank_loss(last_n=long_window)
                avg_nsp_loss = self.metrics.get_avg_nsp_loss(last_n=long_window)
                avg_span_loss = self.metrics.get_avg_span_selection_loss(last_n=long_window) # New
                avg_penalty_reward = self.metrics.get_avg_penalty_reward(last_n=log_interval_window)
                avg_rl_reward = self.metrics.get_avg_rl_reward(last_n=log_interval_window)

                avg_step_time = self.metrics.get_avg_step_time(last_n=log_interval_window)

                log_msg = f"Epoch {epoch+1}, Step {self.metrics.total_steps}, Rank {dist.get_rank() if self.is_distributed else 0}"
                log_msg += f", Loss: {combined_loss_item:.4f}"
                log_msg += f", AvgLoss(L): {avg_combined_loss:.4f}" if avg_combined_loss is not None else ""

                if self.config.use_levenshtein_task:
                    lm_log = f"{avg_lm_comp:.4f}" if avg_lm_comp is not None else "N/A"
                    log_msg += f", AvgLM(L): {lm_log}"

                    rank_log = f"{avg_rank_aux:.4f}" if avg_rank_aux is not None else "N/A"
                    log_msg += f", AvgRankReg(L): {rank_log}"

                    nsp_log = f"{avg_nsp_loss:.4f}" if avg_nsp_loss is not None else "N/A"
                    log_msg += f", AvgNSP(L): {nsp_log}"

                    span_log = f"{avg_span_loss:.4f}" if avg_span_loss is not None else "N/A" # New
                    log_msg += f", AvgSpan(L): {span_log}" # New

                    # Add RL reward to log message
                    rl_reward_log = f"{avg_rl_reward:.4f}" if avg_rl_reward is not None else "N/A"
                    log_msg += f", AvgRLReward: {rl_reward_log}"

                    penalty_log = f"{avg_penalty_reward:.4f}" if avg_penalty_reward is not None else (f"{current_penalty_reward:.4f}" if current_penalty_reward is not None else "N/A")
                    log_msg += f", AvgPenalty: {penalty_log}"

                log_msg += f", LR: {current_lr:.6f}, Step Time: {avg_step_time:.3f}s"
                print(log_msg)
            
            if (not self.is_distributed or dist.get_rank() == 0) and \
               val_loader is not None and \
               self.metrics.total_steps % self.config.eval_every == 0:
                val_loss = self.evaluate(val_loader)
                is_best = val_loss < self.metrics.best_val_loss
                print(f"Validation Loss (Rank {dist.get_rank() if self.is_distributed else 0}): {val_loss:.4f} {'(Best!)' if is_best else ''}")
                if self.config.use_levenshtein_task: # Multi-task mode
                    if self.metrics.val_lm_losses: print(f"  Val LM Comp: {self.metrics.val_lm_losses[-1]:.4f}")
                    if self.metrics.val_rank_aux_losses: print(f"  Val RankReg Aux: {self.metrics.val_rank_aux_losses[-1]:.4f}")
                    if self.metrics.val_nsp_losses: print(f"  Val NSP: {self.metrics.val_nsp_losses[-1]:.4f}")
                    if self.metrics.val_span_selection_losses: print(f"  Val SpanSelect: {self.metrics.val_span_selection_losses[-1]:.4f}") # New
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

        if self.config.use_levenshtein_task: # Multi-task mode
            if self.metrics.rank_aux_losses:
                 ax_flat[plot_idx].plot(self.metrics.rank_aux_losses, 'c-', alpha=0.7, label='RankReg Aux Loss (Train)')
            if self.metrics.nsp_losses:
                 ax_flat[plot_idx].plot(self.metrics.nsp_losses, 'y-', alpha=0.7, label='NSP Aux Loss (Train)')
            if self.metrics.span_selection_losses:
                 ax_flat[plot_idx].plot(self.metrics.span_selection_losses, 'brown', linestyle='-', alpha=0.7, label='SpanSelect Aux Loss (Train)')
            ax_flat[plot_idx].set_title('Auxiliary Task Losses (Training)')
            ax_flat[plot_idx].set_xlabel('Step'); ax_flat[plot_idx].set_ylabel('Loss'); ax_flat[plot_idx].grid(True, alpha=0.3); ax_flat[plot_idx].legend()
        plot_idx += 1

        if self.config.use_levenshtein_task: # For validation metrics as well
            val_steps = np.arange(len(self.metrics.val_losses)) * self.config.eval_every
            ax_val_main = ax_flat[plot_idx]

            if self.metrics.val_lm_losses:
                 ax_val_main.plot(val_steps[:len(self.metrics.val_lm_losses)], self.metrics.val_lm_losses, 'r--', alpha=0.7, label='LM Loss Comp (Val)')
            if self.metrics.val_rank_aux_losses:
                ax_val_main.plot(val_steps[:len(self.metrics.val_rank_aux_losses)], self.metrics.val_rank_aux_losses, 'm--', alpha=0.7, label='RankReg Aux Loss (Val)')
            if self.metrics.val_nsp_losses:
                 ax_val_main.plot(val_steps[:len(self.metrics.val_nsp_losses)], self.metrics.val_nsp_losses, 'orange', linestyle='--', alpha=0.7, label='NSP Aux Loss (Val)')
            if self.metrics.val_span_selection_losses:
                 ax_val_main.plot(val_steps[:len(self.metrics.val_span_selection_losses)], self.metrics.val_span_selection_losses, 'brown', linestyle='--', alpha=0.7, label='SpanSelect Aux Loss (Val)')

            ax_val_main.set_title('Validation Loss Components')
            ax_val_main.set_xlabel('Step'); ax_val_main.set_ylabel('Loss'); ax_val_main.grid(True, alpha=0.3)
            ax_val_main.legend(loc='best')
        plot_idx += 1
        
        if self.metrics.learning_rates:
            ax_flat[plot_idx].plot(self.metrics.learning_rates, 'darkorange', alpha=0.7, label='Learning Rate')
            ax_flat[plot_idx].set_title('Learning Rate')
            ax_flat[plot_idx].set_xlabel('Step'); ax_flat[plot_idx].set_ylabel('LR'); ax_flat[plot_idx].grid(True, alpha=0.3); ax_flat[plot_idx].legend()
        plot_idx += 1

        if self.metrics.rl_rewards:
            ax_flat[plot_idx].plot(self.metrics.rl_rewards, 'purple', alpha=0.7, label='RL Self-Critique Reward')
            ax_flat[plot_idx].set_title('RL Reward (Higher is Better)')
            ax_flat[plot_idx].set_xlabel('Step'); ax_flat[plot_idx].set_ylabel('Reward (-MSE)'); ax_flat[plot_idx].grid(True, alpha=0.3); ax_flat[plot_idx].legend()
        plot_idx += 1
        
        for i in range(plot_idx, len(ax_flat)): fig.delaxes(ax_flat[i]) # Remove unused subplots
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_sample(self, prompt: str = "", task_id_byte: int = TASK_ID_LM,
                        max_length: int = 100, temperature: float = 0.8,
                        top_k: int = 50, top_p: float | None = None):
        # TASK_ID_LM should be defined at module level now

        if not self.data_builder:
            print("Warning: DataBuilder not available in Trainer, cannot generate sample.")
            return ""
        if not hasattr(self.data_builder, 'cls_token_id') or self.data_builder.cls_token_id is None:
            raise ValueError("DataBuilder does not have a CLS token ID, which is needed for generation prefix.")
        if not hasattr(self.data_builder, '_tokenize_text'):
            print("Warning: DataBuilder does not have _tokenize_text method; cannot generate sample.")
            return ""

        cls_id = self.data_builder.cls_token_id

        if isinstance(prompt, dict):
            content_prompt_string = prompt.get("prompt", "").strip()
        elif isinstance(prompt, str):
            content_prompt_string = prompt.strip()
        else:
            content_prompt_string = ""

        # Check if the (stripped) prompt string starts with "[CLS]"
        # Case sensitive check for "[CLS]"
        cls_prefix_str = "[CLS]"
        if content_prompt_string.startswith(cls_prefix_str):
            # Remove the "[CLS]" prefix and any immediate space after it
            content_prompt_string = content_prompt_string[len(cls_prefix_str):].lstrip()

        # Tokenize the (potentially modified) content_prompt_string
        content_tokens = self.data_builder._tokenize_text(content_prompt_string) if content_prompt_string else []

        # Always start with [task_id_byte, cls_id]
        final_initial_tokens = [task_id_byte, cls_id] + content_tokens

        x = torch.tensor([final_initial_tokens], dtype=torch.long).to(self.config.device)

        self.model.eval()
        model_to_generate_from = self.model.module if isinstance(self.model, DDP) else self.model
        with torch.no_grad():
            autocast_context = torch.amp.autocast('cuda') if self.config.use_amp else contextlib.suppress()
            with autocast_context:
                generated_ids = model_to_generate_from.generate(
                    idx=x,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
            generated_text = self.data_builder.decode_tokens(generated_ids[0])
        self.model.train()
        return generated_text
    
    def calculate_perplexity(self, dataloader, max_batches=None):
        self.model.eval()
        total_loss, num_batches = 0, 0
        # lm_ignore_idx for perplexity calculation should be consistent with training
        lm_ignore_idx_ppl = getattr(self.config, 'lm_ignore_idx', -1)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches: break

                if self.config.use_levenshtein_task : # Multi-task batch
                    # Unpack 6 items, ignoring the ones not needed for perplexity
                    input_ids, next_token_lm_targets, _, _, task_type_flags, _ = batch
                    # For perplexity, only consider pure LM task items (type 0)
                    lm_mask = (task_type_flags == TASK_ID_LM)
                    if not lm_mask.any(): continue # Skip batch if no LM items
                    input_ids = input_ids[lm_mask]
                    # For perplexity, targets are next_token_lm_targets for pure LM items
                    targets_for_ppl = next_token_lm_targets[lm_mask]
                else: # Single-task LM batch
                    input_ids, targets_for_ppl = batch

                input_ids = input_ids.to(self.config.device)
                targets_for_ppl = targets_for_ppl.to(self.config.device)
                
                autocast_context = torch.amp.autocast('cuda') if self.config.use_amp and self.config.scaler is not None else contextlib.suppress()
                with autocast_context:
                    # Model's forward now returns a dictionary.
                    model_outputs = self.model(input_ids, targets=None, force_disable_prefix_attention=True)
                    lm_logits = model_outputs['lm_logits']

                # Calculate loss for perplexity using the logits and targets_for_ppl
                # Ensure ignore_index is correctly applied
                loss = F.cross_entropy(
                    lm_logits.view(-1, lm_logits.size(-1)),
                    targets_for_ppl.view(-1),
                    ignore_index=lm_ignore_idx_ppl
                )
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    num_batches += 1
        self.model.train()
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return math.exp(avg_loss) if avg_loss != float('inf') else float('inf')
    
    def generate_inference_sample(self, prompts: Optional[List[Dict[str, Any]]] = None,
                                max_length: Optional[int] = None,
                                temperature: Optional[float] = None,
                                top_k: Optional[int] = None,
                                top_p: Optional[float | None] = None):
        # Ensure module-level TASK_ID constants are accessible: TASK_ID_LM, TASK_ID_UNSHUFFLE, TASK_ID_NSP

        if not self.data_builder:
            return [{"type": "error", "prompt": "N/A", "text": "No data_builder provided."}]

        # Use prompts from config if not provided, otherwise use provided prompts
        # self.config.inference_prompts is now expected to be a list of dicts
        prompts_to_use = prompts if prompts is not None else self.config.inference_prompts

        # Use generation parameters from config if not overridden by method arguments
        max_len = max_length if max_length is not None else self.config.inference_max_length
        temp = temperature if temperature is not None else self.config.inference_temperature
        tk = top_k if top_k is not None else self.config.inference_top_k
        tp = top_p if top_p is not None else self.config.inference_top_p # top_p from config can be None

        all_generated_outputs = []

        # Task string to byte ID mapping
        task_to_id_map = {
            "lm": TASK_ID_LM,
            "unshuffle": TASK_ID_UNSHUFFLE,
            "nsp": TASK_ID_NSP
            # Add other task string identifiers if they are used in config
        }

        self.model.eval() # Ensure model is in eval mode

        for prompt_entry in prompts_to_use:
            if not isinstance(prompt_entry, dict) or "prompt" not in prompt_entry or "task" not in prompt_entry:
                all_generated_outputs.append({
                    'type': 'error',
                    'prompt': str(prompt_entry),
                    'text': 'Invalid prompt entry format in config. Expected {"task": "name", "prompt": "text"}.'
                })
                continue

            prompt_text = prompt_entry["prompt"]
            task_str = prompt_entry["task"].lower() # Ensure lowercase for map lookup
            prompt_type = prompt_entry.get("type", "standard") # Get type, default to "standard"

            task_id_byte = task_to_id_map.get(task_str)
            if task_id_byte is None:
                all_generated_outputs.append({
                    'type': prompt_type,
                    'prompt': prompt_text,
                    'text': f"Generation failed: Unknown task type '{task_str}' in prompt entry."
                })
                continue

            try:
                generated_text = self.generate_sample(
                    prompt=prompt_entry.get("prompt", ""),
                    task_id_byte=task_id_byte,
                    max_length=max_len,
                    temperature=temp,
                    top_k=tk,
                    top_p=tp
                )
                all_generated_outputs.append({
                    'type': prompt_type,
                    'prompt': prompt_entry.get("prompt", ""),
                    'text': generated_text
                })
            except Exception as e:
                all_generated_outputs.append({
                    'type': prompt_type,
                    'prompt': prompt_text,
                    'text': f"Generation failed: {str(e)}"
                })

        self.model.train() # Return model to train mode
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
