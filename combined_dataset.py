import torch
from torch.utils.data import Dataset
import random
from typing import List, Tuple, Union
from levenshtein_dataset import LevenshteinDataset
from nsp_dataset_new import NSPDataset
from span_selection_dataset import SpanSelectionDataset

TASK_ID_LM = 48 # ord('0')

class CombinedMultiTaskDataset(Dataset):
    """
    Combined dataset that mixes three tasks:
    - 25% Levenshtein task (shuffled word order)
    - 25% NSP task (with CLS/SEP tokens)
    - 50% standard teacher forcing (LM task)
    """
    
    def __init__(self,
                 raw_documents: List[str],
                 tokenizer_fn: callable,
                 seq_len: int,
                 cls_token_id: int,
                 sep_token_id: int,
                 lm_ignore_idx: int = -1,
                 input_pad_id: int = 0,
                 task_distribution: Tuple[float, float, float, float] = (0.20, 0.20, 0.40, 0.20)):
        """
        Args:
            raw_documents: List of documents/sentences
            tokenizer_fn: Function to convert text to list of token IDs
            seq_len: Maximum sequence length
            cls_token_id: ID for CLS token
            sep_token_id: ID for SEP token
            lm_ignore_idx: Token ID to use for ignoring positions in LM loss
            input_pad_id: Token ID for padding input sequences
            task_distribution: (rank_ratio, nsp_ratio, lm_ratio, span_selection_ratio)
        """
        self.tokenizer_fn = tokenizer_fn
        self.seq_len = seq_len
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.lm_ignore_idx = lm_ignore_idx
        self.input_pad_id = input_pad_id
        
        # Task distribution
        self.rank_ratio, self.nsp_ratio, self.lm_ratio, self.span_ratio = task_distribution
        if abs(sum(task_distribution) - 1.0) > 1e-6:
            raise ValueError("Task distribution must sum to 1.0")
        
        # Create individual datasets
        self.levenshtein_dataset = LevenshteinDataset(
            raw_documents, tokenizer_fn, seq_len, cls_token_id, 
            lm_ignore_idx, input_pad_id, shuffle_percentage=1.0
        )
        
        self.nsp_dataset = NSPDataset(
            raw_documents, tokenizer_fn, seq_len, cls_token_id, sep_token_id,
            lm_ignore_idx, input_pad_id
        )

        # TODO: Need to get mask_token_id from data_builder
        # For now, let's assume a placeholder value that will be updated.
        # This will require a change in data_builder.py to create and pass this ID.
        placeholder_mask_id = 258
        self.span_selection_dataset = SpanSelectionDataset(
            raw_documents, tokenizer_fn, seq_len,
            mask_token_id=placeholder_mask_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            lm_ignore_idx=lm_ignore_idx,
            input_pad_id=input_pad_id
        )
        
        # Standard LM dataset (sentences without shuffling)
        self.raw_documents = raw_documents
        
        # Length is based on the document count
        self.length = len(raw_documents)

        # Initialize and populate task-specific index lists
        self.lm_indices = []
        self.lev_indices = []
        self.nsp_indices = []

        # Ratios from task_distribution: (lev_ratio, nsp_ratio, lm_ratio)
        # self.lev_ratio = task_distribution[0] (e.g. 0.25)
        # self.nsp_ratio = task_distribution[1] (e.g. 0.25)
        # self.lm_ratio  = task_distribution[2] (e.g. 0.50)
        
        # The current __getitem__ logic determines task type based on index in a cycle.
        # To accommodate (0.2, 0.2, 0.4, 0.2), a cycle length of 10 works well.
        cycle_length = 10
        
        # Calculate number of items for each task in a cycle based on ratios
        num_rank_in_cycle = round(cycle_length * self.rank_ratio) # 2
        num_nsp_in_cycle = round(cycle_length * self.nsp_ratio)   # 2
        num_span_in_cycle = round(cycle_length * self.span_ratio) # 2
        # num_lm_in_cycle = cycle_length - num_rank_in_cycle - num_nsp_in_cycle - num_span_in_cycle # 4
        
        # Boundaries for assigning indices to task types
        rank_boundary = num_rank_in_cycle
        nsp_boundary = rank_boundary + num_nsp_in_cycle
        span_boundary = nsp_boundary + num_span_in_cycle

        # This part of the code that pre-calculates index lists is no longer needed
        # as the task is determined dynamically in __getitem__. We can remove it.
        # self.lm_indices = [] ... etc.
        
    def _create_lm_sample(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a standard language modeling sample with task ID and 5-tuple output."""
        # Tokenize text
        tokens = self.tokenizer_fn(text)
        
        # 1. Prepare input_tokens (with TASK_ID_LM and CLS)
        content_tokens = tokens # actual content tokens from tokenizer_fn(text)
        max_content_len = self.seq_len - 2 # For TASK_ID_LM and CLS
        if len(content_tokens) > max_content_len:
            content_tokens = content_tokens[:max_content_len]
        
        final_input_tokens_list = [TASK_ID_LM, self.cls_token_id] + content_tokens
        padded_input_tokens = self._pad_tokens(final_input_tokens_list, self.seq_len, self.input_pad_id)

        # 2. Prepare next_token_lm_targets
        # Targets are shifted version of final_input_tokens_list.
        # Prediction for TASK_ID_LM is CLS. Prediction for CLS is first content token.
        # Last token in final_input_tokens_list predicts lm_ignore_idx.
        temp_lm_targets = final_input_tokens_list[1:] + [self.lm_ignore_idx]
        next_token_lm_targets_list = self._pad_tokens(temp_lm_targets, self.seq_len, self.lm_ignore_idx)

        # 3. Prepare placeholder for sequence targets (float, all ignore_idx)
        # This corresponds to rank_regression_targets from LevenshteinDataset
        # Needs to be float32 to match the dtype from LevenshteinDataset's rank targets.
        rank_ignore_val_float = float(self.lm_ignore_idx)
        aux_sequence_targets_list = [rank_ignore_val_float] * self.seq_len

        # 4. Auxiliary scalar value (placeholder)
        auxiliary_scalar_value_tensor = torch.tensor(0.0, dtype=torch.float)

        # 5. Task type flag
        task_type_flag_tensor = torch.tensor(0.0, dtype=torch.float) # Type 0 for LM

        # 6. Prepare true ranks of the original text for the RL self-critique step
        # This needs to align with the *input* tokens, not the target tokens.
        # The ranks are for the characters in `final_input_tokens_list`.
        true_ranks_of_original = []
        if len(text) > 0:
            # The first two tokens are [TASK_ID, CLS], they don't have a rank from the text.
            # We can assign them an ignore value.
            true_ranks_of_original.append(rank_ignore_val_float) # Rank for TASK_ID
            true_ranks_of_original.append(rank_ignore_val_float) # Rank for CLS

            # Ranks for the actual content tokens
            for i in range(len(content_tokens)):
                rank = (i + 1.0) / len(text) # 1-based normalized rank
                true_ranks_of_original.append(rank)

        # Pad the ranks list to seq_len
        padded_true_ranks = true_ranks_of_original + [rank_ignore_val_float] * (self.seq_len - len(true_ranks_of_original))
        original_text_ranks_tensor = torch.tensor(padded_true_ranks, dtype=torch.float32)

        return (
            torch.tensor(padded_input_tokens, dtype=torch.long),
            torch.tensor(next_token_lm_targets_list, dtype=torch.long),
            torch.tensor(aux_sequence_targets_list, dtype=torch.float32),
            auxiliary_scalar_value_tensor,
            task_type_flag_tensor,
            original_text_ranks_tensor # New 6th item
        )
    
    def _pad_tokens(self, tokens: List[int], max_len: int, pad_id: int) -> List[int]:
        """Pad tokens to max_len."""
        if len(tokens) > max_len:
            return tokens[:max_len]
        return tokens + [pad_id] * (max_len - len(tokens))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Deterministic task selection based on index to ensure proper batch composition.
        # This creates a repeating pattern that ensures the correct distribution.
        # For (0.2, 0.2, 0.4, 0.2) on a cycle of 10:
        # Rank: 0, 1 | NSP: 2, 3 | Span: 4, 5 | LM: 6, 7, 8, 9
        
        cycle_length = 10
        position_in_cycle = idx % cycle_length
        
        rank_boundary = int(cycle_length * self.rank_ratio)
        nsp_boundary = rank_boundary + int(cycle_length * self.nsp_ratio)
        span_boundary = nsp_boundary + int(cycle_length * self.span_ratio)

        if position_in_cycle < rank_boundary:
            # Rank Regression Task
            if len(self.levenshtein_dataset) > 0:
                return self.levenshtein_dataset[idx % len(self.levenshtein_dataset)]
        
        elif position_in_cycle < nsp_boundary:
            # NSP Task
            if len(self.nsp_dataset) > 0:
                return self.nsp_dataset[idx % len(self.nsp_dataset)]

        elif position_in_cycle < span_boundary:
            # Span Selection Task
            if len(self.span_selection_dataset) > 0:
                return self.span_selection_dataset[idx % len(self.span_selection_dataset)]

        # Default to LM task, which also serves as a fallback
        return self._create_lm_sample(self.raw_documents[idx % len(self.raw_documents)])

    def update_lev_shuffle_parameters(self, min_p: float, max_p: float):
        if hasattr(self, 'levenshtein_dataset') and \
           hasattr(self.levenshtein_dataset, 'update_shuffle_range') and \
           callable(getattr(self.levenshtein_dataset, 'update_shuffle_range')):
            self.levenshtein_dataset.update_shuffle_range(min_p, max_p)
        else:
            # This warning helps if LevenshteinDataset's structure changes or is missing
            print("Warning: CombinedMultiTaskDataset's Levenshtein dataset not found or does not support shuffle range update.")