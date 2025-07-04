import torch
from torch.utils.data import Dataset
import random
from typing import List, Tuple, Union
from levenshtein_dataset import LevenshteinDataset
from nsp_dataset_new import NSPDataset


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
                 task_distribution: Tuple[float, float, float] = (0.25, 0.25, 0.5)):
        """
        Args:
            raw_documents: List of documents/sentences
            tokenizer_fn: Function to convert text to list of token IDs
            seq_len: Maximum sequence length
            cls_token_id: ID for CLS token
            sep_token_id: ID for SEP token
            lm_ignore_idx: Token ID to use for ignoring positions in LM loss
            input_pad_id: Token ID for padding input sequences
            task_distribution: (levenshtein_ratio, nsp_ratio, lm_ratio)
        """
        self.tokenizer_fn = tokenizer_fn
        self.seq_len = seq_len
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.lm_ignore_idx = lm_ignore_idx
        self.input_pad_id = input_pad_id
        
        # Task distribution
        self.lev_ratio, self.nsp_ratio, self.lm_ratio = task_distribution
        if abs(sum(task_distribution) - 1.0) > 1e-6:
            raise ValueError("Task distribution must sum to 1.0")
        
        # Create individual datasets
        self.levenshtein_dataset = LevenshteinDataset(
            raw_documents, tokenizer_fn, seq_len, cls_token_id, 
            lm_ignore_idx, input_pad_id, shuffle_percentage=1.0  # All shuffled for Lev task
        )
        
        self.nsp_dataset = NSPDataset(
            raw_documents, tokenizer_fn, seq_len, cls_token_id, sep_token_id,
            lm_ignore_idx, input_pad_id
        )
        
        # Standard LM dataset (sentences without shuffling)
        self.raw_documents = raw_documents
        
        # Length is based on the document count
        self.length = len(raw_documents)
        
    def _create_lm_sample(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a standard language modeling sample."""
        # Tokenize text
        tokens = self.tokenizer_fn(text)
        
        # Add CLS token at the beginning
        tokens_with_cls = [self.cls_token_id] + tokens
        
        # Truncate if too long
        if len(tokens_with_cls) > self.seq_len:
            tokens_with_cls = tokens_with_cls[:self.seq_len]
        
        # Create input and target sequences
        input_tokens = tokens_with_cls[:-1] if len(tokens_with_cls) > 1 else tokens_with_cls
        target_tokens = tokens_with_cls[1:] if len(tokens_with_cls) > 1 else [self.lm_ignore_idx]
        
        # Pad to sequence length
        input_tokens = self._pad_tokens(input_tokens, self.seq_len, self.input_pad_id)
        target_tokens = self._pad_tokens(target_tokens, self.seq_len, self.lm_ignore_idx)
        
        return (
            torch.tensor(input_tokens, dtype=torch.long),
            torch.tensor(target_tokens, dtype=torch.long),
            torch.tensor(0.0, dtype=torch.float),  # No auxiliary task value
            torch.tensor(0.0, dtype=torch.float),  # Task type flag (0.0 for LM)
        )
    
    def _pad_tokens(self, tokens: List[int], max_len: int, pad_id: int) -> List[int]:
        """Pad tokens to max_len."""
        if len(tokens) > max_len:
            return tokens[:max_len]
        return tokens + [pad_id] * (max_len - len(tokens))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Randomly choose which task to use
        task_choice = random.random()
        
        if task_choice < self.lev_ratio:
            # Levenshtein task
            lev_idx = idx % len(self.levenshtein_dataset)
            return self.levenshtein_dataset[lev_idx]
        elif task_choice < self.lev_ratio + self.nsp_ratio:
            # NSP task
            nsp_idx = idx % len(self.nsp_dataset)
            return self.nsp_dataset[nsp_idx]
        else:
            # Standard LM task
            doc_idx = idx % len(self.raw_documents)
            return self._create_lm_sample(self.raw_documents[doc_idx])