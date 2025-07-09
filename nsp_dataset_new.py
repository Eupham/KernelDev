import torch
from torch.utils.data import Dataset
import random
import re

# Import text utilities for shuffling
try:
    from text_utils import shuffle_words_in_sentence
except ImportError:
    print("Warning: text_utils.py not found, using placeholder shuffle function.")
    def shuffle_words_in_sentence(s):
        words = s.split()
        shuffled_words = words.copy()
        random.shuffle(shuffled_words)
        return ' '.join(shuffled_words), words, shuffled_words

TASK_ID_NSP = 50 # ord('2')

def _truncate_or_pad_tokens(tokens: list[int], max_len: int, pad_id: int) -> list[int]:
    """Truncate or pad tokens to max_len."""
    if len(tokens) > max_len:
        return tokens[:max_len]
    return tokens + [pad_id] * (max_len - len(tokens))


class NSPDataset(Dataset):
    """Dataset for Next Sentence Prediction task with 3 classes."""
    
    def __init__(self,
                 raw_documents: list[str],
                 tokenizer_fn: callable,
                 seq_len: int,
                 cls_token_id: int,
                 sep_token_id: int,
                 lm_ignore_idx: int = -1,
                 input_pad_id: int = 0):
        """
        Args:
            raw_documents: List of documents/paragraphs to split into sentences
            tokenizer_fn: Function to convert text to list of token IDs
            seq_len: Maximum sequence length
            cls_token_id: ID for CLS token
            sep_token_id: ID for SEP token
            lm_ignore_idx: Token ID to use for ignoring positions in LM loss
            input_pad_id: Token ID for padding input sequences
        """
        self.tokenizer_fn = tokenizer_fn
        self.seq_len = seq_len
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.lm_ignore_idx = lm_ignore_idx
        self.input_pad_id = input_pad_id
        
        # Split documents into sentence pairs
        self.sentence_pairs = self._create_sentence_pairs(raw_documents)
        
    def _create_sentence_pairs(self, documents: list[str]) -> list[tuple[str, str]]:
        """Create sentence pairs from documents."""
        pairs = []
        
        for doc in documents:
            # Simple sentence splitting using periods
            sentences = [s.strip() for s in re.split(r'[.!?]+', doc) if s.strip()]
            
            # Create consecutive sentence pairs
            for i in range(len(sentences) - 1):
                pairs.append((sentences[i], sentences[i + 1]))
                
        return pairs
    
    def __len__(self):
        return len(self.sentence_pairs)
    
    def __getitem__(self, idx):
        sent_a, sent_b = self.sentence_pairs[idx]
        
        # Randomly choose NSP class
        nsp_class = random.randint(0, 2)
        
        if nsp_class == 0:  # Correct order
            first_sent = sent_a
            second_sent = sent_b
        elif nsp_class == 1:  # Out of order (reversed)
            first_sent = sent_b
            second_sent = sent_a
        else:  # nsp_class == 2: Garbled/shuffled
            first_sent, _, _ = shuffle_words_in_sentence(sent_a, shuffle_probability=1.0)
            second_sent, _, _ = shuffle_words_in_sentence(sent_b, shuffle_probability=1.0)
        
        # Tokenize sentences
        first_tokens = self.tokenizer_fn(first_sent)
        second_tokens = self.tokenizer_fn(second_sent)
        
        # 1. Prepare input_tokens (with TASK_ID_NSP and CLS)
        content_tokens = first_tokens + [self.sep_token_id] + second_tokens + [self.sep_token_id]
        max_content_len = self.seq_len - 2 # Account for TASK_ID and CLS
        if len(content_tokens) > max_content_len:
            content_tokens = content_tokens[:max_content_len]
        
        final_input_tokens_list = [TASK_ID_NSP, self.cls_token_id] + content_tokens
        padded_input_tokens = _truncate_or_pad_tokens(final_input_tokens_list, self.seq_len, self.input_pad_id)

        # 2. Prepare next_token_lm_targets (all ignore_idx)
        next_token_lm_targets_list = [self.lm_ignore_idx] * self.seq_len

        # 3. Prepare unshuffle_seq_targets (placeholder: all ignore_idx)
        unshuffle_seq_targets_list = [self.lm_ignore_idx] * self.seq_len

        # 4. Auxiliary scalar value (NSP class label)
        auxiliary_scalar_value_tensor = torch.tensor(float(nsp_class), dtype=torch.float)

        # 5. Task type flag
        task_type_flag_tensor = torch.tensor(2.0, dtype=torch.float) # Type 2 for NSP

        return (
            torch.tensor(padded_input_tokens, dtype=torch.long),
            torch.tensor(next_token_lm_targets_list, dtype=torch.long),
            torch.tensor(unshuffle_seq_targets_list, dtype=torch.long),
            auxiliary_scalar_value_tensor,
            task_type_flag_tensor
        )