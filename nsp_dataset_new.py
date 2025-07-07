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
            first_sent, _, _ = shuffle_words_in_sentence(sent_a)
            second_sent, _, _ = shuffle_words_in_sentence(sent_b)
        
        # Tokenize sentences
        first_tokens = self.tokenizer_fn(first_sent)
        second_tokens = self.tokenizer_fn(second_sent)
        
        # Create input sequence: [CLS] first_sent [SEP] second_sent [SEP]
        input_tokens = [self.cls_token_id] + first_tokens + [self.sep_token_id] + second_tokens + [self.sep_token_id]
        
        # Truncate and pad
        input_tokens = _truncate_or_pad_tokens(input_tokens, self.seq_len, self.input_pad_id)
        
        # Create targets - all ignored for NSP task (no LM loss)
        lm_targets = [self.lm_ignore_idx] * self.seq_len
        
        return (
            torch.tensor(input_tokens, dtype=torch.long),
            torch.tensor(lm_targets, dtype=torch.long),
            torch.tensor(float(nsp_class), dtype=torch.float),  # NSP class label
            torch.tensor(2.0, dtype=torch.float),  # Task type flag (2.0 for NSP)
        )