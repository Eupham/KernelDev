import torch
from torch.utils.data import Dataset
import random
import re
from typing import List, Tuple

TASK_ID_SPAN_SELECT = ord('3')

def _pad_tokens(tokens: List[int], max_len: int, pad_id: int) -> List[int]:
    """Pad or truncate tokens to a specific length."""
    if len(tokens) > max_len:
        return tokens[:max_len]
    return tokens + [pad_id] * (max_len - len(tokens))

class SpanSelectionDataset(Dataset):
    """
    Dataset for the Span Reconstruction / Candidate Selection task.
    This version uses a concatenation approach for the input.
    Input format: [TASK_ID] [CLS] masked_text [SEP] cand1 [SEP] cand2 ... [SEP] cand_n
    """
    def __init__(self,
                 raw_documents: List[str],
                 tokenizer_fn: callable,
                 seq_len: int,
                 n_candidates: int = 4,
                 min_span_len: int = 5,
                 max_span_len: int = 50,
                 mask_token_id: int = -1,
                 cls_token_id: int = -1,
                 sep_token_id: int = -1,
                 lm_ignore_idx: int = -1,
                 input_pad_id: int = 0):

        if mask_token_id == -1 or cls_token_id == -1 or sep_token_id == -1:
            raise ValueError("mask_token_id, cls_token_id, and sep_token_id must be provided.")

        self.documents = [doc for doc in raw_documents if len(doc) > (min_span_len * 2)]
        self.tokenizer_fn = tokenizer_fn
        self.seq_len = seq_len
        self.n_candidates = n_candidates
        self.min_span_len = min_span_len
        self.max_span_len = max_span_len
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.lm_ignore_idx = lm_ignore_idx
        self.input_pad_id = input_pad_id

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        # 1. Select a primary document and a random span to cut
        main_doc = self.documents[idx]

        actual_max_span_len = min(self.max_span_len, len(main_doc) - self.min_span_len - 1)
        if actual_max_span_len < self.min_span_len:
            return self.__getitem__(random.randint(0, len(self) - 1))

        span_len = random.randint(self.min_span_len, actual_max_span_len)
        span_start = random.randint(0, len(main_doc) - span_len)

        true_span_text = main_doc[span_start : span_start + span_len]

        # 2. Find n-1 distractor spans
        candidate_spans_text = [true_span_text]
        distractor_indices = random.sample(range(len(self.documents)), self.n_candidates)

        for dist_idx in distractor_indices:
            if len(candidate_spans_text) >= self.n_candidates:
                break
            if dist_idx == idx: continue

            dist_doc = self.documents[dist_idx]
            if len(dist_doc) < span_len: continue

            dist_span_start = random.randint(0, len(dist_doc) - span_len)
            distractor_span = dist_doc[dist_span_start : dist_span_start + span_len]
            candidate_spans_text.append(distractor_span)

        while len(candidate_spans_text) < self.n_candidates:
            candidate_spans_text.append("...")

        # 3. Shuffle candidates and find the new target index
        target_index = 0
        shuffled_mapping = list(range(len(candidate_spans_text)))
        random.shuffle(shuffled_mapping)
        shuffled_candidates_text = [candidate_spans_text[i] for i in shuffled_mapping]
        target_index = shuffled_mapping.index(0)

        # 4. Tokenize and create the single concatenated input sequence
        tokenized_masked_text = self.tokenizer_fn(main_doc[:span_start]) + [self.mask_token_id] + self.tokenizer_fn(main_doc[span_start + span_len:])

        full_sequence = [TASK_ID_SPAN_SELECT, self.cls_token_id] + tokenized_masked_text

        for cand_text in shuffled_candidates_text:
            full_sequence += [self.sep_token_id] + self.tokenizer_fn(cand_text)

        padded_input_tokens = _pad_tokens(full_sequence, self.seq_len, self.input_pad_id)

        # 5. Create dummy tensors for the other 5 tuple items to match batch format
        dummy_lm_targets = torch.full((self.seq_len,), self.lm_ignore_idx, dtype=torch.long)
        dummy_rank_targets = torch.full((self.seq_len,), float(self.lm_ignore_idx), dtype=torch.float32)
        auxiliary_scalar_value = torch.tensor(float(target_index), dtype=torch.float32)
        task_type_flag = torch.tensor(3.0, dtype=torch.float32)
        dummy_true_ranks = torch.full((self.seq_len,), float(self.lm_ignore_idx), dtype=torch.float32)

        return (
            torch.tensor(padded_input_tokens, dtype=torch.long),
            dummy_lm_targets,
            dummy_rank_targets,
            auxiliary_scalar_value,
            task_type_flag,
            dummy_true_ranks
        )
