import torch
from torch.utils.data import Dataset
import re # For sentence splitting if needed, though text_utils handles word splitting

# Assuming text_utils.py is in the same directory or accessible in PYTHONPATH
try:
    from text_utils import shuffle_words_in_sentence, word_levenshtein_distance
except ImportError:
    # Fallback for environments where text_utils might not be directly importable
    # This is a simplified placeholder; direct import is preferred.
    print("Warning: text_utils.py not found, using placeholder shuffle/distance functions.")
    def shuffle_words_in_sentence(s): return s, s.split(), s.split()
    def word_levenshtein_distance(s1, s2): return len(s1) + len(s2)


def _truncate_or_pad_tokens(tokens: list[int], max_len: int, pad_id: int) -> list[int]:
    if len(tokens) > max_len:
        return tokens[:max_len]
    return tokens + [pad_id] * (max_len - len(tokens))

class LevenshteinDataset(Dataset):
    def __init__(self,
                 raw_documents_or_sentences: list[str],
                 tokenizer_fn: callable,
                 seq_len: int,
                 cls_token_id: int,
                 lm_ignore_idx: int = -1, # For masking targets
                 input_pad_id: int = 0):  # For padding input token sequences
        """
        Args:
            raw_documents_or_sentences: List of strings (sentences or short documents).
            tokenizer_fn: Function to convert text to list of token IDs.
            seq_len: Maximum sequence length for model inputs.
            cls_token_id: Integer ID for the CLS token.
            lm_ignore_idx: Token ID to use for ignoring positions in LM loss.
            input_pad_id: Token ID for padding input sequences.
        """
        self.tokenizer_fn = tokenizer_fn
        self.seq_len = seq_len
        self.cls_token_id = cls_token_id
        self.lm_ignore_idx = lm_ignore_idx
        self.input_pad_id = input_pad_id

        self.examples = []
        self._prepare_examples(raw_documents_or_sentences)

    def _prepare_examples(self, documents: list[str]):
        for doc_text in documents:
            if not doc_text.strip():
                continue

            # For this dataset, we treat each document/line as a single "sentence"
            # for shuffling. If finer-grained sentence splitting is needed from
            # longer documents, it should happen before passing to this Dataset.
            original_sentence_text = doc_text

            shuffled_sentence_text, original_words, shuffled_words = shuffle_words_in_sentence(original_sentence_text)

            # Only include if there are words to process
            if not original_words:
                continue

            true_lev_dist = word_levenshtein_distance(original_words, shuffled_words)

            self.examples.append({
                "original_text": original_sentence_text,
                "shuffled_text": shuffled_sentence_text,
                "levenshtein_distance": float(true_lev_dist) # Target for shuffled
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        original_text = example["original_text"]
        shuffled_text = example["shuffled_text"]
        target_lev_dist = example["levenshtein_distance"]

        # Process Original Sentence
        original_tokens = self.tokenizer_fn(original_text)
        # Prepend CLS, then truncate/pad to ensure CLS is [0] if sequence is too long AFTER CLS
        original_tokens_cls_list = [self.cls_token_id] + original_tokens
        if len(original_tokens_cls_list) > self.seq_len:
            # Ensure CLS is always the first token, truncate the rest
            final_original_tokens_list = [self.cls_token_id] + original_tokens_cls_list[1:self.seq_len]
        else:
            final_original_tokens_list = original_tokens_cls_list
        padded_original_tokens = _truncate_or_pad_tokens(final_original_tokens_list, self.seq_len, self.input_pad_id)

        # Create LM targets: shift by 1, pad, and mask CLS
        temp_lm_targets = []
        for i in range(self.seq_len):
            if i < len(padded_original_tokens) - 1: # Target for token at padded_original_tokens[i] is padded_original_tokens[i+1]
                current_token_is_cls_at_start = (i == 0 and padded_original_tokens[i] == self.cls_token_id)
                next_token_is_padding = (padded_original_tokens[i+1] == self.input_pad_id)

                if current_token_is_cls_at_start: # Mask target for CLS token
                    temp_lm_targets.append(self.lm_ignore_idx)
                elif padded_original_tokens[i] == self.input_pad_id: # If current input token is padding, its target is also ignore
                    temp_lm_targets.append(self.lm_ignore_idx)
                elif next_token_is_padding : # If next token (target) would be padding, also ignore
                    temp_lm_targets.append(self.lm_ignore_idx)
                else:
                    temp_lm_targets.append(padded_original_tokens[i+1])
            else: # Handles the last token (which has no next token) and any explicit padding
                temp_lm_targets.append(self.lm_ignore_idx)

        # Ensure it's exactly seq_len by truncating if somehow longer, then padding
        lm_targets_padded = _truncate_or_pad_tokens(temp_lm_targets[:self.seq_len], self.seq_len, self.lm_ignore_idx)


        # Process Shuffled Sentence
        shuffled_tokens = self.tokenizer_fn(shuffled_text)
        shuffled_tokens_cls_list = [self.cls_token_id] + shuffled_tokens
        if len(shuffled_tokens_cls_list) > self.seq_len:
            # Ensure CLS is always the first token
            final_shuffled_tokens_list = [self.cls_token_id] + shuffled_tokens_cls_list[1:self.seq_len]
        else:
            final_shuffled_tokens_list = shuffled_tokens_cls_list
        padded_shuffled_tokens = _truncate_or_pad_tokens(final_shuffled_tokens_list, self.seq_len, self.input_pad_id)

        # Target for original sentence's coherence score (predicted by Levenshtein head) is 0.0
        target_coherence_score = 0.0

        return (
            torch.tensor(padded_original_tokens, dtype=torch.long),
            torch.tensor(lm_targets_padded, dtype=torch.long),
            torch.tensor(padded_shuffled_tokens, dtype=torch.long),
            torch.tensor(target_lev_dist, dtype=torch.float32),
            torch.tensor(target_coherence_score, dtype=torch.float32)
        )

if __name__ == '__main__':
    # Example Usage (requires text_utils.py and a tokenizer)
    print("LevenshteinDataset module loading...")

    # Mock tokenizer for testing
    def mock_tokenizer(text: str) -> list[int]:
        return [ord(c) for c in text[:50]] # Simple tokenizer for testing, limit length

    cls_id = 256 # Example CLS ID
    pad_id = 0   # Example Input PAD ID
    lm_ignore = -1 # LM ignore index
    seq_len = 30

    sample_sentences = [
        "This is the first sentence for testing.",
        "Another example sentence is here.",
        "Short one.",
        "A very very very very very very long sentence to test truncation of everything properly.",
        "Test with CLS in text should not be an issue for tokenizer." # CLS here is text, not id
    ]

    dataset = LevenshteinDataset(sample_sentences, mock_tokenizer, seq_len, cls_id, lm_ignore_idx=lm_ignore, input_pad_id=pad_id)
    print(f"Created dataset with {len(dataset)} examples.")

    if len(dataset) > 0:
        for i in range(min(len(dataset), 2)): # Print first 2 samples
            orig_tok, lm_tgt, shuf_tok, lev_dist, coh_score = dataset[i]
            print(f"--- Sample {i} ---")
            print(f"  Original Tokens (CLS): {orig_tok.tolist()}")
            print(f"  LM Targets:            {lm_tgt.tolist()}")
            print(f"  Shuffled Tokens (CLS): {shuf_tok.tolist()}")
            print(f"  Target Levenshtein Dist (for shuffled): {lev_dist.item()}")
            print(f"  Target Coherence Score (for original):  {coh_score.item()}")

            # Try to decode parts for readability
            # This requires a mock_detokenizer or knowledge of how CLS/PAD are handled by it
            print(f"  Original (approx): CLS + '{''.join([chr(t) for t in orig_tok.tolist()[1:15] if t!=pad_id and t!=cls_id])}'...")
            print(f"  Shuffled (approx): CLS + '{''.join([chr(t) for t in shuf_tok.tolist()[1:15] if t!=pad_id and t!=cls_id])}'...")
