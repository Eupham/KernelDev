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

        self.sentences = raw_documents_or_sentences # Store raw data directly
        # self.examples list and call to _prepare_examples are removed

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        original_sentence_text = self.sentences[idx]

        # Handle potential empty strings if not filtered upstream, though DataBuilder tries to.
        if not original_sentence_text.strip():
            # Return a "degenerate" sample that should ideally be filtered by a collate_fn
            # or represent a very small loss.
            # All tokens will be padding, CLS will be masked in targets.
            # Distances will be 0.
            original_sentence_text = "" # Ensure it's empty string not just whitespace

        # 1. Shuffle words and calculate Levenshtein distance
        shuffled_sentence_text, original_words, shuffled_words = shuffle_words_in_sentence(original_sentence_text)

        true_target_lev_dist = float(word_levenshtein_distance(original_words, shuffled_words))

        # Normalize the Levenshtein distance for the shuffled sentence
        num_original_words = len(original_words)
        if num_original_words > 0:
            normalized_target_lev_dist_for_shuffled = true_target_lev_dist / num_original_words
        else:
            normalized_target_lev_dist_for_shuffled = 0.0

        target_coherence_score = 0.0  # Target for original sentence's coherence score is always 0

        # 2. Tokenize, Prepend CLS, Truncate/Pad for Original Sentence
        original_tokens = self.tokenizer_fn(original_sentence_text)
        original_tokens_cls_list = [self.cls_token_id] + original_tokens
        # Truncate content if too long, keeping CLS at the beginning
        if len(original_tokens_cls_list) > self.seq_len:
            final_original_tokens_list = [self.cls_token_id] + original_tokens_cls_list[1:self.seq_len]
        else:
            final_original_tokens_list = original_tokens_cls_list
        padded_original_tokens = _truncate_or_pad_tokens(final_original_tokens_list, self.seq_len, self.input_pad_id)

        # 3. Create LM Targets for Original Sentence
        temp_lm_targets = []
        for i in range(self.seq_len):
            # Target for token at padded_original_tokens[i] is padded_original_tokens[i+1]
            # unless it's CLS, current is padding, or next is padding
            if i < len(padded_original_tokens) - 1:
                is_current_cls = (i == 0 and padded_original_tokens[i] == self.cls_token_id)
                is_current_pad = (padded_original_tokens[i] == self.input_pad_id)
                is_next_pad = (padded_original_tokens[i+1] == self.input_pad_id)

                if is_current_cls or is_current_pad or is_next_pad:
                    temp_lm_targets.append(self.lm_ignore_idx)
                else:
                    temp_lm_targets.append(padded_original_tokens[i+1])
            else: # Current token is the last in seq_len or already into padding territory
                temp_lm_targets.append(self.lm_ignore_idx)

        # Ensure lm_targets_padded is exactly seq_len
        # temp_lm_targets should already be seq_len due to loop range, but an explicit truncate/pad is safer.
        lm_targets_padded = _truncate_or_pad_tokens(temp_lm_targets[:self.seq_len], self.seq_len, self.lm_ignore_idx)


        # 4. Tokenize, Prepend CLS, Truncate/Pad for Shuffled Sentence
        shuffled_tokens = self.tokenizer_fn(shuffled_sentence_text)
        shuffled_tokens_cls_list = [self.cls_token_id] + shuffled_tokens
        # Truncate content if too long, keeping CLS
        if len(shuffled_tokens_cls_list) > self.seq_len:
            final_shuffled_tokens_list = [self.cls_token_id] + shuffled_tokens_cls_list[1:self.seq_len]
        else:
            final_shuffled_tokens_list = shuffled_tokens_cls_list
        padded_shuffled_tokens = _truncate_or_pad_tokens(final_shuffled_tokens_list, self.seq_len, self.input_pad_id)

        # 5. Return Tensors
        return (
            torch.tensor(padded_original_tokens, dtype=torch.long),
            torch.tensor(lm_targets_padded, dtype=torch.long),
            torch.tensor(padded_shuffled_tokens, dtype=torch.long),
            torch.tensor(normalized_target_lev_dist_for_shuffled, dtype=torch.float32),
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
