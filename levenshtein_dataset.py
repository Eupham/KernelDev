import torch
from torch.utils.data import Dataset
import re # For sentence splitting if needed, though text_utils handles word splitting
import random # New import

# Assuming text_utils.py is in the same directory or accessible in PYTHONPATH
try:
    from text_utils import shuffle_words_in_sentence, word_levenshtein_distance
except ImportError:
    # Fallback for environments where text_utils might not be directly importable
    # This is a simplified placeholder; direct import is preferred.
    print("Warning: text_utils.py not found, using placeholder shuffle/distance functions.")
    def shuffle_words_in_sentence(s): return s, s.split(), s.split()
    def word_levenshtein_distance(s1, s2): return len(s1) + len(s2)

TASK_ID_LEV_UNSHUFFLE = 49 # ord('1')

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
                 input_pad_id: int = 0,  # For padding input token sequences
                 shuffle_percentage: float = 0.25): # New parameter
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

        if not 0.0 <= shuffle_percentage <= 1.0:
            raise ValueError("shuffle_percentage must be between 0.0 and 1.0")
        self.shuffle_percentage = shuffle_percentage
        self.current_min_shuffle_p: float = 0.05
        self.current_max_shuffle_p: float = 0.05 # Start with a narrow range at the minimum

        self.sentences = raw_documents_or_sentences # Store raw data directly
        # self.examples list and call to _prepare_examples are removed

    def __len__(self):
        return len(self.sentences)

    def update_shuffle_range(self, min_p: float, max_p: float):
        # Clamp probabilities to be between 0.0 and 1.0
        min_p_clamped = max(0.0, min(min_p, 1.0))
        max_p_clamped = max(0.0, min(max_p, 1.0))

        # Ensure min_p is not greater than max_p after clamping
        self.current_min_shuffle_p = min(min_p_clamped, max_p_clamped)
        self.current_max_shuffle_p = max(min_p_clamped, max_p_clamped)

        # Optional: print for confirmation, can be removed later
        # print(f"LevenshteinDataset: Shuffle prob range updated to [{self.current_min_shuffle_p:.4f}, {self.current_max_shuffle_p:.4f}]")

    def __getitem__(self, idx):
        # TASK_ID_LEV_UNSHUFFLE = 49 # Define or import

        original_sentence_text = self.sentences[idx]
        if not original_sentence_text.strip():
            original_sentence_text = " " # Avoid empty tokenization issues

        selected_probability = random.uniform(self.current_min_shuffle_p, self.current_max_shuffle_p)
        shuffled_sentence_text, original_words_list_for_lev_metric_only, _ = \
            shuffle_words_in_sentence(original_sentence_text, selected_probability)
        # Note: original_words_list_for_lev_metric_only and the third return from shuffle_words_in_sentence
        # are not directly used in the returned tensors if we are removing scalar Lev distance.

        # 1. Prepare input_tokens with Task ID (shuffled content)
        tokens_shuffled_content = self.tokenizer_fn(shuffled_sentence_text)
        max_content_len = self.seq_len - 2 # For TASK_ID and CLS
        if len(tokens_shuffled_content) > max_content_len:
            tokens_shuffled_content = tokens_shuffled_content[:max_content_len]
        final_input_tokens_list = [TASK_ID_LEV_UNSHUFFLE, self.cls_token_id] + tokens_shuffled_content
        padded_input_tokens = _truncate_or_pad_tokens(final_input_tokens_list, self.seq_len, self.input_pad_id)

        # 2. Prepare next_token_lm_targets (all ignore_idx for this task type 1)
        next_token_lm_targets_list = [self.lm_ignore_idx] * self.seq_len

        # 3. Prepare unshuffle_target_tokens (shifted original sentence)
        original_tokens = self.tokenizer_fn(original_sentence_text) # Tokenize the original sentence

        # Shift for next-token prediction style target:
        if not original_tokens: # Handle case where original_sentence_text tokenizes to empty
            shifted_original_tokens = [self.lm_ignore_idx]
        else:
            shifted_original_tokens = original_tokens[1:] + [self.lm_ignore_idx]

        unshuffle_target_tokens_list = _truncate_or_pad_tokens(shifted_original_tokens, self.seq_len, self.lm_ignore_idx)

        # 4. Auxiliary scalar value (placeholder)
        auxiliary_scalar_value = torch.tensor(0.0, dtype=torch.float32)

        # 5. Task type flag
        task_type_flag_tensor = torch.tensor(1.0, dtype=torch.float32) # Type 1 for Levenshtein/Unshuffle

        return (
            torch.tensor(padded_input_tokens, dtype=torch.long),
            torch.tensor(next_token_lm_targets_list, dtype=torch.long),
            torch.tensor(unshuffle_target_tokens_list, dtype=torch.long),
            auxiliary_scalar_value,
            task_type_flag_tensor
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

    dataset = LevenshteinDataset(
        sample_sentences,
        mock_tokenizer,
        seq_len,
        cls_id,
        lm_ignore_idx=lm_ignore,
        input_pad_id=pad_id,
        shuffle_percentage=0.5 # Example different from default
    )
    print(f"Created dataset with {len(dataset)} examples. Shuffle Pct: {dataset.shuffle_percentage}")

    if len(dataset) > 0:
        print("\n--- Testing a few samples from the dataset ---")
        for i in range(min(len(dataset), 5)): # Print first 5 samples
            input_toks, lm_tgts, lev_dist_target, is_shuf_flag = dataset[i]
            item_type = "Shuffled" if is_shuf_flag.item() == 1.0 else "Original"

            print(f"\n--- Sample {i} ({item_type}) ---")
            print(f"  Input Tokens:          {input_toks.tolist()}")
            print(f"  LM Targets:            {lm_tgts.tolist()}")
            print(f"  Target Levenshtein:    {lev_dist_target.item():.4f}")
            print(f"  Is Shuffled Flag:      {is_shuf_flag.item()}")

            # Try to decode parts for readability
            # This requires a mock_detokenizer or knowledge of how CLS/PAD are handled by it
            approx_decoded_text = "".join([chr(t) for t in input_toks.tolist()[1:25] if t != pad_id and t != cls_id and 32 <= t <= 126])
            print(f"  Input Text (approx):   CLS + '{approx_decoded_text}'...")

            if item_type == "Original":
                non_ignored_targets = sum(1 for t_id in lm_tgts.tolist() if t_id != lm_ignore)
                print(f"  Num Valid LM Targets:  {non_ignored_targets}")
            else: # Shuffled
                all_ignored = all(t_id == lm_ignore for t_id in lm_tgts.tolist())
                print(f"  LM Targets All Ignored: {all_ignored}")
