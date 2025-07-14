import torch
from torch.utils.data import Dataset
import re
import random

# Assuming text_utils.py is in the same directory or accessible in PYTHONPATH
try:
    # shuffle_words_in_sentence now returns a 4th item: permuted_word_source_indices
    from text_utils import shuffle_words_in_sentence, word_levenshtein_distance
except ImportError:
    print("Warning: text_utils.py not found, using placeholder shuffle/distance functions.")
    # Placeholder needs to match the new 4-tuple return type
    def shuffle_words_in_sentence(s, p): return s, s.split(), s.split(), list(range(len(s.split())))
    def word_levenshtein_distance(s1, s2): return len(s1) + len(s2)

TASK_ID_LEV_UNSHUFFLE = 49 # ord('1') # Represents this auxiliary task

def _truncate_or_pad_tokens(tokens: list[int], max_len: int, pad_id: int) -> list[int]:
    """Ensure token list is exactly max_len long by truncating or padding."""
    if len(tokens) > max_len:
        return tokens[:max_len]
    return tokens + [pad_id] * (max_len - len(tokens))

class LevenshteinDataset(Dataset):
    def __init__(self,
                 raw_documents_or_sentences: list[str],
                 tokenizer_fn: callable,
                 seq_len: int,
                 cls_token_id: int,
                 lm_ignore_idx: int = -1,
                 input_pad_id: int = 0,
                 shuffle_percentage: float = 0.25): # Default shuffle_percentage
        self.tokenizer_fn = tokenizer_fn
        self.seq_len = seq_len
        self.cls_token_id = cls_token_id
        self.lm_ignore_idx = lm_ignore_idx # Used for LM targets and as basis for rank ignore
        self.input_pad_id = input_pad_id
        self.rank_ignore_idx_float = float(self.lm_ignore_idx) # For padding rank targets

        if not 0.0 <= shuffle_percentage <= 1.0: # Validate shuffle_percentage
            raise ValueError("shuffle_percentage must be between 0.0 and 1.0")
        # The 'shuffle_percentage' param is currently not used directly to set range,
        # but kept for potential future use or compatibility.
        # Defaulting to a dynamic range, starting narrow.
        self.current_min_shuffle_p: float = 0.05
        self.current_max_shuffle_p: float = 0.05

        self.sentences = raw_documents_or_sentences

    def __len__(self):
        return len(self.sentences)

    def update_shuffle_range(self, min_p: float, max_p: float):
        min_p_clamped = max(0.0, min(min_p, 1.0))
        max_p_clamped = max(0.0, min(max_p, 1.0))
        self.current_min_shuffle_p = min(min_p_clamped, max_p_clamped)
        self.current_max_shuffle_p = max(min_p_clamped, max_p_clamped)

    def __getitem__(self, idx):
        raw_sentence_text = self.sentences[idx]

        # 1. Canonicalize original sentence: strip, split by whitespace, join with single space
        # This ensures consistent character counts and space handling for rank calculation.
        original_words_list = [word for word in re.split(r'\s+', raw_sentence_text.strip()) if word]
        if not original_words_list: # Handle empty or all-space sentences
            canonical_original_sentence = " " # Use a single space as canonical form
            original_words_list = [" "] # Update word list accordingly
        else:
            canonical_original_sentence = " ".join(original_words_list)

        N_original_chars = len(canonical_original_sentence)

        # 2. Shuffle words of the canonical original sentence
        selected_probability = random.uniform(self.current_min_shuffle_p, self.current_max_shuffle_p)
        # shuffle_words_in_sentence now returns: shuffled_text, original_words, final_shuffled_words, permuted_indices
        shuffled_sentence_text, _, final_shuffled_word_list, permuted_word_source_indices = \
            shuffle_words_in_sentence(canonical_original_sentence, selected_probability)

        # 3. Prepare input tokens for the model (based on shuffled_sentence_text)
        tokens_shuffled_content = self.tokenizer_fn(shuffled_sentence_text)

        max_content_len = self.seq_len - 2 # Account for [TASK_ID, CLS_ID] prefix
        if len(tokens_shuffled_content) > max_content_len:
            tokens_shuffled_content = tokens_shuffled_content[:max_content_len]
        actual_content_len = len(tokens_shuffled_content) # Number of actual content tokens

        final_input_tokens_list = [TASK_ID_LEV_UNSHUFFLE, self.cls_token_id] + tokens_shuffled_content
        padded_input_tokens = _truncate_or_pad_tokens(final_input_tokens_list, self.seq_len, self.input_pad_id)

        # 4. Prepare next_token_lm_targets (all ignore_idx for this task type)
        # This task does not train the main LM head.
        next_token_lm_targets_list = [self.lm_ignore_idx] * self.seq_len

        # 5. Prepare rank_regression_targets
        # These are float values representing normalized original ranks, aligned with tokens_shuffled_content.
        target_ranks_for_shuffled_chars = []
        if N_original_chars > 0: # Proceed only if there's a basis for ranks (non-empty canonical sentence)
            # Calculate 0-based start char index of each word in canonical_original_sentence
            char_starts_of_original_words = []
            current_char_pos_in_original = 0
            for word_text in original_words_list:
                char_starts_of_original_words.append(current_char_pos_in_original)
                current_char_pos_in_original += len(word_text)
                current_char_pos_in_original += 1 # Account for the space after (even for the last word, for length consistency)

            # Iterate over the words in their shuffled order to reconstruct ranks
            char_cursor_in_shuffled_text = 0
            for i_shuf_word, current_shuffled_word_text in enumerate(final_shuffled_word_list):
                if char_cursor_in_shuffled_text >= actual_content_len: break # Stop if we've filled ranks for all input tokens

                original_word_idx = permuted_word_source_indices[i_shuf_word]
                original_char_start_for_this_word_block = char_starts_of_original_words[original_word_idx]

                # Add ranks for characters in the current word
                for k_char_in_block in range(len(current_shuffled_word_text)):
                    if char_cursor_in_shuffled_text >= actual_content_len: break

                    original_0_char_idx = original_char_start_for_this_word_block + k_char_in_block
                    rank = (original_0_char_idx + 1.0) / N_original_chars # 1-based normalized rank
                    target_ranks_for_shuffled_chars.append(rank)
                    char_cursor_in_shuffled_text += 1

                if char_cursor_in_shuffled_text >= actual_content_len: break

                # Add rank for the space after this word (if not the last word in sequence)
                if i_shuf_word < len(final_shuffled_word_list) - 1:
                    original_space_0_idx = char_starts_of_original_words[original_word_idx] + len(original_words_list[original_word_idx])
                    rank_for_space = (original_space_0_idx + 1.0) / N_original_chars
                    target_ranks_for_shuffled_chars.append(rank_for_space)
                    char_cursor_in_shuffled_text += 1

        # Create the final rank target sequence for the model (length self.seq_len)
        # Padded with rank_ignore_idx_float.
        rank_regression_targets = [self.rank_ignore_idx_float] * self.seq_len

        # Place calculated ranks into the target sequence, corresponding to input token positions.
        # Ranks start after [TASK_ID, CLS_ID] prefix, i.e., at index 2.
        len_to_copy = min(actual_content_len, len(target_ranks_for_shuffled_chars))
        for i in range(len_to_copy):
            rank_regression_targets[2 + i] = target_ranks_for_shuffled_chars[i]

        # 6. Auxiliary scalar value (dummy, as this task now produces a sequence of ranks)
        auxiliary_scalar_value = torch.tensor(0.0, dtype=torch.float32)

        # 7. Task type flag (identifies this as auxiliary task type 1)
        task_type_flag_tensor = torch.tensor(1.0, dtype=torch.float32)

        # 8. Add dummy placeholder for original text ranks (for 6-tuple consistency)
        original_text_ranks_placeholder = torch.full((self.seq_len,), self.rank_ignore_idx_float, dtype=torch.float32)

        return (
            torch.tensor(padded_input_tokens, dtype=torch.long),
            torch.tensor(next_token_lm_targets_list, dtype=torch.long),
            torch.tensor(rank_regression_targets, dtype=torch.float32),
            auxiliary_scalar_value,
            task_type_flag_tensor,
            original_text_ranks_placeholder # New 6th item
        )

if __name__ == '__main__':
    print("LevenshteinDataset module loading (now Rank Regression variant)...")

    def mock_tokenizer(text: str) -> list[int]: # Character-level tokenizer
        return [ord(c) for c in text]

    cls_id = 0 # Placeholder
    pad_id = 1 # Placeholder for input padding
    lm_ignore = -100 # Standard ignore index for PyTorch loss functions
    seq_len = 35

    sample_sentences = [
        "cat dog", # Target: d o g   c a t -> ranks for these chars
        "short example",
        "This is a slightly longer sentence for testing purposes.",
        "重複 重複", # Test with duplicate words "repeat repeat"
        "  leading and trailing spaces  ",
        "word", # Single word
        "" # Empty string
    ]

    dataset = LevenshteinDataset(
        sample_sentences,
        mock_tokenizer,
        seq_len,
        cls_id,
        lm_ignore_idx=lm_ignore, # Used for rank_ignore_idx_float basis
        input_pad_id=pad_id
    )
    # Update shuffle range for testing
    dataset.update_shuffle_range(min_p=0.5, max_p=1.0)
    print(f"Created Rank Regression Dataset with {len(dataset)} examples.")
    print(f"Shuffle prob range: [{dataset.current_min_shuffle_p:.2f}, {dataset.current_max_shuffle_p:.2f}]")
    print(f"Rank ignore index (float): {dataset.rank_ignore_idx_float}")


    if len(dataset) > 0:
        print(f"\n--- Testing a few samples (max 5) ---")
        for i in range(min(len(dataset), 5)):
            # Unpack the 5 tensors returned by __getitem__
            input_tokens_tensor, lm_targets_tensor, rank_targets_tensor, aux_scalar_tensor, task_flag_tensor = dataset[i]

            print(f"\n--- Sample {i} ---")
            # Decode input tokens for readability (excluding task/cls/pad)
            # Assuming tokenizer maps ord(c) to token id and vice-versa for chr(t)
            # This mock decoding is specific to the mock_tokenizer.
            input_token_list = input_tokens_tensor.tolist()
            shuffled_text_approx = "".join([
                chr(t) for t in input_token_list[2:]
                if t != pad_id and t != cls_id and t >= 32 and t <= 126 # Printable ASCII
            ])

            print(f"  Input Tokens (IDs):    {input_token_list}")
            print(f"  Shuffled Text (approx):'{shuffled_text_approx}'")
            # print(f"  LM Targets:            {lm_targets_tensor.tolist()}") # Usually all ignored

            rank_target_list = rank_targets_tensor.tolist()
            formatted_ranks = [f"{r:.2f}" if r != dataset.rank_ignore_idx_float else "ign" for r in rank_target_list]
            print(f"  Rank Regression Targets: {formatted_ranks}")
            # print(f"  Auxiliary Scalar:      {aux_scalar_tensor.item():.4f}")
            # print(f"  Task Type Flag:        {task_flag_tensor.item()}")

            # Find the original sentence for context
            raw_original_sentence = dataset.sentences[i]
            original_words = [word for word in re.split(r'\s+', raw_original_sentence.strip()) if word]
            if not original_words: canonical_original = " "
            else: canonical_original = " ".join(original_words)
            print(f"  Original Sentence (raw): '{raw_original_sentence}'")
            print(f"  Original Sentence (canon): '{canonical_original}' (len: {len(canonical_original)})")

            # Verify length of ranks vs length of tokenized shuffled text
            num_content_tokens = 0
            for t_idx in range(2, len(input_token_list)):
                if input_token_list[t_idx] == pad_id:
                    break
                num_content_tokens +=1

            num_valid_ranks = 0
            for r_idx in range(2, len(rank_target_list)):
                if rank_target_list[r_idx] == dataset.rank_ignore_idx_float:
                    # Allow for some padding at the end of ranks if content tokens were shorter than max_content_len
                    # but ranks list was prepared up to actual_content_len
                    pass
                num_valid_ranks +=1 # Counts all up to seq_len - 2

            # More precise count of non-ignored ranks
            num_non_ignored_ranks = sum(1 for r in rank_target_list[2:] if r != dataset.rank_ignore_idx_float)

            print(f"  Num Content Tokens (in input): {num_content_tokens}")
            print(f"  Num Non-Ignored Ranks:    {num_non_ignored_ranks}")
            if num_content_tokens != num_non_ignored_ranks:
                 print(f"  WARNING: Mismatch between content token count ({num_content_tokens}) and non-ignored rank count ({num_non_ignored_ranks})")
