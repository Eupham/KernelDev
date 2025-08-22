"""
Data Processing and Task Management for Hierarchical Attention Models

This module handles dataset loading, tokenization, and task-specific data preparation
for both teacher forcing and cocktail party tasks. It supports multiple dataset sources
and generates the metadata required for hierarchical attention patterns.

Key Components:
- DataBuilder: Main class for dataset management and tokenization
- Task-specific collation functions for teacher forcing and cocktail party
- Metadata generation for attention pattern control (in_span, span_id, is_prefix)
- UTF-8 byte-level tokenization with special token support
- Streaming dataset support for large corpora

Special Tokens:
- [PAD]: Padding token (ID: 0)
- [CLS]: Classification/prefix separator (ID: 1)
- [MASK]: Masked token in context (ID: 2)  
- [SPAN]: Span start marker (ID: 3)
- [ES]: Span end marker (ID: 4)
- [MASKQ]: Query aggregator token (ID: 5)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from typing import Optional, Dict, Any, List
import random

# =============================================================================
# Configuration Constants
# =============================================================================

SPECIAL_TOKENS = {
    '[PAD]': 0,
    '[CLS]': 1,
    '[MASK]': 2,
    '[SPAN]': 3,
    '[ES]': 4,
    '[MASKQ]': 5,
}
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)

# =============================================================================
# Dataset Classes
# =============================================================================

class OnTheFlyTokenizedDataset(Dataset):
    def __init__(self, raw_data, seq_len=512, tokenizer_fn=None):
        self.raw_data = raw_data
        self.seq_len = seq_len
        self.tokenizer_fn = tokenizer_fn

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        text = self.raw_data[idx]['text']
        text = f"[CLS] {text}"
        tokens = self.tokenizer_fn(text)

        tokens = tokens[:self.seq_len + 1]

        if len(tokens) < self.seq_len + 1:
            padding = [SPECIAL_TOKENS['[PAD]']] * (self.seq_len + 1 - len(tokens))
            tokens.extend(padding)

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

# =============================================================================
# Main Data Builder
# =============================================================================

class DataBuilder:
    def __init__(
        self,
        dataset_name: str = "allenai/c4",
        dataset_config: str = "en",
        seq_len: int = 512,
        max_samples: Optional[int] = 2000,
        vocab_size: int = 256,
        max_eval_tokens: int = 50000,
        on_the_fly_tokenization: bool = False,
        task_configs: dict = None
    ):
        self.on_the_fly_tokenization = on_the_fly_tokenization
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.seq_len = seq_len
        self.max_samples = max_samples if max_samples is not None else float('inf')
        self.vocab_size = vocab_size + NUM_SPECIAL_TOKENS
        self.max_eval_tokens = max_eval_tokens
        self.task_configs = task_configs or {}

        print(f"Using UTF-8 byte tokenization with vocabulary size: {self.vocab_size}")
        print(f"Max evaluation tokens per split: {self.max_eval_tokens}")
        if self.max_samples != float('inf'):
            print(f"Will attempt to load up to {self.max_samples} samples from the dataset.")
        else:
            print("Will attempt to load all available samples from the dataset.")

    def _tokenize_text(self, text: str) -> list:
        """
        Tokenize text using UTF-8 byte-level tokenization with proper multibyte support.
        
        This function properly handles multibyte UTF-8 characters by encoding the entire
        character and including all bytes in the token sequence.
        
        Optimized to handle large texts efficiently by using string replacements
        for special tokens first, then processing the resulting text.
        """
        # Step 1: Pre-process text to replace special tokens with unique markers
        # Use characters that don't appear in normal text
        special_markers = {}
        processed_text = text
        
        # Replace special tokens with unique single-byte markers
        # Use control characters (0x01-0x06) that won't appear in normal UTF-8 text
        for i, (token_str, token_id) in enumerate(SPECIAL_TOKENS.items(), 1):
            marker_char = chr(i)  # chr(1), chr(2), etc.
            special_markers[marker_char] = token_id
            processed_text = processed_text.replace(token_str, marker_char)
        
        # Step 2: Tokenize the processed text
        tokens = []
        for char in processed_text:
            if char in special_markers:
                # This is a special token marker
                tokens.append(special_markers[char])
            else:
                # Regular character - encode as UTF-8 bytes
                utf8_bytes = char.encode('utf-8')
                for byte_val in utf8_bytes:
                    tokens.append(byte_val + NUM_SPECIAL_TOKENS)
        
        return tokens

    def _detokenize_bytes(self, tokens: list, skip_special_tokens=False) -> str:
        """
        Detokenize UTF-8 byte tokens back to text with proper multibyte reconstruction.
        
        This function reconstructs multibyte UTF-8 characters by collecting bytes
        and decoding them properly.
        """
        special_token_map = {v: k for k, v in SPECIAL_TOKENS.items()}
        byte_sequence = []
        text_parts = []
        
        for t in tokens:
            if t in special_token_map:
                # Process any pending byte sequence first
                if byte_sequence:
                    try:
                        decoded_text = bytes(byte_sequence).decode('utf-8', errors='replace')
                        text_parts.append(decoded_text)
                        byte_sequence = []
                    except UnicodeDecodeError:
                        # Handle corrupted sequences gracefully
                        text_parts.append('�' * len(byte_sequence))
                        byte_sequence = []
                
                # Add special token if not skipping
                if not skip_special_tokens:
                    text_parts.append(special_token_map[t])
            else:
                # Collect bytes for UTF-8 reconstruction
                byte_val = t - NUM_SPECIAL_TOKENS
                if 0 <= byte_val <= 255:  # Valid byte range
                    byte_sequence.append(byte_val)
        
        # Process any remaining byte sequence
        if byte_sequence:
            try:
                decoded_text = bytes(byte_sequence).decode('utf-8', errors='replace')
                text_parts.append(decoded_text)
            except UnicodeDecodeError:
                # Handle corrupted sequences gracefully
                text_parts.append('�' * len(byte_sequence))
        
        return "".join(text_parts)

    def _process_iterable_dataset(self, dataset_iterable, dataset_name_logging: str) -> list:
        samples = []
        processed_count = 0
        if self.max_samples == float('inf') and not dataset_iterable:
             print(f"Warning: Dataset iterable for {dataset_name_logging} is empty or None when expecting all samples.")
             return []
        if not dataset_iterable:
            print(f"Warning: Dataset iterable for {dataset_name_logging} is empty or None.")
            return []

        for i, sample_data in enumerate(dataset_iterable):
            if processed_count >= self.max_samples:
                print(f"Reached max_samples ({self.max_samples}) for {dataset_name_logging}.")
                break

            text_content = ""
            if 'text' in sample_data:
                text_content = sample_data['text']
            elif 'content' in sample_data:
                text_content = sample_data['content']
            else:
                for key, value in sample_data.items():
                    if isinstance(value, str) and value.strip():
                        text_content = value
                        break

            if text_content and text_content.strip():
                samples.append({'text': text_content})
                processed_count += 1

            if (i + 1) % 500 == 0 and (i+1) > 0:
                print(f"Raw iterated {i+1} items from {dataset_name_logging}, processed {processed_count} valid samples...")
        
        print(f"Finished processing {dataset_name_logging}. Total valid samples extracted: {len(samples)}")
        return samples

    def load_raw_dataset(self):
        print(f"Loading dataset: {self.dataset_name}/{self.dataset_config}")
        loaded_samples = []

        # Attempt 1: C4 'en' (streaming)
        try:
            print("Attempting Method 1: Load C4 'en' (streaming)...")
            dataset_stream = load_dataset(
                self.dataset_name, name=self.dataset_config, streaming=True, split='train'
            )
            print("C4 'en' (streaming) load_dataset call succeeded. Processing samples...")
            loaded_samples = self._process_iterable_dataset(dataset_stream, "C4 'en' streaming")
            
            if len(loaded_samples) < self.max_samples and self.max_samples != float('inf'):
                if len(loaded_samples) == 0 and self.max_samples > 0:
                    raise ValueError(f"Streaming C4 'en' yielded 0 samples when {self.max_samples} were requested.")
                print(f"Streaming C4 'en' loaded {len(loaded_samples)} samples, less than requested {self.max_samples}. Will try non-streaming.")
                # Do not reset loaded_samples here, keep what was loaded and attempt non-streaming to supplement or replace
                if len(loaded_samples) == 0 : # Only raise if completely empty, otherwise proceed to non-streaming to try and get more
                    raise ValueError("Triggering non-streaming C4 'en' due to 0 samples from stream.")
            print(f"Successfully processed {len(loaded_samples)} samples from C4 'en' stream.")
        except Exception as e_c4_en_stream:
            print(f"Method 1 (C4 'en' streaming) failed: {e_c4_en_stream}")
            loaded_samples = [] # Ensure it's empty if this path failed before trying non-streaming

            # Attempt 1.5: C4 'en' (non-streaming, sliced)
            if not loaded_samples or (len(loaded_samples) < self.max_samples and self.max_samples != float('inf')):
                try:
                    print("Attempting Method 1.5: Load C4 'en' (non-streaming, sliced)...")
                    fetch_n = int(self.max_samples * 1.5) if self.max_samples != float('inf') else 5000
                    fetch_n = max(fetch_n, 100) # Ensure a minimum fetch
                    print(f"Will try to fetch up to {fetch_n} records for non-streaming C4 'en'.")
                    dataset_non_stream = load_dataset(
                        self.dataset_name, name=self.dataset_config, split=f'train[:{fetch_n}]'
                    )
                    print("C4 'en' (non-streaming) load_dataset call succeeded. Processing samples...")
                    loaded_samples = self._process_iterable_dataset(dataset_non_stream, "C4 'en' non-streaming")
                    if not loaded_samples and self.max_samples > 0:
                        raise ValueError("Non-streaming C4 'en' also yielded no samples.")
                    print(f"Successfully loaded {len(loaded_samples)} samples via C4 'en' non-streaming.")
                except Exception as e_c4_en_non_stream:
                    print(f"Method 1.5 (C4 'en' non-streaming) failed: {e_c4_en_non_stream}")
                    loaded_samples = [] # Ensure empty if this also failed

        if loaded_samples and (len(loaded_samples) >= self.max_samples or self.max_samples == float('inf')):
            print(f"Successfully loaded {len(loaded_samples)} samples using C4 'en'.")
        elif loaded_samples: # Loaded some, but less than max_samples
             print(f"Loaded {len(loaded_samples)} (less than {self.max_samples}) from C4 'en'. Proceeding with these or trying other datasets.")
        else: # No samples from C4 'en'
            print("All C4 'en' attempts (streaming/non-streaming) failed or yielded no samples.")

        # If not enough samples from C4 'en', try other methods
        if not loaded_samples or (len(loaded_samples) < self.max_samples and self.max_samples != float('inf')):
            print(f"Attempting other datasets as C4 'en' yielded {len(loaded_samples)}/{self.max_samples} samples.")
            # Attempt 2: C4 without 'en' config (streaming)
            try:
                print("Attempting Method 2: Load C4 (no config) (streaming)...")
                dataset_m2 = load_dataset(self.dataset_name, streaming=True, split='train')
                print("C4 (no config, streaming) load_dataset call succeeded. Processing samples...")
                loaded_samples = self._process_iterable_dataset(dataset_m2, "C4 (no config) streaming")
                if not loaded_samples and self.max_samples > 0:
                    raise ValueError("Method 2 (C4 no config, streaming) yielded no samples.")
                print(f"Successfully loaded {len(loaded_samples)} samples using C4 (no config).")
            except Exception as e_method2:
                print(f"Method 2 (C4 no config, streaming) failed: {e_method2}")
                loaded_samples = []

        # If still not enough, try wikitext
        if not loaded_samples or (len(loaded_samples) < self.max_samples and self.max_samples != float('inf')):
            print(f"Attempting wikitext as previous methods yielded {len(loaded_samples)}/{self.max_samples} samples.")
            # Attempt 3: Wikitext (streaming)
            try:
                print("Attempting Method 3: Load wikitext (streaming)...")
                dataset_m3 = load_dataset("wikitext", "wikitext-2-raw-v1", streaming=True, split='train')
                print("Wikitext (streaming) load_dataset call succeeded. Processing samples...")
                loaded_samples = self._process_iterable_dataset(dataset_m3, "wikitext streaming")
                if not loaded_samples and self.max_samples > 0:
                    raise ValueError("Method 3 (wikitext, streaming) yielded no samples.")
                print(f"Successfully loaded {len(loaded_samples)} samples using wikitext.")
            except Exception as e_method3:
                print(f"Method 3 (wikitext, streaming) failed: {e_method3}")
                loaded_samples = []

        # Final check and fallback
        if loaded_samples and (len(loaded_samples) >= self.max_samples or self.max_samples == float('inf')):
            print(f"Final dataset loaded with {len(loaded_samples)} samples.")
        elif loaded_samples: # Loaded some, but maybe not enough
            print(f"Warning: Final dataset loaded with {len(loaded_samples)} samples, requested {self.max_samples}. Using available data.")
        else:
            print("All primary dataset loading methods failed or yielded no usable samples. Falling back to simple text dataset...")
            return self._create_fallback_dataset()

        # Split data for train/validation/test
        train_split = int(0.8 * len(loaded_samples))
        val_split = int(0.9 * len(loaded_samples))
        # Ensure validation and test sets are not empty if train_split is too large
        if train_split == len(loaded_samples) and len(loaded_samples) > 0: train_split = len(loaded_samples) -2 # min 1 for val, 1 for test
        if val_split <= train_split and val_split < len(loaded_samples) -1 : val_split = train_split + 1
        if val_split >= len(loaded_samples): val_split = len(loaded_samples) -1
        if train_split < 0: train_split = 0

        final_train_data = loaded_samples[:train_split]
        final_val_data = loaded_samples[train_split:val_split] if val_split > train_split else []
        final_test_data = loaded_samples[val_split:] if val_split < len(loaded_samples) else []

        # Handle cases where splits might be empty due to small loaded_samples size
        if not final_train_data and loaded_samples: final_train_data = loaded_samples # Use all for train if splits fail
        if not final_val_data and final_train_data: final_val_data = final_train_data[:max(1, len(final_train_data)//10)] # 10% of train for val
        if not final_test_data and final_train_data: final_test_data = final_train_data[:max(1, len(final_train_data)//10)] # 10% of train for test

        print(f"Returning dataset splits: train={len(final_train_data)}, val={len(final_val_data)}, test={len(final_test_data)}")
        return {
            'train': final_train_data,
            'validation': final_val_data,
            'test': final_test_data
        }

    def _create_fallback_dataset(self):
        # ... (method content as in original file, ensure it uses self.max_samples correctly)
        sample_texts = [
            "The quick brown fox jumps over the lazy dog. This is a classic pangram used in typing practice.",
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
            "Natural language processing enables computers to understand and generate human language.",
            "Transformers have revolutionized the field of natural language processing with their attention mechanisms.",
            "GPT models are based on the transformer architecture and are trained on large amounts of text data.",
            "Flash attention is an efficient implementation of the attention mechanism that reduces memory usage.",
            "PyTorch is a popular deep learning framework that provides dynamic computation graphs.",
            "CUDA enables parallel computing on NVIDIA GPUs for accelerated machine learning workloads.",
            "Tokenization is the process of converting text into numerical tokens that models can process.",
        ]
        
        # Limit the number of repetitions to prevent excessive text size
        # Cap at reasonable number to avoid tokenization performance issues
        max_reasonable_repetitions = min(1000, self.max_samples // 10 if self.max_samples != float('inf') else 100)
        num_repetitions = max(1, max_reasonable_repetitions)
        full_sample_texts = sample_texts * num_repetitions
        
        # Ensure fallback provides a reasonable number of texts for splitting, but cap it
        num_fallback_texts = min(max(20, len(full_sample_texts)), 10000)  # Cap at 10,000 texts
        text_block = '\n'.join(full_sample_texts[:num_fallback_texts])

        # Ensure splits are somewhat reasonable even for small text_block
        len_block = len(text_block)
        train_end = int(0.8 * len_block)
        val_end = int(0.9 * len_block)

        return {
            'train': [{'text': text_block[:train_end]}],
            'validation': [{'text': text_block[train_end:val_end]}],
            'test': [{'text': text_block[val_end:]}]
        }
    
    def tokenize_dataset(self, dataset):
        # ... (method content as in original file, ensure robust to empty splits)
        tokenized_data = {}
        for split_name, split_data_list in dataset.items():
            print(f"Tokenizing {split_name} split...")
            all_text = ""
            if not isinstance(split_data_list, list):
                print(f"Warning: {split_name} data is not a list (type: {type(split_data_list)}), skipping tokenization.")
                tokenized_data[split_name] = [] # Ensure key exists with empty list
                continue
            if not split_data_list: # Handle empty list
                print(f"Warning: {split_name} data list is empty. Skipping tokenization.")
                tokenized_data[split_name] = []
                continue

            for item in split_data_list:
                if isinstance(item, dict) and 'text' in item:
                    text_content = item['text']
                    if text_content and text_content.strip():
                        all_text += text_content + "\n"
            
            print(f"Text length for {split_name}: {len(all_text)} characters")
            if not all_text.strip():
                print(f"Warning: No text content found for {split_name} split. Resulting tokens will be empty.")
                tokens = []
            else:
                tokens = self._tokenize_text(all_text)
            print(f"Tokenized to {len(tokens)} byte tokens")
            tokenized_data[split_name] = tokens
        return tokenized_data
    
    def create_datasets(self):
        raw_dataset = self.load_raw_dataset()
        if self.on_the_fly_tokenization:
            datasets = {}
            for split_name, data in raw_dataset.items():
                if data:
                    datasets[split_name] = OnTheFlyTokenizedDataset(data, self.seq_len, self._tokenize_text)
                    print(f"{split_name} dataset (on-the-fly): {len(datasets[split_name])} samples")
                else:
                    print(f"Warning: {split_name} split has no data. Skipping dataset creation.")
            return datasets
        else:
            # ... (original pre-tokenization logic)
            tokenized_data = self.tokenize_dataset(raw_dataset)

            datasets = {}
            for split_name, tokens in tokenized_data.items():
                if not tokens:
                    print(f"Warning: {split_name} split has no tokens. Skipping dataset creation.")
                    continue

                current_max_eval_tokens = self.max_eval_tokens
                if self.max_samples != float('inf') and split_name in ['validation', 'test']:
                    scaled_max_tokens = int(self.max_samples * 0.2 * self.seq_len)
                    current_max_eval_tokens = min(self.max_eval_tokens, scaled_max_tokens)
                    current_max_eval_tokens = max(current_max_eval_tokens, self.seq_len * 2 + 1)

                if len(tokens) > self.seq_len:
                    if split_name in ['validation', 'test']:
                        if len(tokens) > current_max_eval_tokens:
                            tokens = tokens[:current_max_eval_tokens]
                            print(f"Limited {split_name} to {len(tokens)} tokens for faster evaluation (target: {current_max_eval_tokens})")

                    # This part needs to be adapted. Let's assume the original TokenizedDataset is still defined for this path.
                    from torch.utils.data import Dataset
                    class TokenizedDataset(Dataset):
                        def __init__(self, tokenized_data, seq_len=512):
                            self.data = tokenized_data
                            self.seq_len = seq_len
                        def __len__(self):
                            return max(1, len(self.data) - self.seq_len)
                        def __getitem__(self, idx):
                            x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
                            y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
                            return x, y

                    datasets[split_name] = TokenizedDataset(tokens, self.seq_len)
                    print(f"{split_name} dataset: {len(datasets[split_name])} samples")
                else:
                    print(f"Warning: {split_name} split has insufficient tokens ({len(tokens)}) for seq_len {self.seq_len}. Skipping dataset.")
            return datasets

    def _collate_fn_teacher_forcing(self, batch):
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        return inputs, targets

    def _collate_fn_cocktail_party(self, batch):
        task_config = self.task_configs.get('cocktail_party', {})
        num_distractors = task_config.get('num_distractors', 3)
        min_span_size = task_config.get('min_span_size', 10)
        max_span_size = task_config.get('max_span_size', 50)

        batch_inputs = []
        batch_correct_indices = []
        batch_attn_masks = []
        
        # Pre-process all items to ensure consistent batch structure
        valid_items = []
        
        for i in range(len(batch)):
            original_tokens_padded, _ = batch[i]
            original_tokens_padded = original_tokens_padded.tolist()

            # 1. Strip padding to get the true sequence
            pad_id = SPECIAL_TOKENS['[PAD]']
            try:
                first_pad_idx = original_tokens_padded.index(pad_id)
                original_tokens = original_tokens_padded[:first_pad_idx]
            except ValueError:
                original_tokens = original_tokens_padded

            # 2. Find [CLS] to separate task instructions from context
            cls_token = SPECIAL_TOKENS['[CLS]']
            try:
                cls_idx = original_tokens.index(cls_token)
                task_prefix = original_tokens[:cls_idx + 1]  # Include [CLS]
                context_tokens = original_tokens[cls_idx + 1:]  # Context after [CLS]
            except ValueError:
                # No [CLS] found, treat everything as context (fallback)
                task_prefix = []
                context_tokens = original_tokens

            # 3. Sample span only from context (not from task instructions)
            span_size = random.randint(min_span_size, max_span_size)
            if len(context_tokens) <= span_size:
                continue  # Skip if context too short
                
            span_start_in_context = random.randint(0, len(context_tokens) - span_size)
            true_span = context_tokens[span_start_in_context : span_start_in_context + span_size]

            # 4. Create distractors from other batch items (also from context only)
            distractors = []
            attempts = 0
            available_indices = [j for j in range(len(batch)) if i != j]
            
            # If no other batch items available, skip this item or create synthetic distractors
            if not available_indices:
                print(f"Warning: Batch size too small for cocktail party task (need at least 2 items)")
                continue
                
            while len(distractors) < num_distractors and attempts < len(batch) * num_distractors * 2:
                distractor_idx = random.choice(available_indices)
                distractor_tokens_padded, _ = batch[distractor_idx]
                distractor_tokens_padded = distractor_tokens_padded.tolist()
                
                try:
                    first_pad_idx_dist = distractor_tokens_padded.index(pad_id)
                    distractor_tokens = distractor_tokens_padded[:first_pad_idx_dist]
                except ValueError:
                    distractor_tokens = distractor_tokens_padded

                # Find [CLS] in distractor to get context only
                try:
                    cls_idx_dist = distractor_tokens.index(cls_token)
                    distractor_context = distractor_tokens[cls_idx_dist + 1:]
                except ValueError:
                    distractor_context = distractor_tokens

                if len(distractor_context) > span_size:
                    distractor_start = random.randint(0, len(distractor_context) - span_size)
                    distractor_span = distractor_context[distractor_start : distractor_start + span_size]
                    distractors.append(distractor_span)
                attempts += 1

            # If we couldn't get enough distractors, pad with copies/variations
            while len(distractors) < num_distractors:
                if distractors:
                    distractors.append(distractors[0])  # Reuse first distractor as fallback
                else:
                    # Fallback: create a distractor from a different part of the same context
                    if len(context_tokens) > span_size * 2:
                        alt_start = random.randint(0, len(context_tokens) - span_size)
                        if alt_start != span_start_in_context:
                            alt_span = context_tokens[alt_start : alt_start + span_size]
                            distractors.append(alt_span)
                        else:
                            distractors.append(true_span)  # Last resort
                    else:
                        distractors.append(true_span)  # Last resort

            valid_items.append({
                'task_prefix': task_prefix,
                'context_tokens': context_tokens,
                'span_start_in_context': span_start_in_context,
                'span_size': span_size,
                'true_span': true_span,
                'distractors': distractors[:num_distractors]  # Ensure exact count
            })

        # Now process all valid items with consistent structure
        for item in valid_items:
            all_spans_with_labels = [(item['true_span'], 1)] + [(d, 0) for d in item['distractors']]
            random.shuffle(all_spans_with_labels)
            correct_idx = [label for _, label in all_spans_with_labels].index(1)

            # Calculate wrapper length to reserve space (including [MASKQ])
            wrapper_tokens = []
            for span_toks, _ in all_spans_with_labels:
                wrapper_tokens.extend([SPECIAL_TOKENS['[SPAN]']] + span_toks + [SPECIAL_TOKENS['[ES]']])
            wrapper_len = len(wrapper_tokens) + 1  # +1 for [MASKQ]

            # Extend max tokens for cocktail party task to accommodate special tokens and distractors
            required_space = len(item['task_prefix']) + wrapper_len
            if required_space >= self.seq_len:
                # Instead of skipping, extend the sequence length for this batch
                extended_seq_len = required_space + min(50, len(item['context_tokens']))  # Add some context
                available_context_len = extended_seq_len - required_space
            else:
                extended_seq_len = self.seq_len
                available_context_len = self.seq_len - required_space
            
            # Create masked context (mask the span we sampled)
            masked_context = (item['context_tokens'][:item['span_start_in_context']] + 
                            [SPECIAL_TOKENS['[MASK]']] + 
                            item['context_tokens'][item['span_start_in_context'] + item['span_size']:])
            
            # Truncate context if needed, but preserve all islands and [MASKQ]
            truncated_masked_context = masked_context[:available_context_len]

            # Stitch final sequence together: {prefix}[CLS]{context}[SPAN]...[ES][SPAN]...[ES][MASKQ]
            final_sequence = item['task_prefix'] + truncated_masked_context + wrapper_tokens + [SPECIAL_TOKENS['[MASKQ]']]

            # Pad to the extended length to accommodate all special tokens
            if len(final_sequence) < extended_seq_len:
                final_sequence.extend([pad_id] * (extended_seq_len - len(final_sequence)))

            # Build metadata tensors: in_span, span_id, is_prefix (use extended length)
            in_span = torch.zeros(extended_seq_len, dtype=torch.bool)
            span_ids = torch.zeros(extended_seq_len, dtype=torch.long)
            is_prefix = torch.zeros(extended_seq_len, dtype=torch.bool)
            
            # Mark prefix tokens (task instructions + [CLS])
            prefix_len = len(item['task_prefix'])
            if prefix_len > 0:
                is_prefix[:prefix_len] = True
            
            # Mark span tokens and assign span IDs
            start_of_spans = prefix_len + len(truncated_masked_context)
            current_pos = start_of_spans
            span_token = SPECIAL_TOKENS['[SPAN]']
            es_token = SPECIAL_TOKENS['[ES]']
            
            for span_idx, (span_toks, _) in enumerate(all_spans_with_labels):
                if current_pos >= extended_seq_len:
                    break
                    
                # Find [SPAN] token
                if current_pos < extended_seq_len and final_sequence[current_pos] == span_token:
                    span_start = current_pos
                    span_end = min(current_pos + len(span_toks) + 2, extended_seq_len)  # +2 for [SPAN] and [ES]
                    
                    # Mark all tokens in this span (including [SPAN] and [ES])
                    in_span[span_start:span_end] = True
                    span_ids[span_start:span_end] = span_idx + 1  # Use 1-based span IDs
                    
                    current_pos += len(span_toks) + 2
                else:
                    current_pos += len(span_toks) + 2
            
            # Mark [MASKQ] specially (it should see all spans but not be in a span itself)
            maskq_token = SPECIAL_TOKENS['[MASKQ]']
            try:
                maskq_idx = final_sequence.index(maskq_token)
                # [MASKQ] is not in_span but has special access patterns
                span_ids[maskq_idx] = -1  # Special marker for [MASKQ]
            except ValueError:
                pass
            
            # Return metadata instead of old attention mask
            batch_inputs.append(torch.tensor(final_sequence, dtype=torch.long))
            batch_correct_indices.append(torch.tensor(correct_idx, dtype=torch.long))
            # Return metadata tensors for attention calculation
            batch_attn_masks.append({
                'in_span': in_span,
                'span_id': span_ids,
                'is_prefix': is_prefix
            })

        if not batch_inputs:
            return torch.empty(0), torch.empty(0), {}

        inputs = torch.stack(batch_inputs)
        correct_indices = torch.stack(batch_correct_indices)
        
        # Stack the metadata tensors
        batch_size = len(batch_attn_masks)
        in_span_batch = torch.stack([mask_dict['in_span'] for mask_dict in batch_attn_masks])
        span_id_batch = torch.stack([mask_dict['span_id'] for mask_dict in batch_attn_masks])
        is_prefix_batch = torch.stack([mask_dict['is_prefix'] for mask_dict in batch_attn_masks])
        
        metadata = {
            'in_span': in_span_batch,
            'span_id': span_id_batch, 
            'is_prefix': is_prefix_batch
        }

        return inputs, correct_indices, metadata


    def create_dataloaders(
        self, batch_size: int = 8, num_workers: int = 0, shuffle_train: bool = True
    ) -> Dict[str, Dict[str, DataLoader]]:
        datasets = self.create_datasets()
        dataloaders = {}
        if not datasets:
            print("Warning: No datasets were created. Returning empty dataloaders dict.")
            return dataloaders

        # Optimize DataLoader settings for speed
        optimized_kwargs = {
            'num_workers': num_workers,
            'pin_memory': torch.cuda.is_available(),
            'persistent_workers': num_workers > 0,  # Keep workers alive between epochs
            'prefetch_factor': 4 if num_workers > 0 else None,  # Increase prefetch for speed
        }
        
        # Remove None values
        optimized_kwargs = {k: v for k, v in optimized_kwargs.items() if v is not None}

        for split_name, dataset_obj in datasets.items():
            if not dataset_obj:
                print(f"Skipping DataLoader for {split_name} as dataset is empty or invalid.")
                continue

            dataloaders[split_name] = {}
            shuffle = shuffle_train if split_name == 'train' else False

            # Teacher forcing dataloader
            dataloaders[split_name]['teacher_forcing'] = DataLoader(
                dataset_obj, batch_size=batch_size, shuffle=shuffle,
                collate_fn=self._collate_fn_teacher_forcing,
                **optimized_kwargs
            )

            # Cocktail party dataloader
            if 'cocktail_party' in self.task_configs:
                # Use half batch size for cocktail party due to memory requirements
                cocktail_batch_size = max(1, batch_size // 2)
                dataloaders[split_name]['cocktail_party'] = DataLoader(
                    dataset_obj, batch_size=cocktail_batch_size, shuffle=shuffle,
                    collate_fn=self._collate_fn_cocktail_party,
                    **optimized_kwargs
                )

        return dataloaders

    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def decode_tokens(self, tokens, skip_special_tokens=False):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().tolist()
        return self._detokenize_bytes(tokens, skip_special_tokens=skip_special_tokens)


def create_data_builder(
    dataset_name: str = "allenai/c4", dataset_config: str = "en",
    seq_len: int = 512, max_samples: Optional[int] = 2000,
    max_eval_tokens: int = 50000,
    on_the_fly_tokenization: bool = False,
    task_configs: dict = None
) -> DataBuilder:
    return DataBuilder(
        dataset_name=dataset_name, dataset_config=dataset_config,
        seq_len=seq_len, max_samples=max_samples,
        max_eval_tokens=max_eval_tokens,
        on_the_fly_tokenization=on_the_fly_tokenization,
        task_configs=task_configs
    )

if __name__ == "__main__":
    # ... (main test block as in original file)
    print("Testing DataBuilder...")
    data_builder = create_data_builder(
        dataset_name="allenai/c4", dataset_config="en",
        seq_len=128, max_samples=500
    )
    dataloaders = data_builder.create_dataloaders(batch_size=4)
    if 'train' in dataloaders and dataloaders['train']:
        train_loader = dataloaders['train']
        print(f"Number of training batches: {len(train_loader)}")
        try:
            for batch_idx, (x, y) in enumerate(train_loader):
                print(f"Batch {batch_idx}: Input shape: {x.shape}, Target shape: {y.shape}")
                sample_text = data_builder.decode_tokens(x[0][:50])
                print(f"Sample text: {sample_text}")
                if batch_idx >= 0: break
        except Exception as e:
            print(f"Error during dataloader iteration test: {e}")
    else:
        print("Train dataloader not created or empty.")
    print("DataBuilder test completed!")
