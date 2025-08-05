import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from typing import Optional, Dict, Any, List
import random

# Define special tokens
BIO_TAGS = {
    'O': 0,
    'B-ORIG': 1,
    'I-ORIG': 2,
    'PAD': -100,
}
NUM_BIO_TAGS = 3

SPECIAL_TOKENS = {
    '[PAD]': 0,
    '[CLS]': 1,
    '[MASK]': 2,
    '[SPAN]': 3,
    '[ES]': 4,
}
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)

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
        # This is a simplified tokenizer. A real implementation would use a pre-trained tokenizer.
        tokens = []
        i = 0
        while i < len(text):
            found = False
            for token_str, token_id in SPECIAL_TOKENS.items():
                if text[i:].startswith(token_str):
                    tokens.append(token_id)
                    i += len(token_str)
                    found = True
                    break
            if not found:
                tokens.append(text[i].encode('utf-8')[0] + NUM_SPECIAL_TOKENS)
                i += 1
        return tokens

    def _detokenize_bytes(self, tokens: list, skip_special_tokens=False) -> str:
        special_token_map = {v: k for k, v in SPECIAL_TOKENS.items()}
        decoded_tokens = []
        for t in tokens:
            if t in special_token_map:
                if not skip_special_tokens:
                    decoded_tokens.append(special_token_map[t])
            else:
                decoded_tokens.append(chr(t - NUM_SPECIAL_TOKENS))
        return "".join(decoded_tokens)

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
                self.dataset_name, name=self.dataset_config, streaming=True, split='train', trust_remote_code=True
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
                        self.dataset_name, name=self.dataset_config, split=f'train[:{fetch_n}]', trust_remote_code=True
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
                dataset_m2 = load_dataset(self.dataset_name, streaming=True, split='train', trust_remote_code=True)
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
        num_repetitions = (self.max_samples // 10 if self.max_samples != float('inf') and self.max_samples > 10 else 100)
        num_repetitions = max(num_repetitions, 1) # Ensure at least one repetition
        full_sample_texts = sample_texts * num_repetitions
        
        # Ensure fallback provides a reasonable number of texts for splitting
        num_fallback_texts = max(20, len(full_sample_texts))
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

        for i in range(len(batch)):
            original_tokens, _ = batch[i]
            original_tokens = original_tokens.tolist()

            span_size = random.randint(min_span_size, max_span_size)
            if len(original_tokens) <= span_size:
                continue

            span_start = random.randint(0, len(original_tokens) - span_size)
            true_span = original_tokens[span_start : span_start + span_size]

            distractors = []
            for _ in range(num_distractors):
                distractor_idx = random.choice([j for j in range(len(batch)) if i != j])
                distractor_tokens, _ = batch[distractor_idx]
                distractor_tokens = distractor_tokens.tolist()
                if len(distractor_tokens) <= span_size:
                    continue
                distractor_start = random.randint(0, len(distractor_tokens) - span_size)
                distractor_span = distractor_tokens[distractor_start : distractor_start + span_size]
                distractors.append(distractor_span)

            if not distractors:
                continue

            # Create the masked sequence with [CLS] token at the beginning
            masked_sequence = [SPECIAL_TOKENS['[CLS]']] + original_tokens[:span_start] + [SPECIAL_TOKENS['[MASK]']] + original_tokens[span_start + span_size:]

            all_spans_with_labels = [(true_span, 1)] + [(d, 0) for d in distractors]
            random.shuffle(all_spans_with_labels)

            spans, is_positive = zip(*all_spans_with_labels)
            correct_idx = is_positive.index(1)

            # Append spans to the sequence
            for span in spans:
                masked_sequence.extend([SPECIAL_TOKENS['[SPAN]']] + span + [SPECIAL_TOKENS['[ES]']])

            # Truncate or pad the sequence
            masked_sequence = masked_sequence[:self.seq_len]
            seq_padding = [SPECIAL_TOKENS['[PAD]']] * (self.seq_len - len(masked_sequence))
            masked_sequence.extend(seq_padding)

            batch_inputs.append(torch.tensor(masked_sequence, dtype=torch.long))
            batch_correct_indices.append(torch.tensor(correct_idx, dtype=torch.long))

        if not batch_inputs:
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

        inputs = torch.stack(batch_inputs)
        correct_indices = torch.stack(batch_correct_indices)

        # Generate attention mask
        attention_mask = self._generate_attention_mask(inputs)

        # The 'spans' tensor is no longer needed in this format
        # We return the attention_mask instead.
        # To avoid changing the signature of the forward pass in `train_loop.py` for all tasks,
        # we will return a tuple of 4 elements, with the second element being the attention_mask.
        # The original `spans` tensor is the second element.
        return inputs, attention_mask, correct_indices, None


    def _collate_fn_soft_jigsaw(self, batch):
        task_config = self.task_configs.get('soft_jigsaw', {})
        M = task_config.get('M', 5)

        batch_inputs = []
        batch_p_star = []

        for item in batch:
            text, _ = item
            text_tokens = text.tolist()

            # Split tokens into segments of roughly equal size
            segment_len = len(text_tokens) // M
            if segment_len == 0:
                continue

            segments = [text_tokens[i:i + segment_len] for i in range(0, len(text_tokens), segment_len)][:M]
            if len(segments) < M:
                continue

            indexed_segments = list(enumerate(segments))
            random.shuffle(indexed_segments)

            shuffled_segments = [s for _, s in indexed_segments]
            original_indices = [i for i, _ in indexed_segments]

            p_star = torch.zeros(M, M)
            for i in range(M):
                p_star[i, original_indices[i]] = 1

            # Create input sequence
            input_tokens = [SPECIAL_TOKENS['[CLS]']]
            for segment in shuffled_segments:
                input_tokens.extend([SPECIAL_TOKENS['[SPAN]']] + segment + [SPECIAL_TOKENS['[ES]']])

            if len(input_tokens) > self.seq_len:
                input_tokens = input_tokens[:self.seq_len]
            else:
                input_tokens += [SPECIAL_TOKENS['[PAD]']] * (self.seq_len - len(input_tokens))

            batch_inputs.append(torch.tensor(input_tokens, dtype=torch.long))
            batch_p_star.append(p_star)

        if not batch_inputs:
            return None, None, None

        inputs = torch.stack(batch_inputs)
        p_star = torch.stack(batch_p_star)

        # Generate attention mask
        attention_mask = self._generate_attention_mask(inputs)

        return inputs, attention_mask, p_star

    def _collate_fn_distractor(self, batch):
        task_config = self.task_configs.get('distractor_loc', {})
        L_min = task_config.get('L_min', 10)
        L_max = task_config.get('L_max', 50)
        sigma_scale = task_config.get('sigma_scale', 4.0)

        batch_x_prime = []
        batch_m_star = []
        batch_c = []
        batch_l = []

        for i in range(len(batch)):
            original_tokens, _ = batch[i]
            original_tokens = original_tokens.tolist()
            T = self.seq_len

            if T <= L_max:
                continue

            L = random.randint(L_min, L_max)
            s = random.randint(0, T - L)

            # Sample distractor from another example
            distractor_idx = random.choice([j for j in range(len(batch)) if i != j])
            distractor_tokens, _ = batch[distractor_idx]
            distractor_tokens = distractor_tokens.tolist()

            distractor_start = random.randint(0, len(distractor_tokens) - L) if len(distractor_tokens) > L else 0
            distractor_span = distractor_tokens[distractor_start : distractor_start + L]

            # Pad distractor if necessary
            if len(distractor_span) < L:
                distractor_span.extend([SPECIAL_TOKENS['[PAD]']] * (L - len(distractor_span)))

            x_prime_list = original_tokens[:s] + distractor_span + original_tokens[s + L:]
            x_prime_list = x_prime_list[:T]

            # Create soft mask
            center_pos = s + L / 2.0
            sigma = L / sigma_scale if sigma_scale > 0 else 1.0

            indices = torch.arange(T, dtype=torch.float32)
            m_star = torch.exp(-((indices - center_pos)**2) / (2 * sigma**2))

            if m_star.max() > 0:
                m_star = m_star / m_star.max()

            # Create center and length targets
            c = (s + L / 2.0) / T
            l = L / T

            batch_x_prime.append(torch.tensor(x_prime_list, dtype=torch.long))
            batch_m_star.append(m_star)
            batch_c.append(torch.tensor(c, dtype=torch.float32))
            batch_l.append(torch.tensor(l, dtype=torch.float32))

        if not batch_x_prime:
            return None, None, None, None

        x_prime_tensor = torch.stack(batch_x_prime)
        m_star_tensor = torch.stack(batch_m_star)
        c_tensor = torch.stack(batch_c)
        l_tensor = torch.stack(batch_l)

        return x_prime_tensor, m_star_tensor, c_tensor, l_tensor

    def create_dataloaders(
        self, batch_size: int = 8, num_workers: int = 0, shuffle_train: bool = True
    ) -> Dict[str, Dict[str, DataLoader]]:
        datasets = self.create_datasets()
        dataloaders = {}
        if not datasets:
            print("Warning: No datasets were created. Returning empty dataloaders dict.")
            return dataloaders

        for split_name, dataset_obj in datasets.items():
            if not dataset_obj:
                print(f"Skipping DataLoader for {split_name} as dataset is empty or invalid.")
                continue

            dataloaders[split_name] = {}
            shuffle = shuffle_train if split_name == 'train' else False

            # Teacher forcing dataloader
            dataloaders[split_name]['teacher_forcing'] = DataLoader(
                dataset_obj, batch_size=16, shuffle=shuffle,
                num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                collate_fn=self._collate_fn_teacher_forcing
            )

            # Cocktail party dataloader
            if 'cocktail_party' in self.task_configs:
                dataloaders[split_name]['cocktail_party'] = DataLoader(
                    dataset_obj, batch_size=8, shuffle=shuffle,
                    num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                    collate_fn=self._collate_fn_cocktail_party
                )

            # Soft jigsaw dataloader
            if 'soft_jigsaw' in self.task_configs:
                dataloaders[split_name]['soft_jigsaw'] = DataLoader(
                    dataset_obj, batch_size=8, shuffle=shuffle,
                    num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                    collate_fn=self._collate_fn_soft_jigsaw
                )

            if 'distractor_loc' in self.task_configs:
                dataloaders[split_name]['distractor_loc'] = DataLoader(
                    dataset_obj, batch_size=4, shuffle=shuffle,
                    num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                    collate_fn=self._collate_fn_distractor
                )

        return dataloaders

    def _generate_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Generates a custom attention mask for span-based tasks.
        - Tokens within a [SPAN]...[ES] block can attend to each other.
        - Tokens outside of spans can attend to all other tokens.
        - [CLS] token can attend to all other tokens.
        """
        batch_size, seq_len = input_ids.shape
        span_id = torch.zeros_like(input_ids)

        in_span = False
        current_span_id = 1
        for b in range(batch_size):
            for t in range(seq_len):
                token = input_ids[b, t].item()
                if token == SPECIAL_TOKENS['[SPAN]']:
                    in_span = True
                    span_id[b, t] = current_span_id
                elif token == SPECIAL_TOKENS['[ES]']:
                    if in_span:
                        span_id[b, t] = current_span_id
                        in_span = False
                        current_span_id += 1
                elif in_span:
                    span_id[b, t] = current_span_id

        same_span = span_id[:, :, None] == span_id[:, None, :]

        # Allow tokens outside spans to attend to each other
        outside_span = span_id == 0
        outside_mask = outside_span[:, :, None] | outside_span[:, None, :]

        # Combine the masks
        attention_mask = same_span | outside_mask

        # CLS token can attend to everything and be attended by everything
        cls_token_mask = (input_ids == SPECIAL_TOKENS['[CLS]'])
        attention_mask |= cls_token_mask[:, :, None]
        attention_mask |= cls_token_mask[:, None, :]

        return attention_mask

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
