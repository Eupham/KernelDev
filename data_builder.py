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

# Updated Special Tokens as per instructions
SPECIAL_TOKENS = {
    '[PAD]': 0,
    '[CLS]': 1,
    '[MASK]': 2,
    '[SPAN]': 3,
    '[ES]': 4,
    '[MASKQ]': 5,
}
NUM_SPECIAL_TOKENS = 6

class OnTheFlyTokenizedDataset(Dataset):
    def __init__(self, raw_data, seq_len=512, tokenizer_fn=None):
        self.raw_data = raw_data
        self.seq_len = seq_len
        self.tokenizer_fn = tokenizer_fn

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        text = self.raw_data[idx]['text']
        # The collate functions will now handle prepending [CLS]
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
        vocab_size: int = 256, # This is the base vocab size for bytes
        max_eval_tokens: int = 50000,
        on_the_fly_tokenization: bool = False,
        task_configs: dict = None,
        bidirectional_prefix_len: int = 1,
    ):
        self.on_the_fly_tokenization = on_the_fly_tokenization
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.seq_len = seq_len
        self.max_samples = max_samples if max_samples is not None else float('inf')
        self.vocab_size = vocab_size + NUM_SPECIAL_TOKENS
        self.max_eval_tokens = max_eval_tokens
        self.task_configs = task_configs or {}
        self.bidirectional_prefix_len = bidirectional_prefix_len

        print(f"Using UTF-8 byte tokenization with vocabulary size: {self.vocab_size}")
        print(f"Max evaluation tokens per split: {self.max_eval_tokens}")
        if self.max_samples != float('inf'):
            print(f"Will attempt to load up to {self.max_samples} samples from the dataset.")
        else:
            print("Will attempt to load all available samples from the dataset.")

    def _tokenize_text(self, text: str) -> list:
        tokens = []
        i = 0
        text_bytes = text.encode('utf-8')
        byte_idx = 0
        while byte_idx < len(text_bytes):
            # Check for special tokens by decoding a small slice of bytes
            # This is imperfect but a reasonable heuristic for byte-based tokenization
            found_special = False
            # To check for special tokens, we look at the string representation
            # We need to be careful with byte boundaries.
            # A simpler way is to do this on the string before encoding.
            # Let's stick to the user's request of byte-accurate tokenization.
            # The original implementation was character-based for special tokens, which is flawed.
            # A better approach is to split the text by special tokens first.

            # Let's refine this. We can't reliably find special tokens in a byte stream.
            # We should operate on the string level for special tokens.
            import re
            # Create a regex to split by special tokens, keeping them
            special_tokens_pattern = f"({'|'.join(re.escape(k) for k in SPECIAL_TOKENS.keys())})"
            parts = re.split(special_tokens_pattern, text)

            final_tokens = []
            for part in parts:
                if not part:
                    continue
                if part in SPECIAL_TOKENS:
                    final_tokens.append(SPECIAL_TOKENS[part])
                else:
                    # This part is normal text, encode to bytes and shift
                    encoded_bytes = part.encode('utf-8')
                    final_tokens.extend([b + NUM_SPECIAL_TOKENS for b in encoded_bytes])
            return final_tokens

    def _detokenize_bytes(self, tokens: list, skip_special_tokens=False) -> str:
        special_token_map = {v: k for k, v in SPECIAL_TOKENS.items()}
        byte_buffer = []
        result_parts = []

        for t in tokens:
            if t in special_token_map:
                if byte_buffer:
                    try:
                        result_parts.append(bytes(byte_buffer).decode('utf-8', errors='replace'))
                    except UnicodeDecodeError:
                        result_parts.append("<<DECODE_ERROR>>")
                    byte_buffer = []

                if not skip_special_tokens:
                    result_parts.append(special_token_map[t])
            else:
                byte_buffer.append(t - NUM_SPECIAL_TOKENS)

        if byte_buffer:
            try:
                result_parts.append(bytes(byte_buffer).decode('utf-8', errors='replace'))
            except UnicodeDecodeError:
                result_parts.append("<<DECODE_ERROR>>")

        return "".join(result_parts)

    def _make_roles(self, ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T = ids.shape
        device = ids.device

        # is_prefix
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        is_prefix = (positions < self.bidirectional_prefix_len).bool()

        # is_mask_marker and is_maskq
        is_mask_marker = (ids == SPECIAL_TOKENS['[MASK]']).bool()
        is_maskq = (ids == SPECIAL_TOKENS['[MASKQ]']).bool()

        # in_span and span_id
        in_span = torch.zeros_like(ids, dtype=torch.bool)
        span_id = torch.full_like(ids, -1, dtype=torch.long)

        for b in range(B):
            current_span_idx = 0
            is_in_span = False
            for t in range(T):
                token_id = ids[b, t].item()
                if token_id == SPECIAL_TOKENS['[PAD]']: # Stop processing at first pad token
                    break
                if token_id == SPECIAL_TOKENS['[SPAN]']:
                    is_in_span = True
                    current_span_idx += 1

                if is_in_span:
                    in_span[b, t] = True
                    span_id[b, t] = current_span_idx

                if token_id == SPECIAL_TOKENS['[ES]']:
                    is_in_span = False

        return {
            'is_prefix': is_prefix.contiguous(),
            'is_mask_marker': is_mask_marker.contiguous(),
            'is_maskq': is_maskq.contiguous(),
            'in_span': in_span.contiguous(),
            'span_id': span_id.contiguous(),
        }

    # ... (load_raw_dataset and helpers are fine, no changes needed there) ...
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
                if len(loaded_samples) == 0 :
                    raise ValueError("Triggering non-streaming C4 'en' due to 0 samples from stream.")
            print(f"Successfully processed {len(loaded_samples)} samples from C4 'en' stream.")
        except Exception as e_c4_en_stream:
            print(f"Method 1 (C4 'en' streaming) failed: {e_c4_en_stream}")
            loaded_samples = []

            if not loaded_samples or (len(loaded_samples) < self.max_samples and self.max_samples != float('inf')):
                try:
                    print("Attempting Method 1.5: Load C4 'en' (non-streaming, sliced)...")
                    fetch_n = int(self.max_samples * 1.5) if self.max_samples != float('inf') else 5000
                    fetch_n = max(fetch_n, 100)
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
                    loaded_samples = []

        if loaded_samples and (len(loaded_samples) >= self.max_samples or self.max_samples == float('inf')):
            print(f"Successfully loaded {len(loaded_samples)} samples using C4 'en'.")
        elif loaded_samples:
             print(f"Loaded {len(loaded_samples)} (less than {self.max_samples}) from C4 'en'. Proceeding with these or trying other datasets.")
        else:
            print("All C4 'en' attempts (streaming/non-streaming) failed or yielded no samples.")

        if not loaded_samples or (len(loaded_samples) < self.max_samples and self.max_samples != float('inf')):
            print(f"Attempting other datasets as C4 'en' yielded {len(loaded_samples)}/{self.max_samples} samples.")
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

        if not loaded_samples or (len(loaded_samples) < self.max_samples and self.max_samples != float('inf')):
            print(f"Attempting wikitext as previous methods yielded {len(loaded_samples)}/{self.max_samples} samples.")
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

        if loaded_samples and (len(loaded_samples) >= self.max_samples or self.max_samples == float('inf')):
            print(f"Final dataset loaded with {len(loaded_samples)} samples.")
        elif loaded_samples:
            print(f"Warning: Final dataset loaded with {len(loaded_samples)} samples, requested {self.max_samples}. Using available data.")
        else:
            print("All primary dataset loading methods failed or yielded no usable samples. Falling back to simple text dataset...")
            return self._create_fallback_dataset()

        train_split = int(0.8 * len(loaded_samples))
        val_split = int(0.9 * len(loaded_samples))
        if train_split == len(loaded_samples) and len(loaded_samples) > 0: train_split = len(loaded_samples) -2
        if val_split <= train_split and val_split < len(loaded_samples) -1 : val_split = train_split + 1
        if val_split >= len(loaded_samples): val_split = len(loaded_samples) -1
        if train_split < 0: train_split = 0

        final_train_data = loaded_samples[:train_split]
        final_val_data = loaded_samples[train_split:val_split] if val_split > train_split else []
        final_test_data = loaded_samples[val_split:] if val_split < len(loaded_samples) else []

        if not final_train_data and loaded_samples: final_train_data = loaded_samples
        if not final_val_data and final_train_data: final_val_data = final_train_data[:max(1, len(final_train_data)//10)]
        if not final_test_data and final_train_data: final_test_data = final_train_data[:max(1, len(final_train_data)//10)]

        print(f"Returning dataset splits: train={len(final_train_data)}, val={len(final_val_data)}, test={len(final_test_data)}")
        return {
            'train': final_train_data,
            'validation': final_val_data,
            'test': final_test_data
        }

    def _create_fallback_dataset(self):
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
        ]
        num_repetitions = (self.max_samples // 10 if self.max_samples != float('inf') and self.max_samples > 10 else 100)
        num_repetitions = max(num_repetitions, 1)
        full_sample_texts = sample_texts * num_repetitions
        
        num_fallback_texts = max(20, len(full_sample_texts))
        text_block = '\n'.join(full_sample_texts[:num_fallback_texts])

        len_block = len(text_block)
        train_end = int(0.8 * len_block)
        val_end = int(0.9 * len_block)

        return {
            'train': [{'text': text_block[:train_end]}],
            'validation': [{'text': text_block[train_end:val_end]}],
            'test': [{'text': text_block[val_end:]}]
        }

    def tokenize_dataset(self, dataset):
        tokenized_data = {}
        for split_name, split_data_list in dataset.items():
            print(f"Tokenizing {split_name} split...")
            all_text = ""
            if not isinstance(split_data_list, list) or not split_data_list:
                tokenized_data[split_name] = []
                continue

            for item in split_data_list:
                if isinstance(item, dict) and 'text' in item and item['text']:
                    all_text += item['text'] + "\n"
            
            tokens = self._tokenize_text(all_text) if all_text.strip() else []
            tokenized_data[split_name] = tokens
        return tokenized_data

    def create_datasets(self):
        raw_dataset = self.load_raw_dataset()
        # OnTheFlyTokenizedDataset is simplified and collate functions will handle tokenization logic
        return raw_dataset

    def _collate_fn_teacher_forcing(self, batch):
        batch_tokens = [item[0].tolist() for item in batch]

        # Add [CLS] token, truncate, and pad
        processed_batch = []
        for tokens in batch_tokens:
            final_tokens = [SPECIAL_TOKENS['[CLS]']] + tokens
            final_tokens = final_tokens[:self.seq_len + 1]
            pad_len = (self.seq_len + 1) - len(final_tokens)
            final_tokens.extend([SPECIAL_TOKENS['[PAD]']] * pad_len)
            processed_batch.append(final_tokens)

        tokens_tensor = torch.tensor(processed_batch, dtype=torch.long)
        inputs = tokens_tensor[:, :-1]
        targets = tokens_tensor[:, 1:]

        # For plain teacher forcing, roles can be None. The model will use SDPA.
        return inputs, targets, None

    def _collate_fn_cocktail_party(self, batch):
        task_config = self.task_configs.get('cocktail_party', {})
        num_distractors = task_config.get('num_distractors', 3)
        min_span_size = task_config.get('min_span_size', 10)
        max_span_size = task_config.get('max_span_size', 50)

        batch_inputs, batch_correct_indices = [], []

        for i in range(len(batch)):
            # In on-the-fly mode, batch items are dicts {'text': ...}
            # This collate needs raw text, so we adjust the dataset handling.
            # Assuming batch items are (tokens, _) for now.
            original_tokens, _ = batch[i]
            original_tokens = original_tokens.tolist()

            pad_id = SPECIAL_TOKENS['[PAD]']
            try:
                first_pad_idx = original_tokens.index(pad_id)
                original_tokens = original_tokens[:first_pad_idx]
            except ValueError:
                pass # no padding

            span_size = random.randint(min_span_size, max_span_size)
            if len(original_tokens) <= span_size: continue

            span_start = random.randint(0, len(original_tokens) - span_size)
            true_span = original_tokens[span_start : span_start + span_size]

            distractors = []
            for _ in range(num_distractors):
                dist_idx = random.choice([j for j in range(len(batch)) if i != j])
                dist_tokens, _ = batch[dist_idx]
                dist_tokens = dist_tokens.tolist()
                try: first_pad_idx_dist = dist_tokens.index(pad_id); dist_tokens = dist_tokens[:first_pad_idx_dist]
                except ValueError: pass
                if len(dist_tokens) <= span_size: continue
                dist_start = random.randint(0, len(dist_tokens) - span_size)
                distractors.append(dist_tokens[dist_start : dist_start + span_size])

            if not distractors: continue

            all_spans = [true_span] + distractors
            correct_idx = 0

            indexed_spans = list(enumerate(all_spans))
            random.shuffle(indexed_spans)

            shuffled_spans = [span for _, span in indexed_spans]
            correct_idx = [idx for idx, _ in indexed_spans].index(0)

            # Build sequence: prefix [CLS] context [MASK] context [SPAN]...[ES]... [MASKQ]
            prefix = [SPECIAL_TOKENS['[CLS]']]
            masked_context = original_tokens[:span_start] + [SPECIAL_TOKENS['[MASK]']] + original_tokens[span_start + span_size:]

            spans_str = []
            for span_toks in shuffled_spans:
                spans_str.extend([SPECIAL_TOKENS['[SPAN]']] + span_toks + [SPECIAL_TOKENS['[ES]']])

            # Leave space for [MASKQ]
            available_len = self.seq_len - len(prefix) - len(spans_str) - 1
            final_context = prefix + masked_context[:available_len]

            final_sequence = final_context + spans_str + [SPECIAL_TOKENS['[MASKQ]']]
            final_sequence = final_sequence[:self.seq_len]

            pad_len = self.seq_len - len(final_sequence)
            final_sequence.extend([pad_id] * pad_len)

            batch_inputs.append(torch.tensor(final_sequence, dtype=torch.long))
            batch_correct_indices.append(torch.tensor(correct_idx, dtype=torch.long))

        if not batch_inputs:
            return torch.empty(0), torch.empty(0), None

        inputs = torch.stack(batch_inputs)
        correct_indices = torch.stack(batch_correct_indices)
        roles = self._make_roles(inputs)

        return inputs, correct_indices, roles

    def _collate_fn_soft_jigsaw(self, batch):
        task_config = self.task_configs.get('soft_jigsaw', {})
        M = task_config.get('M', 5)

        batch_inputs, batch_p_star = [], []

        for item in batch:
            text_tokens, _ = item
            text = self.decode_tokens(text_tokens.tolist(), skip_special_tokens=True)
            sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 10]

            if len(sentences) < M: continue

            start_index = random.randint(0, len(sentences) - M)
            original_sentences = sentences[start_index : start_index + M]

            indexed_sentences = list(enumerate(original_sentences))
            random.shuffle(indexed_sentences)

            shuffled_sentences = [s for _, s in indexed_sentences]
            original_indices = [i for i, _ in indexed_sentences]

            p_star = torch.zeros(M, M)
            for i in range(M): p_star[i, original_indices[i]] = 1

            # Format: [CLS] + spans + [MASKQ]
            spans_text = "".join(f"[SPAN]{seg}[ES]" for seg in shuffled_sentences)
            input_text = f"[CLS]{spans_text}[MASKQ]"
            tokens = self._tokenize_text(input_text)

            tokens = tokens[:self.seq_len]
            pad_len = self.seq_len - len(tokens)
            tokens.extend([SPECIAL_TOKENS['[PAD]']] * pad_len)

            batch_inputs.append(torch.tensor(tokens, dtype=torch.long))
            batch_p_star.append(p_star)

        if not batch_inputs:
            return None, None, None

        inputs = torch.stack(batch_inputs)
        p_star = torch.stack(batch_p_star)
        roles = self._make_roles(inputs)

        return inputs, p_star, roles

    def _collate_fn_distractor(self, batch):
        # This task remains mostly the same, just returning None for roles
        task_config = self.task_configs.get('distractor_loc', {})
        L_min, L_max = task_config.get('L_min', 10), task_config.get('L_max', 50)

        batch_x_prime, batch_m_star, batch_c, batch_l = [], [], [], []

        for i in range(len(batch)):
            original_tokens, _ = batch[i]
            original_tokens = original_tokens.tolist()
            T = self.seq_len
            if T <= L_max: continue

            L = random.randint(L_min, L_max)
            s = random.randint(0, T - L)

            dist_idx = random.choice([j for j in range(len(batch)) if i != j])
            dist_tokens, _ = batch[dist_idx]
            dist_tokens = dist_tokens.tolist()
            dist_start = random.randint(0, len(dist_tokens) - L) if len(dist_tokens) > L else 0
            distractor_span = dist_tokens[dist_start : dist_start + L]
            if len(distractor_span) < L: distractor_span.extend([SPECIAL_TOKENS['[PAD]']] * (L - len(distractor_span)))

            x_prime_list = original_tokens[:s] + distractor_span + original_tokens[s + L:]
            x_prime_list = x_prime_list[:T]

            center_pos = s + L / 2.0
            sigma = L / 4.0
            indices = torch.arange(T, dtype=torch.float32)
            m_star = torch.exp(-((indices - center_pos)**2) / (2 * sigma**2))
            if m_star.max() > 0: m_star /= m_star.max()

            batch_x_prime.append(torch.tensor(x_prime_list, dtype=torch.long))
            batch_m_star.append(m_star)
            batch_c.append(torch.tensor((s + L / 2.0) / T, dtype=torch.float32))
            batch_l.append(torch.tensor(L / T, dtype=torch.float32))

        if not batch_x_prime: return None, None, None, None, None

        return torch.stack(batch_x_prime), torch.stack(batch_m_star), torch.stack(batch_c), torch.stack(batch_l), None

    def create_dataloaders(
        self, batch_size: int = 8, num_workers: int = 0, shuffle_train: bool = True
    ) -> Dict[str, Dict[str, DataLoader]]:

        raw_datasets = self.create_datasets()
        dataloaders = {}
        if not raw_datasets:
            print("Warning: No datasets were created.")
            return dataloaders

        class RawTextDataset(Dataset):
            def __init__(self, data, seq_len, tokenizer_fn):
                self.data = data
                self.seq_len = seq_len
                self._tokenize_text = tokenizer_fn

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                text = self.data[idx]['text']
                tokens = self._tokenize_text(text)
                tokens = tokens[:self.seq_len + 1]
                pad_len = (self.seq_len + 1) - len(tokens)
                tokens.extend([SPECIAL_TOKENS['[PAD]']] * pad_len)
                x = torch.tensor(tokens[:-1], dtype=torch.long)
                y = torch.tensor(tokens[1:], dtype=torch.long)
                return x, y

        for split_name, data in raw_datasets.items():
            if not data: continue

            dataset_obj = RawTextDataset(data, self.seq_len, self._tokenize_text)

            dataloaders[split_name] = {}
            shuffle = shuffle_train if split_name == 'train' else False

            collate_map = {
                'teacher_forcing': (self._collate_fn_teacher_forcing, 16),
                'cocktail_party': (self._collate_fn_cocktail_party, 8),
                'soft_jigsaw': (self._collate_fn_soft_jigsaw, 8),
                'distractor_loc': (self._collate_fn_distractor, 4)
            }

            for task, (collate_fn, b_size) in collate_map.items():
                if task in self.task_configs:
                    dataloaders[split_name][task] = DataLoader(
                        dataset_obj, batch_size=b_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                        collate_fn=collate_fn
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
    task_configs: dict = None,
    bidirectional_prefix_len: int = 1
) -> DataBuilder:
    return DataBuilder(
        dataset_name=dataset_name, dataset_config=dataset_config,
        seq_len=seq_len, max_samples=max_samples,
        max_eval_tokens=max_eval_tokens,
        on_the_fly_tokenization=on_the_fly_tokenization,
        task_configs=task_configs,
        bidirectional_prefix_len=bidirectional_prefix_len
    )

if __name__ == "__main__":
    print("Testing DataBuilder...")
    # Example task configs for testing
    test_task_configs = {
        'teacher_forcing': {},
        'cocktail_party': {},
        'soft_jigsaw': {},
        'distractor_loc': {}
    }
    data_builder = create_data_builder(
        seq_len=128, max_samples=100, task_configs=test_task_configs
    )
    dataloaders = data_builder.create_dataloaders(batch_size=4)

    for split in ['train', 'validation']:
        if split in dataloaders:
            print(f"\n--- Testing {split} dataloaders ---")
            for task, loader in dataloaders[split].items():
                print(f"Testing task: {task}")
                try:
                    batch = next(iter(loader))
                    print(f"  Batch loaded successfully for {task}.")
                    # Unpack based on task
                    if task == 'teacher_forcing':
                        inputs, targets, roles = batch
                        print(f"  Inputs shape: {inputs.shape}")
                        print(f"  Targets shape: {targets.shape}")
                        print(f"  Roles is None: {roles is None}")
                    elif task == 'cocktail_party':
                        inputs, correct_idx, roles = batch
                        print(f"  Inputs shape: {inputs.shape}")
                        print(f"  Correct indices shape: {correct_idx.shape}")
                        print(f"  Roles keys: {roles.keys()}")
                        assert roles['is_maskq'].any(), "[MASKQ] missing in cocktail party"
                    elif task == 'soft_jigsaw':
                        inputs, p_star, roles = batch
                        if inputs is not None:
                            print(f"  Inputs shape: {inputs.shape}")
                            print(f"  P_star shape: {p_star.shape}")
                            print(f"  Roles keys: {roles.keys()}")
                            assert roles['is_maskq'].any(), "[MASKQ] missing in soft jigsaw"
                        else:
                            print("  Skipped (not enough data for a batch).")
                    elif task == 'distractor_loc':
                        x_prime, m_star, c, l, roles = batch
                        if x_prime is not None:
                            print(f"  x_prime shape: {x_prime.shape}")
                            print(f"  m_star shape: {m_star.shape}")
                            print(f"  Roles is None: {roles is None}")
                        else:
                            print("  Skipped (not enough data for a batch).")

                except StopIteration:
                    print(f"  Could not retrieve a batch for {task} (dataloader might be empty).")
                except Exception as e:
                    print(f"  ERROR during dataloader iteration test for task {task}: {e}")

    print("\nDataBuilder test completed!")
