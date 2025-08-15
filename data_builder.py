import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from typing import Optional, Dict, Any, List
import random

# Define special tokens according to the new spec
SPECIAL_TOKENS = {
    '[PAD]': 0,
    '[CLS]': 1,
    '[MASK]': 2,
    '[SPAN]': 3,
    '[ES]': 4,
    '[MASKQ]': 5
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
        # The collate functions will handle adding special tokens like [CLS]
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
        task_configs: dict = None,
        model_config: dict = None,
    ):
        self.on_the_fly_tokenization = on_the_fly_tokenization
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.seq_len = seq_len
        self.max_samples = max_samples if max_samples is not None else float('inf')
        self.vocab_size = vocab_size + NUM_SPECIAL_TOKENS
        self.max_eval_tokens = max_eval_tokens
        self.task_configs = task_configs or {}
        self.model_config = model_config or {}
        self.bidir_prefix_len = self.model_config.get('bidirectional_prefix_len', 1)

        print(f"Using UTF-8 byte tokenization with vocabulary size: {self.vocab_size}")
        print(f"Max evaluation tokens per split: {self.max_eval_tokens}")
        if self.max_samples != float('inf'):
            print(f"Will attempt to load up to {self.max_samples} samples from the dataset.")
        else:
            print("Will attempt to load all available samples from the dataset.")

    def _tokenize_text(self, text: str) -> list:
        tokens = []
        i = 0
        sorted_special_tokens = sorted(SPECIAL_TOKENS.items(), key=lambda x: len(x[0]), reverse=True)
        while i < len(text):
            found = False
            for token_str, token_id in sorted_special_tokens:
                if text[i:].startswith(token_str):
                    tokens.append(token_id)
                    i += len(token_str)
                    found = True
                    break
            if not found:
                char_bytes = text[i].encode('utf-8')
                for byte in char_bytes:
                    tokens.append(byte + NUM_SPECIAL_TOKENS)
                i += 1
        return tokens

    def _detokenize_bytes(self, tokens: list, skip_special_tokens=False) -> str:
        special_token_map = {v: k for k, v in SPECIAL_TOKENS.items()}
        byte_buffer = []
        result_parts = []
        for t in tokens:
            if t in special_token_map:
                if byte_buffer:
                    try:
                        result_parts.append(bytes(byte_buffer).decode('utf-8', errors='replace'))
                    finally:
                        byte_buffer = []
                if not skip_special_tokens:
                    result_parts.append(special_token_map[t])
            elif t >= NUM_SPECIAL_TOKENS:
                byte_buffer.append(t - NUM_SPECIAL_TOKENS)

        if byte_buffer:
            try:
                result_parts.append(bytes(byte_buffer).decode('utf-8', errors='replace'))
            finally:
                byte_buffer = []

        return "".join(result_parts)

    def _make_roles(self, ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        is_prefix = torch.zeros_like(ids, dtype=torch.bool)
        if self.bidir_prefix_len > 0:
            is_prefix[:, :self.bidir_prefix_len] = True

        is_mask_marker = (ids == SPECIAL_TOKENS['[MASK]'])
        is_maskq = (ids == SPECIAL_TOKENS['[MASKQ]'])

        in_span = torch.zeros_like(ids, dtype=torch.bool)
        span_id = torch.full_like(ids, -1, dtype=torch.long)

        for i in range(ids.shape[0]):
            current_span_id = 1
            in_current_span = False
            for j in range(ids.shape[1]):
                token_id = ids[i, j].item()
                if token_id == SPECIAL_TOKENS['[SPAN]']:
                    in_current_span = True

                if in_current_span:
                    in_span[i, j] = True
                    span_id[i, j] = current_span_id

                if token_id == SPECIAL_TOKENS['[ES]']:
                    if in_current_span:
                        in_current_span = False
                        current_span_id += 1

        return {
            'is_prefix': is_prefix.to(ids.device),
            'is_mask_marker': is_mask_marker.to(ids.device),
            'is_maskq': is_maskq.to(ids.device),
            'in_span': in_span.to(ids.device),
            'span_id': span_id.to(ids.device),
        }

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

        try:
            print("Attempting Method 1: Load C4 'en' (streaming)...")
            dataset_stream = load_dataset(
                self.dataset_name, name=self.dataset_config, streaming=True, split='train', trust_remote_code=True
            )
            loaded_samples = self._process_iterable_dataset(dataset_stream, "C4 'en' streaming")
        except Exception as e_c4_en_stream:
            print(f"Method 1 (C4 'en' streaming) failed: {e_c4_en_stream}")

        if not loaded_samples:
            try:
                print("Attempting Method 1.5: Load C4 'en' (non-streaming, sliced)...")
                fetch_n = int(self.max_samples * 1.5) if self.max_samples != float('inf') else 5000
                dataset_non_stream = load_dataset(
                    self.dataset_name, name=self.dataset_config, split=f'train[:{fetch_n}]', trust_remote_code=True
                )
                loaded_samples = self._process_iterable_dataset(dataset_non_stream, "C4 'en' non-streaming")
            except Exception as e_c4_en_non_stream:
                print(f"Method 1.5 (C4 'en' non-streaming) failed: {e_c4_en_non_stream}")

        if not loaded_samples:
            print("All C4 'en' attempts failed. Falling back to other datasets.")
            try:
                print("Attempting Method 3: Load wikitext (streaming)...")
                dataset_m3 = load_dataset("wikitext", "wikitext-2-raw-v1", streaming=True, split='train')
                loaded_samples = self._process_iterable_dataset(dataset_m3, "wikitext streaming")
            except Exception as e_method3:
                print(f"Method 3 (wikitext, streaming) failed: {e_method3}")

        if not loaded_samples:
            print("All loading methods failed. Falling back to simple text dataset.")
            return self._create_fallback_dataset()

        train_split = int(0.8 * len(loaded_samples))
        val_split = int(0.9 * len(loaded_samples))
        return {
            'train': loaded_samples[:train_split],
            'validation': loaded_samples[train_split:val_split],
            'test': loaded_samples[val_split:]
        }

    def _create_fallback_dataset(self):
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.", "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.", "Natural language processing enables computers to understand human language.",
        ]
        num_repetitions = (self.max_samples // 4 if self.max_samples != float('inf') else 100)
        full_sample_texts = sample_texts * max(1, num_repetitions)
        text_block = '\n'.join(full_sample_texts)
        len_block = len(text_block)
        train_end, val_end = int(0.8 * len_block), int(0.9 * len_block)
        return {
            'train': [{'text': text_block[:train_end]}],
            'validation': [{'text': text_block[train_end:val_end]}],
            'test': [{'text': text_block[val_end:]}]
        }
    
    def tokenize_dataset(self, dataset):
        tokenized_data = {}
        for split_name, split_data_list in dataset.items():
            all_text = "".join(item.get('text', '') + "\n" for item in split_data_list if item.get('text', '').strip())
            tokenized_data[split_name] = self._tokenize_text(all_text) if all_text.strip() else []
        return tokenized_data
    
    def create_datasets(self):
        raw_dataset = self.load_raw_dataset()
        datasets = {}
        for split_name, data in raw_dataset.items():
            if data:
                # Always use on-the-fly for task-based learning
                datasets[split_name] = OnTheFlyTokenizedDataset(data, self.seq_len, self._tokenize_text)
        return datasets

    def _collate_fn_teacher_forcing(self, batch):
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        return inputs, targets, None # Roles are None

    def _collate_fn_cocktail_party(self, batch):
        task_config = self.task_configs.get('cocktail_party', {})
        num_distractors = task_config.get('num_distractors', 3)
        min_span_size = task_config.get('min_span_size', 10)
        max_span_size = task_config.get('max_span_size', 50)
        pad_id = SPECIAL_TOKENS['[PAD]']

        batch_inputs, batch_correct_indices = [], []

        for i in range(len(batch)):
            original_tokens_padded, _ = batch[i]
            try:
                first_pad_idx = original_tokens_padded.tolist().index(pad_id)
                original_tokens = original_tokens_padded[:first_pad_idx].tolist()
            except ValueError:
                original_tokens = original_tokens_padded.tolist()

            span_size = random.randint(min_span_size, max_span_size)
            if len(original_tokens) <= span_size: continue
            span_start = random.randint(1, len(original_tokens) - span_size -1) # Avoid CLS
            true_span = original_tokens[span_start : span_start + span_size]

            distractors = []
            for _ in range(num_distractors):
                distractor_idx = random.choice([j for j in range(len(batch)) if i != j])
                dist_tokens_padded, _ = batch[distractor_idx]
                try:
                    first_pad_idx_dist = dist_tokens_padded.tolist().index(pad_id)
                    distractor_tokens = dist_tokens_padded[:first_pad_idx_dist].tolist()
                except ValueError:
                    distractor_tokens = dist_tokens_padded.tolist()
                if len(distractor_tokens) <= span_size: continue
                distractor_start = random.randint(1, len(distractor_tokens) - span_size-1)
                distractors.append(distractor_tokens[distractor_start : distractor_start + span_size])

            if not distractors: continue

            all_spans = [true_span] + distractors
            correct_idx = 0

            indices = list(range(len(all_spans)))
            random.shuffle(indices)

            shuffled_spans = [all_spans[i] for i in indices]
            correct_idx = indices.index(correct_idx)

            spans_tokens = []
            for span_toks in shuffled_spans:
                spans_tokens.extend([SPECIAL_TOKENS['[SPAN]']] + span_toks + [SPECIAL_TOKENS['[ES]']])

            context = original_tokens[:span_start] + [SPECIAL_TOKENS['[MASK]']] + original_tokens[span_start + span_size:]

            available_len = self.seq_len - len(spans_tokens) - 1 # for MASKQ
            context = [SPECIAL_TOKENS['[CLS]']] + context
            final_context = context[:available_len]

            final_sequence = final_context + spans_tokens + [SPECIAL_TOKENS['[MASKQ]']]

            if len(final_sequence) < self.seq_len:
                final_sequence.extend([pad_id] * (self.seq_len - len(final_sequence)))

            if final_sequence.count(SPECIAL_TOKENS['[MASKQ]']) != 1: continue
            if len(shuffled_spans) == 0: continue

            batch_inputs.append(torch.tensor(final_sequence, dtype=torch.long))
            batch_correct_indices.append(torch.tensor(correct_idx, dtype=torch.long))

        if not batch_inputs: return None, None, None

        inputs = torch.stack(batch_inputs)
        correct_indices = torch.stack(batch_correct_indices)
        roles = self._make_roles(inputs)

        assert (roles['is_maskq'].sum(dim=1) == 1).all(), "Each sequence must have exactly one [MASKQ]"

        return inputs, correct_indices, roles

    def _collate_fn_soft_jigsaw(self, batch):
        task_config = self.task_configs.get('soft_jigsaw', {})
        M = task_config.get('M', 5)
        batch_inputs, batch_p_star = [], []

        for item in batch:
            text_tokens, _ = item
            text = self.decode_tokens(text_tokens.tolist(), skip_special_tokens=True)
            sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.split()) > 3]

            if len(sentences) < M: continue

            start_index = random.randint(0, len(sentences) - M)
            original_sentences = sentences[start_index : start_index + M]

            indexed_sentences = list(enumerate(original_sentences))
            random.shuffle(indexed_sentences)
            shuffled_sentences = [s for _, s in indexed_sentences]
            original_indices = [i for i, _ in indexed_sentences]

            p_star = torch.zeros(M, M)
            for i in range(M): p_star[i, original_indices[i]] = 1

            wrapped_segments = "".join(f"[SPAN]{seg}[ES]" for seg in shuffled_sentences)
            input_text = f"[CLS]{wrapped_segments}[MASKQ]"
            tokens = self._tokenize_text(input_text)[:self.seq_len]

            if tokens.count(SPECIAL_TOKENS['[MASKQ]']) == 0:
                if len(tokens) == self.seq_len: tokens[-1] = SPECIAL_TOKENS['[MASKQ]']
                else: tokens.append(SPECIAL_TOKENS['[MASKQ]'])

            tokens.extend([SPECIAL_TOKENS['[PAD]']] * (self.seq_len - len(tokens)))

            batch_inputs.append(torch.tensor(tokens, dtype=torch.long))
            batch_p_star.append(p_star)

        if not batch_inputs: return None, None, None

        inputs = torch.stack(batch_inputs)
        p_star = torch.stack(batch_p_star)
        roles = self._make_roles(inputs)
        assert (roles['is_maskq'].sum(dim=1) == 1).all(), "Each sequence must have exactly one [MASKQ]"

        return inputs, p_star, roles

    def _collate_fn_distractor(self, batch):
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

            distractor_idx = random.choice([j for j in range(len(batch)) if i != j])
            distractor_tokens, _ = batch[distractor_idx]
            distractor_span = distractor_tokens.tolist()[s:s+L]

            x_prime_list = original_tokens[:s] + distractor_span + original_tokens[s + L:]
            x_prime_list = x_prime_list[:T]

            center_pos, sigma = s + L / 2.0, L / 4.0
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
        datasets = self.create_datasets()
        dataloaders = {}
        if not datasets: return dataloaders

        for split_name, dataset_obj in datasets.items():
            if not dataset_obj: continue
            dataloaders[split_name] = {}
            shuffle = shuffle_train if split_name == 'train' else False

            task_collate_map = {
                'teacher_forcing': (self._collate_fn_teacher_forcing, 16),
                'cocktail_party': (self._collate_fn_cocktail_party, 8),
                'soft_jigsaw': (self._collate_fn_soft_jigsaw, 8),
                'distractor_loc': (self._collate_fn_distractor, 4)
            }

            for task_name, (collate_fn, bs) in task_collate_map.items():
                if task_name in self.task_configs:
                    dataloaders[split_name][task_name] = DataLoader(
                        dataset_obj, batch_size=bs, shuffle=shuffle,
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
    on_the_fly_tokenization: bool = True, # Default to True for task-based learning
    task_configs: dict = None,
    model_config: dict = None
) -> DataBuilder:
    return DataBuilder(
        dataset_name=dataset_name, dataset_config=dataset_config,
        seq_len=seq_len, max_samples=max_samples,
        max_eval_tokens=max_eval_tokens,
        on_the_fly_tokenization=on_the_fly_tokenization,
        task_configs=task_configs,
        model_config=model_config
    )

if __name__ == "__main__":
    print("Testing DataBuilder...")
    task_configs = {
        'teacher_forcing': {}, 'cocktail_party': {}, 'soft_jigsaw': {}, 'distractor_loc': {}
    }
    model_config = {'bidirectional_prefix_len': 1}
    data_builder = create_data_builder(
        seq_len=128, max_samples=100, task_configs=task_configs, model_config=model_config
    )
    dataloaders = data_builder.create_dataloaders(batch_size=4)
    for split in ['train', 'validation']:
        if split in dataloaders:
            for task, loader in dataloaders[split].items():
                print(f"\n--- Testing {split}/{task} ---")
                try:
                    batch = next(iter(loader))
                    print(f"Batch loaded for {task}. Number of items: {len(batch)}")
                    for i, item in enumerate(batch):
                        if item is not None:
                            print(f"Item {i} shape/type: {item.shape if isinstance(item, torch.Tensor) else type(item)}")
                        else:
                            print(f"Item {i} is None")
                except StopIteration:
                    print("Could not get a batch.")
                except Exception as e:
                    print(f"Error during dataloader iteration test for {task}: {e}")
    print("\nDataBuilder test completed!")
