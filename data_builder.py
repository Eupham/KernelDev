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
    '[MASKQ]': 5,
}
NUM_SPECIAL_TOKENS = 6
VOCAB_SIZE_BYTES = 256

def _make_roles(ids: torch.Tensor, bidir_prefix_len: int) -> Dict[str, torch.Tensor]:
    """
    Constructs role tensors from token IDs.
    """
    B, T = ids.shape
    device = ids.device

    is_prefix = torch.zeros((B, T), dtype=torch.bool, device=device)
    if bidir_prefix_len > 0:
        is_prefix[:, :bidir_prefix_len] = True

    is_mask_marker = (ids == SPECIAL_TOKENS['[MASK]'])
    is_maskq = (ids == SPECIAL_TOKENS['[MASKQ]'])

    in_span = torch.zeros((B, T), dtype=torch.bool, device=device)
    span_id = torch.full((B, T), -1, dtype=torch.long, device=device)

    span_token_id = SPECIAL_TOKENS['[SPAN]']
    es_token_id = SPECIAL_TOKENS['[ES]']

    for i in range(B):
        current_span_id = 1
        is_inside_span = False
        for j in range(T):
            token_id = ids[i, j].item()
            if token_id == span_token_id:
                is_inside_span = True
            elif token_id == es_token_id:
                if is_inside_span:
                    is_inside_span = False
                    current_span_id += 1
            elif is_inside_span:
                in_span[i, j] = True
                span_id[i, j] = current_span_id

    return {
        'is_prefix': is_prefix.contiguous(),
        'is_mask_marker': is_mask_marker.contiguous(),
        'is_maskq': is_maskq.contiguous(),
        'in_span': in_span.contiguous(),
        'span_id': span_id.contiguous(),
    }

class Tokenizer:
    def __init__(self):
        self.special_tokens_map = SPECIAL_TOKENS
        self.rev_special_tokens_map = {v: k for k, v in self.special_tokens_map.items()}

    def encode(self, text: str) -> list[int]:
        tokens = []
        i = 0
        while i < len(text):
            found_special = False
            for token_str, token_id in self.special_tokens_map.items():
                if text[i:].startswith(token_str):
                    tokens.append(token_id)
                    i += len(token_str)
                    found_special = True
                    break

            if not found_special:
                char = text[i]
                for byte in char.encode('utf-8'):
                    tokens.append(byte + NUM_SPECIAL_TOKENS)
                i += 1
        return tokens

    def decode(self, tokens: list[int], skip_special_tokens: bool = False) -> str:
        result_parts = []
        byte_buffer = bytearray()
        for t in tokens:
            if t < NUM_SPECIAL_TOKENS:
                if byte_buffer:
                    result_parts.append(byte_buffer.decode('utf-8', errors='replace'))
                    byte_buffer = bytearray()
                if not skip_special_tokens:
                    result_parts.append(self.rev_special_tokens_map.get(t, ''))
            else:
                byte_buffer.append(t - NUM_SPECIAL_TOKENS)

        if byte_buffer:
            result_parts.append(byte_buffer.decode('utf-8', errors='replace'))

        return "".join(result_parts)

class OnTheFlyTokenizedDataset(Dataset):
    def __init__(self, raw_data, seq_len=512, tokenizer=None):
        self.raw_data = raw_data
        self.seq_len = seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        text = self.raw_data[idx]['text']
        text = f"[CLS] {text}"
        tokens = self.tokenizer.encode(text)
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
        vocab_size: int = 262,
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
        self.vocab_size = vocab_size
        self.max_eval_tokens = max_eval_tokens
        self.task_configs = task_configs or {}
        self.tokenizer = Tokenizer()
        self.bidirectional_prefix_len = bidirectional_prefix_len

        print(f"Using UTF-8 byte tokenization with vocabulary size: {self.vocab_size}")
        print(f"Max evaluation tokens per split: {self.max_eval_tokens}")
        if self.max_samples != float('inf'):
            print(f"Will attempt to load up to {self.max_samples} samples from the dataset.")
        else:
            print("Will attempt to load all available samples from the dataset.")

    def _tokenize_text(self, text: str) -> list:
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens, skip_special_tokens=False):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().tolist()
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

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
            print("All C4 'en' attempts failed. Trying other datasets.")
            try:
                print("Attempting Method 2: Load C4 (no config) (streaming)...")
                dataset_m2 = load_dataset(self.dataset_name, streaming=True, split='train', trust_remote_code=True)
                loaded_samples = self._process_iterable_dataset(dataset_m2, "C4 (no config) streaming")
            except Exception as e_method2:
                print(f"Method 2 (C4 no config, streaming) failed: {e_method2}")
                try:
                    print("Attempting Method 3: Load wikitext (streaming)...")
                    dataset_m3 = load_dataset("wikitext", "wikitext-2-raw-v1", streaming=True, split='train')
                    loaded_samples = self._process_iterable_dataset(dataset_m3, "wikitext streaming")
                except Exception as e_method3:
                    print(f"Method 3 (wikitext, streaming) failed: {e_method3}")

        if not loaded_samples:
            print("All dataset loading methods failed. Falling back to simple text dataset...")
            return self._create_fallback_dataset()

        train_split = int(0.8 * len(loaded_samples))
        val_split = int(0.9 * len(loaded_samples))
        if train_split >= val_split: val_split = train_split + 1

        final_train_data = loaded_samples[:train_split]
        final_val_data = loaded_samples[train_split:val_split]
        final_test_data = loaded_samples[val_split:]

        return {
            'train': final_train_data, 'validation': final_val_data, 'test': final_test_data
        }

    def _create_fallback_dataset(self):
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.", "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.", "Natural language processing enables computers to understand human language.",
            "Transformers have revolutionized natural language processing.", "GPT models are based on the transformer architecture.",
            "Flash attention is an efficient implementation of the attention mechanism.", "PyTorch is a popular deep learning framework.",
            "CUDA enables parallel computing on NVIDIA GPUs.", "Tokenization is the process of converting text into numerical tokens."
        ]
        num_repetitions = (self.max_samples // 10 if self.max_samples != float('inf') else 100)
        full_sample_texts = sample_texts * max(num_repetitions, 1)
        text_block = '\n'.join(full_sample_texts)
        train_end = int(0.8 * len(text_block))
        val_end = int(0.9 * len(text_block))
        return {
            'train': [{'text': text_block[:train_end]}],
            'validation': [{'text': text_block[train_end:val_end]}],
            'test': [{'text': text_block[val_end:]}]
        }

    def create_datasets(self):
        raw_dataset = self.load_raw_dataset()
        datasets = {}
        for split_name, data in raw_dataset.items():
            if data:
                datasets[split_name] = OnTheFlyTokenizedDataset(data, self.seq_len, self.tokenizer)
        return datasets

    def _collate_fn_teacher_forcing(self, batch):
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        roles = _make_roles(inputs, self.bidirectional_prefix_len)
        return inputs, targets, roles

    def _collate_fn_cocktail_party(self, batch):
        task_config = self.task_configs.get('cocktail_party', {})
        num_distractors = task_config.get('num_distractors', 3)
        min_span_size = task_config.get('min_span_size', 100)
        max_span_size = task_config.get('max_span_size', 200)

        batch_inputs, batch_correct_indices = [], []
        pad_id = SPECIAL_TOKENS['[PAD]']

        for i in range(len(batch)):
            original_tokens, _ = batch[i]
            original_tokens = original_tokens.tolist()
            try:
                first_pad_idx = original_tokens.index(pad_id)
                original_tokens = original_tokens[:first_pad_idx]
            except ValueError:
                pass

            if len(original_tokens) <= span_size: continue
            span_size = random.randint(min_span_size, max_span_size)
            span_start = random.randint(0, len(original_tokens) - span_size)
            true_span = original_tokens[span_start : span_start + span_size]

            distractors = []
            possible_indices = [j for j in range(len(batch)) if i != j]
            if not possible_indices: continue

            for _ in range(num_distractors):
                dist_idx = random.choice(possible_indices)
                dist_tokens, _ = batch[dist_idx]
                dist_tokens = dist_tokens.tolist()
                try:
                    dist_tokens = dist_tokens[:dist_tokens.index(pad_id)]
                except ValueError:
                    pass
                if len(dist_tokens) <= span_size: continue
                dist_start = random.randint(0, len(dist_tokens) - span_size)
                distractors.append(dist_tokens[dist_start : dist_start + span_size])

            if not distractors: continue

            all_spans = [(true_span, 1)] + [(d, 0) for d in distractors]
            random.shuffle(all_spans)
            correct_idx = [label for _, label in all_spans].index(1)

            wrapper_tokens = []
            for span_toks, _ in all_spans:
                wrapper_tokens.extend([SPECIAL_TOKENS['[SPAN]']] + span_toks + [SPECIAL_TOKENS['[ES]']])
            wrapper_tokens.append(SPECIAL_TOKENS['[MASKQ]'])

            available_len = self.seq_len - len(wrapper_tokens)
            if available_len < 0: continue

            context = original_tokens[:span_start] + [SPECIAL_TOKENS['[MASK]']] + original_tokens[span_start+span_size:]
            final_sequence = context[:available_len] + wrapper_tokens

            if len(final_sequence) < self.seq_len:
                final_sequence.extend([pad_id] * (self.seq_len - len(final_sequence)))

            assert final_sequence.count(SPECIAL_TOKENS['[MASKQ]']) == 1
            batch_inputs.append(torch.tensor(final_sequence, dtype=torch.long))
            batch_correct_indices.append(torch.tensor(correct_idx, dtype=torch.long))

        if not batch_inputs:
            return torch.empty(0, self.seq_len, dtype=torch.long), torch.empty(0, dtype=torch.long), None

        inputs = torch.stack(batch_inputs)
        correct_indices = torch.stack(batch_correct_indices)
        roles = _make_roles(inputs, self.bidirectional_prefix_len)
        return inputs, correct_indices, roles

    def _collate_fn_soft_jigsaw(self, batch):
        task_config = self.task_configs.get('soft_jigsaw', {})
        M = task_config.get('M', 5)
        batch_inputs, batch_p_star = [], []
        pad_id = SPECIAL_TOKENS['[PAD]']

        for item in batch:
            tokens, _ = item
            text = self.decode_tokens(tokens.tolist(), skip_special_tokens=True)
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
            if len(sentences) < M: continue

            start_idx = random.randint(0, len(sentences) - M)
            original_sentences = sentences[start_idx : start_idx + M]

            indexed_sentences = list(enumerate(original_sentences))
            random.shuffle(indexed_sentences)
            shuffled_sentences = [s for _, s in indexed_sentences]
            original_indices = [i for i, _ in indexed_sentences]

            p_star = torch.zeros(M, M)
            for i in range(M): p_star[i, original_indices[i]] = 1

            wrapped_segs = "".join(f"[SPAN]{s}[ES]" for s in shuffled_sentences)
            final_text = f"[CLS]{wrapped_segs}[MASKQ]"
            tokens = self._tokenize_text(final_text)

            tokens = tokens[:self.seq_len]
            if tokens[-1] != SPECIAL_TOKENS['[MASKQ]']:
                if SPECIAL_TOKENS['[MASKQ]'] in tokens: tokens.remove(SPECIAL_TOKENS['[MASKQ]'])
                tokens = tokens[:self.seq_len-1] + [SPECIAL_TOKENS['[MASKQ]']]

            if len(tokens) < self.seq_len:
                tokens.extend([pad_id] * (self.seq_len - len(tokens)))

            assert tokens.count(SPECIAL_TOKENS['[MASKQ]']) == 1
            batch_inputs.append(torch.tensor(tokens, dtype=torch.long))
            batch_p_star.append(p_star)

        if not batch_inputs:
            return torch.empty(0, self.seq_len, dtype=torch.long), torch.empty(0, M, M), None

        inputs = torch.stack(batch_inputs)
        p_star = torch.stack(batch_p_star)
        roles = _make_roles(inputs, self.bidirectional_prefix_len)
        return inputs, p_star, roles

    def _collate_fn_distractor(self, batch):
        task_config = self.task_configs.get('distractor_loc', {})
        L_min, L_max = task_config.get('L_min', 10), task_config.get('L_max', 50)

        batch_x, batch_m, batch_c, batch_l = [], [], [], []
        pad_id = SPECIAL_TOKENS['[PAD]']

        for i in range(len(batch)):
            orig, _ = batch[i]
            orig = orig.tolist()
            if len(orig) <= L_max: continue

            L = random.randint(L_min, L_max)
            s = random.randint(0, len(orig) - L)

            dist_idx = random.choice([j for j in range(len(batch)) if i != j])
            dist, _ = batch[dist_idx]
            dist_span = dist.tolist()[:L]

            x_prime = orig[:s] + dist_span + orig[s+L:]
            x_prime = x_prime[:self.seq_len]
            x_prime.extend([pad_id] * (self.seq_len - len(x_prime)))

            center = s + L / 2.0
            sigma = L / 4.0
            indices = torch.arange(self.seq_len, dtype=torch.float32)
            m_star = torch.exp(-((indices - center)**2) / (2 * sigma**2))

            batch_x.append(torch.tensor(x_prime, dtype=torch.long))
            batch_m.append(m_star / m_star.max() if m_star.max() > 0 else m_star)
            batch_c.append(torch.tensor(center / self.seq_len, dtype=torch.float32))
            batch_l.append(torch.tensor(L / self.seq_len, dtype=torch.float32))

        if not batch_x:
            return None, None, None, None, None

        x_prime_tensor = torch.stack(batch_x)
        roles = _make_roles(x_prime_tensor, self.bidirectional_prefix_len)
        return x_prime_tensor, torch.stack(batch_m), torch.stack(batch_c), torch.stack(batch_l), roles

    def create_dataloaders(self, batch_size: int = 8, num_workers: int = 0, shuffle_train: bool = True):
        datasets = self.create_datasets()
        dataloaders = {}
        for split, ds in datasets.items():
            if not ds: continue
            dataloaders[split] = {}
            shuffle = shuffle_train if split == 'train' else False

            collate_map = {
                'teacher_forcing': (self._collate_fn_teacher_forcing, 16),
                'cocktail_party': (self._collate_fn_cocktail_party, 8),
                'soft_jigsaw': (self._collate_fn_soft_jigsaw, 8),
                'distractor_loc': (self._collate_fn_distractor, 4)
            }

            for task, (collate_fn, bs_mult) in collate_map.items():
                if task in self.task_configs:
                    dataloaders[split][task] = DataLoader(
                        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available(), collate_fn=collate_fn
                    )
        return dataloaders

    def get_vocab_size(self) -> int:
        return self.vocab_size

def create_data_builder(
    dataset_name: str, dataset_config: str, seq_len: int,
    max_samples: int, max_eval_tokens: int, on_the_fly_tokenization: bool,
    task_configs: dict, bidirectional_prefix_len: int, vocab_size: int
) -> DataBuilder:
    return DataBuilder(
        dataset_name=dataset_name, dataset_config=dataset_config,
        seq_len=seq_len, max_samples=max_samples,
        max_eval_tokens=max_eval_tokens, vocab_size=vocab_size,
        on_the_fly_tokenization=on_the_fly_tokenization,
        task_configs=task_configs,
        bidirectional_prefix_len=bidirectional_prefix_len
    )

if __name__ == "__main__":
    print("Testing DataBuilder...")
    # This main block is for basic testing and may need updates to reflect new collate outputs
    try:
        data_builder = DataBuilder(
            seq_len=128, max_samples=100, task_configs={'teacher_forcing': {}}
        )
        dataloaders = data_builder.create_dataloaders(batch_size=4)
        if 'train' in dataloaders and 'teacher_forcing' in dataloaders['train']:
            train_loader = dataloaders['train']['teacher_forcing']
            print(f"Number of training batches: {len(train_loader)}")
            for batch_idx, (x, y, r) in enumerate(train_loader):
                print(f"Batch {batch_idx}: Input shape: {x.shape}, Target shape: {y.shape}")
                print(f"Roles keys: {r.keys()}")
                sample_text = data_builder.decode_tokens(x[0][:50])
                print(f"Sample text: {sample_text}")
                if batch_idx >= 0: break
        else:
            print("Train dataloader for teacher_forcing not created or empty.")
    except Exception as e:
        print(f"Error during DataBuilder test: {e}")
    print("DataBuilder test completed!")
