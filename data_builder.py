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

class TeacherForcingDataset(Dataset):
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

class NextKDataset(Dataset):
    def __init__(self, raw_data, seq_len=512, k=5, tokenizer_fn=None):
        self.raw_data = raw_data
        self.seq_len = seq_len
        self.k = k
        self.tokenizer_fn = tokenizer_fn

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        text = self.raw_data[idx]['text']
        text = f"[CLS] {text}"
        tokens = self.tokenizer_fn(text)

        prefix = tokens[:self.seq_len]
        next_k = tokens[self.seq_len:self.seq_len + self.k]

        if len(prefix) < self.seq_len:
            prefix.extend([SPECIAL_TOKENS['[PAD]']] * (self.seq_len - len(prefix)))
        if len(next_k) < self.k:
            next_k.extend([SPECIAL_TOKENS['[PAD]']] * (self.k - len(next_k)))

        x_prefix = torch.tensor(prefix, dtype=torch.long)
        y_nextk = torch.tensor(next_k, dtype=torch.long)
        return x_prefix, y_nextk

class CocktailPartyDataset(Dataset):
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

class SoftJigsawDataset(Dataset):
    def __init__(self, raw_data, seq_len=512, tokenizer_fn=None):
        self.raw_data = raw_data
        self.seq_len = seq_len
        self.tokenizer_fn = tokenizer_fn

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return self.raw_data[idx]['text'], self.raw_data[idx]['text']

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

        try:
            print("Attempting Method 1: Load C4 'en' (streaming)...")
            dataset_stream = load_dataset(
                self.dataset_name, name=self.dataset_config, streaming=True, split='train', trust_remote_code=True
            )
            print("C4 'en' (streaming) load_dataset call succeeded. Processing samples...")
            loaded_samples = self._process_iterable_dataset(dataset_stream, "C4 'en' streaming")
        except Exception as e_c4_en_stream:
            print(f"Method 1 (C4 'en' streaming) failed: {e_c4_en_stream}")

        if not loaded_samples:
            print("All dataset loading methods failed. Falling back to simple text dataset...")
            loaded_samples = self._create_fallback_dataset()

        task_data = {task: [] for task in self.task_configs.keys()}
        if 'teacher_forcing' not in task_data:
            task_data['teacher_forcing'] = []

        num_tasks = len(task_data)
        task_names = list(task_data.keys())

        for i, sample in enumerate(loaded_samples):
            task_name = task_names[i % num_tasks]
            task_data[task_name].append(sample)

        final_data = {}
        for task_name, samples in task_data.items():
            if not samples:
                final_data[task_name] = {'train': [], 'validation': [], 'test': []}
                continue

            train_split = int(0.8 * len(samples))
            val_split = int(0.9 * len(samples))
            final_data[task_name] = {
                'train': samples[:train_split],
                'validation': samples[train_split:val_split],
                'test': samples[val_split:]
            }
            print(f"Task {task_name} data splits: train={len(final_data[task_name]['train'])}, val={len(final_data[task_name]['validation'])}, test={len(final_data[task_name]['test'])}")

        return final_data

    def _create_fallback_dataset(self):
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
        ]
        num_repetitions = (self.max_samples // 10 if self.max_samples != float('inf') and self.max_samples > 10 else 100)
        full_sample_texts = sample_texts * num_repetitions
        return [{'text': text} for text in full_sample_texts]
    
    def tokenize_dataset(self, dataset):
        pass
    
    def create_datasets(self):
        raw_dataset_by_task = self.load_raw_dataset()
        datasets = {}

        for task_name, task_data_splits in raw_dataset_by_task.items():
            datasets[task_name] = {}
            for split_name, data in task_data_splits.items():
                if not data:
                    continue

                dataset_class = None
                kwargs = {'raw_data': data, 'seq_len': self.seq_len, 'tokenizer_fn': self._tokenize_text}

                if task_name == 'teacher_forcing':
                    dataset_class = TeacherForcingDataset
                elif task_name == 'next_k':
                    dataset_class = NextKDataset
                    kwargs['k'] = self.task_configs.get('next_k', {}).get('k', 5)
                elif task_name == 'cocktail_party':
                    dataset_class = CocktailPartyDataset
                elif task_name == 'soft_jigsaw':
                    dataset_class = SoftJigsawDataset

                if dataset_class:
                    datasets[task_name][split_name] = dataset_class(**kwargs)
                    print(f"Task '{task_name}', split '{split_name}' dataset created with {len(datasets[task_name][split_name])} samples.")
        return datasets

    def _collate_fn_teacher_forcing(self, batch):
        inputs, targets = zip(*batch)
        return torch.stack(inputs), torch.stack(targets)

    def _collate_fn_next_k(self, batch):
        x, y = zip(*batch)
        return torch.stack(x), torch.stack(y)

    def _collate_fn_cocktail_party(self, batch):
        task_config = self.task_configs.get('cocktail_party', {})
        num_distractors = task_config.get('num_distractors', 3)
        min_span_size = task_config.get('min_span_size', 10)
        max_span_size = task_config.get('max_span_size', 50)

        batch_inputs, batch_spans, batch_correct_indices = [], [], []

        for i in range(len(batch)):
            original_tokens, _ = batch[i]
            original_tokens = original_tokens.tolist()
            span_size = random.randint(min_span_size, max_span_size)
            if len(original_tokens) <= span_size: continue
            span_start = random.randint(0, len(original_tokens) - span_size)
            true_span = original_tokens[span_start : span_start + span_size]

            distractors = []
            for _ in range(num_distractors):
                distractor_idx = random.choice([j for j in range(len(batch)) if i != j])
                distractor_tokens, _ = batch[distractor_idx]
                distractor_tokens = distractor_tokens.tolist()
                if len(distractor_tokens) <= span_size: continue
                distractor_start = random.randint(0, len(distractor_tokens) - span_size)
                distractor_span = distractor_tokens[distractor_start : distractor_start + span_size]
                distractors.append(distractor_span)

            if not distractors: continue

            masked_sequence = original_tokens[:span_start] + [SPECIAL_TOKENS['[MASK]']] + original_tokens[span_start + span_size:]
            all_spans_with_labels = [(true_span, 1)] + [(d, 0) for d in distractors]
            random.shuffle(all_spans_with_labels)
            spans, is_positive = zip(*all_spans_with_labels)
            correct_idx = is_positive.index(1)

            masked_sequence = masked_sequence[:self.seq_len]
            masked_sequence += [SPECIAL_TOKENS['[PAD]']] * (self.seq_len - len(masked_sequence))

            padded_spans = []
            max_len_span = max(len(s) for s in spans) if spans else 0
            for s in spans:
                padded_s = s + [SPECIAL_TOKENS['[PAD]']] * (max_len_span - len(s))
                padded_spans.append(padded_s)

            batch_inputs.append(torch.tensor(masked_sequence, dtype=torch.long))
            batch_spans.append(torch.tensor(padded_spans, dtype=torch.long))
            batch_correct_indices.append(torch.tensor(correct_idx, dtype=torch.long))

        if not batch_inputs:
            return torch.empty(0), torch.empty(0), torch.empty(0)

        max_batch_span_len = max(s.size(1) for s in batch_spans) if batch_spans else 0
        padded_batch_spans = []
        for s in batch_spans:
            padding = max_batch_span_len - s.size(1)
            padded_s = torch.nn.functional.pad(s, (0, padding), 'constant', SPECIAL_TOKENS['[PAD]']) if padding > 0 else s
            padded_batch_spans.append(padded_s)

        return torch.stack(batch_inputs), torch.stack(padded_batch_spans), torch.stack(batch_correct_indices)

    def _collate_fn_soft_jigsaw(self, batch):
        task_config = self.task_configs.get('soft_jigsaw', {})
        M = task_config.get('M', 5)
        batch_inputs, batch_p_star = [], []

        for item in batch:
            text, _ = item
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) < M: continue

            start_index = random.randint(0, len(sentences) - M)
            original_sentences = sentences[start_index : start_index + M]

            indexed_sentences = list(enumerate(original_sentences))
            random.shuffle(indexed_sentences)
            shuffled_sentences = [s for _, s in indexed_sentences]
            original_indices = [i for i, _ in indexed_sentences]

            p_star = torch.zeros(M, M)
            for i in range(M): p_star[i, original_indices[i]] = 1

            input_text = f"[CLS] " + f" [SPAN] ".join(shuffled_sentences)
            tokens = self._tokenize_text(input_text)
            tokens = tokens[:self.seq_len] + [SPECIAL_TOKENS['[PAD]']] * (self.seq_len - len(tokens))

            batch_inputs.append(torch.tensor(tokens, dtype=torch.long))
            batch_p_star.append(p_star)

        if not batch_inputs: return None
        return torch.stack(batch_inputs), torch.stack(batch_p_star)

    def create_dataloaders(
        self, batch_size: int = 8, num_workers: int = 0, shuffle_train: bool = True
    ) -> Dict[str, Dict[str, DataLoader]]:
        datasets_by_task = self.create_datasets()
        dataloaders = {}

        for split_name in ['train', 'validation', 'test']:
            dataloaders[split_name] = {}
            shuffle = shuffle_train if split_name == 'train' else False

            for task_name, task_datasets in datasets_by_task.items():
                if split_name not in task_datasets or not task_datasets[split_name]:
                    continue

                dataset_obj = task_datasets[split_name]

                collate_fn_map = {
                    'teacher_forcing': self._collate_fn_teacher_forcing,
                    'next_k': self._collate_fn_next_k,
                    'cocktail_party': self._collate_fn_cocktail_party,
                    'soft_jigsaw': self._collate_fn_soft_jigsaw,
                }

                batch_size_map = {
                    'teacher_forcing': 8,
                    'next_k': 4,
                    'cocktail_party': 8,
                    'soft_jigsaw': 8,
                }

                task_batch_size = batch_size_map.get(task_name, batch_size)
                collate_fn = collate_fn_map.get(task_name)

                if collate_fn:
                    dataloaders[split_name][task_name] = DataLoader(
                        dataset_obj, batch_size=task_batch_size, shuffle=shuffle,
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
