import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset # Keep this import
import numpy as np
from typing import Optional, Dict, Any


class TokenizedDataset(Dataset):
    # ... (class content as in original file)
    def __init__(self, tokenized_data, seq_len=512):
        self.data = tokenized_data
        self.seq_len = seq_len
        
    def __len__(self):
        return max(1, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class DataBuilder:
    # ... (constructor and other methods like _tokenize_text, _detokenize_bytes as in original file)
    def __init__(
        self,
        dataset_name: str = "allenai/c4",
        dataset_config: str = "en",
        seq_len: int = 512,
        max_samples: Optional[int] = 2000,
        vocab_size: int = 256,
        max_eval_tokens: int = 50000
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.seq_len = seq_len
        self.max_samples = max_samples if max_samples is not None else float('inf')
        self.vocab_size = vocab_size
        self.max_eval_tokens = max_eval_tokens
        
        print(f"Using UTF-8 byte tokenization with vocabulary size: {self.vocab_size}")
        print(f"Max evaluation tokens per split: {self.max_eval_tokens}")
        if self.max_samples != float('inf'):
            print(f"Will attempt to load up to {self.max_samples} samples from the dataset.")
        else:
            print("Will attempt to load all available samples from the dataset.")

    def _tokenize_text(self, text: str) -> list:
        return list(text.encode('utf-8'))
    
    def _detokenize_bytes(self, tokens: list) -> str:
        try:
            byte_data = bytes(tokens)
            return byte_data.decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Warning: Error decoding tokens: {e}")
            return f"[DECODE_ERROR: {tokens[:10]}...]"

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
        # ... (method content as in original file, ensure robust to empty tokenized_data)
        raw_dataset = self.load_raw_dataset()
        tokenized_data = self.tokenize_dataset(raw_dataset)
        
        datasets = {}
        for split_name, tokens in tokenized_data.items():
            if not tokens:
                print(f"Warning: {split_name} split has no tokens. Skipping dataset creation.")
                continue

            current_max_eval_tokens = self.max_eval_tokens
            if self.max_samples != float('inf') and split_name in ['validation', 'test']:
                scaled_max_tokens = int(self.max_samples * 0.2 * self.seq_len) # Estimate based on 20% of max_samples
                current_max_eval_tokens = min(self.max_eval_tokens, scaled_max_tokens)
                current_max_eval_tokens = max(current_max_eval_tokens, self.seq_len * 2 + 1) # Ensure at least two full sequences + 1 for target

            if len(tokens) > self.seq_len:
                if split_name in ['validation', 'test']:
                    if len(tokens) > current_max_eval_tokens:
                        tokens = tokens[:current_max_eval_tokens]
                        print(f"Limited {split_name} to {len(tokens)} tokens for faster evaluation (target: {current_max_eval_tokens})")
                
                datasets[split_name] = TokenizedDataset(tokens, self.seq_len)
                print(f"{split_name} dataset: {len(datasets[split_name])} samples")
            else:
                print(f"Warning: {split_name} split has insufficient tokens ({len(tokens)}) for seq_len {self.seq_len}. Skipping dataset.")
        return datasets

    def create_dataloaders(
        self, batch_size: int = 8, num_workers: int = 0, shuffle_train: bool = True
    ) -> Dict[str, DataLoader]:
        # ... (method content as in original file, ensure robust to empty datasets dict)
        datasets = self.create_datasets()
        dataloaders = {}
        if not datasets:
            print("Warning: No datasets were created. Returning empty dataloaders dict.")
            return dataloaders

        for split_name, dataset_obj in datasets.items():
            if not dataset_obj:
                print(f"Skipping DataLoader for {split_name} as dataset is empty or invalid.")
                continue
            shuffle = shuffle_train if split_name == 'train' else False
            dataloaders[split_name] = DataLoader(
                dataset_obj, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, pin_memory=torch.cuda.is_available()
            )
            print(f"{split_name} dataloader: {len(dataloaders[split_name])} batches")
        return dataloaders

    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def decode_tokens(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().tolist()
        return self._detokenize_bytes(tokens)


def create_data_builder(
    dataset_name: str = "allenai/c4", dataset_config: str = "en",
    seq_len: int = 512, max_samples: Optional[int] = 2000,
    max_eval_tokens: int = 50000
) -> DataBuilder:
    return DataBuilder(
        dataset_name=dataset_name, dataset_config=dataset_config,
        seq_len=seq_len, max_samples=max_samples,
        max_eval_tokens=max_eval_tokens
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
