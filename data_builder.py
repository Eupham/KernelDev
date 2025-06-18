import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset # Keep this import
import numpy as np
from typing import Optional, Dict, Any
from nsp_dataset import NSPDataset # Import NSPDataset


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
        vocab_size: int = 256, # Original vocab size, typically 256 for byte tokenization
        max_eval_tokens: int = 50000,
        nsp_task: bool = False, # New parameter for NSP task
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.seq_len = seq_len
        self.max_samples = max_samples if max_samples is not None else float('inf')
        self.nsp_task = nsp_task

        self.cls_token_id = 256
        self.sep_token_id = 257
        # Adjust vocab_size for CLS and SEP tokens
        if vocab_size == 256: # Default byte tokenizer size
            self.vocab_size = 258 # Accommodate CLS and SEP
            print(f"Original vocab_size was 256, updated to {self.vocab_size} for CLS/SEP tokens.")
        elif vocab_size < 258:
            print(f"Warning: Original vocab_size {vocab_size} is less than 258. "
                  f"CLS_TOKEN_ID ({self.cls_token_id}) or SEP_TOKEN_ID ({self.sep_token_id}) might collide or be out of bounds "
                  f"if not already accounted for. Setting vocab_size to 258.")
            self.vocab_size = 258
        else: # vocab_size >= 258
            self.vocab_size = vocab_size
            print(f"Using provided vocab_size: {self.vocab_size}. Ensure it accounts for CLS/SEP if NSP is active.")

        self.max_eval_tokens = max_eval_tokens

        print(f"Effective vocabulary size: {self.vocab_size}")
        print(f"Max evaluation tokens per split: {self.max_eval_tokens}")
        if self.max_samples != float('inf'):
            print(f"Will attempt to load up to {self.max_samples} samples from the dataset.")
        else:
            print("Will attempt to load all available samples from the dataset.")

    def _tokenize_text(self, text: str) -> list:
        return list(text.encode('utf-8'))
    
    def _detokenize_bytes(self, tokens: list) -> str:
        try:
            # Handle special tokens for decoding if they are in the list
            processed_tokens = []
            for t_id in tokens:
                if self.nsp_task: # Only apply special decoding if NSP task is configured
                    if t_id == self.cls_token_id:
                        # This part is tricky as bytes() expects integers 0-255
                        # We can't directly convert 256/257 to a byte.
                        # So, for string representation, we map them to text tags.
                        # This means _detokenize_bytes is now lossy for CLS/SEP if used outside debugging.
                        # Actual model input should remain integer IDs.
                        # This function is mostly for inspection.
                        pass # Will be handled by string joining later
                    elif t_id == self.sep_token_id:
                        pass # Will be handled by string joining later
                    elif t_id < 0 or t_id > 255: # Other special tokens like pad, or out of byte range
                        pass # Will be handled by string joining later
                    else:
                        processed_tokens.append(t_id)
                else:
                    if 0 <= t_id <= 255:
                         processed_tokens.append(t_id)

            byte_data = bytes(processed_tokens)
            decoded_text = byte_data.decode('utf-8', errors='replace')

            # Re-insert string representations for special tokens if NSP task
            if self.nsp_task:
                final_str_parts = []
                current_byte_idx = 0
                for t_id in tokens:
                    if t_id == self.cls_token_id:
                        final_str_parts.append("[CLS]")
                    elif t_id == self.sep_token_id:
                        final_str_parts.append("[SEP]")
                    elif t_id < 0 or t_id > 255: # e.g. pad_token_id = -1
                        final_str_parts.append(f"[PAD:{t_id}]")
                    else:
                        # This assumes one byte token corresponds to one character after potential multi-byte decoding
                        # This part is complex due to utf-8 variable byte length.
                        # A simpler approach for _detokenize_bytes for inspection:
                        # Just convert byte range and use placeholders for others.
                        pass # Byte tokens are handled by byte_data.decode above.
                # This improved version handles byte tokens first, then inserts placeholders.
                # However, the initial version of just decoding valid bytes is safer.
                # Let's stick to a simpler version for now: decode valid bytes, and if special tokens were present,
                # it implies the string output here is mainly for byte-tokens.
                # A truly accurate detokenization would need to know where byte sequences were interrupted by special tokens.

                # Fallback to simpler detokenization for inspection if NSP tokens are present:
                if any(t_id in [self.cls_token_id, self.sep_token_id] for t_id in tokens):
                    return " ".join(
                        "[CLS]" if t == self.cls_token_id else \
                        "[SEP]" if t == self.sep_token_id else \
                        f"[PAD:{t}]" if t < 0 or t > 255 else \
                        chr(t) if chr(t).isprintable() or chr(t) in ['\n', '\t', ' '] else f"[{t:02x}]"
                        for t in tokens
                    )
            return decoded_text

        except Exception as e:
            print(f"Warning: Error decoding tokens: {e}")
            return f"[DECODE_ERROR: {tokens[:10]}...]"

    def _segment_text_to_sentences(self, text: str) -> list[str]:
        """Segments text into sentences using basic punctuation."""
        if not text:
            return []
        # Use regex for more robust sentence splitting
        # This pattern splits by '.', '!', '?' followed by space or end of string.
        # It tries to handle common abbreviations by not splitting if preceded by a single capital letter (e.g., Mr. Smith).
        import re
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|!)\s', text)

        # Filter out very short or empty sentences and strip whitespace
        min_sentence_length = 5 # Minimum number of characters for a sentence
        processed_sentences = [s.strip() for s in sentences if s and len(s.strip()) >= min_sentence_length]

        # Further cleanup: if a "sentence" ends with an abbreviation that wasn't caught,
        # and the next "sentence" starts lowercase, they might belong together.
        # This is complex; for now, the regex handles common cases.
        return processed_sentences

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
                if self.nsp_task:
                    sentences = self._segment_text_to_sentences(text_content)
                    if sentences: # Only add if there's at least one sentence
                        samples.append({'text': text_content, 'sentences': sentences}) # Store original text and sentences
                        processed_count += 1
                else:
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
            if not isinstance(split_data_list, list):
                print(f"Warning: {split_name} data is not a list (type: {type(split_data_list)}), skipping tokenization.")
                tokenized_data[split_name] = []
                continue
            if not split_data_list:
                print(f"Warning: {split_name} data list is empty. Skipping tokenization.")
                tokenized_data[split_name] = []
                continue

            if self.nsp_task:
                # For NSP, process list of documents, each with a list of sentences
                # Output: list[list[list[int]]] (list of docs, each doc is list of tokenized sentences)
                tokenized_docs_for_split = []
                num_sentences_total = 0
                for doc_item in split_data_list:
                    if isinstance(doc_item, dict) and 'sentences' in doc_item:
                        doc_sentences_tokenized = []
                        for sentence_str in doc_item['sentences']:
                            if sentence_str and sentence_str.strip():
                                tokenized_sentence = self._tokenize_text(sentence_str)
                                if tokenized_sentence: # Ensure sentence is not empty after tokenization
                                    doc_sentences_tokenized.append(tokenized_sentence)
                                    num_sentences_total +=1
                        if doc_sentences_tokenized: # Only add doc if it has tokenized sentences
                             tokenized_docs_for_split.append(doc_sentences_tokenized)
                tokenized_data[split_name] = tokenized_docs_for_split
                print(f"Tokenized {split_name} for NSP: {len(tokenized_docs_for_split)} documents, {num_sentences_total} total sentences.")
            else:
                # Standard tokenization: concatenate all text and tokenize once
                all_text = ""
                for item in split_data_list:
                    if isinstance(item, dict) and 'text' in item:
                        text_content = item['text']
                        if text_content and text_content.strip():
                            all_text += text_content + "\n"

                print(f"Text length for {split_name} (standard task): {len(all_text)} characters")
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
        # tokenized_data will be list[list[list[int]]] for NSP, or list[int] for standard
        tokenized_data_for_splits = self.tokenize_dataset(raw_dataset)

        datasets = {}
        for split_name, data_content in tokenized_data_for_splits.items():
            if not data_content: # data_content can be empty list of docs (for NSP) or empty list of tokens
                print(f"Warning: {split_name} split has no tokenized content. Skipping dataset creation.")
                continue

            if self.nsp_task:
                # data_content is list[list[list[int]]] (list of docs, each doc is list of tokenized sentences)
                if not any(doc for doc in data_content): # Check if all documents are empty or list itself is empty
                    print(f"Warning: {split_name} split for NSP has no sentences after tokenization. Skipping dataset.")
                    continue
                
                # For NSP, max_eval_tokens is not directly applicable here as NSPDataset creates examples internally.
                # We could limit the number of documents passed to NSPDataset for eval splits if needed.
                # For now, pass all processed documents.
                datasets[split_name] = NSPDataset(
                    documents=data_content,
                    seq_len=self.seq_len,
                    cls_token_id=self.cls_token_id,
                    sep_token_id=self.sep_token_id,
                    pad_token_id=-1 # Standard pad_token_id for LM loss
                )
                print(f"{split_name} NSP dataset: {len(datasets[split_name])} examples")

            else: # Standard TokenizedDataset
                # data_content is list[int] (concatenated tokens for the split)
                tokens = data_content
                current_max_eval_tokens = self.max_eval_tokens
                if self.max_samples != float('inf') and split_name in ['validation', 'test']:
                    # Estimate based on 20% of max_samples, ensure it's reasonable
                    scaled_max_tokens = int(self.max_samples * 0.2 * self.seq_len)
                    current_max_eval_tokens = min(self.max_eval_tokens, scaled_max_tokens)
                    # Ensure at least a few full sequences for eval
                    current_max_eval_tokens = max(current_max_eval_tokens, self.seq_len * 5 + 1)

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
        max_eval_tokens: int = 50000,
        nsp_task: bool = False, # Added nsp_task
) -> DataBuilder:
    return DataBuilder(
        dataset_name=dataset_name, dataset_config=dataset_config,
        seq_len=seq_len, max_samples=max_samples,
        max_eval_tokens=max_eval_tokens,
        nsp_task=nsp_task # Pass to constructor
    )

if __name__ == "__main__":
    # ... (main test block as in original file)
    print("Testing DataBuilder (Standard Task)...")
    data_builder_std = create_data_builder(
        dataset_name="allenai/c4", dataset_config="en", # Using C4 for more text
        seq_len=128, max_samples=200, # Reduced max_samples for faster test
        nsp_task=False
    )
    dataloaders_std = data_builder_std.create_dataloaders(batch_size=2)
    if 'train' in dataloaders_std and dataloaders_std['train']:
        train_loader_std = dataloaders_std['train']
        print(f"Number of standard training batches: {len(train_loader_std)}")
        try:
            for batch_idx, (x, y) in enumerate(train_loader_std):
                print(f"Std Batch {batch_idx}: Input shape: {x.shape}, Target shape: {y.shape}")
                if x.numel() > 0: # Check if tensor is not empty
                    sample_text = data_builder_std.decode_tokens(x[0][:30]) # Shorter sample
                    print(f"Std Sample text: {sample_text}")
                if batch_idx >= 0: break # Only show first batch
        except Exception as e:
            print(f"Error during standard dataloader iteration test: {e}")
            raise
    else:
        print("Standard train dataloader not created or empty.")
    print("DataBuilder standard test completed!")

    print("\nTesting DataBuilder (NSP Task)...")
    data_builder_nsp = create_data_builder(
        dataset_name="allenai/c4", dataset_config="en", # Using C4 for more text
        seq_len=64, max_samples=100, # Further reduced for very fast NSP test
        nsp_task=True
    )
    # Ensure model's cls_token_id would be set here in a real scenario
    # model.cls_token_id = data_builder_nsp.cls_token_id

    dataloaders_nsp = data_builder_nsp.create_dataloaders(batch_size=2)
    if 'train' in dataloaders_nsp and dataloaders_nsp['train']:
        train_loader_nsp = dataloaders_nsp['train']
        print(f"Number of NSP training batches: {len(train_loader_nsp)}")
        try:
            for batch_idx, (input_ids, lm_target_ids, nsp_label) in enumerate(train_loader_nsp):
                print(f"NSP Batch {batch_idx}: Inputs: {input_ids.shape}, Targets: {lm_target_ids.shape}, NSP Label: {nsp_label.shape}")
                if input_ids.numel() > 0:
                    sample_tokens = input_ids[0][:30].tolist() # Shorter sample
                    decoded_sample = data_builder_nsp.decode_tokens(sample_tokens)
                    print(f"NSP Sample tokens: {sample_tokens}")
                    print(f"NSP Sample decoded: {decoded_sample}")
                if batch_idx >= 0: break # Only show first batch
        except Exception as e:
            print(f"Error during NSP dataloader iteration test: {e}")
            raise
    else:
        print("NSP train dataloader not created or empty.")
    print("DataBuilder NSP test completed!")
