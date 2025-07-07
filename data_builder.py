import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset # Keep this import
import numpy as np
from typing import Optional, Dict, Any
from levenshtein_dataset import LevenshteinDataset


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
        seq_len: Optional[Any] = 512,
        max_samples: Optional[int] = 2000,
        vocab_size: int = 256,
        max_eval_tokens: Optional[Any] = 50000,
        use_levenshtein_task: bool = False,
        levenshtein_shuffle_percentage: float = 0.25,
        max_train_tokens: Optional[Any] = None, # New parameter
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config

        # Robust initialization for self.seq_len
        default_seq_len = 512
        if seq_len is None:
            # print(f"Debug: seq_len param is None. Defaulting to {default_seq_len}.")
            self.seq_len = default_seq_len
        else:
            try:
                self.seq_len = int(seq_len)
                if self.seq_len <= 0:
                    # print(f"Debug: seq_len {self.seq_len} is not positive. Defaulting to {default_seq_len}.")
                    self.seq_len = default_seq_len
            except ValueError:
                # print(f"Debug: Could not convert seq_len '{seq_len}' to int. Defaulting to {default_seq_len}.")
                self.seq_len = default_seq_len
        # print(f"Debug: Final self.seq_len: {self.seq_len}")

        self.max_samples = max_samples if max_samples is not None else float('inf')
        self.use_levenshtein_task = use_levenshtein_task
        if levenshtein_shuffle_percentage is None:
            self.levenshtein_shuffle_percentage = 0.25  # Default value if None is passed
        else:
            self.levenshtein_shuffle_percentage = levenshtein_shuffle_percentage

        self.cls_token_id = None # Default to None
        self.sep_token_id = None # Default to None
        if self.use_levenshtein_task:
            self.cls_token_id = 256 # Fixed ID for Levenshtein task's CLS
            self.sep_token_id = 257 # Fixed ID for SEP token (for NSP task)
            original_vocab_size_param = vocab_size
            if original_vocab_size_param == 256: # Default byte vocab size
                self.vocab_size = 258 # Accommodate CLS (256) and SEP (257)
            elif self.sep_token_id >= original_vocab_size_param: # If SEP ID is outside original range
                print(f"Warning: vocab_size {original_vocab_size_param} from config might be too small for sep_token_id {self.sep_token_id}. Adjusting vocab_size.")
                self.vocab_size = self.sep_token_id + 1
            else: # tokens are within original vocab, assume it's accounted for (user configured)
                self.vocab_size = original_vocab_size_param
            print(f"Levenshtein task active: cls_token_id={self.cls_token_id}, sep_token_id={self.sep_token_id}, effective vocab_size={self.vocab_size}")
        else: # Not using Levenshtein task
            self.vocab_size = vocab_size # Use vocab_size as passed

        self.max_eval_tokens = max_eval_tokens if max_eval_tokens is not None else float('inf')

        if max_train_tokens is None:
            self.max_train_tokens = float('inf')
        else:
            try:
                self.max_train_tokens = float(max_train_tokens)
            except ValueError:
                print(f"Warning: Could not convert max_train_tokens value '{max_train_tokens}' to float. Defaulting to float('inf').")
                self.max_train_tokens = float('inf')

        print(f"Effective vocabulary size: {self.vocab_size}")
        print(f"Max train tokens: {self.max_train_tokens}")
        print(f"Max evaluation tokens per split: {self.max_eval_tokens}")
        if self.max_samples != float('inf'):
            print(f"Will attempt to load up to {self.max_samples} samples from the dataset.")
        else:
            print("Will attempt to load all available samples from the dataset.")

    def _tokenize_text(self, text: str) -> list:
        return list(text.encode('utf-8'))
    
    def _detokenize_bytes(self, tokens: list) -> str:
        try:
            string_parts = []
            current_byte_sequence = []

            for token in tokens:
                if self.cls_token_id is not None and token == self.cls_token_id: # Check if cls_token_id is defined
                    if current_byte_sequence:
                        string_parts.append(bytes(current_byte_sequence).decode('utf-8', errors='replace'))
                        current_byte_sequence = []
                    string_parts.append("[CLS]")
                elif self.sep_token_id is not None and token == self.sep_token_id: # Check if sep_token_id is defined
                    if current_byte_sequence:
                        string_parts.append(bytes(current_byte_sequence).decode('utf-8', errors='replace'))
                        current_byte_sequence = []
                    string_parts.append("[SEP]")
                elif 0 <= token <= 255:
                    current_byte_sequence.append(token)
                else: # Special tokens other than CLS (e.g., padding -1) or unknown tokens
                    if current_byte_sequence:
                        string_parts.append(bytes(current_byte_sequence).decode('utf-8', errors='replace'))
                        current_byte_sequence = []
                    # Represent other special tokens (like pad_token_id = -1 from NSPDataset)
                    # or any unexpected token.
                    if token == -1: # Common padding value for LM targets
                        string_parts.append("[PAD]")
                    else:
                        string_parts.append(f"[UNK_TOKEN:{token}]")

            # After the loop, decode any remaining byte sequence
            if current_byte_sequence:
                string_parts.append(bytes(current_byte_sequence).decode('utf-8', errors='replace'))

            return "".join(string_parts)

        except Exception as e:
            print(f"Warning: Error decoding tokens: {e}")
            # Provide more context in the error if possible
            problematic_part = []
            for t in tokens[:20]: # Show first 20 tokens that might be causing issues
                if isinstance(t, int):
                    if 0 <= t <=255:
                        problematic_part.append(hex(t))
                    else:
                        problematic_part.append(str(t))
                else:
                    problematic_part.append(str(type(t)))


            return f"[DECODE_ERROR for tokens: {problematic_part} ... (Total: {len(tokens)})]"

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
        if self.max_samples == float('inf') and dataset_iterable is None:
             print(f"Warning: Dataset iterable for {dataset_name_logging} is empty or None when expecting all samples.")
             return []
        if dataset_iterable is None:
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
                # For Levenshtein task, we don't need to pre-segment sentences here.
                # The LevenshteinDataset will treat each 'text' field as a unit for shuffling.
                # If NSP task specific sentence storage was here, it's removed or made generic.
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

            if self.use_levenshtein_task:
                # For Levenshtein, we need a list of text strings for each split.
                # The raw_dataset (output of load_raw_dataset) is Dict[str, List[Dict[str, str]]],
                # where each inner dict has a 'text' key.
                current_split_texts = []
                for item_dict in split_data_list: # split_data_list is List[Dict[str, str]]
                    if isinstance(item_dict, dict) and 'text' in item_dict and item_dict['text'].strip():
                        current_split_texts.append(item_dict['text'])
                tokenized_data[split_name] = current_split_texts # This is List[str]
                print(f"Processed {split_name} for Levenshtein: {len(current_split_texts)} text items.")
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
        tokenized_or_raw_data = self.tokenize_dataset(raw_dataset) # This is now Dict[str, List[str]] if Levenshtein

        datasets = {}
        for split_name, data_for_split in tokenized_or_raw_data.items():
            if not data_for_split:
                print(f"Warning: {split_name} split has no data after tokenization/processing. Skipping dataset creation.")
                continue

            if self.use_levenshtein_task:
                # Use combined multi-task dataset
                # data_for_split is List[str] here
                if self.cls_token_id is None or self.sep_token_id is None: # Both tokens must be set
                    raise ValueError("cls_token_id and sep_token_id must be set in DataBuilder for multi-task, but use_levenshtein_task is True.")
                
                # Import the combined dataset
                from combined_dataset import CombinedMultiTaskDataset
                
                # Limit number of raw sentences for eval splits if max_eval_tokens is a concern
                if split_name in ['validation', 'test'] and len(data_for_split) * self.seq_len > self.max_eval_tokens:
                    if self.max_eval_tokens == float('inf'):
                        approx_docs_to_keep = float('inf')
                    else:
                        approx_docs_to_keep = self.max_eval_tokens // self.seq_len

                    if approx_docs_to_keep == 0 and self.max_eval_tokens > 0 and self.max_eval_tokens != float('inf'):
                        approx_docs_to_keep = 1

                    if approx_docs_to_keep < len(data_for_split): # This condition handles approx_docs_to_keep = float('inf') correctly
                         print(f"Limiting {split_name} for multi-task to approx {int(approx_docs_to_keep)} documents from {len(data_for_split)} for faster evaluation.")
                         data_for_split = data_for_split[:int(approx_docs_to_keep)] # Slicing requires int

                    if not data_for_split and len(tokenized_or_raw_data[split_name]) > 0 :
                         data_for_split = tokenized_or_raw_data[split_name][:1]

                if not data_for_split:
                    print(f"Warning: {split_name} split became empty after limiting for multi-task eval. Skipping.")
                    continue

                datasets[split_name] = CombinedMultiTaskDataset(
                    raw_documents=data_for_split,
                    tokenizer_fn=self._tokenize_text,
                    seq_len=self.seq_len,
                    cls_token_id=self.cls_token_id,
                    sep_token_id=self.sep_token_id,
                    lm_ignore_idx=-1,
                    input_pad_id=0,    # Assuming 0 is a safe padding ID not overlapping with CLS/SEP
                    task_distribution=(0.25, 0.25, 0.5)  # 25% Lev, 25% NSP, 50% LM
                )
                print(f"{split_name} Combined multi-task dataset: {len(datasets[split_name])} examples")
            else: # Standard TokenizedDataset for LM
                # data_for_split is list[int] (concatenated tokens)
                tokens = data_for_split
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
                            if current_max_eval_tokens != float('inf'):
                                tokens = tokens[:int(current_max_eval_tokens)] # Ensure int for slicing
                            # If current_max_eval_tokens is float('inf'), no slicing occurs, all tokens are kept.
                            print(f"Limited {split_name} to {len(tokens)} tokens for faster evaluation (target was: {current_max_eval_tokens})")

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
            print(f"\n--- Preparing {split_name}_dataloader ---")
            print(f"Dataset object: {dataset_obj}")
            if dataset_obj is not None:
                try:
                    try:
                        dataset_len = len(dataset_obj)
                        print(f"len(dataset): {dataset_len}")
                    except TypeError:
                        # Handle datasets that don't support len() (e.g., generators, IterableDatasets)
                        print(f"len(dataset): unknown (iterable dataset)")
                    
                    if hasattr(dataset_obj, 'seq_len'):
                        print(f"dataset.seq_len: {dataset_obj.seq_len}")
                    if isinstance(dataset_obj, LevenshteinDataset):
                        # LevenshteinDataset stores input_pad_id directly
                        if hasattr(dataset_obj, 'input_pad_id'):
                             print(f"dataset.input_pad_id: {dataset_obj.input_pad_id}")
                except Exception as e:
                    print(f"Error getting dataset attributes for {split_name}_dataset: {e}")

            current_shuffle_status = shuffle_train if split_name == 'train' else False
            current_pin_memory = torch.cuda.is_available()

            print(f"batch_size: {batch_size}")
            print(f"shuffle: {current_shuffle_status}")
            print(f"num_workers: {num_workers}")
            print(f"collate_fn: None (using default)") # DataLoader uses default_collate if None
            print(f"pin_memory: {current_pin_memory}")

            if dataset_obj is None: # Re-check after prints, before DataLoader call
                print(f"Skipping DataLoader for {split_name} as dataset is None or invalid after attribute checks.")
                print(f"--- {split_name}_dataloader preparation skipped ---")
                continue

            try:
                dataloaders[split_name] = DataLoader(
                    dataset_obj, batch_size=batch_size, shuffle=current_shuffle_status,
                    num_workers=num_workers, pin_memory=current_pin_memory
                )
            except (TypeError, ValueError) as e:
                # Handle datasets that don't support shuffling (e.g., IterableDatasets)
                if current_shuffle_status and ("shuffle" in str(e) or "len()" in str(e)):
                    print(f"Warning: Dataset doesn't support shuffling. Creating DataLoader without shuffle.")
                    dataloaders[split_name] = DataLoader(
                        dataset_obj, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=current_pin_memory
                    )
                else:
                    raise  # Re-raise if it's a different error
                    
            try:
                batch_count = len(dataloaders[split_name])
                print(f"{split_name} dataloader created: {batch_count} batches")
            except TypeError:
                # Handle datasets that don't support len() (e.g., generators, IterableDatasets)
                print(f"{split_name} dataloader created: unknown number of batches (iterable dataset)")
            print(f"--- {split_name}_dataloader preparation complete ---")
        return dataloaders

    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def decode_tokens(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().tolist()
        return self._detokenize_bytes(tokens)


def create_data_builder(
    dataset_name: str = "allenai/c4", dataset_config: str = "en",
    seq_len: Optional[Any] = 512, # Keep Optional[Any] as it's passed to DataBuilder constructor
    max_samples: Optional[int] = 2000,
    max_eval_tokens: Optional[Any] = 50000,
    use_levenshtein_task: bool = False,
    levenshtein_shuffle_percentage: float = 0.25,
    max_train_tokens: Optional[Any] = None,
) -> DataBuilder:
    return DataBuilder(
        dataset_name=dataset_name, dataset_config=dataset_config,
        seq_len=seq_len, max_samples=max_samples,
        max_eval_tokens=max_eval_tokens,
        use_levenshtein_task=use_levenshtein_task,
        levenshtein_shuffle_percentage=levenshtein_shuffle_percentage,
        max_train_tokens=max_train_tokens
    )

if __name__ == "__main__":
    # ... (main test block as in original file)
    print("Testing DataBuilder (Standard Task)...")
    data_builder_std = create_data_builder(
        dataset_name="allenai/c4", dataset_config="en", # Using C4 for more text
        seq_len=128, max_samples=200, # Reduced max_samples for faster test
        use_levenshtein_task=False # Explicitly false for this standard test
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

    print("\nTesting DataBuilder (Levenshtein Task)...")
    data_builder_lev = create_data_builder(
        dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", # Using a smaller dataset for test
        seq_len=64, max_samples=50, # Small values for quick test
        use_levenshtein_task=True,
        levenshtein_shuffle_percentage=0.5 # Test with a specific shuffle percentage
    )
    print(f"DataBuilder for Levenshtein created with shuffle_percentage: {data_builder_lev.levenshtein_shuffle_percentage}")

    dataloaders_lev = data_builder_lev.create_dataloaders(batch_size=2)
    if 'train' in dataloaders_lev and dataloaders_lev['train']:
        train_loader_lev = dataloaders_lev['train']
        print(f"Number of Levenshtein training batches: {len(train_loader_lev)}")
        try:
            # LevenshteinDataset now returns 4 items
            for batch_idx, (input_tokens, lm_targets, lev_dist_target, is_shuffled_flag) in enumerate(train_loader_lev):
                item_type = "Shuffled" if is_shuffled_flag[0].item() == 1.0 else "Original"
                print(f"Lev Batch {batch_idx} (Item 0 Type: {item_type}):")
                print(f"  Input Tokens: {input_tokens.shape}")
                print(f"  LM Targets:   {lm_targets.shape}")
                print(f"  Lev Dist Target: {lev_dist_target.shape}, Values: {lev_dist_target.tolist()}")
                print(f"  Is Shuffled Flag: {is_shuffled_flag.shape}, Values: {is_shuffled_flag.tolist()}")

                if input_tokens.numel() > 0:
                    sample_text = data_builder_lev.decode_tokens(input_tokens[0][:30])
                    print(f"  Sample Input Decoded (approx): {sample_text}")

                if lm_targets[0,0].item() != -1 : # If first target is not ignore (means it's an original sentence)
                    self_is_shuffled = is_shuffled_flag[0].item()
                    assert self_is_shuffled == 0.0, "LM targets should be ignored for shuffled items!"

                if batch_idx >= 1: break # Show a couple of batches
        except Exception as e:
            print(f"Error during Levenshtein dataloader iteration test: {e}")
            raise
    else:
        print("Levenshtein train dataloader not created or empty.")
    print("DataBuilder Levenshtein test completed!")
