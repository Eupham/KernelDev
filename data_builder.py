import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset # Keep this import
import numpy as np
from typing import Optional, Dict, Any, List
import random
import re


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


class NSPDataset(Dataset):
    def __init__(self, documents: list[dict], tokenizer_fn, seq_len: int, is_eval: bool = False, max_samples_for_eval: int = None):
        self.tokenizer_fn = tokenizer_fn
        self.seq_len = seq_len
        self.is_eval = is_eval
        self.pad_token_id = 0  # Assuming 0 for padding input_ids
        self.lm_ignore_idx = -100 # For ignoring LM labels

        self._prepare_examples(documents, is_eval, max_samples_for_eval)

    def _split_document(self, doc_text: str) -> list[str]:
        if not doc_text:
            return []
        # Split by sentence-ending punctuation followed by space or newline
        sentences = re.split(r'(?<=[.!?])(?:\s+|\n)', doc_text)
        processed_sentences = []
        for s in sentences:
            s_stripped = s.strip()
            if not s_stripped:
                continue
            # Re-add punctuation if it was stripped by split (though current regex keeps it)
            # No, current regex (?<=[.!?]) keeps the delimiter as part of the preceding sentence.
            # Filter by word count
            if len(s_stripped.split()) >= 3:
                processed_sentences.append(s_stripped)
        return processed_sentences

    def _prepare_examples(self, documents: list[dict], is_eval: bool, max_samples_for_eval: int):
        self.all_sentences = []  # List of tuples (doc_idx, sent_idx_in_doc, sentence_text)
        self.positive_candidates = []  # List of tuples (doc_idx, sent_A_text, sent_B_text)

        for doc_idx, doc_dict in enumerate(documents):
            doc_text = doc_dict.get('text', '')
            if not doc_text:
                continue

            sents = self._split_document(doc_text)

            current_doc_sents = []
            for sent_idx, sent_text in enumerate(sents):
                self.all_sentences.append((doc_idx, sent_idx, sent_text))
                current_doc_sents.append(sent_text)

            if len(current_doc_sents) >= 2:
                for i in range(len(current_doc_sents) - 1):
                    self.positive_candidates.append((doc_idx, current_doc_sents[i], current_doc_sents[i+1]))

        if is_eval and max_samples_for_eval is not None:
            if len(self.positive_candidates) > max_samples_for_eval // 2:
                random.shuffle(self.positive_candidates)
                self.positive_candidates = self.positive_candidates[:max_samples_for_eval // 2]

            # Ensure all_sentences is also capped for eval if it's very large,
            # though negative sampling primarily relies on its diversity rather than sheer count matching positives.
            # Let's cap it relative to max_samples_for_eval to prevent excessive memory/time if docs are huge.
            # A cap of max_samples_for_eval * 2 seems reasonable for sentence diversity.
            if len(self.all_sentences) > max_samples_for_eval * 2: # Max number of sentences to pick from for negative examples
                random.shuffle(self.all_sentences)
                self.all_sentences = self.all_sentences[:max_samples_for_eval * 2]


        self.num_positive_samples = len(self.positive_candidates)

        if not self.all_sentences: # Edge case: no sentences found at all
            print("Warning: No sentences extracted from documents. NSPDataset will be empty.")
            self.current_epoch_samples = 0
            return

        if not is_eval:
            # For training, aim for roughly 50/50 positive/negative samples.
            # If no positive samples, all will be negative (random pairs from all_sentences).
            self.current_epoch_samples = self.num_positive_samples * 2 if self.num_positive_samples > 0 else len(self.all_sentences)
        else:
            # For eval, cap total samples by max_samples_for_eval.
            # If positive_candidates were capped, this reflects that.
            # If no positive_candidates, it will be min(max_samples_for_eval, len(all_sentences))
            num_potential_samples = self.num_positive_samples * 2 if self.num_positive_samples > 0 else len(self.all_sentences)
            if max_samples_for_eval is not None:
                 self.current_epoch_samples = min(max_samples_for_eval, num_potential_samples)
            else: # Should not happen if is_eval is true and design is followed, but as fallback:
                 self.current_epoch_samples = num_potential_samples

        if self.num_positive_samples == 0 and self.current_epoch_samples > 0:
            print("Warning: No positive sentence pairs found. NSPDataset will only provide random sentence pairs.")


    def __len__(self):
        return self.current_epoch_samples

    def __getitem__(self, idx: int):
        if not self.all_sentences: # Should not happen if __len__ is 0
            raise IndexError("NSPDataset has no samples.")

        is_positive_sample = (idx % 2 == 0 and self.num_positive_samples > 0) or \
                             (self.num_positive_samples == 0 and len(self.all_sentences) >=2) # if no positives, all are "negative"

        sent_A_text, sent_B_text, nsp_label = "", "", 1

        if is_positive_sample and self.num_positive_samples > 0:
            # Positive Sample
            _, sent_A_text, sent_B_text = self.positive_candidates[idx // 2 % self.num_positive_samples]
            nsp_label = 0
        else:
            # Negative Sample or no positive pairs available
            if len(self.all_sentences) < 2 and self.num_positive_samples == 0 : # Need at least 2 sentences for a random pair
                 # This case should ideally be prevented by __len__ being 0 if not enough sentences.
                 # Fallback: return dummy data or raise error. For now, let's try to make a dummy sample.
                 # This might happen if max_samples_for_eval is 1 and no positives.
                 print(f"Warning: Trying to create a negative sample but only {len(self.all_sentences)} available.")
                 if not self.all_sentences: # Should be caught by initial check
                    # This indicates a logic error if reached.
                    # For robustness, create a completely dummy sample.
                    sent_A_text = "Dummy sentence A."
                    sent_B_text = "Dummy sentence B."
                 else: # Only one sentence available
                    sent_A_text = self.all_sentences[0][2]
                    sent_B_text = self.all_sentences[0][2] # Use the same sentence if only one exists
                 nsp_label = 1 # Still a "negative" pair as it's not a true next sentence

            else: # Standard negative sampling
                sent_A_doc_idx, sent_A_sent_idx, sent_A_text = random.choice(self.all_sentences)

                # Try to find a B that is not A's true next sentence
                for _ in range(10): # Max 10 retries to find a different sentence
                    sent_B_doc_idx, sent_B_sent_idx, sent_B_text = random.choice(self.all_sentences)
                    if sent_A_doc_idx != sent_B_doc_idx or \
                       (sent_A_doc_idx == sent_B_doc_idx and sent_B_sent_idx != sent_A_sent_idx + 1):
                        break
                # If loop finishes, sent_B_text is the last choice, which is acceptable.
                nsp_label = 1

        tokens_A = self.tokenizer_fn(sent_A_text)
        tokens_B = self.tokenizer_fn(sent_B_text)

        # Truncate tokens
        # Max length for A is roughly half, B takes the rest. B can be shorter.
        # Add 1 for potential CLS token if tokenizer adds it (our byte tokenizer doesn't)
        # Add 1 for potential SEP token if tokenizer adds it (our byte tokenizer doesn't)
        # seq_len includes space for these if they were used. With byte tokenizer, it's just content.

        # Simple truncation: A takes up to half, B takes remaining.
        # This needs to be smarter if [CLS] and [SEP] tokens are explicitly added.
        # For now, assuming tokenizer_fn just returns list of byte values.
        # And seq_len is the absolute max for the combined sequence.

        max_tok_A = self.seq_len // 2
        max_tok_B = self.seq_len - len(tokens_A[:max_tok_A]) # B gets what's left

        tokens_A = tokens_A[:max_tok_A]
        tokens_B = tokens_B[:max_tok_B]

        # Ensure B is not truncated to be empty if A is very long and seq_len is small.
        # B must have at least one token if its original text was non-empty.
        # This logic might need refinement if very short seq_len and long tokens_A.
        if not tokens_B and sent_B_text: # If B became empty due to A's length, but B had content
            if tokens_A: # If A has tokens, steal one for B if possible
                tokens_B = [tokens_A.pop()] + tokens_B # Give B the last token of A
            else: # Both A and B somehow ended up empty, though text existed.
                  # This case is unlikely with byte tokenization unless texts were just spaces.
                  # Or if seq_len is extremely small (e.g. 0 or 1).
                  # Add a padding token to B to ensure it's not empty if it had content.
                  tokens_B = [self.pad_token_id]


        input_ids = tokens_A + tokens_B
        token_type_ids = [0] * len(tokens_A) + [1] * len(tokens_B)

        # Create LM labels: predict next token in A, B is masked.
        # Shifted A for labels: tokens_A[1:] + [pad/mask_for_last_A_token]
        # B part of labels is all self.lm_ignore_idx
        lm_labels = []
        if tokens_A:
            lm_labels.extend(tokens_A[1:])
            lm_labels.append(self.lm_ignore_idx) # Or a pad_token_id if predicting last token of A is desired. For now, ignore.

        lm_labels.extend([self.lm_ignore_idx] * len(tokens_B))

        # Pad to self.seq_len
        padding_len = self.seq_len - len(input_ids)
        input_ids.extend([self.pad_token_id] * padding_len)
        token_type_ids.extend([0] * padding_len) # Pad token_type_ids with 0 (or any other aribtrary type for padding)
        lm_labels.extend([self.lm_ignore_idx] * padding_len)

        # Ensure all lists are exactly self.seq_len
        input_ids = input_ids[:self.seq_len]
        token_type_ids = token_type_ids[:self.seq_len]
        lm_labels = lm_labels[:self.seq_len]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'lm_labels': torch.tensor(lm_labels, dtype=torch.long),
            'nsp_label': torch.tensor(nsp_label, dtype=torch.long)
        }


class DataBuilder:
    # ... (constructor and other methods like _tokenize_text, _detokenize_bytes as in original file)
    def __init__(
        self,
        data_cfg: Dict[str, Any]
    ):
        self.dataset_name = data_cfg.get("dataset_name", "allenai/c4")
        self.dataset_config = data_cfg.get("dataset_config", "en")
        self.seq_len = data_cfg.get("seq_len", 512)
        self.max_samples = data_cfg.get("max_samples")
        if self.max_samples is None: # Handle None for max_samples to mean infinity
            self.max_samples = float('inf')
        self.vocab_size = data_cfg.get("vocab_size", 256) # Default vocab_size for byte tokenizer
        self.max_eval_tokens = data_cfg.get("max_eval_tokens", 50000) # This might be interpreted as max_eval_samples for NSP
        self.enable_nsp = data_cfg.get('enable_nsp', False)

        print(f"DataBuilder initialized with enable_nsp: {self.enable_nsp}")
        print(f"Using dataset: {self.dataset_name}/{self.dataset_config}, seq_len: {self.seq_len}")
        print(f"Using UTF-8 byte tokenization with vocabulary size: {self.vocab_size}")

        if self.enable_nsp:
            print(f"Max evaluation samples per split (for NSP): {self.max_eval_tokens}") # Re-interpreting max_eval_tokens as samples
        else:
            print(f"Max evaluation tokens per split (for LM): {self.max_eval_tokens}")

        if self.max_samples != float('inf'):
            print(f"Will attempt to load up to {self.max_samples} documents from the dataset.")
        else:
            print("Will attempt to load all available documents from the dataset.")

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
        raw_dataset = self.load_raw_dataset() # This loads documents like {'text': "..."}
        datasets = {}

        if self.enable_nsp:
            print("NSP is enabled. Creating NSPDatasets.")
            for split_name in ['train', 'validation', 'test']:
                docs_for_split = raw_dataset.get(split_name, [])
                if not docs_for_split:
                    print(f"Warning: No documents found for NSP {split_name} split. Skipping.")
                    datasets[split_name] = None
                    continue

                is_eval_split = split_name in ['validation', 'test']
                # For NSP, max_eval_tokens is interpreted as max_samples_for_eval
                max_s_eval = self.max_eval_tokens if is_eval_split else None
                
                print(f"Creating NSPDataset for {split_name} with {len(docs_for_split)} documents. is_eval={is_eval_split}, max_samples_for_eval={max_s_eval}")
                nsp_dataset = NSPDataset(
                    documents=docs_for_split,
                    tokenizer_fn=self._tokenize_text,
                    seq_len=self.seq_len,
                    is_eval=is_eval_split,
                    max_samples_for_eval=max_s_eval
                )
                if len(nsp_dataset) > 0:
                    datasets[split_name] = nsp_dataset
                    print(f"NSPDataset for {split_name} created with {len(nsp_dataset)} samples.")
                else:
                    print(f"Warning: NSPDataset for {split_name} resulted in 0 samples. Skipping.")
                    datasets[split_name] = None
        else:
            print("NSP is disabled. Creating TokenizedDatasets for standard LM.")
            tokenized_data = self.tokenize_dataset(raw_dataset) # This processes texts into single token streams per split

            for split_name, tokens in tokenized_data.items():
                if not tokens:
                    print(f"Warning: {split_name} split has no tokens. Skipping TokenizedDataset creation.")
                    datasets[split_name] = None
                    continue

                current_max_eval_tokens = self.max_eval_tokens
                # Adjust max_eval_tokens for TokenizedDataset if based on overall document count (max_samples)
                if self.max_samples != float('inf') and split_name in ['validation', 'test']:
                    # This scaling might not be directly applicable if max_samples refers to documents
                    # and max_eval_tokens refers to tokens. Let's keep it simple for now.
                    # scaled_max_tokens = int(self.max_samples * 0.2 * self.seq_len)
                    # current_max_eval_tokens = min(self.max_eval_tokens, scaled_max_tokens)
                    # Ensure at least a few sequences can be formed
                    current_max_eval_tokens = max(current_max_eval_tokens, self.seq_len * 2 + 1)


                if len(tokens) > self.seq_len:
                    # For TokenizedDataset, tokens are truncated for evaluation splits if they exceed current_max_eval_tokens
                    if split_name in ['validation', 'test'] and len(tokens) > current_max_eval_tokens:
                        tokens_for_dataset = tokens[:current_max_eval_tokens]
                        print(f"Limited {split_name} (LM) to {len(tokens_for_dataset)} tokens for faster evaluation (target: {current_max_eval_tokens})")
                    else:
                        tokens_for_dataset = tokens

                    datasets[split_name] = TokenizedDataset(tokens_for_dataset, self.seq_len)
                    print(f"TokenizedDataset for {split_name} (LM) created with {len(datasets[split_name])} samples.")
                else:
                    print(f"Warning: {split_name} split (LM) has insufficient tokens ({len(tokens)}) for seq_len {self.seq_len}. Skipping dataset.")
                    datasets[split_name] = None
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


def create_data_builder(data_config: Dict[str, Any]) -> DataBuilder:
    return DataBuilder(data_cfg=data_config)

if __name__ == "__main__":
    # ... (main test block as in original file)
    print("Testing DataBuilder...")

    # Test with NSP disabled (default behavior)
    print("\n--- Testing with NSP Disabled ---")
    data_cfg_lm = {
        "dataset_name": "allenai/c4", # "wikitext",
        "dataset_config": "en", #"wikitext-2-raw-v1",
        "seq_len": 128,
        "max_samples": 50, # Using a small number for faster testing
        "max_eval_tokens": 10000, # For LM, this is token count
        "enable_nsp": False
    }
    data_builder_lm = create_data_builder(data_cfg_lm)
    dataloaders_lm = data_builder_lm.create_dataloaders(batch_size=2)

    if 'train' in dataloaders_lm and dataloaders_lm['train']:
        train_loader_lm = dataloaders_lm['train']
        print(f"Number of LM training batches: {len(train_loader_lm)}")
        try:
            for batch_idx, (x, y) in enumerate(train_loader_lm):
                print(f"LM Batch {batch_idx}: Input shape: {x.shape}, Target shape: {y.shape}")
                # sample_text_x = data_builder_lm.decode_tokens(x[0][:50])
                # sample_text_y = data_builder_lm.decode_tokens(y[0][:50])
                # print(f"LM Sample X: {sample_text_x}")
                # print(f"LM Sample Y: {sample_text_y}")
                if batch_idx >= 0: break # Check first batch
        except Exception as e:
            print(f"Error during LM dataloader iteration test: {e}")
    else:
        print("LM Train dataloader not created or empty.")

    # Test with NSP enabled
    print("\n--- Testing with NSP Enabled ---")
    data_cfg_nsp = {
        "dataset_name": "allenai/c4", #"wikitext",
        "dataset_config": "en", #"wikitext-2-raw-v1",
        "seq_len": 128,
        "max_samples": 50, # Number of documents to load
        "max_eval_tokens": 20, # For NSP, this is max_samples_for_eval
        "enable_nsp": True
    }
    data_builder_nsp = create_data_builder(data_cfg_nsp)
    dataloaders_nsp = data_builder_nsp.create_dataloaders(batch_size=2)

    if 'train' in dataloaders_nsp and dataloaders_nsp['train']:
        train_loader_nsp = dataloaders_nsp['train']
        print(f"Number of NSP training batches: {len(train_loader_nsp)}")
        try:
            for batch_idx, batch_data in enumerate(train_loader_nsp):
                print(f"NSP Batch {batch_idx}:")
                print(f"  Input IDs shape: {batch_data['input_ids'].shape}")
                print(f"  Token Type IDs shape: {batch_data['token_type_ids'].shape}")
                print(f"  LM Labels shape: {batch_data['lm_labels'].shape}")
                print(f"  NSP Label: {batch_data['nsp_label']}")
                # decoded_sample = data_builder_nsp.decode_tokens(batch_data['input_ids'][0])
                # print(f"  Decoded sample input: {decoded_sample[:100]}...")
                if batch_idx >= 0: break # Check first batch
        except Exception as e:
            print(f"Error during NSP dataloader iteration test: {e}")
    else:
        print("NSP Train dataloader not created or empty.")

    print("\nDataBuilder test completed!")
