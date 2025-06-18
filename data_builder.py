import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset # Keep this import
import numpy as np
from typing import Optional, Dict, Any, List, Tuple # Added Tuple
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
    def __init__(self, documents: list[dict], tokenizer_fn, seq_len: int, is_eval: bool = False, max_samples_for_eval: int = None,
                 enable_word_order_task: bool = False, word_shuffle_probability: float = 0.0):
        self.tokenizer_fn = tokenizer_fn
        self.seq_len = seq_len
        self.is_eval = is_eval
        self.enable_word_order_task = enable_word_order_task
        self.word_shuffle_probability = word_shuffle_probability
        self.pad_token_id = 0
        self.lm_ignore_idx = -100
        self.print_counter = 0 # TEMP DEBUG
        self.max_prints = 5    # TEMP DEBUG

        self._prepare_examples(documents, is_eval, max_samples_for_eval)

    @staticmethod
    def _calculate_levenshtein_word_level(s1: List[str], s2: List[str]) -> int:
        if len(s1) < len(s2):
            return NSPDataset._calculate_levenshtein_word_level(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1_word in enumerate(s1): # Renamed c1 to c1_word for clarity
            current_row = [i + 1]
            for j, c2_word in enumerate(s2): # Renamed c2 to c2_word for clarity
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1_word != c2_word)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def _split_document(self, doc_text: str) -> list[str]:
        if not doc_text:
            return []
        sentences = re.split(r'(?<=[.!?])(?:\s+|\n)', doc_text)
        processed_sentences = []
        for s in sentences:
            s_stripped = s.strip()
            if not s_stripped:
                continue
            if len(s_stripped.split()) >= 3:
                processed_sentences.append(s_stripped)
        return processed_sentences

    def _prepare_examples(self, documents: list[dict], is_eval: bool, max_samples_for_eval: int):
        self.all_sentences = []
        self.positive_candidates = []
        self.sentences_by_doc: Dict[int, List[Tuple[int, str]]] = {}

        for doc_idx, doc_dict in enumerate(documents):
            doc_text = doc_dict.get('text', '')
            if not doc_text:
                continue

            sents = self._split_document(doc_text)
            self.sentences_by_doc[doc_idx] = []

            current_doc_sents_text_only = []
            for sent_idx, sent_text in enumerate(sents):
                self.all_sentences.append((doc_idx, sent_idx, sent_text))
                self.sentences_by_doc[doc_idx].append((sent_idx, sent_text))
                current_doc_sents_text_only.append(sent_text)

            if len(current_doc_sents_text_only) >= 2:
                for i in range(len(current_doc_sents_text_only) - 1):
                    self.positive_candidates.append((doc_idx, current_doc_sents_text_only[i], current_doc_sents_text_only[i+1]))

        if is_eval and max_samples_for_eval is not None:
            if not isinstance(self.positive_candidates, list):
                self.positive_candidates = list(self.positive_candidates)

            if len(self.positive_candidates) > max_samples_for_eval // 2:
                random.shuffle(self.positive_candidates)
                self.positive_candidates = self.positive_candidates[:max_samples_for_eval // 2]

            if len(self.all_sentences) > max_samples_for_eval * 2:
                random.shuffle(self.all_sentences)
                self.all_sentences = self.all_sentences[:max_samples_for_eval * 2]

        self.num_positive_samples = len(self.positive_candidates)

        if not self.all_sentences:
            print("Warning: No sentences extracted from documents. NSPDataset will be empty.")
            self.current_epoch_samples = 0
            return

        if not is_eval:
            self.current_epoch_samples = self.num_positive_samples * 2 if self.num_positive_samples > 0 else len(self.all_sentences)
        else:
            num_potential_samples = self.num_positive_samples * 2 if self.num_positive_samples > 0 else len(self.all_sentences)
            if max_samples_for_eval is not None:
                 self.current_epoch_samples = min(max_samples_for_eval, num_potential_samples)
            else:
                 self.current_epoch_samples = num_potential_samples

        if self.num_positive_samples == 0 and self.current_epoch_samples > 0:
            print("Warning: No positive sentence pairs found. NSPDataset will only provide random sentence pairs.")

    def __len__(self):
        return self.current_epoch_samples

    def __getitem__(self, idx: int):
        if not self.all_sentences:
            raise IndexError("NSPDataset has no samples.")

        is_positive_sample = (idx % 2 == 0 and self.num_positive_samples > 0) or \
                             (self.num_positive_samples == 0 and len(self.all_sentences) >=2)

        original_sent_A_text, sent_B_text, nsp_label = "", "", 1

        if is_positive_sample and self.num_positive_samples > 0:
            _, original_sent_A_text, sent_B_text = self.positive_candidates[idx // 2 % self.num_positive_samples]
            nsp_label = 0
        else: # Negative NSP sample
            if len(self.all_sentences) < 2 and self.num_positive_samples == 0 :
                 print(f"Warning: Trying to create a negative sample but only {len(self.all_sentences)} available.")
                 if not self.all_sentences:
                    original_sent_A_text = "Dummy sentence A."
                    sent_B_text = "Dummy sentence B."
                 else:
                    original_sent_A_text = self.all_sentences[0][2]
                    sent_B_text = self.all_sentences[0][2]
                 nsp_label = 1
            else:
                sent_A_doc_idx, sent_A_sent_idx, original_sent_A_text = random.choice(self.all_sentences)

                # Attempt In-Document Negative Sampling
                doc_sentences_tuples = self.sentences_by_doc.get(sent_A_doc_idx, [])
                in_doc_negative_candidates = [
                    s_text for s_idx, s_text in doc_sentences_tuples
                    if s_idx != sent_A_sent_idx and s_idx != (sent_A_sent_idx + 1)
                ]

                used_in_doc_negative_flag = False # TEMP DEBUG
                if in_doc_negative_candidates:
                    sent_B_text = random.choice(in_doc_negative_candidates)
                    used_in_doc_negative_flag = True # TEMP DEBUG
                else:
                    # Fallback to Global Random Sampling
                    global_fallback_sent_B_text = "Dummy fallback B sentence."
                    if self.all_sentences:
                        if self.all_sentences[0][2] != original_sent_A_text or len(self.all_sentences) == 1:
                             global_fallback_sent_B_text = self.all_sentences[0][2]
                        elif len(self.all_sentences) > 1:
                             global_fallback_sent_B_text = self.all_sentences[1][2]

                    temp_sent_B_text_global = global_fallback_sent_B_text
                    chosen_B_text_global_assigned = False
                    for _ in range(10):
                        sent_B_doc_idx_global, sent_B_sent_idx_global, chosen_B_text_global = random.choice(self.all_sentences)
                        if sent_A_doc_idx != sent_B_doc_idx_global or \
                           (sent_A_doc_idx == sent_B_doc_idx_global and sent_B_sent_idx_global != sent_A_sent_idx + 1):
                            temp_sent_B_text_global = chosen_B_text_global
                            chosen_B_text_global_assigned = True
                            break
                    if not chosen_B_text_global_assigned and 'chosen_B_text_global' in locals(): # if loop finished but a choice was made
                        temp_sent_B_text_global = chosen_B_text_global

                    sent_B_text = temp_sent_B_text_global
                nsp_label = 1

        # TEMP DEBUG PRINT for NSP
        if hasattr(self, 'print_counter') and self.print_counter < self.max_prints and not self.is_eval :
            print("\n--- Sample NSP Pair ---")
            print(f"Index: {idx}")
            print(f"NSP Label: {nsp_label} (0=IsNext, 1=IsNotNext/Random)")
            print(f"Sentence A (original): {original_sent_A_text}")
            # We'll print shuffled A later if WOD is active for this sample
            print(f"Sentence B: {sent_B_text}")
            if nsp_label == 1 and 'used_in_doc_negative_flag' in locals(): # Check if flag exists (only for negative samples)
                 print(f"Used In-Doc Negative: {used_in_doc_negative_flag}")
            # print("------------------------\n") # Delay this print until after WOD info
        # END TEMP DEBUG PRINT

        original_words = original_sent_A_text.split(' ')
        input_words = list(original_words)
        gate_lm_loss_for_this_sample = False
        word_order_score_label = 1.0

        if self.enable_word_order_task and random.random() < self.word_shuffle_probability and len(original_words) > 1:
            shuffled_input_words = list(original_words)
            random.shuffle(shuffled_input_words)

            if shuffled_input_words != original_words:
                input_words = shuffled_input_words
                gate_lm_loss_for_this_sample = True
                edit_dist = NSPDataset._calculate_levenshtein_word_level(original_words, input_words)
                normalized_dist = edit_dist / len(original_words) if len(original_words) > 0 else 0.0
                word_order_score_label = max(0.0, 1.0 - normalized_dist)

        input_sentence_for_tokenizer = ' '.join(input_words)

        # TEMP DEBUG PRINT for WOD (continued)
        if hasattr(self, 'print_counter') and self.print_counter < self.max_prints and not self.is_eval: # Check counter again
            if input_sentence_for_tokenizer != original_sent_A_text:
                print(f"Sentence A (shuffled for WOD input): {input_sentence_for_tokenizer}")
            print(f"WOD Score Label: {word_order_score_label}")
            print(f"LM Labels Gated for A: {gate_lm_loss_for_this_sample}")
            print("------------------------\n")
            if idx % 2 != 0 or nsp_label == 0 : # Increment only once per effective sample pair log
                self.print_counter += 1
        # END TEMP DEBUG PRINT


        tokens_A = self.tokenizer_fn(input_sentence_for_tokenizer)
        tokens_B = self.tokenizer_fn(sent_B_text)

        max_tok_A = self.seq_len // 2
        max_tok_B = self.seq_len - len(tokens_A[:max_tok_A])

        tokens_A = tokens_A[:max_tok_A]
        tokens_B = tokens_B[:max_tok_B]

        if not tokens_B and sent_B_text:
            if tokens_A:
                tokens_B = [tokens_A.pop()] + tokens_B
            else:
                  tokens_B = [self.pad_token_id]

        input_ids = tokens_A + tokens_B
        token_type_ids = [0] * len(tokens_A) + [1] * len(tokens_B)

        lm_labels = []
        if gate_lm_loss_for_this_sample:
            lm_labels.extend([self.lm_ignore_idx] * len(tokens_A))
        else:
            if tokens_A:
                lm_labels.extend(tokens_A[1:])
                lm_labels.append(self.lm_ignore_idx)

        lm_labels.extend([self.lm_ignore_idx] * len(tokens_B))

        padding_len = self.seq_len - len(input_ids)
        input_ids.extend([self.pad_token_id] * padding_len)
        token_type_ids.extend([0] * padding_len)
        lm_labels.extend([self.lm_ignore_idx] * padding_len)

        input_ids = input_ids[:self.seq_len]
        token_type_ids = token_type_ids[:self.seq_len]
        lm_labels = lm_labels[:self.seq_len]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'lm_labels': torch.tensor(lm_labels, dtype=torch.long),
            'nsp_label': torch.tensor(nsp_label, dtype=torch.long),
            'word_order_score_label': torch.tensor(word_order_score_label, dtype=torch.float)
        }


class DataBuilder:
    def __init__(
        self,
        data_cfg: Dict[str, Any]
    ):
        self.dataset_name = data_cfg.get("dataset_name", "allenai/c4")
        self.dataset_config = data_cfg.get("dataset_config", "en")
        self.seq_len = data_cfg.get("seq_len", 512)
        self.max_samples = data_cfg.get("max_samples")
        if self.max_samples is None:
            self.max_samples = float('inf')
        self.vocab_size = data_cfg.get("vocab_size", 256)
        self.max_eval_tokens = data_cfg.get("max_eval_tokens", 50000)
        self.enable_nsp = data_cfg.get('enable_nsp', False)
        self.enable_word_order_task = data_cfg.get('enable_word_order_task', False)
        self.word_shuffle_probability = data_cfg.get('word_shuffle_probability', 0.15)

        print(f"DataBuilder initialized with: enable_nsp={self.enable_nsp}, enable_word_order_task={self.enable_word_order_task}")
        if self.enable_word_order_task:
            print(f"  Word shuffle probability: {self.word_shuffle_probability}")
        print(f"Using dataset: {self.dataset_name}/{self.dataset_config}, seq_len: {self.seq_len}")
        print(f"Using UTF-8 byte tokenization with vocabulary size: {self.vocab_size}")

        if self.enable_nsp or self.enable_word_order_task:
            print(f"Max evaluation samples per split (for NSP/WOD): {self.max_eval_tokens}")
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
            if not isinstance(split_data_list, list):
                print(f"Warning: {split_name} data is not a list (type: {type(split_data_list)}), skipping tokenization.")
                tokenized_data[split_name] = []
                continue
            if not split_data_list:
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
        datasets = {}

        if self.enable_nsp: # Also implies WOD can be enabled if NSPDataset handles it
            print("NSP or WOD is enabled. Creating NSPDatasets.")
            for split_name in ['train', 'validation', 'test']:
                docs_for_split = raw_dataset.get(split_name, [])
                if not docs_for_split:
                    print(f"Warning: No documents found for {split_name} split. Skipping.")
                    datasets[split_name] = None
                    continue

                is_eval_split = split_name in ['validation', 'test']
                max_s_eval = self.max_eval_tokens if is_eval_split else None
                
                print(f"Creating NSPDataset for {split_name} with {len(docs_for_split)} documents. is_eval={is_eval_split}, max_samples_for_eval={max_s_eval}")
                dataset_instance = NSPDataset( # Renamed from nsp_dataset to generic dataset_instance
                    documents=docs_for_split,
                    tokenizer_fn=self._tokenize_text,
                    seq_len=self.seq_len,
                    is_eval=is_eval_split,
                    max_samples_for_eval=max_s_eval,
                    enable_word_order_task=self.enable_word_order_task,
                    word_shuffle_probability=self.word_shuffle_probability
                )
                if len(dataset_instance) > 0:
                    datasets[split_name] = dataset_instance
                    print(f"NSPDataset for {split_name} created with {len(dataset_instance)} samples.")
                else:
                    print(f"Warning: NSPDataset for {split_name} resulted in 0 samples. Skipping.")
                    datasets[split_name] = None
        else: # Only standard LM if NSP is off (current logic means WOD also off)
            print("NSP and WOD are disabled. Creating TokenizedDatasets for standard LM.")
            tokenized_data = self.tokenize_dataset(raw_dataset)

            for split_name, tokens in tokenized_data.items():
                if not tokens:
                    print(f"Warning: {split_name} split has no tokens. Skipping TokenizedDataset creation.")
                    datasets[split_name] = None
                    continue

                current_max_eval_tokens = self.max_eval_tokens
                if self.max_samples != float('inf') and split_name in ['validation', 'test']:
                    current_max_eval_tokens = max(current_max_eval_tokens, self.seq_len * 2 + 1)

                if len(tokens) > self.seq_len:
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
    print("Testing DataBuilder...")

    print("\n--- Testing with NSP Disabled ---")
    data_cfg_lm = {
        "dataset_name": "allenai/c4",
        "dataset_config": "en",
        "seq_len": 128,
        "max_samples": 50,
        "max_eval_tokens": 10000,
        "enable_nsp": False,
        "enable_word_order_task": False # Ensure WOD is off for this test
    }
    data_builder_lm = create_data_builder(data_cfg_lm)
    dataloaders_lm = data_builder_lm.create_dataloaders(batch_size=2)

    if 'train' in dataloaders_lm and dataloaders_lm['train']:
        train_loader_lm = dataloaders_lm['train']
        print(f"Number of LM training batches: {len(train_loader_lm)}")
        try:
            for batch_idx, (x, y) in enumerate(train_loader_lm):
                print(f"LM Batch {batch_idx}: Input shape: {x.shape}, Target shape: {y.shape}")
                if batch_idx >= 0: break
        except Exception as e:
            print(f"Error during LM dataloader iteration test: {e}")
    else:
        print("LM Train dataloader not created or empty.")

    print("\n--- Testing with NSP Enabled & WOD Enabled ---")
    data_cfg_nsp_wod = {
        "dataset_name": "allenai/c4",
        "dataset_config": "en",
        "seq_len": 128,
        "max_samples": 50,
        "max_eval_tokens": 20,
        "enable_nsp": True, # NSPDataset will be used
        "enable_word_order_task": True,
        "word_shuffle_probability": 0.50 # Increased for better chance of seeing shuffled samples
    }
    data_builder_nsp_wod = create_data_builder(data_cfg_nsp_wod)
    # Initialize print counter for NSPDataset if it's going to be used
    # This is a bit of a hack for testing; ideally, the dataset object would be accessed directly
    # For now, assume the DataBuilder will create it and it will have the counter.
    # data_builder_nsp_wod.datasets['train'].print_counter = 0 # This won't work as datasets not created yet

    dataloaders_nsp_wod = data_builder_nsp_wod.create_dataloaders(batch_size=2)

    if 'train' in dataloaders_nsp_wod and dataloaders_nsp_wod['train']:
        # Access the dataset object to reset counter for testing, if possible and needed
        # This assumes 'train' dataset exists and is an NSPDataset instance
        # For robust testing, this might need a more direct way to access or pass debug flags
        if hasattr(dataloaders_nsp_wod['train'].dataset, 'print_counter'):
             dataloaders_nsp_wod['train'].dataset.print_counter = 0


        train_loader_nsp_wod = dataloaders_nsp_wod['train']
        print(f"Number of NSP/WOD training batches: {len(train_loader_nsp_wod)}")
        try:
            for batch_idx, batch_data in enumerate(train_loader_nsp_wod):
                print(f"NSP/WOD Batch {batch_idx}:")
                print(f"  Input IDs shape: {batch_data['input_ids'].shape}")
                print(f"  Token Type IDs shape: {batch_data['token_type_ids'].shape}")
                print(f"  LM Labels shape: {batch_data['lm_labels'].shape}")
                print(f"  NSP Label: {batch_data['nsp_label']}")
                print(f"  WOD Score Label: {batch_data['word_order_score_label']}")
                if batch_idx >= 0: break
        except Exception as e:
            print(f"Error during NSP/WOD dataloader iteration test: {e}")
    else:
        print("NSP/WOD Train dataloader not created or empty.")

    print("\nDataBuilder test completed!")

[end of data_builder.py]
