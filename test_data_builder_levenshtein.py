import unittest
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_builder import DataBuilder, create_data_builder
from levenshtein_dataset import LevenshteinDataset
from text_utils import shuffle_words_in_sentence, word_levenshtein_distance

# Mock Tokenizer for testing
class MockTokenizer:
    def __init__(self, vocab_size=256):
        self.vocab = {f"<token_{i}>": i for i in range(vocab_size)}
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = 0 # Assuming 0 is pad for mock

    def encode(self, text, add_special_tokens=False):
        # Simple space-based tokenization for mock
        return [self.vocab.get(word, 0) for word in text.split()]

    def decode(self, token_ids):
        return " ".join([self.ids_to_tokens.get(id, "<unk>") for id in token_ids])

class TestLevenshteinDataset(unittest.TestCase):

    def setUp(self):
        self.cls_token_id = 256
        self.pad_token_id_for_lm_targets = -1
        self.pad_token_id_for_inputs = 0 # Common practice for input padding
        self.seq_len = 30 # Max sequence length for inputs and targets
        self.mock_tokenizer = MockTokenizer(vocab_size=255) # vocab up to 254, CLS is 256

        # Sample documents (raw text strings)
        self.raw_documents = [
            "This is the first document.",
            "Another example document is here for testing purposes.",
            "Short one.",
            "A very long document that will surely exceed the sequence length and require truncation to test that specific functionality of the dataset preparation.",
            "Shuffle me please, this is fun."
        ]

        # Manually create tokenized versions for direct dataset instantiation if needed
        # For LevenshteinDataset, it expects raw text, so this is more for reference
        self.tokenized_docs_example = [
            self.mock_tokenizer.encode("This is the first document."),
            self.mock_tokenizer.encode("Another example document is here for testing purposes.")
        ]


    def test_prepare_examples_levenshtein(self):
        print("\nRunning test_prepare_examples_levenshtein")
        dataset = LevenshteinDataset(
            raw_text_data=self.raw_documents,
            tokenizer=self.mock_tokenizer, # Not directly used by _prepare_examples
            seq_len=self.seq_len,
            cls_token_id=self.cls_token_id,
            pad_token_id_for_lm_targets=self.pad_token_id_for_lm_targets,
            pad_token_id_for_inputs=self.pad_token_id_for_inputs
        )
        # _prepare_examples is called in __init__
        self.assertTrue(len(dataset.examples) > 0)

        for example in dataset.examples:
            self.assertIn("original_text", example)
            self.assertIn("shuffled_text", example)
            self.assertIn("levenshtein_distance", example)

            orig_words = example["original_text"].split()
            shuf_words = example["shuffled_text"].split()

            if example["original_text"] == example["shuffled_text"]:
                # This can happen if sentence is too short to shuffle meaningfully
                self.assertEqual(example["levenshtein_distance"], 0)
            else:
                # Recompute to verify. Note: shuffle_words_in_sentence has randomness,
                # so we check consistency with its output, not a fixed shuffle.
                # The dataset stores the *actual* shuffle it produced.
                expected_dist = word_levenshtein_distance(orig_words, shuf_words)
                self.assertEqual(example["levenshtein_distance"], expected_dist)
            self.assertGreaterEqual(example["levenshtein_distance"], 0)

    def test_levenshtein_dataset_getitem(self):
        print("\nRunning test_levenshtein_dataset_getitem")
        dataset = LevenshteinDataset(
            raw_text_data=self.raw_documents,
            tokenizer=self.mock_tokenizer,
            seq_len=self.seq_len,
            cls_token_id=self.cls_token_id,
            pad_token_id_for_lm_targets=self.pad_token_id_for_lm_targets,
            pad_token_id_for_inputs=self.pad_token_id_for_inputs
        )

        self.assertTrue(len(dataset) > 0)

        # Get a sample item
        original_tokens_cls, lm_targets, shuffled_tokens_cls, \
        target_lev_distance, target_coherence_score = dataset[0]

        # 1. Verify 5 tensors are returned
        self.assertIsInstance(original_tokens_cls, torch.Tensor)
        self.assertIsInstance(lm_targets, torch.Tensor)
        self.assertIsInstance(shuffled_tokens_cls, torch.Tensor)
        self.assertIsInstance(target_lev_distance, torch.Tensor)
        self.assertIsInstance(target_coherence_score, torch.Tensor)

        # 2. Check shapes
        self.assertEqual(original_tokens_cls.shape, (self.seq_len,))
        self.assertEqual(lm_targets.shape, (self.seq_len,))
        self.assertEqual(shuffled_tokens_cls.shape, (self.seq_len,))
        self.assertEqual(target_lev_distance.shape, torch.Size([])) # Scalar
        self.assertEqual(target_coherence_score.shape, torch.Size([])) # Scalar

        # 3. Check dtypes
        self.assertEqual(original_tokens_cls.dtype, torch.long)
        self.assertEqual(lm_targets.dtype, torch.long)
        self.assertEqual(shuffled_tokens_cls.dtype, torch.long)
        self.assertEqual(target_lev_distance.dtype, torch.float32) # Or long, check dataset impl. float is typical for loss.
        self.assertEqual(target_coherence_score.dtype, torch.float32)

        # 4. Assert CLS token presence
        self.assertEqual(original_tokens_cls[0].item(), self.cls_token_id)
        self.assertEqual(shuffled_tokens_cls[0].item(), self.cls_token_id)

        # 5. Assert target_coherence_score is 0.0 for the original text's aux output
        self.assertEqual(target_coherence_score.item(), 0.0)

        # 6. Verify LM targets
        #    - CLS position should be padded
        self.assertEqual(lm_targets[0].item(), self.pad_token_id_for_lm_targets)
        #    - Other tokens should be shifted version of original_tokens_cls (excluding CLS)
        #      or padded if original sequence was shorter than seq_len.
        for i in range(1, self.seq_len):
            if original_tokens_cls[i].item() == self.pad_token_id_for_inputs : # End of original input sequence
                self.assertEqual(lm_targets[i-1].item(), self.pad_token_id_for_lm_targets) # Previous actual token's target also padded
                self.assertEqual(lm_targets[i].item(), self.pad_token_id_for_lm_targets)   # Current and subsequent also padded
            elif i < self.seq_len -1 and original_tokens_cls[i+1].item() != self.pad_token_id_for_inputs: # If next token is not padding
                 self.assertEqual(lm_targets[i].item(), original_tokens_cls[i+1].item())
            elif i == self.seq_len -1 or original_tokens_cls[i+1].item() == self.pad_token_id_for_inputs: # last token before padding or end of seq
                 self.assertEqual(lm_targets[i].item(), self.pad_token_id_for_lm_targets)


        # 7. Test truncation (using the long document)
        long_doc_idx = -1 # Assuming the last doc is the long one
        for i, ex in enumerate(dataset.examples):
            if "truncation" in ex["original_text"]: # Quick check
                long_doc_idx = i
                break
        self.assertNotEqual(long_doc_idx, -1, "Long document for truncation test not found in examples.")

        original_tokens_cls_trunc, _, shuffled_tokens_cls_trunc, _, _ = dataset[long_doc_idx]

        # Check if actual content (non-CLS, non-pad) is less than seq_len - 1
        # For original_tokens_cls_trunc
        original_content_len = sum(1 for t_id in original_tokens_cls_trunc[1:] if t_id != self.pad_token_id_for_inputs)
        self.assertLessEqual(original_content_len, self.seq_len - 1) # -1 for CLS

        # For shuffled_tokens_cls_trunc
        shuffled_content_len = sum(1 for t_id in shuffled_tokens_cls_trunc[1:] if t_id != self.pad_token_id_for_inputs)
        self.assertLessEqual(shuffled_content_len, self.seq_len - 1) # -1 for CLS


class TestDataBuilderLevenshtein(unittest.TestCase):

    def setUp(self):
        self.base_vocab_size = 256 # From DataBuilder default
        self.cls_token_id_expected = 256 # Expected CLS ID when added
        self.dataset_name = "hf-internal-testing/tiny_c4_ संस्कृत" # Small, multilingual
        self.dataset_config = "sa" # Sanskrit part
        self.seq_len = 64
        self.max_samples = 20 # Keep this very small for CI tests
        self.data_builder_lev = create_data_builder(
            dataset_name=self.dataset_name,
            dataset_config=self.dataset_config,
            seq_len=self.seq_len,
            max_samples=self.max_samples,
            use_levenshtein_task=True # Enable Levenshtein task
        )
        self.data_builder_no_lev = create_data_builder(
            dataset_name=self.dataset_name,
            dataset_config=self.dataset_config,
            seq_len=self.seq_len,
            max_samples=self.max_samples,
            use_levenshtein_task=False # Disable Levenshtein task
        )

    def test_vocab_and_special_tokens_levenshtein(self):
        print("\nRunning test_vocab_and_special_tokens_levenshtein")
        # Test with Levenshtein task enabled
        self.assertTrue(self.data_builder_lev.use_levenshtein_task)
        self.assertIsNotNone(self.data_builder_lev.cls_token_id)
        self.assertEqual(self.data_builder_lev.cls_token_id, self.cls_token_id_expected)
        # Vocab size should be base + 1 (for CLS) if CLS ID is at the boundary
        self.assertEqual(self.data_builder_lev.get_vocab_size(), self.base_vocab_size + 1)
        self.assertFalse(hasattr(self.data_builder_lev, 'sep_token_id'))

        # Test with Levenshtein task disabled
        self.assertFalse(self.data_builder_no_lev.use_levenshtein_task)
        self.assertIsNone(self.data_builder_no_lev.cls_token_id)
        self.assertEqual(self.data_builder_no_lev.get_vocab_size(), self.base_vocab_size)

    def test_tokenize_dataset_levenshtein_mode(self):
        print("\nRunning test_tokenize_dataset_levenshtein_mode")
        # This test verifies that when use_levenshtein_task=True,
        # tokenize_dataset returns a dictionary of lists of raw text strings.
        tokenized_output = self.data_builder_lev.tokenize_dataset()
        self.assertIsInstance(tokenized_output, dict)
        self.assertIn('train', tokenized_output)
        self.assertIsInstance(tokenized_output['train'], list)
        if len(tokenized_output['train']) > 0:
            self.assertIsInstance(tokenized_output['train'][0], str)

        # Compare with standard tokenization (use_levenshtein_task=False)
        standard_tokenized_output = self.data_builder_no_lev.tokenize_dataset()
        self.assertIsInstance(standard_tokenized_output, dict)
        self.assertIn('train', standard_tokenized_output)
        self.assertIsInstance(standard_tokenized_output['train'], list)
        if len(standard_tokenized_output['train']) > 0:
            # Expect list of lists of ints for standard tokenization
            self.assertIsInstance(standard_tokenized_output['train'][0], list)
            if len(standard_tokenized_output['train'][0]) > 0:
                 self.assertIsInstance(standard_tokenized_output['train'][0][0], int)


    @unittest.skipIf(os.environ.get("CI_SKIP_DATALOADER_TESTS") == "true", "Skipping dataloader tests in CI env")
    def test_create_dataloaders_levenshtein_mode(self):
        print("\nRunning test_create_dataloaders_levenshtein_mode")
        # Ensure DataBuilder is set up for Levenshtein task
        db_lev = create_data_builder(
            dataset_name=self.dataset_name,
            dataset_config=self.dataset_config,
            seq_len=self.seq_len,
            max_samples=self.max_samples, # Small number of samples
            use_levenshtein_task=True
        )

        dataloaders = db_lev.create_dataloaders(batch_size=2)
        self.assertTrue('train' in dataloaders)

        if 'train' in dataloaders and len(dataloaders['train']) > 0:
            batch_count = 0
            for batch in dataloaders['train']:
                # 1. Check for 5 tensor components
                self.assertEqual(len(batch), 5)

                original_tokens_cls, lm_targets, shuffled_tokens_cls, \
                target_lev_distance, target_coherence_score = batch

                # 2. Check shapes (assuming batch_size=2, seq_len=self.seq_len)
                self.assertEqual(original_tokens_cls.shape, (2, self.seq_len))
                self.assertEqual(lm_targets.shape, (2, self.seq_len))
                self.assertEqual(shuffled_tokens_cls.shape, (2, self.seq_len))
                self.assertEqual(target_lev_distance.shape, (2,)) # Batch of scalars
                self.assertEqual(target_coherence_score.shape, (2,)) # Batch of scalars

                # 3. Check dtypes
                self.assertEqual(original_tokens_cls.dtype, torch.long)
                self.assertEqual(lm_targets.dtype, torch.long)
                self.assertEqual(shuffled_tokens_cls.dtype, torch.long)
                self.assertEqual(target_lev_distance.dtype, torch.float32)
                self.assertEqual(target_coherence_score.dtype, torch.float32)

                # 4. Decode some original_tokens_cls to check CLS presence
                sample_decoded = db_lev.decode_tokens(original_tokens_cls[0])
                # The mock tokenizer in DataBuilder might use byte-level, so CLS might not be "[CLS]" string
                # Instead, check the token ID
                self.assertEqual(original_tokens_cls[0, 0].item(), db_lev.cls_token_id)
                self.assertEqual(shuffled_tokens_cls[0, 0].item(), db_lev.cls_token_id)

                batch_count += 1
                if batch_count >= 1: # Check one batch is enough
                    break
            self.assertTrue(batch_count > 0, "Levenshtein Dataloader (train) did not yield any batches.")
        else:
            print(f"Warning: Levenshtein Dataloader (train) is empty for {self.dataset_name}/{self.dataset_config} with {self.max_samples} samples. Test might not be comprehensive.")

if __name__ == '__main__':
    unittest.main()
