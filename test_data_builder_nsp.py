import unittest
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_builder import DataBuilder, create_data_builder
from nsp_dataset import NSPDataset # Assuming nsp_dataset.py is in the same directory or accessible

class TestDataBuilderNSP(unittest.TestCase):

    def test_segment_text_to_sentences(self):
        print("\nRunning test_segment_text_to_sentences")
        db = DataBuilder(nsp_task=True) # nsp_task enables sentence segmentation logic if used internally

        text1 = "This is the first sentence. This is the second one! Is this the third? Yes."
        expected1 = ["This is the first sentence.", "This is the second one!", "Is this the third?", "Yes."]
        self.assertEqual(db._segment_text_to_sentences(text1), expected1)

        text2 = "Mr. Smith went to Washington. He visited the White House."
        expected2 = ["Mr. Smith went to Washington.", "He visited the White House."]
        # The regex might split "Mr." depending on its exact form. Current regex might be okay.
        # Let's test current behavior. If it splits "Mr.", the test will show.
        # Current regex: r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|!)\s'
        # This should correctly handle "Mr. Smith"
        self.assertEqual(db._segment_text_to_sentences(text2), expected2)

        text3 = "One. Two! Three? Four.Five." # No space after "Four."
        expected3 = ["One.", "Two!", "Three?", "Four.Five."] # Current regex needs space to split.
        self.assertEqual(db._segment_text_to_sentences(text3), expected3)

        text4 = "This is a sentence.... another one." # Multiple dots
        expected4 = ["This is a sentence.... another one."] # May treat multiple dots as part of sentence or split differently
                                                       # Current regex will likely keep them together if no space follows the sequence of dots.
        self.assertEqual(db._segment_text_to_sentences(text4), expected4)

        text5 = ""
        expected5 = []
        self.assertEqual(db._segment_text_to_sentences(text5), expected5)

        text6 = "Short one." # Test min_sentence_length (default 5)
        expected6 = ["Short one."]
        self.assertEqual(db._segment_text_to_sentences(text6), expected6)

        text7 = "Tiny" # Shorter than min_sentence_length
        expected7 = []
        self.assertEqual(db._segment_text_to_sentences(text7), expected7)

    def test_nsp_dataset_creation_and_getitem(self):
        print("\nRunning test_nsp_dataset_creation_and_getitem")
        cls_id = 256
        sep_id = 257
        pad_id_for_lm = -1
        seq_len = 20 # Keep it reasonable for manual trace

        doc1_s1 = [10, 11, 12]       # len 3
        doc1_s2 = [13, 14, 15, 16]  # len 4
        doc2_s1 = [20, 21]          # len 2
        doc2_s2 = [22, 23, 24]      # len 3

        # For negative sampling, add more diverse sentences
        doc3_s1 = [30,31,32,33,34,35,36,37,38,39] # len 10 (will cause truncation)

        documents = [
            [doc1_s1, doc1_s2], # Doc 1
            [doc2_s1, doc2_s2], # Doc 2
            [doc3_s1]           # Doc 3 (for negative sampling pool)
        ]

        dataset = NSPDataset(documents, seq_len, cls_id, sep_id, pad_id_for_lm, nsp_neg_prob=0.0) # Force positive
        # Expected positive examples: (d1s1,d1s2), (d2s1,d2s2)
        # Total examples = number of sentences that can be sentence_a
        # Doc1: s1 (can be A) -> (s1,s2) label 1
        # Doc2: s1 (can be A) -> (s2,s2) label 1
        # Doc3: s1 (can be A) -> no s2, so must be negative.
        # If nsp_neg_prob=0.0, doc3_s1 cannot form a positive pair. It will try to form a negative.
        # Let's adjust expectations for _create_examples logic for nsp_neg_prob=0.0
        # It will try to make positive if possible. If not (e.g. last sentence in doc), it must make negative.
        # So, with nsp_neg_prob=0.0:
        # (d1s1, d1s2, 1), (d2s1, d2s2, 1). doc1_s2 and doc2_s2 and doc3_s1 will form negative pairs.
        # This means 2 positive, 3 negative if all_sentences_tokenized is diverse enough.
        # For testing, let's make nsp_neg_prob=0.0 to ensure positive pairs are chosen when possible.
        # Number of examples from create_examples:
        # d1s1 -> (d1s1,d1s2,1)
        # d1s2 -> (d1s2, random_neg, 0)
        # d2s1 -> (d2s1,d2s2,1)
        # d2s2 -> (d2s2, random_neg, 0)
        # d3s1 -> (d3s1, random_neg, 0)
        # Total = 5 examples
        self.assertEqual(len(dataset), 5)

        # Example 1: (doc1_s1, doc1_s2, 1)
        # [CLS] d1s1 [SEP] d1s2 [SEP]
        # [256, 10,11,12, 257, 13,14,15,16, 257] -> len 10
        inputs, lm_targets, nsp_label = dataset[0]
        self.assertEqual(nsp_label.item(), 1)
        expected_input_ids = [cls_id] + doc1_s1 + [sep_id] + doc1_s2 + [sep_id]
        expected_input_ids_padded = expected_input_ids + [0] * (seq_len - len(expected_input_ids))
        self.assertEqual(inputs.tolist(), expected_input_ids_padded)

        # LM targets for example 1:
        # Input: [256, 10, 11, 12, 257, 13, 14, 15, 16, 257, 0, 0, ...]
        # Target (shifted): [10, 11, 12, 257, 13, 14, 15, 16, 257, 0, 0, ..., PAD_ID]
        # Masked:
        # CLS (idx 0) -> PAD_ID
        # SEP1 (idx 1+len(doc1_s1)=4) -> PAD_ID
        # SEP2 (idx 1+len(doc1_s1)+1+len(doc1_s2)=1+3+1+4=9) -> PAD_ID
        # Padding (idx 10 onwards) -> PAD_ID
        expected_lm_targets = [pad_id_for_lm, 11, 12, sep_id, pad_id_for_lm, 14, 15, 16, sep_id, pad_id_for_lm]
        expected_lm_targets_padded = expected_lm_targets + [pad_id_for_lm] * (seq_len - len(expected_lm_targets))
        self.assertEqual(lm_targets.tolist(), expected_lm_targets_padded)


        # Test truncation: sent_a long, sent_b short
        # sent_a = doc3_s1 (len 10), sent_b = doc2_s1 (len 2)
        # [CLS] doc3_s1 [SEP] doc2_s1 [SEP]
        # [256] + 10 tokens + [257] + 2 tokens + [257] = 1+10+1+2+1 = 15 tokens
        # If seq_len = 10:
        # Max content = 10 - 3 = 7
        # A(10) + B(2) = 12 > 7
        # Truncate B: B_len_new = max(0, 7 - 10) = 0. B becomes []
        # A(10) + B(0) = 10 > 7
        # Truncate A: A_len_new = max(0, 7 - 0) = 7. A becomes A[:7]
        # Final: [CLS] A[:7] [SEP] [] [SEP] = 1+7+1+0+1 = 10
        dataset_trunc_A = NSPDataset([ [doc3_s1, doc2_s1] ], seq_len=10, cls_token_id=cls_id, sep_token_id=sep_id, pad_token_id=pad_id_for_lm, nsp_neg_prob=0.0)
        inputs_trunc_A, _, _ = dataset_trunc_A[0] # Should be (doc3_s1, doc2_s1, 1)
        expected_input_trunc_A = [cls_id] + doc3_s1[:7] + [sep_id] + [] + [sep_id]
        self.assertEqual(inputs_trunc_A.tolist(), expected_input_trunc_A)

        # Test truncation: sent_a short, sent_b long
        # sent_a = doc2_s1 (len 2), sent_b = doc3_s1 (len 10)
        # [CLS] doc2_s1 [SEP] doc3_s1 [SEP]
        # [256] + 2 tokens + [257] + 10 tokens + [257] = 1+2+1+10+1 = 15 tokens
        # If seq_len = 10:
        # Max content = 10 - 3 = 7
        # A(2) + B(10) = 12 > 7
        # Truncate B: B_len_new = max(0, 7 - 2) = 5. B becomes B[:5]
        # A(2) + B(5) = 7. Fits.
        # Final: [CLS] A(2) [SEP] B[:5] [SEP] = 1+2+1+5+1 = 10
        dataset_trunc_B = NSPDataset([ [doc2_s1, doc3_s1] ], seq_len=10, cls_token_id=cls_id, sep_token_id=sep_id, pad_token_id=pad_id_for_lm, nsp_neg_prob=0.0)
        inputs_trunc_B, _, _ = dataset_trunc_B[0]
        expected_input_trunc_B = [cls_id] + doc2_s1 + [sep_id] + doc3_s1[:5] + [sep_id]
        self.assertEqual(inputs_trunc_B.tolist(), expected_input_trunc_B)

    @unittest.skipIf(os.environ.get("CI_SMALL_DATASET_TEST") != "true", "Skipping full DataBuilder NSP test unless CI_SMALL_DATASET_TEST=true")
    def test_data_builder_nsp_mode(self):
        print("\nRunning test_data_builder_nsp_mode")
        # Using a very small public dataset for this test to avoid large downloads in CI
        # Example: 'hf-internal-testing/tiny-wikitext2' or create a local dummy dataset
        # For now, let's assume a fallback or ensure test runs where 'wikitext' is available and small.
        # The DataBuilder has a fallback mechanism if primary datasets fail.
        # To make test faster, use max_samples.
        db = create_data_builder(
            dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", # Smaller than C4
            seq_len=64,
            max_samples=50, # Very small number of documents to process
            nsp_task=True
        )

        self.assertEqual(db.vocab_size, 258) # 256 byte tokens + CLS + SEP
        self.assertEqual(db.cls_token_id, 256)
        self.assertEqual(db.sep_token_id, 257)

        dataloaders = db.create_dataloaders(batch_size=2)
        self.assertTrue('train' in dataloaders)

        if 'train' in dataloaders and len(dataloaders['train']) > 0 :
            batch_count = 0
            for input_ids, lm_targets, nsp_labels in dataloaders['train']:
                self.assertEqual(input_ids.shape[0], 2) # Batch size
                self.assertEqual(input_ids.shape[1], db.seq_len)
                self.assertEqual(lm_targets.shape[0], 2)
                self.assertEqual(lm_targets.shape[1], db.seq_len)
                self.assertEqual(nsp_labels.shape[0], 2)

                # Check for CLS and SEP tokens in decoded output
                sample_decoded = db.decode_tokens(input_ids[0])
                self.assertTrue("[CLS]" in sample_decoded)
                self.assertTrue("[SEP]" in sample_decoded)

                # Check that targets for CLS/SEP are padded (-1)
                # First token is CLS, its target should be -1
                self.assertEqual(lm_targets[0, 0].item(), -1)

                # Find first SEP and check its target
                sep_indices = (input_ids[0] == db.sep_token_id).nonzero(as_tuple=True)[0]
                if len(sep_indices) > 0:
                    first_sep_idx = sep_indices[0].item()
                    self.assertEqual(lm_targets[0, first_sep_idx].item(), -1)
                if len(sep_indices) > 1:
                    second_sep_idx = sep_indices[1].item()
                    self.assertEqual(lm_targets[0, second_sep_idx].item(), -1)

                batch_count += 1
                if batch_count >= 2: # Check a couple of batches
                    break
            self.assertTrue(batch_count > 0, "NSP Dataloader (train) did not yield any batches.")
        else:
            print("Warning: NSP Dataloader (train) is empty. Test might not be comprehensive.")


if __name__ == '__main__':
    unittest.main()
