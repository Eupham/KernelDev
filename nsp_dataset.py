import torch
from torch.utils.data import Dataset
import random

class NSPDataset(Dataset):
    def __init__(self, documents: list[list[list[int]]], seq_len: int,
                 cls_token_id: int, sep_token_id: int,
                 pad_token_id: int = -1, nsp_neg_prob: float = 0.5):
        self.documents = documents
        self.seq_len = seq_len
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id # For LM targets
        self.nsp_neg_prob = nsp_neg_prob

        self.all_sentences_tokenized = []
        for doc in self.documents:
            for sent_tokens in doc:
                if sent_tokens: # Ensure sentence is not empty
                    self.all_sentences_tokenized.append(sent_tokens)

        if not self.all_sentences_tokenized:
            print("Warning: NSPDataset initialized with no sentences to sample for negative pairs.")
            # This can happen if all input documents had <1 or <2 sentences after processing.
            # Consider raising an error or handling this more explicitly based on requirements.

        self.examples = self._create_examples()

    def _create_examples(self) -> list:
        examples = []
        for doc_idx, doc in enumerate(self.documents):
            if not doc or len(doc) == 0: # Skip empty documents
                continue

            for i in range(len(doc)):
                sent_a_tokens = doc[i]

                # Try to form a positive pair
                if (i + 1 < len(doc)) and (random.random() > self.nsp_neg_prob):
                    sent_b_tokens = doc[i+1]
                    nsp_label = 1 # IsNext
                # Form a negative pair
                else:
                    if not self.all_sentences_tokenized: # Should not happen if constructor checks
                         # If truly no other sentences, cannot form a negative pair. Skip this A.
                         # Or, as a fallback, make it a positive pair with itself if no other options?
                         # For now, if no negative candidates, this example might be skipped or lead to error.
                         # Let's assume all_sentences_tokenized has items due to constructor logic.
                         # If it's empty, the loop below for `random_sent_idx` will fail.
                         print("Warning: all_sentences_tokenized is empty during negative pair creation. This should be rare.")
                         continue # Skip creating this example if no negative candidates

                    # Ensure sent_b is not the actual next sentence or sent_a itself
                    while True:
                        random_sent_idx = random.randint(0, len(self.all_sentences_tokenized) - 1)
                        sent_b_tokens_candidate = self.all_sentences_tokenized[random_sent_idx]

                        is_sent_a = (sent_b_tokens_candidate == sent_a_tokens)
                        is_actual_next = False
                        if (i + 1 < len(doc)) and (sent_b_tokens_candidate == doc[i+1]):
                            is_actual_next = True

                        if not is_sent_a and not is_actual_next:
                            sent_b_tokens = sent_b_tokens_candidate
                            break
                        # If all_sentences_tokenized has only one unique sentence (sent_a), this loop will be infinite.
                        # Add a counter to break if too many tries, though this indicates a data problem.
                        if len(self.all_sentences_tokenized) == 1 and is_sent_a:
                            # Fallback: if only one sentence type exists, can't make a distinct negative.
                            # This is an edge case. For now, we'll just use it, label 0.
                            # Ideally, data diversity prevents this.
                            sent_b_tokens = sent_b_tokens_candidate # Use itself, but label 0.
                            break


                    nsp_label = 0 # NotNext

                examples.append((sent_a_tokens, sent_b_tokens, nsp_label))
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sent_a_tokens, sent_b_tokens, nsp_label = self.examples[idx]

        # [CLS] sent_a [SEP] sent_b [SEP]
        input_ids_list = [self.cls_token_id] + sent_a_tokens + [self.sep_token_id] + sent_b_tokens + [self.sep_token_id]

        # Store lengths before truncation for target masking
        len_cls = 1
        len_sent_a = len(sent_a_tokens)
        len_sep1 = 1
        len_sent_b = len(sent_b_tokens)
        # len_sep2 = 1 # The final SEP

        # Truncate if necessary
        # Max length for tokens themselves, excluding CLS and SEPs for this calculation initially
        max_token_content_len = self.seq_len - 3 # CLS, SEP, SEP

        if len(sent_a_tokens) + len(sent_b_tokens) > max_token_content_len:
            # Prioritize truncating sentence B, then sentence A
            if len(sent_b_tokens) > (max_token_content_len - len(sent_a_tokens)):
                sent_b_tokens = sent_b_tokens[:max(0, max_token_content_len - len(sent_a_tokens))] # Truncate B

            if len(sent_a_tokens) + len(sent_b_tokens) > max_token_content_len: # Recheck after B truncate
                sent_a_tokens = sent_a_tokens[:max(0, max_token_content_len - len(sent_b_tokens))] # Truncate A

        # Reconstruct input_ids with potentially truncated sentences
        input_ids_list = [self.cls_token_id] + sent_a_tokens + [self.sep_token_id] + sent_b_tokens + [self.sep_token_id]

        # Ensure final list doesn't exceed seq_len (e.g. if one sentence was very long)
        if len(input_ids_list) > self.seq_len:
            input_ids_list = input_ids_list[:self.seq_len]

        # Update actual lengths after truncation for masking targets
        # The CLS token is always present.
        # The first SEP is present if seq_len > 1.
        # The second SEP is present if it wasn't truncated.

        # Determine actual positions of special tokens after truncation
        pos_cls = 0 # Always at the beginning

        # Position of first SEP
        # sent_a_tokens could be empty if truncated heavily
        pos_sep1 = len_cls + len(sent_a_tokens)
        if pos_sep1 >= self.seq_len -1: # SEP1 is truncated out or is the last token
            pos_sep1 = -1 # Mark as not effectively present for masking lm_targets later
            pos_sep2 = -1

        # Position of second SEP
        if pos_sep1 != -1 :
            pos_sep2 = pos_sep1 + len_sep1 + len(sent_b_tokens)
            if pos_sep2 >= self.seq_len -1: # SEP2 is truncated out or is the last token
                 pos_sep2 = -1 # Mark as not effectively present
        else: # if SEP1 was truncated, SEP2 definitely is
            pos_sep2 = -1


        # Language Modeling Targets
        lm_target_ids_list = input_ids_list[1:] + [self.pad_token_id] # Shifted, add padding for last token's target

        # Mask targets for CLS, SEP, and padding
        # Target for CLS token is padded
        if pos_cls < len(lm_target_ids_list): # Should always be true
             lm_target_ids_list[pos_cls] = self.pad_token_id

        # Target for the token *after* sent_a (which is SEP1) should be padded
        if pos_sep1 != -1 and pos_sep1 < len(lm_target_ids_list):
            lm_target_ids_list[pos_sep1] = self.pad_token_id

        # Target for the token *after* sent_b (which is SEP2) should be padded
        if pos_sep2 != -1 and pos_sep2 < len(lm_target_ids_list):
            lm_target_ids_list[pos_sep2] = self.pad_token_id

        # Padding input_ids and lm_target_ids to seq_len
        padding_len = self.seq_len - len(input_ids_list)
        input_ids_padded = input_ids_list + [0] * padding_len # Pad with 0, ensure embedding layer handles padding_idx=0
        lm_target_ids_padded = lm_target_ids_list + [self.pad_token_id] * padding_len

        # Ensure lm_target_ids_padded is also of self.seq_len
        if len(lm_target_ids_padded) > self.seq_len:
             lm_target_ids_padded = lm_target_ids_padded[:self.seq_len]


        return torch.tensor(input_ids_padded, dtype=torch.long), \
               torch.tensor(lm_target_ids_padded, dtype=torch.long), \
               torch.tensor(nsp_label, dtype=torch.long)

if __name__ == '__main__':
    # Example Usage:
    doc1_sent1_tokens = [10, 11, 12]
    doc1_sent2_tokens = [13, 14]
    doc1_sent3_tokens = [15, 16, 17, 18]
    doc2_sent1_tokens = [20, 21]

    documents_tokenized = [
        [doc1_sent1_tokens, doc1_sent2_tokens, doc1_sent3_tokens], # Doc 1
        [doc2_sent1_tokens]  # Doc 2
    ]

    cls_id = 256
    sep_id = 257
    pad_id_for_lm = -1
    seq_length = 15 # Example sequence length

    nsp_dataset = NSPDataset(documents_tokenized, seq_length, cls_id, sep_id, pad_id_for_lm)
    print(f"Number of examples: {len(nsp_dataset)}")

    for i in range(len(nsp_dataset)):
        inputs, lm_targets, nsp_label = nsp_dataset[i]
        print(f"\nExample {i}:")
        print(f"  Input IDs: {inputs.tolist()}")
        print(f"  LM Targets: {lm_targets.tolist()}")
        print(f"  NSP Label: {nsp_label.item()}")

        # Sanity checks
        assert inputs.shape == (seq_length,), f"Input shape error: {inputs.shape}"
        assert lm_targets.shape == (seq_length,), f"Target shape error: {lm_targets.shape}"
        # Check that CLS and SEP targets are padded
        # This needs careful index checking based on actual sentence lengths in the example
        # For instance, input_ids[0] is CLS, so lm_targets[0] should be pad_id_for_lm
        if inputs[0] == cls_id : # If CLS is present
             assert lm_targets[0] == pad_id_for_lm, f"CLS target not padded: {lm_targets[0]}"

        # Find first SEP actual position
        sep1_actual_idx = -1
        for k_idx, token_val in enumerate(inputs.tolist()):
            if k_idx > 0 and token_val == sep_id: # k_idx > 0 to skip CLS if it's same ID as SEP (not here)
                sep1_actual_idx = k_idx
                break
        if sep1_actual_idx != -1:
             assert lm_targets[sep1_actual_idx] == pad_id_for_lm, f"SEP1 target not padded at index {sep1_actual_idx}: {lm_targets[sep1_actual_idx]}"

        # Find second SEP actual position
        sep2_actual_idx = -1
        if sep1_actual_idx != -1:
            for k_idx in range(sep1_actual_idx + 1, len(inputs.tolist())):
                if inputs.tolist()[k_idx] == sep_id:
                    sep2_actual_idx = k_idx
                    break
            if sep2_actual_idx != -1:
                 assert lm_targets[sep2_actual_idx] == pad_id_for_lm, f"SEP2 target not padded at index {sep2_actual_idx}: {lm_targets[sep2_actual_idx]}"


    print("\nNSPDataset test completed.")

    # Test with empty documents
    empty_docs_dataset = NSPDataset([], seq_length, cls_id, sep_id, pad_id_for_lm)
    print(f"Number of examples with empty docs: {len(empty_docs_dataset)}")
    assert len(empty_docs_dataset) == 0

    # Test with documents having few sentences
    short_docs_tokenized = [
        [[1,2,3]], # Doc 1, one sentence
        [[4,5], [6,7]] # Doc 2, two sentences
    ]
    short_docs_dataset = NSPDataset(short_docs_tokenized, seq_length, cls_id, sep_id, pad_id_for_lm)
    print(f"Number of examples with short docs: {len(short_docs_dataset)}")
    # Expected: Doc1Sent1 -> neg, Doc2Sent1 -> pos OR neg, Doc2Sent2 -> neg
    # So, 3 examples.
    assert len(short_docs_dataset) == 1 + 2 # (1 from doc1, 2 from doc2)
    print("NSPDataset short docs test completed.")

    # Test truncation logic
    long_sent_a = list(range(100, 110)) # len 10
    long_sent_b = list(range(200, 210)) # len 10
    # CLS + A(10) + SEP + B(10) + SEP = 1+10+1+10+1 = 23 tokens
    # If seq_len is 15:
    # Max content = 15 - 3 = 12
    # A(10) + B(10) = 20 > 12
    # Truncate B first: max_b_len = 12 - 10 = 2. So B becomes B[:2] = [200, 201]
    # A(10) + B(2) = 12. This fits.
    # Final: CLS, A(10), SEP, B(2), SEP = 1+10+1+2+1 = 15 tokens.

    # Test if seq_len is very small, e.g. 5
    # Max content = 5 - 3 = 2
    # A(10) + B(10) = 20 > 2
    # Truncate B: max_b_len = 2 - 10 = -8. So B becomes B[:0] = [] (empty)
    # A(10) + B(0) = 10 > 2
    # Truncate A: max_a_len = 2 - 0 = 2. So A becomes A[:2] = [100, 101]
    # Final: CLS, A(2), SEP, B(0), SEP = 1+2+1+0+1 = 5.

    test_trunc_dataset = NSPDataset(
        [[long_sent_a, long_sent_b]], # One doc, two long sentences for positive pair
        seq_len=15, cls_token_id=cls_id, sep_token_id=sep_id, pad_token_id=pad_id_for_lm,
        nsp_neg_prob=0.0 # Force positive pair
    )
    assert len(test_trunc_dataset) == 1
    inputs, lm_targets, nsp_label = test_trunc_dataset[0]
    assert inputs.tolist() == [cls_id] + long_sent_a + [sep_id] + long_sent_b[:2] + [sep_id]
    print("Truncation test (seq_len=15) passed.")

    test_trunc_dataset_short = NSPDataset(
        [[long_sent_a, long_sent_b]],
        seq_len=5, cls_token_id=cls_id, sep_token_id=sep_id, pad_token_id=pad_id_for_lm,
        nsp_neg_prob=0.0 # Force positive pair
    )
    assert len(test_trunc_dataset_short) == 1
    inputs_short, _, _ = test_trunc_dataset_short[0]
    assert inputs_short.tolist() == [cls_id] + long_sent_a[:2] + [sep_id] + [] + [sep_id]
    print("Truncation test (seq_len=5) passed.")

    # Test padding of input_ids (e.g. if total length < seq_len)
    # CLS, A(3), SEP, B(2), SEP = 1+3+1+2+1 = 8. If seq_len = 10, 2 padding tokens (0)
    short_a = [1,2,3]
    short_b = [4,5]
    test_pad_dataset = NSPDataset(
        [[short_a, short_b]],
        seq_len=10, cls_token_id=cls_id, sep_token_id=sep_id, pad_token_id=pad_id_for_lm,
        nsp_neg_prob=0.0 # Force positive
    )
    inputs_pad, targets_pad, _ = test_pad_dataset[0]
    expected_inputs_pad = [cls_id] + short_a + [sep_id] + short_b + [sep_id] + [0]*(10-8)
    # Expected targets: [1,2,3,pad, 4,5,pad, pad, pad, pad] (pad = -1)
    # CLS target -> pad_id_for_lm
    # A targets -> 1,2,3
    # SEP1 target -> pad_id_for_lm
    # B targets -> 4,5
    # SEP2 target -> pad_id_for_lm
    # Padding targets -> pad_id_for_lm
    # Shifted: [1,2,3,sep,4,5,sep,0,0,pad]
    # Targets: [2,3,sep(->pad),4,5,sep(->pad),0(->pad),0(->pad),pad(->pad),pad(->pad)]
    # Corrected logic for targets from input:
    # input_ids_list = [cls, 1,2,3, sep, 4,5, sep] (len 8)
    # lm_target_ids_list = [1,2,3, sep, 4,5, sep, pad] (len 8)
    # Masking:
    # lm_target_ids_list[0 (for CLS)] = pad (-1) -> [-1, 2,3, sep, 4,5, sep, pad]
    # pos_sep1 = 1 + 3 = 4. lm_target_ids_list[4 (for SEP1)] = pad (-1) -> [-1, 2,3, sep, -1, 5, sep, pad]
    # pos_sep2 = 4+1+2 = 7. lm_target_ids_list[7 (for SEP2)] = pad (-1) -> [-1, 2,3, sep, -1, 5, sep, -1]
    # Padded targets: [-1, 2,3, sep_id, -1, 5, sep_id, -1, -1, -1]
    # Wait, tokens themselves are targets, not their values.
    # Original: input_ids_padded = [256, 1, 2, 3, 257, 4, 5, 257, 0, 0]
    # Expected targets: [1, 2, 3, 257, 4, 5, 257, 0, 0, -1] (shifted input, last one is pad)
    # Masked targets:
    # lm_targets[0] (for CLS) = -1
    # lm_targets[4] (for first SEP) = -1
    # lm_targets[7] (for second SEP) = -1
    # lm_targets[8] (for first 0 pad) = -1
    # lm_targets[9] (for second 0 pad) = -1
    # So: [-1, 2, 3, 257, -1, 5, 257, -1, -1, -1] -> This is wrong, SEP tokens are valid targets if not masked.
    # The rule is: "Positions in lm_target_ids corresponding to the original positions of [CLS] token and the first [SEP] token in input_ids should get self.pad_token_id (-1).
    # The target for the final [SEP] (if not truncated) and any padding tokens in input_ids should also be self.pad_token_id."

    # Let's re-verify masking:
    # input_ids_padded = [256, 1, 2, 3, 257, 4, 5, 257, 0, 0]
    # lm_target_ids_list (before padding, after shift): [1, 2, 3, 257, 4, 5, 257, -1]
    # pos_cls = 0. lm_target_ids_list[0] = -1. -> [-1, 2, 3, 257, 4, 5, 257, -1]
    # pos_sep1 = 1 + len(short_a) = 1+3 = 4. lm_target_ids_list[4] = -1. -> [-1, 2, 3, 257, -1, 5, 257, -1]
    # pos_sep2 = pos_sep1 + 1 + len(short_b) = 4+1+2 = 7. lm_target_ids_list[7] = -1. -> [-1, 2, 3, 257, -1, 5, 257, -1]
    # lm_target_ids_padded = [-1, 2, 3, 257, -1, 5, 257, -1, -1, -1] -- This looks correct now.

    assert inputs_pad.tolist() == expected_inputs_pad
    expected_targets_pad = [-1, 2, 3, 257, -1, 5, 257, -1, -1, -1]
    assert targets_pad.tolist() == expected_targets_pad
    print("Padding test (seq_len=10) passed.")
