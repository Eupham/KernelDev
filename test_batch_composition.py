#!/usr/bin/env python3
"""Test script to verify batch composition in multi-task training."""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from combined_dataset import CombinedMultiTaskDataset

def test_batch_composition():
    """Test that batches have the correct task distribution."""
    
    # Create test documents with multiple sentences for NSP
    raw_docs = [
        'This is the first sentence. This is the second sentence. This is the third sentence.',
        'Another document starts here. It has multiple sentences too. Testing NSP creation.',
        'Third document for testing. It also has several sentences. This completes the test.',
        'Fourth document with content. Multiple sentences are needed. NSP requires sentence pairs.',
        'Fifth document for variety. More sentences help testing. Final sentence here.'
    ]
    
    def simple_tokenizer(text):
        """Simple tokenizer that converts words to indices."""
        return list(range(len(text.split())))
    
    print("Testing batch composition with default distribution (0.25, 0.25, 0.5)...")
    
    # Test with default distribution
    dataset = CombinedMultiTaskDataset(
        raw_documents=raw_docs,
        tokenizer_fn=simple_tokenizer,
        seq_len=50,
        cls_token_id=256,
        sep_token_id=257,
        task_distribution=(0.25, 0.25, 0.5)
    )
    
    print(f"Dataset info:")
    print(f"  Total length: {len(dataset)}")
    print(f"  NSP pairs: {len(dataset.nsp_dataset)}")
    print(f"  Levenshtein samples: {len(dataset.levenshtein_dataset)}")
    print(f"  Raw documents: {len(dataset.raw_documents)}")
    
    # Test first batch
    batch_size = 8
    task_counts = {0.0: 0, 1.0: 0, 2.0: 0}
    
    print(f"\nTesting first batch (indices 0-{batch_size-1}):")
    for i in range(batch_size):
        input_tokens, lm_targets, auxiliary_value, task_type = dataset[i]
        task_counts[task_type.item()] += 1
        print(f"  Index {i}: Task {task_type.item()} ({'NSP' if task_type.item() == 2.0 else 'Lev' if task_type.item() == 1.0 else 'LM'})")
    
    print(f"\nBatch composition:")
    print(f"  LM tasks: {task_counts[0.0]} (expected: 4)")
    print(f"  Levenshtein tasks: {task_counts[1.0]} (expected: 2)")
    print(f"  NSP tasks: {task_counts[2.0]} (expected: 2)")
    
    # Verify the distribution
    expected_lm, expected_lev, expected_nsp = 4, 2, 2
    actual_lm, actual_lev, actual_nsp = task_counts[0.0], task_counts[1.0], task_counts[2.0]
    
    success = (actual_lm == expected_lm and 
               actual_lev == expected_lev and 
               actual_nsp == expected_nsp)
    
    print(f"\nFirst batch test: {'PASS' if success else 'FAIL'}")
    
    # Test second batch to ensure consistency
    task_counts_2 = {0.0: 0, 1.0: 0, 2.0: 0}
    
    print(f"\nTesting second batch (indices {batch_size}-{2*batch_size-1}):")
    for i in range(batch_size, 2*batch_size):
        input_tokens, lm_targets, auxiliary_value, task_type = dataset[i]
        task_counts_2[task_type.item()] += 1
        print(f"  Index {i}: Task {task_type.item()} ({'NSP' if task_type.item() == 2.0 else 'Lev' if task_type.item() == 1.0 else 'LM'})")
    
    print(f"\nSecond batch composition:")
    print(f"  LM tasks: {task_counts_2[0.0]} (expected: 4)")
    print(f"  Levenshtein tasks: {task_counts_2[1.0]} (expected: 2)")
    print(f"  NSP tasks: {task_counts_2[2.0]} (expected: 2)")
    
    # Verify second batch
    actual_lm_2, actual_lev_2, actual_nsp_2 = task_counts_2[0.0], task_counts_2[1.0], task_counts_2[2.0]
    
    success_2 = (actual_lm_2 == expected_lm and 
                 actual_lev_2 == expected_lev and 
                 actual_nsp_2 == expected_nsp)
    
    print(f"\nSecond batch test: {'PASS' if success_2 else 'FAIL'}")
    
    return success and success_2

def test_different_distributions():
    """Test with different task distributions."""
    
    # Simple test data
    raw_docs = [
        'First sentence. Second sentence. Third sentence.',
        'Another document. More sentences here. Final sentence.',
        'Third document. Additional content. More text here.'
    ]
    
    def simple_tokenizer(text):
        return list(range(len(text.split())))
    
    print("\n" + "="*50)
    print("Testing different task distributions...")
    
    # Test with (0.125, 0.125, 0.75) distribution
    dataset = CombinedMultiTaskDataset(
        raw_documents=raw_docs,
        tokenizer_fn=simple_tokenizer,
        seq_len=30,
        cls_token_id=256,
        sep_token_id=257,
        task_distribution=(0.125, 0.125, 0.75)
    )
    
    batch_size = 8
    task_counts = {0.0: 0, 1.0: 0, 2.0: 0}
    
    print(f"\nTesting distribution (0.125, 0.125, 0.75):")
    for i in range(batch_size):
        input_tokens, lm_targets, auxiliary_value, task_type = dataset[i]
        task_counts[task_type.item()] += 1
        print(f"  Index {i}: Task {task_type.item()}")
    
    print(f"\nBatch composition:")
    print(f"  LM tasks: {task_counts[0.0]} (expected: 6)")
    print(f"  Levenshtein tasks: {task_counts[1.0]} (expected: 1)")
    print(f"  NSP tasks: {task_counts[2.0]} (expected: 1)")
    
    expected_lm, expected_lev, expected_nsp = 6, 1, 1
    actual_lm, actual_lev, actual_nsp = task_counts[0.0], task_counts[1.0], task_counts[2.0]
    
    success = (actual_lm == expected_lm and 
               actual_lev == expected_lev and 
               actual_nsp == expected_nsp)
    
    print(f"\nDifferent distribution test: {'PASS' if success else 'FAIL'}")
    
    return success

if __name__ == "__main__":
    print("Testing batch composition in multi-task training...")
    
    # Run tests
    test1_passed = test_batch_composition()
    test2_passed = test_different_distributions()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Default distribution test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Different distribution test: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\nALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED!")
        sys.exit(1)