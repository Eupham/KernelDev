#!/usr/bin/env python3
"""Comprehensive test for multi-task training batch composition fix."""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from combined_dataset import CombinedMultiTaskDataset

def test_comprehensive():
    """Run comprehensive tests to verify the fix."""
    
    print("="*60)
    print("COMPREHENSIVE MULTI-TASK TRAINING BATCH COMPOSITION TEST")
    print("="*60)
    
    # Test data with multiple sentences for NSP pairs
    raw_docs = [
        'First document sentence one. First document sentence two. First document sentence three.',
        'Second document sentence one. Second document sentence two. Second document sentence three.',
        'Third document sentence one. Third document sentence two. Third document sentence three.',
        'Fourth document sentence one. Fourth document sentence two. Fourth document sentence three.',
        'Fifth document sentence one. Fifth document sentence two. Fifth document sentence three.'
    ]
    
    def tokenizer(text):
        return list(range(len(text.split())))
    
    # Test 1: Default distribution (0.25, 0.25, 0.5)
    print("\n1. Testing default distribution (0.25, 0.25, 0.5)...")
    dataset = CombinedMultiTaskDataset(
        raw_documents=raw_docs,
        tokenizer_fn=tokenizer,
        seq_len=40,
        cls_token_id=256,
        sep_token_id=257,
        task_distribution=(0.25, 0.25, 0.5)
    )
    
    # Test multiple batches
    batch_size = 8
    all_tests_passed = True
    
    for batch_num in range(3):
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        
        task_counts = {0.0: 0, 1.0: 0, 2.0: 0}
        for i in range(start_idx, end_idx):
            _, _, _, task_type = dataset[i]
            task_counts[task_type.item()] += 1
        
        expected_lm, expected_lev, expected_nsp = 4, 2, 2
        actual_lm, actual_lev, actual_nsp = task_counts[0.0], task_counts[1.0], task_counts[2.0]
        
        batch_passed = (actual_lm == expected_lm and 
                       actual_lev == expected_lev and 
                       actual_nsp == expected_nsp)
        
        print(f"  Batch {batch_num}: LM={actual_lm}, Lev={actual_lev}, NSP={actual_nsp} - {'PASS' if batch_passed else 'FAIL'}")
        
        if not batch_passed:
            all_tests_passed = False
    
    # Test 2: Custom distribution (0.125, 0.125, 0.75)
    print("\n2. Testing custom distribution (0.125, 0.125, 0.75)...")
    dataset2 = CombinedMultiTaskDataset(
        raw_documents=raw_docs,
        tokenizer_fn=tokenizer,
        seq_len=40,
        cls_token_id=256,
        sep_token_id=257,
        task_distribution=(0.125, 0.125, 0.75)
    )
    
    task_counts = {0.0: 0, 1.0: 0, 2.0: 0}
    for i in range(batch_size):
        _, _, _, task_type = dataset2[i]
        task_counts[task_type.item()] += 1
    
    expected_lm, expected_lev, expected_nsp = 6, 1, 1
    actual_lm, actual_lev, actual_nsp = task_counts[0.0], task_counts[1.0], task_counts[2.0]
    
    custom_passed = (actual_lm == expected_lm and 
                    actual_lev == expected_lev and 
                    actual_nsp == expected_nsp)
    
    print(f"  Custom distribution: LM={actual_lm}, Lev={actual_lev}, NSP={actual_nsp} - {'PASS' if custom_passed else 'FAIL'}")
    
    if not custom_passed:
        all_tests_passed = False
    
    # Test 3: Verify deterministic pattern
    print("\n3. Testing deterministic pattern consistency...")
    pattern_test_passed = True
    
    # Check that the pattern repeats every 8 samples
    for cycle_start in [0, 8, 16, 24]:
        cycle_tasks = []
        for i in range(cycle_start, cycle_start + 8):
            _, _, _, task_type = dataset[i]
            cycle_tasks.append(task_type.item())
        
        expected_pattern = [2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        pattern_matches = cycle_tasks == expected_pattern
        
        print(f"  Cycle {cycle_start//8}: {cycle_tasks} - {'PASS' if pattern_matches else 'FAIL'}")
        
        if not pattern_matches:
            pattern_test_passed = False
    
    if not pattern_test_passed:
        all_tests_passed = False
    
    # Test 4: Edge cases
    print("\n4. Testing edge cases...")
    
    # Test with empty NSP dataset (should fallback to LM)
    edge_docs = ['Single sentence without period']
    dataset_edge = CombinedMultiTaskDataset(
        raw_documents=edge_docs,
        tokenizer_fn=tokenizer,
        seq_len=20,
        cls_token_id=256,
        sep_token_id=257,
        task_distribution=(0.25, 0.25, 0.5)
    )
    
    # NSP positions should fallback to LM
    nsp_fallback_test = True
    for i in [0, 1]:  # NSP positions
        _, _, _, task_type = dataset_edge[i]
        if task_type.item() != 0.0:  # Should be LM fallback
            nsp_fallback_test = False
            break
    
    print(f"  NSP fallback test: {'PASS' if nsp_fallback_test else 'FAIL'}")
    
    if not nsp_fallback_test:
        all_tests_passed = False
    
    # Test 5: Verify auxiliary values are appropriate
    print("\n5. Testing auxiliary values...")
    aux_test_passed = True
    
    for i in range(8):
        _, _, aux_val, task_type = dataset[i]
        
        if task_type.item() == 0.0:  # LM task
            if aux_val.item() != 0.0:
                aux_test_passed = False
                break
        elif task_type.item() == 1.0:  # Levenshtein task
            if not (0.0 <= aux_val.item() <= 1.0):
                aux_test_passed = False
                break
        elif task_type.item() == 2.0:  # NSP task
            if aux_val.item() not in [0.0, 1.0, 2.0]:
                aux_test_passed = False
                break
    
    print(f"  Auxiliary values test: {'PASS' if aux_test_passed else 'FAIL'}")
    
    if not aux_test_passed:
        all_tests_passed = False
    
    # Final result
    print("\n" + "="*60)
    print("FINAL RESULT:")
    print("="*60)
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\nThe multi-task training batch composition fix is working correctly:")
        print("✅ Deterministic batch composition instead of random selection")
        print("✅ Proper task distribution (1/4 NSP, 1/4 Levenshtein, 1/2 LM)")
        print("✅ Consistent pattern across multiple batches")
        print("✅ Custom distributions work correctly")
        print("✅ Edge cases handled properly")
        print("✅ Auxiliary values are appropriate for each task type")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        print("\nPlease check the implementation.")
        return False

if __name__ == "__main__":
    success = test_comprehensive()
    sys.exit(0 if success else 1)