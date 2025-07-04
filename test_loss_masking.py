#!/usr/bin/env python3
"""Test script to verify loss masking works correctly with deterministic batch composition."""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from combined_dataset import CombinedMultiTaskDataset

def test_loss_masking():
    """Test that loss masking logic works correctly."""
    
    # Create test documents 
    raw_docs = [
        'This is the first sentence. This is the second sentence. This is the third sentence.',
        'Another document starts here. It has multiple sentences too. Testing NSP creation.',
        'Third document for testing. It also has several sentences. This completes the test.'
    ]
    
    def simple_tokenizer(text):
        return list(range(len(text.split())))
    
    print("Testing loss masking with deterministic batch composition...")
    
    dataset = CombinedMultiTaskDataset(
        raw_documents=raw_docs,
        tokenizer_fn=simple_tokenizer,
        seq_len=30,
        cls_token_id=256,
        sep_token_id=257,
        task_distribution=(0.25, 0.25, 0.5)
    )
    
    # Create a batch
    batch_size = 8
    batch_data = []
    for i in range(batch_size):
        batch_data.append(dataset[i])
    
    # Stack batch data
    input_tokens = torch.stack([item[0] for item in batch_data])
    lm_targets = torch.stack([item[1] for item in batch_data]) 
    auxiliary_values = torch.stack([item[2] for item in batch_data])
    task_type_flags = torch.stack([item[3] for item in batch_data])
    
    print(f"Batch shape: {input_tokens.shape}")
    print(f"Task type flags: {task_type_flags}")
    
    # Test the masking logic from training loop
    lm_task_mask = (task_type_flags == 0.0)  # LM task
    lev_task_mask = (task_type_flags == 1.0)  # Levenshtein task
    nsp_task_mask = (task_type_flags == 2.0)  # NSP task
    
    print(f"\nTask masks:")
    print(f"LM task mask: {lm_task_mask}")
    print(f"Levenshtein task mask: {lev_task_mask}")
    print(f"NSP task mask: {nsp_task_mask}")
    
    # Test LM loss masking (should include LM and Levenshtein tasks)
    lm_valid_mask = lm_task_mask | lev_task_mask
    print(f"\nLM valid mask (LM + Lev): {lm_valid_mask}")
    print(f"LM valid mask count: {lm_valid_mask.sum().item()}")
    
    # Test individual task masks
    print(f"\nIndividual task counts:")
    print(f"LM tasks: {lm_task_mask.sum().item()}")
    print(f"Levenshtein tasks: {lev_task_mask.sum().item()}")
    print(f"NSP tasks: {nsp_task_mask.sum().item()}")
    
    # Expected counts
    expected_lm = 4
    expected_lev = 2
    expected_nsp = 2
    
    actual_lm = lm_task_mask.sum().item()
    actual_lev = lev_task_mask.sum().item()
    actual_nsp = nsp_task_mask.sum().item()
    
    print(f"\nExpected vs Actual:")
    print(f"LM: {expected_lm} vs {actual_lm}")
    print(f"Levenshtein: {expected_lev} vs {actual_lev}")
    print(f"NSP: {expected_nsp} vs {actual_nsp}")
    
    # Test auxiliary values for different task types
    print(f"\nAuxiliary values by task type:")
    if lm_task_mask.any():
        lm_aux_values = auxiliary_values[lm_task_mask]
        print(f"LM auxiliary values: {lm_aux_values} (should be 0.0)")
    
    if lev_task_mask.any():
        lev_aux_values = auxiliary_values[lev_task_mask]
        print(f"Levenshtein auxiliary values: {lev_aux_values} (should be distance values)")
    
    if nsp_task_mask.any():
        nsp_aux_values = auxiliary_values[nsp_task_mask]
        print(f"NSP auxiliary values: {nsp_aux_values} (should be class labels 0-2)")
    
    # Verify correctness
    success = (actual_lm == expected_lm and 
               actual_lev == expected_lev and 
               actual_nsp == expected_nsp)
    
    print(f"\nLoss masking test: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == "__main__":
    print("Testing loss masking with deterministic batch composition...")
    
    success = test_loss_masking()
    
    print("\n" + "="*50)
    if success:
        print("LOSS MASKING TEST PASSED!")
        sys.exit(0)
    else:
        print("LOSS MASKING TEST FAILED!")
        sys.exit(1)