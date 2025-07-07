#!/usr/bin/env python3
"""
Test script to verify the full pipeline works with fallback datasets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_builder import DataBuilder

def test_full_pipeline_with_fallback():
    """Test the full pipeline when dataset loading fails and fallback is used"""
    print("=== Testing Full Pipeline with Fallback Dataset ===")
    
    try:
        # Create DataBuilder with a non-existent dataset to trigger fallback
        data_builder = DataBuilder(
            dataset_name="non_existent_dataset",
            dataset_config="non_existent_config",
            seq_len=64,
            max_samples=10,
            vocab_size=256,
            max_eval_tokens=1000,
            use_levenshtein_task=False
        )
        
        print("✅ DataBuilder created successfully")
        
        # Test creating dataloaders (should use fallback dataset)
        dataloaders = data_builder.create_dataloaders(batch_size=2)
        print(f"✅ Dataloaders created: {list(dataloaders.keys())}")
        
        # Test that we can iterate through the dataloaders
        for split_name, dataloader in dataloaders.items():
            print(f"Testing {split_name} dataloader...")
            try:
                batch_count = 0
                for batch_idx, (x, y) in enumerate(dataloader):
                    print(f"  Batch {batch_idx}: Input shape: {x.shape}, Target shape: {y.shape}")
                    batch_count += 1
                    if batch_count >= 2:  # Just test first 2 batches
                        break
                print(f"✅ {split_name} dataloader works: {batch_count} batches tested")
            except Exception as e:
                print(f"❌ Error with {split_name} dataloader: {e}")
                return False
        
        print("✅ SUCCESS: Full pipeline works with fallback dataset")
        return True
        
    except Exception as e:
        print(f"❌ FAILURE: Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_levenshtein_pipeline_with_fallback():
    """Test the full pipeline with Levenshtein task when dataset loading fails"""
    print("\n=== Testing Levenshtein Pipeline with Fallback Dataset ===")
    
    try:
        # Create DataBuilder with Levenshtein task enabled
        data_builder = DataBuilder(
            dataset_name="non_existent_dataset",
            dataset_config="non_existent_config",
            seq_len=64,
            max_samples=10,
            vocab_size=256,
            max_eval_tokens=1000,
            use_levenshtein_task=True
        )
        
        print("✅ DataBuilder with Levenshtein task created successfully")
        
        # Test creating dataloaders (should use fallback dataset)
        dataloaders = data_builder.create_dataloaders(batch_size=2)
        print(f"✅ Dataloaders created: {list(dataloaders.keys())}")
        
        # Test that we can iterate through the dataloaders
        for split_name, dataloader in dataloaders.items():
            print(f"Testing {split_name} dataloader...")
            try:
                batch_count = 0
                for batch_idx, batch_data in enumerate(dataloader):
                    # Levenshtein dataset returns 4 items per batch
                    if len(batch_data) == 4:
                        input_toks, lm_tgts, lev_dist, is_shuf = batch_data
                        print(f"  Batch {batch_idx}: Input: {input_toks.shape}, LM: {lm_tgts.shape}, Lev: {lev_dist.shape}, Shuffle: {is_shuf.shape}")
                    else:
                        print(f"  Batch {batch_idx}: Unexpected batch format with {len(batch_data)} elements")
                    batch_count += 1
                    if batch_count >= 2:  # Just test first 2 batches
                        break
                print(f"✅ {split_name} dataloader works: {batch_count} batches tested")
            except Exception as e:
                print(f"❌ Error with {split_name} dataloader: {e}")
                return False
        
        print("✅ SUCCESS: Levenshtein pipeline works with fallback dataset")
        return True
        
    except Exception as e:
        print(f"❌ FAILURE: Levenshtein pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing Full Pipeline with Fallback Datasets ===")
    
    success1 = test_full_pipeline_with_fallback()
    success2 = test_levenshtein_pipeline_with_fallback()
    
    overall_success = success1 and success2
    
    print(f"\n=== Test Results ===")
    print(f"Standard pipeline: {'PASSED ✅' if success1 else 'FAILED ❌'}")
    print(f"Levenshtein pipeline: {'PASSED ✅' if success2 else 'FAILED ❌'}")
    print(f"Overall: {'PASSED ✅' if overall_success else 'FAILED ❌'}")
    
    if overall_success:
        print("\n🎉 All tests passed! The full pipeline works correctly with fallback datasets.")
    else:
        print("\n❌ Some tests failed. The pipeline needs more work.")
    
    sys.exit(0 if overall_success else 1)