#!/usr/bin/env python3
"""
Test script to reproduce the 'object of type generator has no len()' error
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_builder import DataBuilder

def test_reproduce_issue():
    """Test to reproduce the issue from the GitHub issue"""
    print("=== Testing DataBuilder with allenai/c4/en dataset ===")
    
    # Create DataBuilder with the same configuration as the error
    data_builder = DataBuilder(
        dataset_name="allenai/c4",
        dataset_config="en",
        seq_len=512,
        max_samples=1000000,
        vocab_size=256,
        max_eval_tokens=50000,
        use_levenshtein_task=False
    )
    
    # Try to create dataloaders, which should trigger the error
    try:
        dataloaders = data_builder.create_dataloaders(batch_size=8)
        print("✅ SUCCESS: Dataloaders created without errors!")
        
        # Check if train dataloader exists
        if 'train' in dataloaders:
            print(f"✅ Train dataloader exists")
            try:
                print(f"✅ Train dataloader has {len(dataloaders['train'])} batches")
            except Exception as e:
                print(f"⚠️ Cannot get length of train dataloader: {e}")
        else:
            print("❌ No train dataloader found")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_reproduce_issue()
    sys.exit(0 if success else 1)