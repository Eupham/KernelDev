#!/usr/bin/env python3
"""
Test script to demonstrate the fix for the generator len() issue.
This test verifies that streaming datasets can be processed without
encountering the "object of type 'generator' has no len()" error.
"""

import sys
sys.path.append('/home/runner/work/KernelDev/KernelDev')

from data_builder import DataBuilder

class MockStreamingDataset:
    """
    A mock streaming dataset that raises TypeError when len() is called,
    simulating the behavior of HuggingFace streaming datasets.
    """
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __iter__(self):
        for item in self.data_list:
            yield item
    
    def __len__(self):
        # This is the error that was occurring with streaming datasets
        raise TypeError("object of type 'generator' has no len()")

def test_streaming_dataset_fix():
    """Test that streaming datasets can be processed without len() errors."""
    print("Testing streaming dataset compatibility...")
    
    # Create mock streaming data
    streaming_data = [
        {"text": "First sample text for streaming test."},
        {"text": "Second sample text for streaming test."},
        {"text": "Third sample text for streaming test."},
        {"text": "Fourth sample text for streaming test."},
        {"text": "Fifth sample text for streaming test."},
    ]
    
    # Create mock streaming dataset
    streaming_dataset = MockStreamingDataset(streaming_data)
    
    # Verify that len() call fails (as expected)
    try:
        len(streaming_dataset)
        print("❌ UNEXPECTED: len() should fail on streaming dataset")
        return False
    except TypeError as e:
        print(f"✅ EXPECTED: len() fails on streaming dataset: {e}")
    
    # Test that our fix allows processing of streaming datasets
    try:
        data_builder = DataBuilder(max_samples=3)
        
        # This call should succeed with our fix
        processed_samples = data_builder._process_iterable_dataset(
            streaming_dataset, 
            "Mock Streaming Dataset"
        )
        
        print(f"✅ SUCCESS: Processed {len(processed_samples)} samples from streaming dataset")
        
        # Verify correct number of samples
        expected_count = min(3, len(streaming_data))  # Limited by max_samples
        if len(processed_samples) == expected_count:
            print(f"✅ SUCCESS: Correct sample count ({expected_count})")
            return True
        else:
            print(f"❌ FAILURE: Expected {expected_count} samples, got {len(processed_samples)}")
            return False
            
    except Exception as e:
        print(f"❌ FAILURE: Processing streaming dataset failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Streaming Dataset Fix Test ===")
    success = test_streaming_dataset_fix()
    print(f"\nTest result: {'PASSED ✅' if success else 'FAILED ❌'}")
    
    if success:
        print("\n🎉 The fix successfully resolves the 'object of type generator has no len()' error!")
        print("Streaming datasets can now be processed without issues.")
    else:
        print("\n❌ The fix did not resolve the issue.")
    
    sys.exit(0 if success else 1)