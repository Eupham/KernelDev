#!/usr/bin/env python3
"""
Test script to verify the fix for generator len() issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_builder import DataBuilder

class MockStreamingDataset:
    """
    Mock streaming dataset that simulates HuggingFace streaming datasets
    """
    def __init__(self, data):
        self.data = data
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        # This is the error that was occurring with streaming datasets
        raise TypeError("object of type 'generator' has no len()")

def test_generator_len_fix():
    """Test that our fix handles generator objects without len() errors"""
    print("=== Testing Generator/Streaming Dataset Fix ===")
    
    # Test the _safe_convert_to_list method directly
    data_builder = DataBuilder(max_samples=5)
    
    # Test data
    test_data = [
        {"text": "First sample text"},
        {"text": "Second sample text"}, 
        {"text": "Third sample text"},
        {"text": "Fourth sample text"},
        {"text": "Fifth sample text"}
    ]
    
    # Test with regular list (should work fine)
    print("Testing with regular list...")
    result = data_builder._safe_convert_to_list(test_data, "test_list")
    print(f"✅ Regular list: {len(result)} items")
    
    # Test with mock streaming dataset (the problematic case)
    print("Testing with mock streaming dataset...")
    mock_streaming = MockStreamingDataset(test_data)
    
    # Verify it fails with len()
    try:
        len(mock_streaming)
        print("❌ UNEXPECTED: Mock streaming dataset should fail len()")
        return False
    except TypeError as e:
        print(f"✅ EXPECTED: Mock streaming dataset fails len(): {e}")
    
    # Test our fix
    try:
        result = data_builder._safe_convert_to_list(mock_streaming, "mock_streaming")
        print(f"✅ SUCCESS: Converted mock streaming dataset to list with {len(result)} items")
        
        # Verify the content
        if len(result) == len(test_data):
            print("✅ SUCCESS: Correct number of items converted")
            return True
        else:
            print(f"❌ FAILURE: Expected {len(test_data)} items, got {len(result)}")
            return False
            
    except Exception as e:
        print(f"❌ FAILURE: Our fix failed: {e}")
        return False

def test_process_iterable_dataset():
    """Test that _process_iterable_dataset works with generators"""
    print("\n=== Testing _process_iterable_dataset with generators ===")
    
    data_builder = DataBuilder(max_samples=3)
    
    test_data = [
        {"text": "Sample one for processing"},
        {"text": "Sample two for processing"},
        {"text": "Sample three for processing"},
        {"text": "Sample four for processing"}
    ]
    
    # Test with mock streaming dataset
    mock_streaming = MockStreamingDataset(test_data)
    
    try:
        result = data_builder._process_iterable_dataset(mock_streaming, "test_streaming")
        print(f"✅ SUCCESS: Processed {len(result)} samples from streaming dataset")
        
        # Should be limited by max_samples
        expected_count = min(3, len(test_data))
        if len(result) == expected_count:
            print(f"✅ SUCCESS: Correct sample count (limited by max_samples)")
            return True
        else:
            print(f"❌ FAILURE: Expected {expected_count} samples, got {len(result)}")
            return False
    except Exception as e:
        print(f"❌ FAILURE: Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing Generator/Streaming Dataset Fix ===")
    
    success1 = test_generator_len_fix()
    success2 = test_process_iterable_dataset()
    
    overall_success = success1 and success2
    
    print(f"\n=== Test Results ===")
    print(f"Generator len() fix: {'PASSED ✅' if success1 else 'FAILED ❌'}")
    print(f"Process iterable fix: {'PASSED ✅' if success2 else 'FAILED ❌'}")
    print(f"Overall: {'PASSED ✅' if overall_success else 'FAILED ❌'}")
    
    if overall_success:
        print("\n🎉 All tests passed! The fix successfully handles generator objects.")
    else:
        print("\n❌ Some tests failed. The fix needs more work.")
    
    sys.exit(0 if overall_success else 1)