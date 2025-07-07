#!/usr/bin/env python3
"""
Debug script to understand the streaming dataset issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# First test: Try to load a streaming dataset directly
print("=== Testing direct streaming dataset loading ===")

try:
    import datasets
    print(f"✅ datasets library loaded successfully")
    
    # Try to load a streaming dataset
    print("Loading streaming dataset...")
    streaming_dataset = datasets.load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    print(f"✅ Streaming dataset loaded: {type(streaming_dataset)}")
    
    # Try to call len() on it
    try:
        dataset_len = len(streaming_dataset)
        print(f"✅ len() works: {dataset_len}")
    except Exception as e:
        print(f"❌ len() fails: {e}")
        print("This is expected for streaming datasets")
    
    # Try to iterate over it
    print("Testing iteration...")
    count = 0
    for item in streaming_dataset:
        count += 1
        print(f"Sample {count}: {list(item.keys())}")
        if count >= 3:
            break
    
    print(f"✅ Successfully iterated over {count} samples")
    
except Exception as e:
    print(f"❌ Error with streaming dataset: {e}")

# Second test: Try with non-streaming dataset
print("\n=== Testing non-streaming dataset loading ===")

try:
    regular_dataset = datasets.load_dataset(
        "allenai/c4",
        "en",
        split="train[:1000]",
        streaming=False,
        trust_remote_code=True
    )
    
    print(f"✅ Regular dataset loaded: {type(regular_dataset)}")
    
    # Try to call len() on it
    try:
        dataset_len = len(regular_dataset)
        print(f"✅ len() works: {dataset_len}")
    except Exception as e:
        print(f"❌ len() fails: {e}")
    
except Exception as e:
    print(f"❌ Error with regular dataset: {e}")

# Third test: Use the DataBuilder
print("\n=== Testing DataBuilder ===")

try:
    from data_builder import DataBuilder
    
    print("Creating DataBuilder...")
    data_builder = DataBuilder(
        dataset_name="allenai/c4",
        dataset_config="en",
        seq_len=512,
        max_samples=1000,
        vocab_size=256,
        max_eval_tokens=50000,
        use_levenshtein_task=False
    )
    
    print("✅ DataBuilder created successfully")
    
    # Try to load raw dataset
    print("Loading raw dataset...")
    raw_dataset = data_builder.load_raw_dataset()
    print(f"✅ Raw dataset loaded: {type(raw_dataset)}")
    for split, data in raw_dataset.items():
        print(f"  {split}: {len(data)} samples")
    
    # Try to create dataloaders
    print("Creating dataloaders...")
    dataloaders = data_builder.create_dataloaders(batch_size=2)
    print(f"✅ Dataloaders created: {list(dataloaders.keys())}")
    
except Exception as e:
    print(f"❌ Error with DataBuilder: {e}")
    import traceback
    traceback.print_exc()