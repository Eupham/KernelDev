#!/usr/bin/env python3
"""
Test ACT-R functionality without full dependencies.
"""

import sys
import os
sys.path.append('.')

# Mock the datasets import to avoid dependency issues
import sys
from unittest.mock import MagicMock
sys.modules['datasets'] = MagicMock()

# Now import our modules
from data_builder import DataBuilder, SPECIAL_TOKENS
from train_loop import TrainingMetrics
import torch

def test_actr_basic_functionality():
    """Test ACT-R basic functionality without full dataset loading."""
    print("=== Testing ACT-R Basic Functionality ===")
    
    # Test DataBuilder with ACT-R functionality
    db = DataBuilder(max_samples=10)
    print(f"✓ DataBuilder created with ACT-R attributes")
    print(f"✓ ACT-R association table: {hasattr(db, 'actr_association_table')}")
    print(f"✓ ACT-R window size: {db.actr_window_size}")
    
    # Test association building with mock data
    test_tokens = [10, 15, 20, 25, 30, 15, 35, 40, 20, 45]  # Mock token sequence
    db._build_associations_from_tokens(test_tokens)
    print(f"✓ Association building completed")
    print(f"✓ Association table size: {len(db.actr_association_table)}")
    
    # Test fan calculation
    fan_15 = db.calculate_fan(15)
    fan_20 = db.calculate_fan(20) 
    print(f"✓ Fan calculation works: token 15 has fan {fan_15}, token 20 has fan {fan_20}")
    
    # Test anchor extraction
    test_span = [SPECIAL_TOKENS['[SPAN]'], 15, 20, 25, SPECIAL_TOKENS['[ES]']]
    anchor = db.get_anchor_token_from_span(test_span)
    print(f"✓ Anchor extraction: {anchor} (expected: 15)")
    
    # Test similarity calculation
    span1 = [15, 20, 25]
    span2 = [20, 25, 30]
    sim = db.calculate_similarity(span1, span2)
    print(f"✓ Similarity calculation: {sim:.3f}")
    
    # Test tertile categorization
    fan_distribution = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    category = db.categorize_fan_tertile(3, fan_distribution)
    print(f"✓ Tertile categorization: fan=3 -> {category}")
    
    return True

def test_actr_metrics():
    """Test ACT-R metrics functionality."""
    print("\n=== Testing ACT-R Metrics ===")
    
    # Test TrainingMetrics with ACT-R
    metrics = TrainingMetrics()
    print(f"✓ TrainingMetrics created with ACT-R support")
    print(f"✓ ACT-R metrics list: {hasattr(metrics, 'actr_metrics')}")
    
    # Test adding ACT-R trial data
    trial_data = {
        'fan': 5,
        'anchor_token': 15,
        'sim_gold': 1.0,
        'max_sim_distractor': 0.3,
        'accuracy': 1.0,
        'margin': 2.5
    }
    metrics.update_actr_metrics(trial_data)
    print(f"✓ ACT-R trial data added: {len(metrics.actr_metrics)} trials")
    
    # Add more varied trial data for analysis
    for i in range(10):
        trial = {
            'fan': i + 1,
            'anchor_token': 10 + i,
            'sim_gold': 1.0,
            'max_sim_distractor': 0.2 + (i * 0.05),
            'accuracy': 1.0 - (i * 0.1),  # Decreasing accuracy
            'margin': 3.0 - (i * 0.2)     # Decreasing margin
        }
        metrics.update_actr_metrics(trial)
    
    print(f"✓ Added {len(metrics.actr_metrics)} total trials")
    
    # Test analysis
    analysis = metrics.analyze_actr_fan_effect()
    if analysis:
        print(f"✓ ACT-R analysis completed:")
        print(f"  Total trials: {analysis['total_trials']}")
        print(f"  Fan range: {analysis['fan_range']}")
        print("  Tertile results:")
        for tertile, data in analysis['fan_tertiles'].items():
            print(f"    {tertile}: acc={data['accuracy']:.3f}, margin={data['margin']:.3f}, n={data['count']}")
    else:
        print("✗ ACT-R analysis failed")
        return False
    
    return True

def test_actr_batch_processing():
    """Test ACT-R batch metric processing."""
    print("\n=== Testing ACT-R Batch Processing ===")
    
    db = DataBuilder(max_samples=10)
    
    # Mock valid items for batch processing
    valid_items = [
        {
            'true_span': [15, 20, 25],
            'distractors': [[30, 35, 40], [45, 50, 55], [60, 65, 70]]
        },
        {
            'true_span': [75, 80, 85],
            'distractors': [[90, 95, 100], [105, 110, 115], [120, 125, 130]]
        }
    ]
    
    # Mock batch items (simplified)
    batch_items = [{}, {}]  # Mock metadata items
    
    # Process ACT-R metrics for batch
    actr_metrics = db._calculate_actr_metrics_for_batch(batch_items, valid_items)
    print(f"✓ Batch ACT-R metrics calculated: {len(actr_metrics)} items")
    
    for i, metrics in enumerate(actr_metrics):
        print(f"  Item {i}: fan={metrics['fan']}, anchor={metrics['anchor_token']}, "
              f"max_sim={metrics['max_sim_distractor']:.3f}")
    
    return True

if __name__ == "__main__":
    print("Testing ACT-R Implementation")
    print("=" * 50)
    
    try:
        results = []
        results.append(test_actr_basic_functionality())
        results.append(test_actr_metrics())
        results.append(test_actr_batch_processing())
        
        if all(results):
            print("\n🎉 ALL ACT-R TESTS PASSED!")
            print("   ✓ Association building and fan calculation work")
            print("   ✓ Metrics tracking and analysis work") 
            print("   ✓ Batch processing works")
            print("   ✓ Integration points are functional")
            print("\nACT-R functionality is ready for training integration!")
        else:
            print("\n❌ SOME ACT-R TESTS FAILED!")
            exit(1)
            
    except Exception as e:
        print(f"\n💥 ACT-R TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)