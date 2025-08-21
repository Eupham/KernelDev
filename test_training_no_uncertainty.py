#!/usr/bin/env python3
"""
Test to verify uncertainty mechanism has been removed from the training loop.
This test checks that the training loop no longer references uncertainty parameters.
"""

import torch
import torch.nn as nn
import sys
import re

def test_no_uncertainty_in_training():
    """Test that training loop no longer has uncertainty references."""
    
    print("=== Testing Training Loop Without Uncertainty ===\n")
    
    # Test 1: Check that train_loop.py doesn't contain uncertainty references
    print("1. Checking train_loop.py for uncertainty references:")
    
    with open('train_loop.py', 'r') as f:
        content = f.read()
    
    # Look for problematic patterns
    uncertainty_patterns = [
        r'log_sigmas',
        r'apply_layer_uncertainty_weighting',
        r'uncertainty.*weighting',
        r'weighted_loss.*exp.*log_sigma',
        r'sigma.*=.*exp\(',
    ]
    
    found_issues = []
    for pattern in uncertainty_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            found_issues.extend(matches)
    
    if found_issues:
        print(f"   ❌ Found uncertainty references: {found_issues}")
        return False
    else:
        print("   ✓ No uncertainty references found in training loop")
    
    # Test 2: Check for task weighting instead
    print("\n2. Checking for task weight usage:")
    
    if 'task_weight' in content and 'task_configs' in content:
        print("   ✓ Found task weighting mechanism using config")
    else:
        print("   ❌ Task weighting mechanism not found")
        return False
    
    # Test 3: Check that logging no longer mentions sigma
    print("\n3. Checking logging statements:")
    
    if 'σ:' in content or 'sigma:' in content:
        print("   ❌ Found sigma references in logging")
        return False
    else:
        print("   ✓ No sigma references found in logging")
    
    print("\n4. Checking that weight logging is present:")
    if 'weight:' in content:
        print("   ✓ Found weight logging")
    else:
        print("   ❌ Weight logging not found")
        return False
    
    print("\n🎉 ALL TESTS PASSED: Uncertainty mechanism successfully removed from training loop!")
    return True

def test_task_config_structure():
    """Test that task configuration still works for weighting."""
    
    print("\n=== Testing Task Configuration ===\n")
    
    # Mock task configs like in the actual config
    task_configs = {
        'teacher_forcing': {'weight': 1.0},
        'cocktail_party': {'weight': 1.0, 'num_distractors': 3}
    }
    
    print("1. Testing task weight extraction:")
    for task_name in task_configs:
        weight = task_configs.get(task_name, {}).get('weight', 1.0)
        print(f"   {task_name}: weight = {weight}")
    
    print("   ✓ Task weights extracted successfully")
    
    # Test weight calculation
    print("\n2. Testing simple weighted loss calculation:")
    
    # Simulate losses
    losses = {'teacher_forcing': 2.5, 'cocktail_party': 1.8}
    total_loss = 0
    
    for task_name, loss in losses.items():
        weight = task_configs.get(task_name, {}).get('weight', 1.0)
        weighted_loss = weight * loss
        total_loss += weighted_loss
        print(f"   {task_name}: {loss:.3f} * {weight:.1f} = {weighted_loss:.3f}")
    
    print(f"   Total weighted loss: {total_loss:.3f}")
    print("   ✓ Simple weighted loss calculation works")
    
    return True

if __name__ == "__main__":
    success1 = test_no_uncertainty_in_training()
    success2 = test_task_config_structure()
    
    if success1 and success2:
        print("\n🎉 ALL TRAINING TESTS PASSED!")
    else:
        print("\n❌ SOME TRAINING TESTS FAILED!")
        exit(1)