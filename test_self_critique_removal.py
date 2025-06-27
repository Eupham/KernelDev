#!/usr/bin/env python3
"""
Test to validate that self-critique mechanism has been properly removed.
This test validates the API changes without requiring PyTorch installation.
"""

import inspect
import sys
import os

def test_config_parameters_removed():
    """Test that self-critique parameters are removed from TrainingConfig."""
    print("Testing TrainingConfig parameter removal...")
    
    # Read the train_loop.py file to check for removed parameters
    with open('train_loop.py', 'r') as f:
        content = f.read()
    
    # Check that self-critique parameters are not in the __init__ method
    removed_params = [
        'lm_self_critique_base_penalty',
        'lm_self_critique_reward_max', 
        'self_critique_temperature'
    ]
    
    for param in removed_params:
        if param in content:
            print(f"❌ FAIL: Parameter '{param}' still found in train_loop.py")
            return False
        else:
            print(f"✅ PASS: Parameter '{param}' successfully removed")
    
    return True

def test_train_step_signature():
    """Test that train_step method has correct return signature."""
    print("\nTesting train_step method signature...")
    
    with open('train_loop.py', 'r') as f:
        content = f.read()
    
    # Look for the train_step method signature
    train_step_line = None
    for line_num, line in enumerate(content.split('\n'), 1):
        if 'def train_step(self, batch:' in line:
            train_step_line = line.strip()
            break
    
    if train_step_line is None:
        print("❌ FAIL: Could not find train_step method definition")
        return False
    
    # Should return 4 values instead of 5 (removed d_self_critique_mean)
    expected_return_type = "Tuple[float, Optional[float], Optional[float], Optional[float]]"
    if expected_return_type in train_step_line:
        print(f"✅ PASS: train_step signature updated correctly: {train_step_line}")
        return True
    else:
        print(f"❌ FAIL: train_step signature incorrect: {train_step_line}")
        return False

def test_logging_removal():
    """Test that self-critique logging statements are removed."""
    print("\nTesting self-critique logging removal...")
    
    with open('train_loop.py', 'r') as f:
        content = f.read()
    
    # Check that logging statements are removed
    removed_log_strings = [
        'AvgCritiqueD:',
        'EMA_CritiqueD:',
        'DeltaCritiqueD:'
    ]
    
    for log_str in removed_log_strings:
        if log_str in content:
            print(f"❌ FAIL: Logging string '{log_str}' still found in train_loop.py")
            return False
        else:
            print(f"✅ PASS: Logging string '{log_str}' successfully removed")
    
    return True

def test_config_yaml_cleanup():
    """Test that config.yaml has been cleaned up."""
    print("\nTesting config.yaml cleanup...")
    
    if not os.path.exists('config.yaml'):
        print("❌ FAIL: config.yaml not found")
        return False
    
    with open('config.yaml', 'r') as f:
        content = f.read()
    
    # Check that self-critique parameters are removed
    removed_config_items = [
        'lm_self_critique_base_penalty',
        'lm_self_critique_reward_max',
        'self_critique_temperature'
    ]
    
    for item in removed_config_items:
        if item in content:
            print(f"❌ FAIL: Config item '{item}' still found in config.yaml")
            return False
        else:
            print(f"✅ PASS: Config item '{item}' successfully removed from config.yaml")
    
    return True

def test_documentation_updated():
    """Test that documentation reflects the changes."""
    print("\nTesting documentation updates...")
    
    if not os.path.exists('SELF_CRITIQUE_FIX.md'):
        print("❌ FAIL: SELF_CRITIQUE_FIX.md not found")
        return False
    
    with open('SELF_CRITIQUE_FIX.md', 'r') as f:
        content = f.read()
    
    # Check that documentation mentions the removal approach
    key_phrases = [
        'Complete Removal',
        'entire self-critique mechanism has been removed',
        'Eliminated performance overhead'
    ]
    
    for phrase in key_phrases:
        if phrase in content:
            print(f"✅ PASS: Documentation includes key phrase: '{phrase}'")
        else:
            print(f"❌ FAIL: Documentation missing key phrase: '{phrase}'")
            return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Self-Critique Mechanism Removal")
    print("=" * 60)
    
    tests = [
        test_config_parameters_removed,
        test_train_step_signature,
        test_logging_removal,
        test_config_yaml_cleanup,
        test_documentation_updated
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"Test {test.__name__} failed")
        except Exception as e:
            print(f"Test {test.__name__} raised exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Self-critique mechanism successfully removed.")
        return 0
    else:
        print("❌ SOME TESTS FAILED. Please review the changes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())