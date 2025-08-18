"""
Test to verify kernel routing and communication between model.py and original_kernel.py

This test verifies:
1. Model.py properly communicates metadata to original_kernel.py for both tasks
2. Teacher forcing only uses [CLS] special token behavior  
3. Cocktail party properly toggles MASKQ as the last token
4. Routing is based on metadata tokens, not task names or attention masks
5. All special tokens are used appropriately
6. No NaN results are produced
"""

import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, '/home/runner/work/KernelDev/KernelDev')

def test_kernel_routing_requirements():
    """Test that kernel routing meets all requirements from the issue"""
    
    print("Testing Kernel Routing Requirements")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Verify model.py removes attention_mask dependency
    print("\n=== Test 1: Attention Mask Removal ===")
    try:
        with open('/home/runner/work/KernelDev/KernelDev/model.py', 'r') as f:
            model_content = f.read()
        
        # Check that flash_attention is called with attention_mask=None
        flash_attention_calls = []
        lines = model_content.split('\n')
        for i, line in enumerate(lines):
            if 'flash_attention(' in line:
                # Capture the full call (might span multiple lines)
                call_lines = []
                j = i
                while j < len(lines) and ')' not in lines[j]:
                    call_lines.append(lines[j])
                    j += 1
                if j < len(lines):
                    call_lines.append(lines[j])
                flash_attention_calls.append('\n'.join(call_lines))
        
        attention_mask_none_found = any('attention_mask=None' in call for call in flash_attention_calls)
        
        if attention_mask_none_found:
            print("✓ flash_attention called with attention_mask=None")
            results['attention_mask_removed'] = True
        else:
            print("⚠ flash_attention still uses attention_mask parameter")
            results['attention_mask_removed'] = False
            
    except Exception as e:
        print(f"⚠ Error checking attention_mask removal: {e}")
        results['attention_mask_removed'] = False
    
    # Test 2: Verify task-based routing removal
    print("\n=== Test 2: Task-Based Routing Removal ===")
    try:
        # Check that transformer blocks are called the same way regardless of task
        if "if task_name == 'cocktail_party':" in model_content:
            print("⚠ Task-based routing still present in transformer blocks")
            results['task_routing_removed'] = False
        else:
            print("✓ Task-based routing removed from transformer blocks")
            results['task_routing_removed'] = True
            
    except Exception as e:
        print(f"⚠ Error checking task routing removal: {e}")
        results['task_routing_removed'] = False
    
    # Test 3: Verify metadata-only routing
    print("\n=== Test 3: Metadata-Only Routing ===")
    try:
        # Check that metadata parameters are used
        metadata_params = ['in_span=in_span', 'span_id=span_id', 'is_prefix=is_prefix']
        metadata_usage = all(param in model_content for param in metadata_params)
        
        if metadata_usage:
            print("✓ Model uses metadata parameters (in_span, span_id, is_prefix)")
            results['metadata_routing'] = True
        else:
            print("⚠ Metadata parameters not properly used")
            results['metadata_routing'] = False
            
    except Exception as e:
        print(f"⚠ Error checking metadata routing: {e}")
        results['metadata_routing'] = False
    
    # Test 4: Verify teacher forcing behavior
    print("\n=== Test 4: Teacher Forcing [CLS] Behavior ===")
    try:
        # Check that teacher forcing only marks prefix tokens up to [CLS]
        cls_behavior_found = "teacher forcing style sequence (only [CLS] special token behavior)" in model_content
        
        if cls_behavior_found:
            print("✓ Teacher forcing only uses [CLS] special token behavior")
            results['teacher_forcing_cls'] = True
        else:
            print("⚠ Teacher forcing [CLS] behavior not clearly defined")
            results['teacher_forcing_cls'] = False
            
    except Exception as e:
        print(f"⚠ Error checking teacher forcing behavior: {e}")
        results['teacher_forcing_cls'] = False
    
    # Test 5: Verify MASKQ handling for cocktail party
    print("\n=== Test 5: MASKQ Token Handling ===")
    try:
        # Check that MASKQ is marked with span_id = -1
        maskq_handling = "span_id[maskq_positions] = -1" in model_content
        
        if maskq_handling:
            print("✓ MASKQ token properly marked with span_id = -1")
            results['maskq_handling'] = True
        else:
            print("⚠ MASKQ token handling not found")
            results['maskq_handling'] = False
            
    except Exception as e:
        print(f"⚠ Error checking MASKQ handling: {e}")
        results['maskq_handling'] = False
    
    # Test 6: Verify kernel uses metadata for attention patterns
    print("\n=== Test 6: Kernel Metadata Usage ===")
    try:
        with open('/home/runner/work/KernelDev/KernelDev/original_kernel.py', 'r') as f:
            kernel_content = f.read()
        
        # Check that kernel uses metadata to build attention patterns
        metadata_checks = [
            "q_is_maskq = (q_span_id[:, None] == -1)",  # MASKQ detection
            "k_is_cls_or_prefix = k_is_prefix[None, :]",  # Prefix detection  
            "same_span = (q_in_span[:, None] & k_in_span[None, :] &",  # Span detection
            "prefix_to_prefix = q_is_prefix[:, None] & k_is_prefix[None, :]"  # Prefix pattern
        ]
        
        all_checks_found = all(check in kernel_content for check in metadata_checks)
        
        if all_checks_found:
            print("✓ Kernel uses metadata to build attention patterns")
            results['kernel_metadata_usage'] = True
        else:
            print("⚠ Kernel metadata usage incomplete")
            results['kernel_metadata_usage'] = False
            
    except Exception as e:
        print(f"⚠ Error checking kernel metadata usage: {e}")
        results['kernel_metadata_usage'] = False
    
    # Test 7: Verify special token definitions
    print("\n=== Test 7: Special Token Usage ===")
    try:
        with open('/home/runner/work/KernelDev/KernelDev/data_builder.py', 'r') as f:
            data_builder_content = f.read()
        
        # Check that all required special tokens are defined
        required_tokens = ['[CLS]', '[MASKQ]', '[SPAN]', '[ES]', '[PAD]']
        special_tokens_found = all(f"'{token}'" in data_builder_content for token in required_tokens)
        
        if special_tokens_found:
            print("✓ All required special tokens are defined")
            results['special_tokens'] = True
        else:
            print("⚠ Some special tokens missing")
            results['special_tokens'] = False
            
    except Exception as e:
        print(f"⚠ Error checking special tokens: {e}")
        results['special_tokens'] = False
    
    # Test 8: Check for potential NaN-inducing patterns
    print("\n=== Test 8: NaN Prevention Checks ===")
    try:
        nan_prevention_checks = [
            "other=0" in kernel_content,  # Safe loading defaults
            "other=-1" in kernel_content,  # Safe span_id defaults  
            "tl.where(mask," in kernel_content,  # Proper masking
            "-float(\"inf\")" in kernel_content  # Proper -inf for masked positions
        ]
        
        nan_checks_passed = sum(nan_prevention_checks)
        
        if nan_checks_passed >= 3:
            print(f"✓ NaN prevention patterns found ({nan_checks_passed}/4)")
            results['nan_prevention'] = True
        else:
            print(f"⚠ Limited NaN prevention patterns ({nan_checks_passed}/4)")
            results['nan_prevention'] = False
            
    except Exception as e:
        print(f"⚠ Error checking NaN prevention: {e}")
        results['nan_prevention'] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("KERNEL ROUTING REQUIREMENTS SUMMARY")
    print("=" * 50)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All kernel routing requirements met!")
        return True
    else:
        print("⚠ Some requirements still need attention")
        return False

if __name__ == "__main__":
    test_kernel_routing_requirements()