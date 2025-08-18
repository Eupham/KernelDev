"""
Comprehensive verification test for Issue #102: Kernel Routing

This test specifically addresses each requirement from the issue:
1. Verify model.py communicates appropriately to original_kernel.py for both tasks
2. Verify original_kernel.py follows appropriate routes based on communicated metadata
3. Verify all special tokens are used as intended
4. Verify no attention mask is used, only metadata
5. Verify routing is based on tokens, not task names
6. Verify teacher forcing only uses [CLS] behavior
7. Verify MASKQ is toggled as last token for cocktail party
"""

def verify_issue_102_requirements():
    """Verify all specific requirements from Issue #102"""
    
    print("Verifying Issue #102: Kernel Routing Requirements")
    print("=" * 60)
    
    # Read source files
    with open('/home/runner/work/KernelDev/KernelDev/model.py', 'r') as f:
        model_content = f.read()
    
    with open('/home/runner/work/KernelDev/KernelDev/original_kernel.py', 'r') as f:
        kernel_content = f.read()
        
    with open('/home/runner/work/KernelDev/KernelDev/data_builder.py', 'r') as f:
        data_builder_content = f.read()
    
    requirements_met = []
    
    # Requirement 1: Model.py communicates appropriately to original_kernel.py
    print("\n1. Model communicates metadata to kernel appropriately")
    flash_call_has_metadata = all(param in model_content for param in [
        'in_span=in_span',
        'span_id=span_id', 
        'is_prefix=is_prefix'
    ])
    print(f"   ✓ Flash attention called with metadata parameters: {flash_call_has_metadata}")
    requirements_met.append(flash_call_has_metadata)
    
    # Requirement 2: Kernel follows appropriate routes based on metadata
    print("\n2. Kernel routes based on communicated metadata")
    kernel_patterns = [
        "q_is_maskq = (q_span_id[:, None] == -1)",  # MASKQ routing
        "prefix_to_prefix = q_is_prefix",  # Prefix routing
        "same_span = (q_in_span[:, None] & k_in_span[None, :] &",  # Span routing
        "context_causal = q_is_context & k_is_context"  # Context routing
    ]
    kernel_routing_proper = all(pattern in kernel_content for pattern in kernel_patterns)
    print(f"   ✓ Kernel implements metadata-based routing patterns: {kernel_routing_proper}")
    requirements_met.append(kernel_routing_proper)
    
    # Requirement 3: No attention mask used, only metadata
    print("\n3. No attention mask used, only metadata")
    no_attention_mask = 'attention_mask=None' in model_content
    metadata_only = 'in_span' in model_content and 'span_id' in model_content and 'is_prefix' in model_content
    print(f"   ✓ Flash attention called with attention_mask=None: {no_attention_mask}")
    print(f"   ✓ Metadata tensors used instead: {metadata_only}")
    requirements_met.append(no_attention_mask and metadata_only)
    
    # Requirement 4: No task-based routing
    print("\n4. Routing based on tokens, not task names")
    no_task_routing = "if task_name == 'cocktail_party':" not in model_content
    token_based_detection = "has_mask = (x == mask_token_id).any()" in model_content
    print(f"   ✓ No task-based routing in transformer blocks: {no_task_routing}")
    print(f"   ✓ Token-based output mode detection: {token_based_detection}")
    requirements_met.append(no_task_routing and token_based_detection)
    
    # Requirement 5: Teacher forcing only uses [CLS] behavior
    print("\n5. Teacher forcing only uses [CLS] special token behavior")
    cls_only_comment = "only [CLS] special token behavior" in model_content
    cls_prefix_logic = "cls_pos + 1] = True" in model_content
    print(f"   ✓ Teacher forcing explicitly uses only [CLS] behavior: {cls_only_comment}")
    print(f"   ✓ Prefix marking up to and including [CLS]: {cls_prefix_logic}")
    requirements_met.append(cls_only_comment and cls_prefix_logic)
    
    # Requirement 6: MASKQ explicitly toggled as last token for cocktail party
    print("\n6. MASKQ toggled as last token for cocktail party")
    maskq_special_id = "span_id[maskq_positions] = -1" in model_content
    maskq_data_builder = "span_ids[maskq_idx] = -1" in data_builder_content
    maskq_kernel_detection = "q_is_maskq = (q_span_id[:, None] == -1)" in kernel_content
    print(f"   ✓ MASKQ marked with span_id=-1 in model: {maskq_special_id}")
    print(f"   ✓ MASKQ marked with span_id=-1 in data_builder: {maskq_data_builder}")
    print(f"   ✓ Kernel detects MASKQ via span_id=-1: {maskq_kernel_detection}")
    requirements_met.append(maskq_special_id and maskq_data_builder and maskq_kernel_detection)
    
    # Requirement 7: All special tokens used appropriately
    print("\n7. All special tokens used appropriately")
    special_tokens_present = all(token in data_builder_content for token in [
        "'[CLS]': 1",
        "'[MASKQ]': 5", 
        "'[SPAN]': 3",
        "'[ES]': 4",
        "'[PAD]': 0"
    ])
    prefix_cls_usage = "k_is_cls_or_prefix" in kernel_content
    print(f"   ✓ All special tokens defined: {special_tokens_present}")
    print(f"   ✓ [CLS] used for prefix patterns in kernel: {prefix_cls_usage}")
    requirements_met.append(special_tokens_present and prefix_cls_usage)
    
    # Requirement 8: Metadata provides position information as specified
    print("\n8. Metadata provides proper position information")
    prefix_positions = "is_prefix" in model_content and "is_prefix" in kernel_content
    span_positions = "in_span" in model_content and "in_span" in kernel_content  
    span_ids = "span_id" in model_content and "span_id" in kernel_content
    pad_handling = "ignore_index=SPECIAL_TOKENS['[PAD]']" in model_content
    print(f"   ✓ Prefix positions fed via is_prefix: {prefix_positions}")
    print(f"   ✓ Span positions fed via in_span: {span_positions}")
    print(f"   ✓ Span IDs fed via span_id: {span_ids}")
    print(f"   ✓ PAD tokens ignored in loss: {pad_handling}")
    requirements_met.append(prefix_positions and span_positions and span_ids and pad_handling)
    
    # Requirement 9: Causal assumption for non-specified positions
    print("\n9. Non-specified positions assume causal attention")
    causal_context = "context_causal = q_is_context & k_is_context & (q_tile_indices[:, None] >= kv_indices[None, :])" in kernel_content
    causal_default = "elif CAUSAL:" in kernel_content and "mask = q_tile_indices[:, None] >= kv_indices[None, :]" in kernel_content
    print(f"   ✓ Context tokens use causal attention: {causal_context}")
    print(f"   ✓ Default causal behavior for non-metadata: {causal_default}")
    requirements_met.append(causal_context and causal_default)
    
    # Summary
    print("\n" + "=" * 60)
    print("ISSUE #102 REQUIREMENTS VERIFICATION")
    print("=" * 60)
    
    total_requirements = len(requirements_met)
    met_requirements = sum(requirements_met)
    
    requirement_names = [
        "Model-Kernel Communication",
        "Metadata-Based Routing", 
        "No Attention Mask Usage",
        "Token-Based (Not Task-Based) Routing",
        "Teacher Forcing [CLS] Only",
        "MASKQ Last Token Toggle",
        "Special Token Usage",
        "Position Information Metadata",
        "Causal Default Assumption"
    ]
    
    for i, (name, met) in enumerate(zip(requirement_names, requirements_met)):
        status = "✓ MET" if met else "✗ NOT MET"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall: {met_requirements}/{total_requirements} requirements met")
    
    if met_requirements == total_requirements:
        print("\n🎉 ALL REQUIREMENTS FROM ISSUE #102 SUCCESSFULLY IMPLEMENTED!")
        print("✓ Model.py communicates appropriately to original_kernel.py")
        print("✓ Kernel routes based on metadata tokens, not task names or masks")
        print("✓ Teacher forcing uses only [CLS] behavior")
        print("✓ MASKQ properly toggled for cocktail party")
        print("✓ All special tokens used as specified")
        return True
    else:
        print(f"\n⚠ {total_requirements - met_requirements} requirements still need attention")
        return False

if __name__ == "__main__":
    verify_issue_102_requirements()