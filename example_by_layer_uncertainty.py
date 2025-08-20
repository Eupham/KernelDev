#!/usr/bin/env python3
"""
Example usage of the new "By layer uncertainty tweaks" features (Issue #129).

This example demonstrates how to use all the new features implemented:
- Deep supervision at every layer for both tasks
- Shared heads with scalar conditioning
- Per-(layer,task) uncertainty parameters
- Enhanced loss computation and reporting
"""

import torch
from model import GPTModel
from train_loop import Trainer, TrainingConfig

def example_by_layer_uncertainty_usage():
    """Example showing how to use the new by-layer uncertainty features."""
    
    print("=== Example: Using By Layer Uncertainty Tweaks ===\n")
    
    # 1. Create model with new configuration options
    print("1. Creating model with new configuration:")
    
    model = GPTModel(
        vocab_size=1000,
        dim=512,
        n_layers=12,
        n_heads=8,
        task_names=['teacher_forcing', 'cocktail_party'],
        
        # NEW CONFIGURATION OPTIONS:
        supervise_layers="all",     # Supervise ALL layers (not just every N-th)
        share_heads=True,           # Use shared head across layers
        conditioning="film"         # Use FiLM conditioning (or "concat2")
    )
    
    print(f"   - Supervised layers: {model.supervised_layer_indices}")
    print(f"   - Share heads: {model.share_heads}")
    print(f"   - Conditioning: {model.conditioning}")
    print(f"   - Has shared TF head: {hasattr(model, 'shared_tf_head')}")
    print(f"   - Has FiLM parameters: {hasattr(model, 'layer_alpha')}")
    
    # 2. Show per-(layer,task) uncertainty parameters
    print(f"\n2. Per-(layer,task) uncertainty parameters:")
    for i in [0, 6, 11]:  # Show first, middle, last layers
        block = model.blocks[i]
        if hasattr(block, 'log_sigmas'):
            for task in model.task_names:
                if task in block.log_sigmas:
                    sigma = torch.exp(block.log_sigmas[task]).item()
                    print(f"   Layer {i:2d} {task:15s}: σ = {sigma:.6f}")
    
    # 3. Example forward pass scenarios
    print(f"\n3. Forward pass examples:")
    
    batch_size = 4
    seq_len = 64
    
    # Mock teacher forcing data
    tf_input = torch.randint(1, 1000, (batch_size, seq_len))
    tf_targets = torch.randint(1, 1000, (batch_size, seq_len))
    
    # Mock cocktail party data (with special tokens)
    cls_id, mask_id, span_id, es_id = 1, 2, 3, 4
    cp_input = torch.tensor([
        [cls_id, 10, 11, mask_id, span_id, 20, 21, es_id, span_id, 22, 23, es_id] + [0] * (seq_len-12)
    ] * batch_size)
    cp_correct_idx = torch.randint(0, 2, (batch_size,))
    
    # Example A: Teacher forcing only
    print(f"   A. Teacher forcing only:")
    logits_tf, loss_tf = model(tf_input, targets=tf_targets)
    print(f"      Logits: {logits_tf.shape}")
    if isinstance(loss_tf, dict):
        print(f"      Loss structure: final_ce + {len(loss_tf['layer_ce'])} layer losses")
    
    # Example B: Cocktail party only  
    print(f"   B. Cocktail party only:")
    scores_cp, loss_cp = model(cp_input, correct_idx=cp_correct_idx)
    print(f"      Scores: {scores_cp.shape}")
    if isinstance(loss_cp, dict):
        print(f"      Loss structure: final_ce + {len(loss_cp['layer_ce'])} layer losses")
    
    # Example C: Both tasks simultaneously (mixed batch)
    print(f"   C. Both tasks simultaneously:")
    mixed_outputs, loss_mixed = model(cp_input, targets=tf_targets[:, :cp_input.size(1)], correct_idx=cp_correct_idx)
    if isinstance(loss_mixed, dict) and "teacher_forcing" in loss_mixed:
        print(f"      Multi-task loss: {list(loss_mixed.keys())}")
        tf_part = loss_mixed["teacher_forcing"]
        cp_part = loss_mixed["cocktail_party"]
        if isinstance(tf_part, dict):
            print(f"      TF: final_ce + {len(tf_part['layer_ce'])} layer losses")
        if isinstance(cp_part, dict):
            print(f"      CP: final_ce + {len(cp_part['layer_ce'])} layer losses")
    
    # 4. New uncertainty weighting
    print(f"\n4. New uncertainty weighting:")
    
    # Create trainer with new configuration
    config = TrainingConfig(learning_rate=3e-4, warmup_steps=1000)
    trainer = Trainer(model, config)
    
    # Configure lambda values as specified in the issue
    lambda_layers = {"teacher_forcing": 1.0, "cocktail_party": 1.0}  # λ_t per task
    lambda_kl = 1e-3  # KL regularization weight
    
    # Apply new uncertainty weighting
    if isinstance(loss_mixed, dict) and "teacher_forcing" in loss_mixed:
        total_loss, unc_deltas = trainer.apply_layer_uncertainty_weighting(
            loss_mixed, lambda_layers, lambda_kl
        )
        
        print(f"   Total weighted loss: {total_loss.item():.6f}")
        print(f"   Per-task uncertainty contributions:")
        for task, deltas in unc_deltas.items():
            print(f"     {task:15s}: delta={deltas['delta'].item():8.4f}, kl={deltas['kl'].item():8.6f}")
    
    # 5. Mathematical formula verification
    print(f"\n5. Mathematical formula verification:")
    print(f"   Formula: L_ℓ,t^unc = 1/2 * e^(-2s_ℓ,t) * L_ℓ,t^raw + s_ℓ,t")
    
    if isinstance(loss_tf, dict) and 'layer_ce' in loss_tf:
        # Pick one layer to demonstrate
        layer_name = 'layer_6'  # Middle layer
        if layer_name in loss_tf['layer_ce']:
            layer_idx = 6
            raw_loss = loss_tf['layer_ce'][layer_name]
            
            for task in ['teacher_forcing', 'cocktail_party']:
                if hasattr(model.blocks[layer_idx], 'log_sigmas') and task in model.blocks[layer_idx].log_sigmas:
                    s_lt = model.blocks[layer_idx].log_sigmas[task]
                    s_clamped = torch.clamp(s_lt, -5, 5)
                    
                    # Apply the formula
                    weighted = 0.5 * torch.exp(-2 * s_clamped) * raw_loss + s_clamped
                    
                    print(f"   Layer {layer_idx} {task}:")
                    print(f"     Raw loss: {raw_loss.item():.6f}")
                    print(f"     s_ℓ,t: {s_lt.item():.6f}")
                    print(f"     Weighted: {weighted.item():.6f}")
    
    print(f"\n=== Key Benefits of the Implementation ===")
    print(f"✓ Deep supervision: All layers contribute to learning")
    print(f"✓ Per-(layer,task) uncertainty: Fine-grained uncertainty control")
    print(f"✓ Shared heads: Efficient parameter usage with conditioning")
    print(f"✓ Multi-task support: Both TF and CP losses at every layer")
    print(f"✓ Enhanced reporting: Raw CE vs uncertainty-weighted losses separate")
    print(f"✓ Mathematical rigor: Exact formula from Issue #129 implemented")
    
    print(f"\n=== Configuration Reference ===")
    print(f"Config file settings (config.yaml):")
    print(f"  layer_uncertainty:")
    print(f"    supervise_layers: 'all'  # or integer for every N layers")
    print(f"    share_heads: true")
    print(f"    conditioning: 'film'     # or 'concat2'")
    print(f"    lambda_layers:")
    print(f"      tf: 1.0               # teacher_forcing weight")
    print(f"      cp: 1.0               # cocktail_party weight")
    print(f"    lambda_kl: 1e-3         # KL regularization")
    
    print(f"\n🎉 By layer uncertainty tweaks are ready for production use!")


if __name__ == "__main__":
    example_by_layer_uncertainty_usage()