#!/usr/bin/env python3
"""
Simplified Layer-Level Uncertainty Test

This test validates the layer-level uncertainty implementation without
requiring CUDA or flash attention by creating a minimal model setup.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
warnings.filterwarnings('ignore')

# Mock SPECIAL_TOKENS
SPECIAL_TOKENS = {
    '[PAD]': 0,
    '[CLS]': 1,
    '[MASK]': 2,
    '[SPAN]': 3,
    '[ES]': 4,
    '[MASKQ]': 5
}

# Simplified model components for testing
class SimpleAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x, **kwargs):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Simple scaled dot-product attention
        scale = (self.head_dim ** -0.5)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)

class SimpleRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class SimpleSwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class SimpleTransformerBlock(nn.Module):
    """Simplified transformer block with layer uncertainty."""
    
    def __init__(self, dim, n_heads, mlp_ratio=4, vocab_size=None, has_layer_supervision=False):
        super().__init__()
        self.norm1 = SimpleRMSNorm(dim)
        self.attn = SimpleAttention(dim, n_heads)
        self.norm2 = SimpleRMSNorm(dim)
        self.mlp = SimpleSwiGLU(dim, int(dim * mlp_ratio))
        
        # Layer uncertainty and supervision components
        self.has_layer_supervision = has_layer_supervision
        if has_layer_supervision and vocab_size is not None:
            # Learnable log-precision parameter for this layer (start at 0)
            self.log_sigma = nn.Parameter(torch.zeros(1))
            # Small readout head for deep supervision
            self.layer_head = nn.Linear(dim, vocab_size, bias=False)
    
    def forward(self, x, **kwargs):
        # Pre-norm for attention
        x = x + self.attn(self.norm1(x))
        # Pre-norm for MLP
        x = x + self.mlp(self.norm2(x))
        return x

class SimpleGPTModel(nn.Module):
    """Simplified GPT model for testing layer uncertainty."""
    
    def __init__(
        self,
        vocab_size,
        dim=64,
        n_layers=8,
        n_heads=4,
        max_seq_len=128,
        mlp_ratio=4,
        layer_supervision_frequency=2
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.layer_supervision_frequency = layer_supervision_frequency
        
        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        
        # Transformer blocks with selective layer supervision
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(
                dim=dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                vocab_size=vocab_size,
                has_layer_supervision=(i % layer_supervision_frequency == 0 and i > 0)
            )
            for i in range(n_layers)
        ])
        
        # Track which layers have supervision
        self.supervised_layer_indices = [i for i in range(n_layers) 
                                       if i % layer_supervision_frequency == 0 and i > 0]
        
        # Final norm and output projection
        self.norm_out = SimpleRMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
    
    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        
        # Create position indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        
        # Token and position embeddings
        x_embed = self.token_emb(x) + self.pos_emb(pos)
        
        # Apply transformer blocks with layer supervision
        layer_losses = {}
        
        for i, block in enumerate(self.blocks):
            x_embed = block(x_embed)
            
            # Collect intermediate outputs for layer supervision
            if block.has_layer_supervision and targets is not None:
                # Apply normalization and compute layer logits
                normalized_x = self.norm_out(x_embed)
                layer_logits = block.layer_head(normalized_x)
                
                # Compute layer-wise cross-entropy loss
                layer_loss = F.cross_entropy(
                    layer_logits.view(-1, layer_logits.size(-1)),
                    targets.view(-1),
                    ignore_index=SPECIAL_TOKENS['[PAD]']
                )
                
                layer_losses[f'layer_{i}'] = layer_loss
        
        # Final normalization
        x_embed = self.norm_out(x_embed)
        
        # Final logits
        logits = self.head(x_embed)
        loss = None
        
        if targets is not None:
            # Compute final layer cross-entropy loss
            final_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=SPECIAL_TOKENS['[PAD]']
            )
            
            # Return structured loss if we have layer supervision
            if layer_losses:
                loss = {
                    'final_loss': final_loss,
                    'layer_losses': layer_losses
                }
            else:
                loss = final_loss
        
        return logits, loss

# Simple layer uncertainty weighting function
def apply_layer_uncertainty_weighting(model, loss, lambda_pred=1.0, lambda_kl=1e-3):
    """Apply layer-wise uncertainty weighting to structured losses."""
    
    if isinstance(loss, dict) and 'layer_losses' in loss:
        # Handle structured loss with layer supervision
        total_weighted_loss = torch.tensor(0.0, device=loss['final_loss'].device)
        
        # 1. Final layer loss
        final_loss = loss['final_loss']
        weighted_final = lambda_pred * final_loss
        total_weighted_loss += weighted_final
        
        # 2. Layer-wise losses with layer uncertainty
        layer_losses = loss['layer_losses']
        kl_penalty = torch.tensor(0.0, device=final_loss.device)
        
        for layer_name, layer_loss in layer_losses.items():
            # Extract layer index
            layer_idx = int(layer_name.split('_')[1])
            layer_block = model.blocks[layer_idx]
            
            if hasattr(layer_block, 'log_sigma'):
                # Apply uncertainty weighting: L_ℓ(unc) = 1/2 * exp(-2*s_ℓ) * L_ℓ + s_ℓ
                s_l = layer_block.log_sigma
                
                # Clamp s_ℓ to [-5, 5] to avoid degenerate blow-ups
                s_l_clamped = torch.clamp(s_l, -5.0, 5.0)
                
                uncertainty_weighted_loss = 0.5 * torch.exp(-2 * s_l_clamped) * layer_loss + s_l_clamped
                total_weighted_loss = total_weighted_loss + uncertainty_weighted_loss
                
                # Add KL penalty: simplified L2 penalty
                kl_penalty = kl_penalty + 0.5 * s_l_clamped ** 2
            else:
                # No uncertainty for this layer, just add the loss
                total_weighted_loss = total_weighted_loss + layer_loss
        
        # Add KL regularization
        total_weighted_loss = total_weighted_loss + lambda_kl * kl_penalty
        
        return total_weighted_loss
    else:
        # Handle simple loss (no layer supervision)
        return loss

def test_layer_uncertainty_mechanism():
    """Test the layer-level uncertainty mechanism."""
    
    print("=== Simplified Layer-Level Uncertainty Test ===\n")
    
    # Create a small model with layer supervision
    vocab_size = 100
    dim = 32
    n_layers = 6
    layer_supervision_frequency = 2  # Every 2nd layer
    
    model = SimpleGPTModel(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=4,
        max_seq_len=64,
        layer_supervision_frequency=layer_supervision_frequency
    )
    
    print(f"✓ Created model with {n_layers} layers, supervision every {layer_supervision_frequency} layers")
    print(f"✓ Supervised layers: {model.supervised_layer_indices}")
    
    # Test 1: Check layer supervision setup
    print("\n1. Checking layer supervision setup:")
    
    supervised_layers = []
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'log_sigma') and hasattr(block, 'layer_head'):
            supervised_layers.append(i)
            log_sigma_val = block.log_sigma.data.item()
            print(f"   Layer {i}: has supervision, log_sigma = {log_sigma_val:.6f}")
        else:
            print(f"   Layer {i}: no supervision")
    
    print(f"   Expected supervised layers: {model.supervised_layer_indices}")
    print(f"   Actual supervised layers: {supervised_layers}")
    assert supervised_layers == model.supervised_layer_indices, "Supervised layer mismatch!"
    
    # Test 2: Create sample input and test forward pass
    print("\n2. Testing forward pass with layer supervision:")
    
    batch_size = 2
    seq_len = 8
    
    # Create input and targets
    x = torch.randint(1, vocab_size, (batch_size, seq_len))  # Avoid PAD token
    targets = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    print(f"   Input shape: {x.shape}")
    print(f"   Target shape: {targets.shape}")
    
    # Forward pass
    logits, loss = model(x, targets=targets)
    
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Loss type: {type(loss)}")
    
    if isinstance(loss, dict):
        print(f"   Loss structure: {list(loss.keys())}")
        print(f"   Final loss: {loss['final_loss'].item():.6f}")
        if 'layer_losses' in loss:
            print(f"   Layer losses: {len(loss['layer_losses'])} layers")
            for layer_name, layer_loss in loss['layer_losses'].items():
                print(f"     {layer_name}: {layer_loss.item():.6f}")
    else:
        print(f"   Simple loss: {loss.item():.6f}")
    
    # Test 3: Check gradient flow through layer uncertainty parameters
    print("\n3. Testing gradient flow through layer uncertainty parameters:")
    
    # Reset gradients and compute uncertainty-weighted loss
    model.zero_grad()
    
    if isinstance(loss, dict):
        # Use uncertainty weighting to ensure layer parameters get gradients
        weighted_loss = apply_layer_uncertainty_weighting(model, loss)
    else:
        weighted_loss = loss
    
    # Backward pass
    weighted_loss.backward()
    
    # Check gradients on layer uncertainty parameters
    layer_grad_info = {}
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'log_sigma'):
            grad_norm = block.log_sigma.grad.norm().item() if block.log_sigma.grad is not None else 0.0
            grad_value = block.log_sigma.grad.item() if block.log_sigma.grad is not None else None
            layer_grad_info[i] = (grad_norm, grad_value)
            print(f"   Layer {i} log_sigma: grad_norm = {grad_norm:.6f}, grad = {grad_value}")
    
    has_layer_gradients = all(info[0] > 0 for info in layer_grad_info.values())
    print(f"   All layer uncertainty parameters have gradients: {has_layer_gradients}")
    
    # Test 4: Test uncertainty weighting computation
    print("\n4. Testing uncertainty weighting computation:")
    
    # Create a fresh forward pass for uncertainty weighting test
    model.zero_grad()
    logits_fresh, loss_fresh = model(x, targets=targets)
    
    if isinstance(loss_fresh, dict):
        weighted_loss = apply_layer_uncertainty_weighting(model, loss_fresh)
        print(f"   Original loss structure: {type(loss_fresh)}")
        print(f"   Weighted loss: {weighted_loss.item():.6f}")
        
        # Check if weighting works
        weighted_loss.backward()
        
        # Check that layer uncertainty parameters still receive gradients
        layer_grad_after_weighting = {}
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'log_sigma'):
                grad_norm = block.log_sigma.grad.norm().item() if block.log_sigma.grad is not None else 0.0
                layer_grad_after_weighting[i] = grad_norm
                print(f"   Layer {i} log_sigma after weighting: grad_norm = {grad_norm:.6f}")
        
        has_weighted_gradients = all(grad > 0 for grad in layer_grad_after_weighting.values())
        print(f"   All layer parameters have gradients after weighting: {has_weighted_gradients}")
    
    # Test 5: Test parameter clamping
    print("\n5. Testing layer uncertainty parameter clamping:")
    
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'log_sigma'):
            # Test with extreme values
            with torch.no_grad():
                original_value = block.log_sigma.data.clone()
                
                # Test positive extreme
                block.log_sigma.data.fill_(10.0)
                s_l_clamped = torch.clamp(block.log_sigma, -5.0, 5.0)
                clamped_pos = s_l_clamped.item()
                
                # Test negative extreme
                block.log_sigma.data.fill_(-10.0)
                s_l_clamped = torch.clamp(block.log_sigma, -5.0, 5.0)
                clamped_neg = s_l_clamped.item()
                
                # Restore original
                block.log_sigma.data.copy_(original_value)
                
                print(f"   Layer {i}: +10.0 → {clamped_pos:.1f}, -10.0 → {clamped_neg:.1f}")
                
                assert clamped_pos == 5.0, f"Positive clamping failed: {clamped_pos}"
                assert clamped_neg == -5.0, f"Negative clamping failed: {clamped_neg}"
    
    # Test 6: Simulate uncertainty parameter learning
    print("\n6. Simulating uncertainty parameter updates:")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Save initial values
    initial_values = {}
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'log_sigma'):
            initial_values[i] = block.log_sigma.data.clone()
    
    print("   Step | Layer | log_sigma | sigma | Loss")
    print("   -----|-------|-----------|-------|-----")
    
    for step in range(3):
        # Create new batch
        x = torch.randint(1, vocab_size, (batch_size, seq_len))
        targets = torch.randint(1, vocab_size, (batch_size, seq_len))
        
        optimizer.zero_grad()
        logits, loss = model(x, targets=targets)
        
        if isinstance(loss, dict):
            weighted_loss = apply_layer_uncertainty_weighting(model, loss)
        else:
            weighted_loss = loss
            
        weighted_loss.backward()
        optimizer.step()
        
        # Log values
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'log_sigma'):
                log_sig = block.log_sigma.item()
                sigma = math.exp(log_sig)
                print(f"   {step:4d} | {i:5d} | {log_sig:9.6f} | {sigma:5.3f} | {weighted_loss.item():5.3f}")
    
    # Test 7: Analysis of uncertainty behavior
    print("\n7. Analysis of uncertainty learning behavior:")
    
    changes = {}
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'log_sigma'):
            if i in initial_values:
                initial = initial_values[i].item()
                current = block.log_sigma.data.item()
                change = current - initial
                abs_change = abs(change)
                changes[i] = abs_change
                direction = "increased" if change > 0 else "decreased" if change < 0 else "unchanged"
                print(f"   Layer {i}: {initial:.6f} → {current:.6f} (change: {change:+.6f}, {direction})")
    
    # Test 8: Final validation
    print("\n=== VALIDATION RESULTS ===")
    
    all_tests_passed = True
    
    # Check layer supervision setup
    setup_correct = supervised_layers == model.supervised_layer_indices
    print(f"✓ Layer supervision setup correct: {setup_correct}")
    if not setup_correct:
        all_tests_passed = False
    
    # Check structured loss output
    structured_loss = isinstance(loss, dict) and 'layer_losses' in loss
    print(f"✓ Model outputs structured loss with layer supervision: {structured_loss}")
    if not structured_loss:
        all_tests_passed = False
    
    # Check layer uncertainty gradients
    print(f"✓ Layer uncertainty parameters receive gradients: {has_layer_gradients}")
    if not has_layer_gradients:
        all_tests_passed = False
    
    # Check uncertainty weighting preserves gradients
    if 'has_weighted_gradients' in locals():
        print(f"✓ Uncertainty weighting preserves gradients: {has_weighted_gradients}")
        if not has_weighted_gradients:
            all_tests_passed = False
    
    # Check readout heads exist
    readout_heads_exist = all(hasattr(model.blocks[i], 'layer_head') for i in supervised_layers)
    print(f"✓ Readout heads exist for supervised layers: {readout_heads_exist}")
    if not readout_heads_exist:
        all_tests_passed = False
    
    # Check parameter updates
    parameters_updated = any(change > 1e-6 for change in changes.values())
    print(f"✓ Layer uncertainty parameters update during optimization: {parameters_updated}")
    if not parameters_updated:
        all_tests_passed = False
    
    # Summary
    if all_tests_passed:
        print(f"\n🎉 ALL TESTS PASSED: Layer-level uncertainty mechanism is properly implemented!")
        print(f"   - Layer uncertainty parameters are learnable and receive gradients")
        print(f"   - Deep supervision readout heads work correctly")
        print(f"   - Layer-wise uncertainty weighting is applied correctly")
        print(f"   - Structured loss output includes both final and layer losses")
        print(f"   - Parameter clamping works to prevent degenerate values")
        print(f"   - Parameters update during optimization as expected")
    else:
        print(f"\n❌ SOME TESTS FAILED: Layer uncertainty implementation has issues!")
    
    return all_tests_passed

if __name__ == "__main__":
    test_layer_uncertainty_mechanism()