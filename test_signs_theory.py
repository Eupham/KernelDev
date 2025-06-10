import torch
import torch.nn.functional as F
import math

def generate_hadamard_signs(head_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate random signs for Hadamard transform."""
    return torch.randint(0, 2, (head_dim,), device=device, dtype=dtype) * 2 - 1

def hadamard_transform(x: torch.Tensor, signs: torch.Tensor = None) -> torch.Tensor:
    """Fast Walsh-Hadamard transform with random signs."""
    *batch_dims, head_dim = x.shape
    
    if head_dim & (head_dim - 1) != 0:
        raise ValueError(f"Head dimension {head_dim} must be a power of 2")
    
    if signs is None:
        signs = generate_hadamard_signs(head_dim, x.device, x.dtype)
    
    # Apply random signs
    x_signed = x * signs
    
    # Fast Walsh-Hadamard Transform
    result = x_signed
    stride = 1
    while stride < head_dim:
        result = result.view(*batch_dims, head_dim // (2 * stride), 2, stride)
        left, right = result.chunk(2, dim=-2)
        left, right = left.squeeze(-2), right.squeeze(-2)
        
        result = torch.stack([left + right, left - right], dim=-2)
        result = result.view(*batch_dims, head_dim)
        stride *= 2
    
    return result / math.sqrt(head_dim)

def simple_attention(q, k, v):
    """Simple attention computation."""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)

def test_same_vs_different_signs():
    """Test whether same signs vs different signs matter."""
    print("=== Testing Same vs Different Signs in Attention ===")
    
    # Create test data with outliers
    B, H, T, D = 1, 1, 4, 32
    torch.manual_seed(42)  # For reproducibility
    
    q = torch.randn(B, H, T, D, device='cuda') 
    k = torch.randn(B, H, T, D, device='cuda')
    v = torch.randn(B, H, T, D, device='cuda')
    
    # Add outliers
    q.view(-1)[::100] *= 10  # Every 100th element becomes outlier
    k.view(-1)[::100] *= 10
    
    print(f"Q outliers: min={q.min():.3f}, max={q.max():.3f}")
    print(f"K outliers: min={k.min():.3f}, max={k.max():.3f}")
    
    # Test 1: No transform (baseline)
    out_baseline = simple_attention(q, k, v)
    
    # Test 2: Same signs for Q and K
    signs = generate_hadamard_signs(D, q.device, q.dtype)
    q_same = hadamard_transform(q, signs)
    k_same = hadamard_transform(k, signs)
    out_same = simple_attention(q_same, k_same, v)
    
    print(f"Q transformed (same): min={q_same.min():.3f}, max={q_same.max():.3f}")
    print(f"K transformed (same): min={k_same.min():.3f}, max={k_same.max():.3f}")
    
    # Test 3: Different signs for Q and K
    signs_q = generate_hadamard_signs(D, q.device, q.dtype)
    signs_k = generate_hadamard_signs(D, k.device, k.dtype)
    q_diff = hadamard_transform(q, signs_q)
    k_diff = hadamard_transform(k, signs_k)
    out_diff = simple_attention(q_diff, k_diff, v)
    
    print(f"Q transformed (diff): min={q_diff.min():.3f}, max={q_diff.max():.3f}")
    print(f"K transformed (diff): min={k_diff.min():.3f}, max={k_diff.max():.3f}")
    
    # Compare results
    diff_same = F.mse_loss(out_baseline, out_same).item()
    diff_different = F.mse_loss(out_baseline, out_diff).item()
    diff_same_vs_diff = F.mse_loss(out_same, out_diff).item()
    
    print(f"\nMSE vs baseline:")
    print(f"Same signs: {diff_same:.8f}")
    print(f"Different signs: {diff_different:.8f}")
    print(f"Same vs Different: {diff_same_vs_diff:.8f}")
    
    # Test if transforms actually spread outliers
    def outlier_spread_test(original, transformed, name):
        orig_range = original.max() - original.min()
        trans_range = transformed.max() - transformed.min()
        print(f"{name} - Original range: {orig_range:.3f}, Transformed range: {trans_range:.3f}")
        
    outlier_spread_test(q, q_same, "Q (same signs)")
    outlier_spread_test(k, k_same, "K (same signs)")
    outlier_spread_test(q, q_diff, "Q (diff signs)")
    outlier_spread_test(k, k_diff, "K (diff signs)")

if __name__ == "__main__":
    test_same_vs_different_signs()
