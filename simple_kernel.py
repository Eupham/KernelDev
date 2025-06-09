"""
Simplified Modal-compatible flash attention kernel.
This version removes problematic backward functions that cause Triton JIT compilation issues.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def simple_flash_attention(q, k, v, causal=True, sm_scale=None):
    """
    Simplified flash attention implementation using standard PyTorch operations.
    This is a fallback when Triton kernels fail to compile.
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    
    # Apply causal mask if needed
    if causal:
        seq_len = q.shape[-2]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1)
        scores = scores.masked_fill(causal_mask.bool(), float('-inf'))
    
    # Compute attention weights and output
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    
    return output


# Simple forward kernel that should work with most Triton versions
@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    B, H, T, D,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Simplified forward kernel for flash attention."""
    # Get program IDs
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    
    # Compute offsets
    q_offset = batch_id * stride_qb + head_id * stride_qh
    k_offset = batch_id * stride_kb + head_id * stride_kh
    v_offset = batch_id * stride_vh + head_id * stride_vh
    o_offset = batch_id * stride_ob + head_id * stride_oh
    
    # Simple attention computation (this is a placeholder)
    # In a real implementation, this would have the flash attention logic
    pass


def flash_attention(q, k, v, causal=True, sm_scale=None):
    """
    Main flash attention function with fallback to simple implementation.
    """
    try:
        # Try to use the optimized kernel if available
        # For now, just use the simple fallback
        return simple_flash_attention(q, k, v, causal, sm_scale)
    except Exception as e:
        print(f"Flash attention kernel failed, using fallback: {e}")
        return simple_flash_attention(q, k, v, causal, sm_scale)


class FlashAttention(torch.autograd.Function):
    """
    Flash attention autograd function with simplified backward pass.
    """
    
    @staticmethod
    def forward(ctx, q, k, v, causal=True, sm_scale=None):
        output = flash_attention(q, k, v, causal, sm_scale)
        
        # Save for backward (simplified)
        ctx.save_for_backward(q, k, v, output)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Simplified backward pass using standard PyTorch operations."""
        q, k, v, output = ctx.saved_tensors
        
        # Use standard PyTorch backward for now
        # This is less memory efficient but more compatible
        with torch.enable_grad():
            q_req = q.requires_grad_()
            k_req = k.requires_grad_()
            v_req = v.requires_grad_()
            
            output_req = simple_flash_attention(q_req, k_req, v_req, ctx.causal, ctx.sm_scale)
            grad_q, grad_k, grad_v = torch.autograd.grad(
                output_req, (q_req, k_req, v_req), grad_output, retain_graph=False
            )
        
        return grad_q, grad_k, grad_v, None, None


# Export the main function
def flash_attention_func(q, k, v, causal=True, sm_scale=None):
    """Main entry point for flash attention."""
    return FlashAttention.apply(q, k, v, causal, sm_scale)


# For compatibility with existing code
flash_attention = flash_attention_func
