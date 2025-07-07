#!/usr/bin/env python3
"""
Register autograd function for flash attention.
This allows PyTorch to use our custom autograd function for backpropagation.
"""

import torch

# Register the backward function for flash_attention::forward
def flash_attention_backward_setup_context(ctx, q, k, v, lens, sm_scale, causal, autotune, return_lse, prescale_qk, precision, is_prefix_token_mask, output=None):
    """Set up the context for backward pass."""
    o, lse = output if output else (None, None)
    
    ctx.save_for_backward(q, k, v, lens)
    ctx.sm_scale = sm_scale
    ctx.causal = causal # Saved causal
    ctx.autotune = autotune
    ctx.prescale_qk = prescale_qk
    ctx.precision = precision
    ctx.o = o
    ctx.lse = lse
    ctx.is_prefix_token_mask = is_prefix_token_mask

def flash_attention_backward_adapter(ctx, grad_out, grad_lse=None):
    """Backward function for flash attention."""
    q, k, v, lens = ctx.saved_tensors
    is_prefix_token_mask = ctx.is_prefix_token_mask
    causal = ctx.causal # Retrieved causal
    
    # Call the backward operation
    dq, dk, dv = torch.ops.flash_attention.backward(
        q=q,
        k=k,
        v=v,
        lens=lens,
        o=ctx.o,
        lse=ctx.lse,
        do=grad_out,
        sm_scale=ctx.sm_scale,
        causal=causal, # Passed causal
        autotune=ctx.autotune,
        prescale_qk=ctx.prescale_qk,
        precision=ctx.precision,
        is_prefix_token_mask=is_prefix_token_mask,
    )
    
    # Return gradients for all inputs to forward
    # q, k, v, lens, sm_scale, causal, autotune, return_lse, prescale_qk, precision, is_prefix_token_mask
    # Grads for q, k, v are returned. Need 8 Nones for the rest.
    return dq, dk, dv, None, None, None, None, None, None, None, None

# Register the autograd formula with PyTorch
torch.library.register_autograd(
    "flash_attention::forward",
    flash_attention_backward_adapter,
    setup_context=flash_attention_backward_setup_context,
)

if __name__ == "__main__":
    print("Flash attention autograd function registered!")
