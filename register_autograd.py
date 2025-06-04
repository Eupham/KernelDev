#!/usr/bin/env python3
"""
Register autograd function for flash attention.
This allows PyTorch to use our custom autograd function for backpropagation.
"""

import torch

# Register the backward function for flash_attention::forward
def flash_attention_backward_setup_context(ctx, q, k, v, lens, sm_scale, autotune, return_lse, prescale_qk, precision, output=None):
    """Set up the context for backward pass."""
    o, lse = output if output else (None, None)
    
    ctx.save_for_backward(q, k, v, lens)
    ctx.sm_scale = sm_scale
    ctx.autotune = autotune
    ctx.prescale_qk = prescale_qk
    ctx.precision = precision
    ctx.o = o
    ctx.lse = lse

def flash_attention_backward_adapter(ctx, grad_out, grad_lse=None):
    """Backward function for flash attention."""
    q, k, v, lens = ctx.saved_tensors
    o, lse = ctx.outputs
    
    # Call the backward operation
    dq, dk, dv = torch.ops.flash_attention.backward(
        q=q,
        k=k,
        v=v,
        lens=lens,
        o=o,
        lse=lse,
        do=grad_out,
        sm_scale=ctx.sm_scale,
        autotune=ctx.autotune,
        prescale_qk=ctx.prescale_qk,
        precision=ctx.precision,
    )
    
    # Return gradients for all inputs to forward
    return dq, dk, dv, None, None, None, None, None, None

# Register the autograd formula with PyTorch
torch.library.register_autograd(
    "flash_attention::forward",
    flash_attention_backward_adapter,
    setup_context=flash_attention_backward_setup_context,
)

if __name__ == "__main__":
    print("Flash attention autograd function registered!")
