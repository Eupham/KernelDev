#!/usr/bin/env python3
"""
Register autograd function for flash attention.
This allows PyTorch to use our custom autograd function for backpropagation.
"""

import torch
import torch._dynamo # Added

# Register the backward function for flash_attention::forward
def flash_attention_backward_setup_context(ctx, inputs, output): # output is already correctly named by PyTorch
    # Unpack all arguments of the forward op from the 'inputs' tuple
    # Forward op schema: (q, k, v, lens, sm_scale, causal, autotune, return_lse, prescale_qk, precision, is_prefix_token_mask)
    q, k, v, lens, sm_scale, causal, autotune, return_lse, prescale_qk, precision, is_prefix_token_mask = inputs
    
    # Unpack the forward operator's output
    o, lse = output

    # Save tensors that are inputs to the forward function and needed for backward
    # is_prefix_token_mask is a tensor or None. If None, it's handled by the op.
    # If it's a tensor, it should be saved if needed by backward.
    # The backward op torch.ops.flash_attention.backward needs is_prefix_token_mask.
    tensors_to_save = [q, k, v]
    if lens is not None: # lens can be None
        tensors_to_save.append(lens)
    if is_prefix_token_mask is not None: # is_prefix_token_mask can be None
        tensors_to_save.append(is_prefix_token_mask)
    ctx.save_for_backward(*tensors_to_save)

    # Save other non-tensor parameters or parameters derived from output directly on ctx
    ctx.sm_scale = sm_scale
    ctx.causal = causal
    ctx.autotune = autotune
    # return_lse is not directly used by the backward adapter from ctx, but other parameters are
    ctx.prescale_qk = prescale_qk
    ctx.precision = precision
    # is_prefix_token_mask is now saved above if it's a tensor. If it's None, backward op handles it.
    # Storing it on ctx separately is fine for the adapter to retrieve.
    ctx.is_prefix_token_mask_tensor_saved = is_prefix_token_mask is not None # Flag to know if it was tensor
    if is_prefix_token_mask is None: # Store the None value on ctx if it was None.
         ctx.is_prefix_token_mask = None


    # Store o and lse from the output tuple (these are outputs of fwd, not inputs to save_for_backward)
    ctx.o = o
    ctx.lse = lse

@torch._dynamo.disable # Added decorator
def flash_attention_backward_adapter(ctx, grad_out, grad_lse=None):
    # Unpack saved tensors. The order matters.
    # Based on the new save_for_backward: q, k, v, [lens], [is_prefix_token_mask]
    saved_tensors_list = list(ctx.saved_tensors)
    q = saved_tensors_list.pop(0)
    k = saved_tensors_list.pop(0)
    v = saved_tensors_list.pop(0)

    lens = None
    # Check if lens was saved (it's conditional in setup_context)
    # A simple way is to check based on number of saved tensors or a flag from ctx.
    # Assuming lens is always saved if not None, and mask is saved if not None.
    # Let's rely on the structure: if lens was None, it wasn't added. If mask was None, it wasn't added.
    
    # For simplicity, let's assume fixed number of saved tensors if they are not None.
    # A robust way: save flags on ctx about what was saved.
    # Or, reconstruct based on what was saved.
    # For now, let's assume lens is always passed to save_for_backward (even if None, PyTorch handles it)
    # And is_prefix_token_mask is also always passed (even if None)
    # No, save_for_backward only takes tensors.

    # Let's retrieve them based on the conditional saving logic:
    current_idx = 0
    q = ctx.saved_tensors[current_idx]; current_idx +=1
    k = ctx.saved_tensors[current_idx]; current_idx +=1
    v = ctx.saved_tensors[current_idx]; current_idx +=1

    lens = None
    if hasattr(ctx, 'sm_scale'): # A bit of a hack, means full setup ran
        # Check if lens was intended to be saved. The original forward op takes lens.
        # If ctx.saved_tensors has more items, assume they are lens and then mask.
        # This part is tricky without knowing exactly how many tensors save_for_backward stores if one is None.
        # Let's assume forward op always provides lens and is_prefix_token_mask, even if None.
        # And setup_context saves them if they are not None.

        # Correct unpacking if lens and is_prefix_token_mask are conditionally saved:
        if ctx.is_prefix_token_mask_tensor_saved : # if mask was saved, lens must have been saved before it if lens was not None
            if len(ctx.saved_tensors) == 5 : # q, k, v, lens, mask
                lens = ctx.saved_tensors[3]
                is_prefix_token_mask_tensor = ctx.saved_tensors[4]
            elif len(ctx.saved_tensors) == 4: # q, k, v, mask (lens was None)
                lens = None
                is_prefix_token_mask_tensor = ctx.saved_tensors[3]
            else: # Should not happen with current logic
                is_prefix_token_mask_tensor = None # Fallback
        elif len(ctx.saved_tensors) == 4: # q, k, v, lens (mask was None)
            lens = ctx.saved_tensors[3]
            is_prefix_token_mask_tensor = None
        else: # q, k, v (both lens and mask were None)
             lens = None
             is_prefix_token_mask_tensor = None


    # Retrieve other parameters from ctx
    sm_scale = ctx.sm_scale
    causal = ctx.causal
    autotune = ctx.autotune
    prescale_qk = ctx.prescale_qk
    precision = ctx.precision
    # is_prefix_token_mask is retrieved from ctx.is_prefix_token_mask which was set directly
    # if the input was None, or use the tensor if it was saved.
    is_prefix_token_mask_to_pass = ctx.is_prefix_token_mask # This handles the None case
    if ctx.is_prefix_token_mask_tensor_saved:
         is_prefix_token_mask_to_pass = is_prefix_token_mask_tensor


    # Call the backward operation
    dq, dk, dv = torch.ops.flash_attention.backward(
        q=q,
        k=k,
        v=v,
        lens=lens, # Use unpacked lens
        o=ctx.o,
        lse=ctx.lse,
        do=grad_out,
        sm_scale=sm_scale,
        causal=causal,
        autotune=autotune,
        prescale_qk=prescale_qk,
        precision=precision,
        is_prefix_token_mask=is_prefix_token_mask_to_pass, # Use unpacked mask
    )
    
    # Return gradients for all inputs to forward
    # (q, k, v, lens, sm_scale, causal, autotune, return_lse, prescale_qk, precision, is_prefix_token_mask)
    # Grads for q, k, v are dq, dk, dv. Others are None. Total 11 inputs.
    return dq, dk, dv, None, None, None, None, None, None, None, None # 8 Nones

# Register the autograd formula with PyTorch
torch.library.register_autograd(
    "flash_attention::forward",
    flash_attention_backward_adapter,
    setup_context=flash_attention_backward_setup_context,
)

if __name__ == "__main__":
    print("Flash attention autograd function registered!")
