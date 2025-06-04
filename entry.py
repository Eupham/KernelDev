import torch
import torch.nn.functional as F
import os
import sys

# It's assumed that fwd.py and bwd.py are in the same directory or accessible via PYTHONPATH
from fwd import attention_forward_adapter
from bwd import attention_backward_adapter_op, attention_backward_adapter_op_setup_context

@torch.compile(fullgraph=True, dynamic=True)
def _streaming_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor | None,
    context_size: int,
    back_contexts: int,
    sm_scale: float | None,
    autotune: bool,
    return_lse: bool,
    prescale_qk: bool,
    precision: str,
):
    requires_grad = any(i.requires_grad for i in (q, k, v))
    if sm_scale is None:
        HEAD_DIM = q.size(-1)
        sm_scale = HEAD_DIM**-0.5

    # The custom op is CUDA only. Check if CUDA is available.
    if not q.is_cuda and torch.cuda.is_available(): # q might be on CPU due to device selection logic
        # This case should ideally not be hit if device selection is consistent
        # but as a safeguard:
        raise RuntimeError("_streaming_attention custom op called with CPU tensors when CUDA is available. Ensure tensors are on CUDA.")

    if not torch.cuda.is_available() and q.is_cuda:
         raise RuntimeError("_streaming_attention custom op called with CUDA tensors when CUDA is NOT available.")

    # Only call the CUDA op if inputs are on CUDA and CUDA is available
    if q.is_cuda:
        O, LSE = torch.ops.alexdremov_streaming_attention.forward(
            q=q,
            k=k,
            v=v,
            lens=lens,
            context_size=context_size,
            back_contexts=back_contexts,
            sm_scale=sm_scale,
            autotune=autotune,
            prescale_qk=prescale_qk,
            return_lse=return_lse or requires_grad,
            precision=precision,
        )
    else:
        # Fallback or error if not on CUDA
        # For now, let's raise an error as the op is CUDA-specific
        raise RuntimeError("Streaming attention custom CUDA op called with CPU tensors or CUDA not available.")

    if return_lse:
        return O, LSE
    return O

def streaming_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor | None,
    context_size: int,
    back_contexts: int,
    sm_scale: float | None = None,
    autotune=False,
    return_lse=False,
    prescale_qk=False,
    precision="ieee",
):
    if not torch.compiler.is_compiling():
        for i in (q, k, v):
            torch._dynamo.mark_static(i, 0)
            torch._dynamo.mark_static(i, 1)
            torch._dynamo.mark_static(i, 3)
    if sm_scale is None:
        HEAD_DIM = q.size(-1)
        sm_scale = HEAD_DIM**-0.5

    # If not on CUDA, this will call _streaming_attention which will raise an error
    # as the custom op is CUDA-only.
    # The device of q, k, v should be consistent before calling this.
    result = _streaming_attention(
        q=q,
        k=k,
        v=v,
        lens=lens,
        context_size=context_size,
        back_contexts=back_contexts,
        sm_scale=sm_scale,
        autotune=autotune,
        return_lse=return_lse,
        prescale_qk=prescale_qk,
        precision=precision,
    )
    return result

torch.library.register_autograd(
    "alexdremov_streaming_attention::forward",
    attention_backward_adapter_op,
    setup_context=attention_backward_adapter_op_setup_context,
)

def streaming_attention_reference(
    q, k, v, context_size, back_contexts, lens, scale=None
):
    block_size = context_size
    left_context_blocks_count = back_contexts + 1
    T = q.shape[-2]

    block_idxes = torch.div(torch.arange(T, device=q.device), block_size, rounding_mode="floor")
    block_idxes_diff = block_idxes.unsqueeze(1) - block_idxes.unsqueeze(0)
    attn_mask = (block_idxes_diff >= 0) & (block_idxes_diff < left_context_blocks_count)

    if lens is not None:
        key_padding_mask = (
            torch.arange(T, device=q.device).unsqueeze(0) < lens.unsqueeze(-1)
        ).unsqueeze(-1)
        key_padding_mask_ref = key_padding_mask
        key_padding_mask = key_padding_mask & key_padding_mask.transpose(-1, -2)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) & key_padding_mask.unsqueeze(1)
        res_mask = key_padding_mask_ref.unsqueeze(1)
    else:
        attn_mask = attn_mask.to(q.device)
        res_mask = torch.tensor([True], device=q.device)

    output = F.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attn_mask, scale=scale
        )

    sparsity_fraction = attn_mask.sum().item() / attn_mask.numel() if attn_mask.numel() > 0 else 0.0
    return (
        output,
        res_mask,
        sparsity_fraction,
    )

def test_loss_reduction(dtype=torch.float32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning loss reduction test on {device.upper()} with {dtype}...")

    if device == "cpu":
        print("Skipping loss reduction test for custom op on CPU as it's CUDA-only.")
        # Optionally, run with streaming_attention_reference if a CPU test is desired
        # For now, just skip as the main goal is testing the custom op path.
        return

    B, H, T, HEAD_DIM = 2, 2, 64, 32
    context_size, back_contexts = 16, 1

    q = torch.randn(B, H, T, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, T, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, T, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
    target = torch.randn(B, H, T, HEAD_DIM, device=device, dtype=dtype)

    sm_scale_val = HEAD_DIM**-0.5
    optimizer = torch.optim.SGD([q, k, v], lr=0.01)

    initial_loss = -1.0
    final_loss = -1.0
    loss_decreased = False

    for i in range(5):
        optimizer.zero_grad()
        output = streaming_attention( # This will use the custom op path if on CUDA
            q, k, v,
            lens=None,
            context_size=context_size,
            back_contexts=back_contexts,
            sm_scale=sm_scale_val,
            autotune=False,
            return_lse=False,
            prescale_qk=False,
            precision="ieee"
        )

        loss = F.mse_loss(output, target)
        if i == 0:
            initial_loss = loss.item()
        print(f"Step {i}, Loss: {loss.item()}")

        loss.backward()
        optimizer.step()

        if i == 4: # Last step
            final_loss = loss.item()

        # Check for decrease relative to initial, not just previous step
        if loss.item() < initial_loss:
            loss_decreased = True

    assert loss_decreased, f"Loss did not decrease. Initial: {initial_loss}, Final: {final_loss}"
    print("Loss reduction test passed.")


from torch.autograd import gradcheck

def test_grad_accuracy(dtype_test=torch.float32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning gradient accuracy test on {device.upper()} with {dtype_test}...")

    if device == "cpu":
        print("Skipping gradient accuracy test for custom op on CPU as it's CUDA-only.")
        # Optionally, one could gradcheck streaming_attention_reference on CPU.
        return

    B, H, T, HEAD_DIM = 1, 1, 8, 16
    context_size, back_contexts = 4, 0

    q_double = torch.randn(B, H, T, HEAD_DIM, device=device, dtype=torch.double, requires_grad=True)
    k_double = torch.randn(B, H, T, HEAD_DIM, device=device, dtype=torch.double, requires_grad=True)
    v_double = torch.randn(B, H, T, HEAD_DIM, device=device, dtype=torch.double, requires_grad=True)

    sm_scale_val = HEAD_DIM**-0.5

    def gradcheck_func_wrapper(query_double, key_double, value_double):
        query = query_double.to(dtype_test)
        key = key_double.to(dtype_test)
        value = value_double.to(dtype_test)

        output = streaming_attention(
            query, key, value,
            lens=None,
            context_size=context_size,
            back_contexts=back_contexts,
            sm_scale=sm_scale_val,
            autotune=False,
            return_lse=True,
            prescale_qk=False,
            precision="ieee"
        )
        if isinstance(output, tuple):
            return output[0].to(torch.double)
        return output.to(torch.double)

    inputs_double = (q_double, k_double, v_double)
    is_correct = gradcheck(gradcheck_func_wrapper, inputs_double, eps=1e-3, atol=5e-3, nondet_tol=1e-7, fast_mode=False)
    assert is_correct, "Gradient check failed."
    print("Gradient accuracy test passed.")


if __name__ == "__main__":
    print("Running tests in entry.py...")

    # Test with float32
    test_loss_reduction(dtype=torch.float32)
    test_grad_accuracy(dtype_test=torch.float32)

    # Potentially test with bfloat16 if supported and desired
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        test_loss_reduction(dtype=torch.bfloat16)
        # Gradcheck with bfloat16 is often problematic due to its low precision.
        # It might require very loose tolerances or be skipped.
        # print("\nGradient accuracy test for bfloat16 might be unstable/skipped.")
        # test_grad_accuracy(dtype_test=torch.bfloat16)
    elif torch.cuda.is_available():
        print("\ntorch.bfloat16 not supported on this CUDA device for tests.")


    print("\nAll specified tests in entry.py finished.")

    if torch.cuda.is_available():
        B, H, T_val, D_val = 2, 1, 16, 32
        q_ex = torch.randn(B, H, T_val, D_val, device='cuda', dtype=torch.float32)
        k_ex = torch.randn(B, H, T_val, D_val, device='cuda', dtype=torch.float32)
        v_ex = torch.randn(B, H, T_val, D_val, device='cuda', dtype=torch.float32)
        context_ex, back_ex = 8, 1

        print("\nRunning example usage of streaming_attention...")
        output_ex_no_lse = streaming_attention(q_ex, k_ex, v_ex, None, context_ex, back_ex, return_lse=False)
        print("Example streaming_attention output shape (return_lse=False):", output_ex_no_lse.shape)

        output_ex_with_lse, lse_ex = streaming_attention(q_ex, k_ex, v_ex, None, context_ex, back_ex, return_lse=True)
        print("Example streaming_attention output shape (return_lse=True):", output_ex_with_lse.shape)
        print("Example LSE shape:", lse_ex.shape)
    else:
        print("\nSkipping example usage as CUDA is not available.")
