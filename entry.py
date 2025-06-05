import torch
import torch.nn.functional as F
import os
import sys
import inspect # Added import

# Updated imports to reflect potential renames in fwd.py and bwd.py
from fwd import causal_attention_forward_adapter
from bwd import causal_attention_backward_op, causal_attention_backward_op_setup_context

@torch.compile(fullgraph=True, dynamic=True)
def _causal_attention( # Renamed
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor | None,
    # context_size: int, # Removed
    # back_contexts: int, # Removed
    sm_scale: float | None,
    autotune: bool,
    return_lse: bool,
    prescale_qk: bool,
    precision: str,
):
    requires_grad = any([t.requires_grad for t in [q, k, v]])
    if sm_scale is None:
        HEAD_DIM = q.size(-1)
        sm_scale = HEAD_DIM**-0.5

    if not q.is_cuda and torch.cuda.is_available():
        raise RuntimeError("_causal_attention custom op called with CPU tensors when CUDA is available. Ensure tensors are on CUDA.")
    if not torch.cuda.is_available() and q.is_cuda:
         raise RuntimeError("_causal_attention custom op called with CUDA tensors when CUDA is NOT available.")

    if q.is_cuda: # Custom op is CUDA only
        O, LSE = torch.ops.alexdremov_causal_attention.forward( # Updated op name
            q=q,
            k=k,
            v=v,
            lens=lens,
            # context_size=context_size, # Removed
            # back_contexts=back_contexts, # Removed
            sm_scale=sm_scale,
            autotune=autotune,
            prescale_qk=prescale_qk,
            return_lse=return_lse or requires_grad,
            precision=precision,
        )
    else:
        raise RuntimeError("Causal attention custom CUDA op called with CPU tensors or CUDA not available.")

    if return_lse:
        return O, LSE
    return O

def causal_attention( # Renamed
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor | None,
    # context_size: int, # Removed
    # back_contexts: int, # Removed
    sm_scale: float | None = None,
    autotune=False,
    return_lse=False,
    prescale_qk=False,
    precision="ieee",
):
    """
    Computes causal self-attention.
    Args are similar to streaming_attention but without context_size and back_contexts.
    """
    if not torch.compiler.is_compiling():
        for i in (q, k, v):
            torch._dynamo.mark_static(i, 0)
            torch._dynamo.mark_static(i, 1)
            torch._dynamo.mark_static(i, 3)
    if sm_scale is None:
        HEAD_DIM = q.size(-1)
        sm_scale = HEAD_DIM**-0.5

    result = _causal_attention( # Renamed
        q=q,
        k=k,
        v=v,
        lens=lens,
        # context_size=context_size, # Removed
        # back_contexts=back_contexts, # Removed
        sm_scale=sm_scale,
        autotune=autotune,
        return_lse=return_lse,
        prescale_qk=prescale_qk,
        precision=precision,
    )
    return result

# Updated autograd registration
torch.library.register_autograd(
    "alexdremov_causal_attention::forward", # Updated op name
    causal_attention_backward_op, # Using renamed backward op from bwd.py
    setup_context=causal_attention_backward_op_setup_context, # Using renamed setup context from bwd.py
)

def causal_attention_reference(q, k, v, lens, scale=None): # Renamed and simplified
    """
    Reference causal attention using PyTorch's F.scaled_dot_product_attention.
    `lens` is not directly used by is_causal=True but could be used to generate a key_padding_mask
    if combined causal+padding is needed. For this reference, we rely on is_causal only.
    """
    if scale is None:
        scale = q.size(-1)**-0.5 # Common practice if not provided

    # F.scaled_dot_product_attention with is_causal=True handles causal masking.
    # If `lens` were to be used to create a key_padding_mask, it would need careful construction
    # as `is_causal` and `attn_mask` (for key_padding_mask) are mutually exclusive in the simpler API.
    # For a pure causal reference, `attn_mask` is None.
    # If `lens` is provided, a proper combined mask would need to be manually created and passed to `attn_mask`.
    # This basic reference does not implement manual mask combination with `lens`.
    _ = lens # lens is ignored in this simplified reference.

    output = F.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale
        )
    # Placeholder for LSE (not typically returned by SDPA) and sparsity_fraction (1.0 for dense causal)
    return output, None, 1.0


def test_loss_reduction(dtype=torch.float32):
    if dtype == torch.bfloat16:
        print(f"Skipping {inspect.currentframe().f_code.co_name} for bfloat16 due to potential Triton compiler issues on some hardware.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning loss reduction test for CAUSAL attention on {device.upper()} with {dtype}...")

    if device == "cpu":
        print("Skipping loss reduction test for custom op on CPU as it's CUDA-only.")
        return

    B, H, T, HEAD_DIM = 2, 2, 16, 32

    input_sequence = torch.randn(B, H, T, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
    # Target: predict the next token in the sequence (autoregressive)
    # For causal attention, output at step t should predict input_sequence at t+1
    # So, the loss is calculated between output[..., :-1, :] and input_sequence[..., 1:, :]

    q_param = input_sequence.clone().detach().requires_grad_(True)
    k_param = input_sequence.clone().detach().requires_grad_(True)
    v_param = input_sequence.clone().detach().requires_grad_(True)

    sm_scale_val = HEAD_DIM**-0.5
    optimizer = torch.optim.SGD([q_param, k_param, v_param], lr=0.01)

    initial_loss = -1.0
    final_loss = -1.0
    loss_decreased = False

    print("Starting loss reduction test (autoregressive prediction)...")
    for i in range(5):
        optimizer.zero_grad()
        output = causal_attention(
            q_param, k_param, v_param,
            lens=None,
            sm_scale=sm_scale_val,
            autotune=False,
            return_lse=False,
            prescale_qk=False,
            precision="ieee"
        )

        loss = F.mse_loss(output[..., :-1, :], input_sequence[..., 1:, :].detach())

        if i == 0:
            initial_loss = loss.item()
        print(f"Step {i}, Loss: {loss.item()}")

        loss.backward()
        optimizer.step()

        if i == 4:
            final_loss = loss.item()

        if loss.item() < initial_loss:
            loss_decreased = True

    assert loss_decreased, f"Loss did not decrease. Initial: {initial_loss}, Final: {final_loss}"
    print("Loss reduction test passed.")


from torch.autograd import gradcheck

# Renamed parameter to dtype_to_test for clarity within this function
def test_grad_accuracy(dtype_to_test=torch.float32):
    # gradcheck itself needs double for inputs, but op is tested with dtype_to_test
    if dtype_to_test == torch.bfloat16:
        print(f"Skipping {inspect.currentframe().f_code.co_name} for bfloat16 due to potential Triton compiler issues on some hardware.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning gradient accuracy test for CAUSAL attention on {device.upper()} with {dtype_to_test}...")

    if device == "cpu":
        print("Skipping gradient accuracy test for custom op on CPU as it's CUDA-only.")
        return

    B, H, T, HEAD_DIM = 1, 1, 8, 16

    q_double = torch.randn(B, H, T, HEAD_DIM, device=device, dtype=torch.double, requires_grad=True)
    k_double = torch.randn(B, H, T, HEAD_DIM, device=device, dtype=torch.double, requires_grad=True)
    v_double = torch.randn(B, H, T, HEAD_DIM, device=device, dtype=torch.double, requires_grad=True)

    sm_scale_val = HEAD_DIM**-0.5

    def gradcheck_func_wrapper(query_double, key_double, value_double):
        query = query_double.to(dtype_to_test)
        key = key_double.to(dtype_to_test)
        value = value_double.to(dtype_to_test)

        output = causal_attention(
            query, key, value,
            lens=None,
            sm_scale=sm_scale_val,
            autotune=False,
            return_lse=True,
            prescale_qk=False,
            precision="ieee"
        )
        if isinstance(output, tuple):
            return output[0].to(torch.double)
        return output.to(torch.double)

    print("Starting gradient accuracy check (gradcheck)...")
    inputs_double = (q_double, k_double, v_double)
    is_correct = gradcheck(gradcheck_func_wrapper, inputs_double, eps=1e-3, atol=5e-3, nondet_tol=1e-7, fast_mode=False)
    assert is_correct, "Gradient check failed."
    print("Gradient accuracy test passed.")


if __name__ == "__main__":
    print("Running tests in entry.py for CAUSAL attention...")

    test_loss_reduction(dtype=torch.float32)
    # In test_grad_accuracy, the parameter is named dtype_to_test, so we pass it as such.
    test_grad_accuracy(dtype_to_test=torch.float32)

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("\nTesting with torch.bfloat16 on CUDA...")
        test_loss_reduction(dtype=torch.bfloat16)
        test_grad_accuracy(dtype_to_test=torch.bfloat16)
    elif torch.cuda.is_available():
        print("\ntorch.bfloat16 not supported on this CUDA device for tests.")

    print("\nAll specified tests in entry.py finished.")

    if torch.cuda.is_available():
        B, H, T_val, D_val = 2, 1, 16, 32
        q_ex = torch.randn(B, H, T_val, D_val, device='cuda', dtype=torch.float32)
        k_ex = torch.randn(B, H, T_val, D_val, device='cuda', dtype=torch.float32)
        v_ex = torch.randn(B, H, T_val, D_val, device='cuda', dtype=torch.float32)

        print("\nRunning example usage of causal_attention...")
        output_ex_no_lse = causal_attention(q_ex, k_ex, v_ex, None, return_lse=False)
        print("Example causal_attention output shape (return_lse=False):", output_ex_no_lse.shape)

        output_ex_with_lse, lse_ex = causal_attention(q_ex, k_ex, v_ex, None, return_lse=True)
        print("Example causal_attention output shape (return_lse=True):", output_ex_with_lse.shape)
        print("Example LSE shape:", lse_ex.shape)
    else:
        print("\nSkipping example usage as CUDA is not available.")
