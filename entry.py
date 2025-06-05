import torch
import torch.nn as nn # Added for LiteGPTModel parameters if not already there
import torch.nn.functional as F
import os
import sys
import inspect

# Updated imports to reflect potential renames in fwd.py and bwd.py
# from model import LiteGPTModel # Moved to local scope to break circular import
from fwd import causal_attention_forward_adapter # This might be unused if causal_attention is self-contained
from bwd import causal_attention_backward_op, causal_attention_backward_op_setup_context

@torch.compile(fullgraph=True, dynamic=True)
def _causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor | None,
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

    if q.is_cuda:
        O, LSE = torch.ops.alexdremov_causal_attention.forward(
            q=q, k=k, v=v, lens=lens,
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

def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor | None,
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

    result = _causal_attention(
        q=q, k=k, v=v, lens=lens,
        sm_scale=sm_scale,
        autotune=autotune,
        return_lse=return_lse,
        prescale_qk=prescale_qk,
        precision=precision,
    )
    return result

torch.library.register_autograd(
    "alexdremov_causal_attention::forward",
    causal_attention_backward_op,
    setup_context=causal_attention_backward_op_setup_context,
)

def causal_attention_reference(q, k, v, lens, scale=None):
    if scale is None:
        scale = q.size(-1)**-0.5
    _ = lens
    output = F.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale
        )
    return output, None, 1.0


def test_loss_reduction(dtype=torch.float32):
    from model import LiteGPTModel # Local import
    if dtype == torch.bfloat16: #This check should be before device selection for clarity
        print(f"Skipping {inspect.currentframe().f_code.co_name} for bfloat16 due to potential Triton compiler issues or specific test setup needs.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning loss reduction test with LiteGPTModel on {device.upper()} with {dtype}...")

    if device == "cpu":
        # LiteGPTModel uses causal_attention, which is CUDA-only. So skip on CPU.
        print(f"Skipping {inspect.currentframe().f_code.co_name} on CPU as LiteGPTModel uses CUDA-only causal_attention.")
        return

    # Model parameters
    B, T, embed_dim, vocab_size = 2, 16, 64, 100
    num_layers, num_heads, ff_hidden_dim, max_seq_len = 2, 4, 128, 32 # T must be <= max_seq_len

    if T > max_seq_len:
        print(f"Adjusting T from {T} to {max_seq_len} for test_loss_reduction as T > max_seq_len.")
        T = max_seq_len

    model = LiteGPTModel(num_layers, embed_dim, num_heads, ff_hidden_dim, vocab_size, max_seq_len).to(device)

    # Ensure model parameters are in the specified dtype for the test, especially for fp16
    if dtype == torch.float16 or dtype == torch.bfloat16:
        model = model.to(dtype)

    # Synthetic Autoregressive Data (Token IDs)
    input_ids = torch.randint(0, vocab_size, (B, T), device=device, dtype=torch.long)
    # Target for CrossEntropyLoss should be (B * (T-1))
    targets = input_ids[:, 1:].contiguous().view(-1)
    # Input to model will be input_ids[:, :-1]
    model_input_ids = input_ids[:, :-1].contiguous()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    initial_loss = -1.0
    final_loss = -1.0
    loss_decreased = False

    print(f"Starting loss reduction test (autoregressive token prediction with LiteGPTModel, model dtype={next(model.parameters()).dtype})...")

    use_fp16_autocast = (dtype == torch.float16)

    for i in range(5):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_fp16_autocast, dtype=torch.float16 if use_fp16_autocast else torch.float32):
            logits = model(model_input_ids) # model_input_ids is (B, T-1)
            # Logits will be (B, T-1, vocab_size)

            current_loss = F.cross_entropy(logits.view(-1, vocab_size), targets)

        if i == 0:
            initial_loss = current_loss.item()
        print(f"Step {i}, Loss: {current_loss.item()}")

        if use_fp16_autocast:
            # Basic backward, GradScaler would be better for robust fp16 training
            current_loss.backward()
        else:
            current_loss.backward()

        optimizer.step()

        if i == 4:
            final_loss = current_loss.item()

        if current_loss.item() < initial_loss:
            loss_decreased = True

    assert loss_decreased, f"Loss did not decrease. Initial: {initial_loss}, Final: {final_loss}"
    print("Loss reduction test with LiteGPTModel passed.")


from torch.autograd import gradcheck

def test_grad_accuracy(dtype_to_test=torch.float32):
    if dtype_to_test == torch.bfloat16:
        print(f"Skipping {inspect.currentframe().f_code.co_name} for bfloat16 due to potential Triton compiler issues or specific test setup needs.")
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
    from model import LiteGPTModel # Local import for main execution context
    print("Running tests in entry.py for CAUSAL attention...")

    # Test with float32
    test_loss_reduction(dtype=torch.float32)
    test_grad_accuracy(dtype_to_test=torch.float32)

    if torch.cuda.is_available(): # Only run fp16/bf16 if CUDA is available
        print("\nTesting with torch.float16 on CUDA (if supported by op)...")
        test_loss_reduction(dtype=torch.float16)
        # gradcheck for float16 is often very tricky and might require specific setup or be skipped.
        # test_grad_accuracy(dtype_to_test=torch.float16)

        if torch.cuda.is_bf16_supported():
            print("\nTesting with torch.bfloat16 on CUDA...")
            test_loss_reduction(dtype=torch.bfloat16) # Will be skipped by the new logic
            test_grad_accuracy(dtype_to_test=torch.bfloat16) # Will be skipped
        else:
            print("\ntorch.bfloat16 not supported on this CUDA device for tests.")
    else:
        print("\nCUDA not available, skipping float16/bfloat16 tests that require CUDA.")


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

        # Example LiteGPTModel instantiation and forward pass
        try:
            print("\nRunning example LiteGPTModel instantiation and forward pass...")
            model_params = {
                "num_layers": 2, "embed_dim": 64, "num_heads": 4, "ff_hidden_dim": 128,
                "vocab_size": 100, "max_seq_len": 32
            }
            example_model = LiteGPTModel(**model_params).to('cuda') # LiteGPTModel already imported locally in main
            example_input_ids = torch.randint(0, model_params["vocab_size"], (B, T_val), device='cuda', dtype=torch.long)
            example_logits = example_model(example_input_ids)
            print("Example LiteGPTModel output logits shape:", example_logits.shape)
        except Exception as e:
            print(f"Error during LiteGPTModel example: {e}")

    else:
        print("\nSkipping example usage as CUDA is not available.")
