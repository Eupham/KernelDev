import torch
import torch.nn.functional as F
import time
import argparse
from typing import Optional
from torch.autograd import gradcheck # Import gradcheck

# Assuming flash_attention.py is in the same directory or accessible in PYTHONPATH
from flash_attention import self_attention

# Register a fake op for the case where flash_attention.py might not be fully built/imported in some CI test environments
# This is a fallback and should ideally not be needed if the environment is set up for custom ops.
try:
    torch.ops.alexdremov_flash_attention.forward
    # We don't expect a backward op to be registered anymore
    # torch.ops.alexdremov_flash_attention.backward
except AttributeError:
    print("Warning: Custom op 'alexdremov_flash_attention::forward' not found. This script might not run correctly.")
    if not hasattr(torch.ops, 'alexdremov_flash_attention'):
        # This is a placeholder, actual loading depends on how custom ops are built/registered
        # For Triton, importing flash_attention.py should handle JIT compilation and registration.
        pass


def generate_inputs(batch_size, num_heads, seq_len, head_dim, dtype, device, requires_grad=True, use_lens=False, is_causal=False):
    q = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device, requires_grad=requires_grad)
    k = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device, requires_grad=requires_grad)
    v = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device, requires_grad=requires_grad)

    lens = None
    if use_lens:
        min_len = seq_len // 2 if seq_len > 1 else 1
        lens = torch.randint(min_len, seq_len + 1, (batch_size,), device=device, dtype=torch.int32)
        if batch_size > 0 and not (lens == seq_len).any():
             lens[0] = seq_len

    attn_mask_sdpa = None
    if use_lens: # Only padding mask for SDPA if use_lens is true
        # SDPA expects True where elements are NOT masked.
        attn_mask_sdpa = (torch.arange(seq_len, device=device)[None, :] < lens[:, None]).unsqueeze(1).unsqueeze(2)
        # For SDPA, if both causal and padding, is_causal=True handles causal, attn_mask handles padding.
        # If only causal, is_causal=True, attn_mask=None.
        # If only padding, is_causal=False, attn_mask=padding_mask.

    return q, k, v, lens, attn_mask_sdpa


def test_accuracy(batch_size, num_heads, seq_len, head_dim, dtype, device, use_lens, is_causal, prescale, autotune_custom):
    print(f"\n--- Accuracy Test (Forward Pass Only) ---")
    print(f"Params: B={batch_size}, H={num_heads}, T={seq_len}, D={head_dim}, dtype={dtype}, device={device}, use_lens={use_lens}, is_causal={is_causal}, prescale={prescale}, autotune_custom={autotune_custom}")

    # Generate inputs, requires_grad=False as we only test forward
    q, k, v, lens_custom, attn_mask_sdpa = generate_inputs(batch_size, num_heads, seq_len, head_dim, dtype, device, requires_grad=False, use_lens=use_lens, is_causal=is_causal)

    sm_scale = head_dim ** -0.5
    try:
        output_custom = self_attention(q, k, v, lens_custom, sm_scale=sm_scale, autotune=autotune_custom, prescale=prescale, is_causal=is_causal)
    except Exception as e:
        print(f"Error in custom self_attention (forward pass): {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        # For F.scaled_dot_product_attention:
        # - is_causal flag handles causal masking.
        # - attn_mask (if provided) handles padding. It should be a boolean mask where True means "attend" and False means "mask".
        output_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_sdpa, is_causal=is_causal, scale=sm_scale)
    except Exception as e:
        print(f"Error in PyTorch SDPA (forward pass): {e}")
        import traceback
        traceback.print_exc()
        return False

    fwd_atol = 1e-5 if dtype == torch.float32 else 1e-2
    fwd_rtol = 1e-4 if dtype == torch.float32 else 1e-1
    try:
        fwd_match = torch.allclose(output_custom, output_ref, atol=fwd_atol, rtol=fwd_rtol)
        print(f"Forward output match with SDPA: {fwd_match}")
        if not fwd_match:
            print("Forward pass output does NOT match SDPA.")
            # print("Custom output sample:", output_custom[0,0,0,:min(5, output_custom.shape[-1])])
            # print("Ref output sample:", output_ref[0,0,0,:min(5, output_ref.shape[-1])])
            # print("Max difference:", (output_custom - output_ref).abs().max())
        return fwd_match
    except Exception as e:
        print(f"Error during forward pass torch.allclose: {e}")
        return False

def benchmark_speed(batch_size, num_heads, seq_len, head_dim, dtype, device, use_lens, is_causal, prescale, autotune_custom, num_repeats=20, num_warmup=5):
    print(f"\n--- Speed Benchmark (Forward Pass Only, using time.perf_counter) ---")
    print(f"Params: B={batch_size}, H={num_heads}, T={seq_len}, D={head_dim}, dtype={dtype}, device={device}, use_lens={use_lens}, is_causal={is_causal}, prescale={prescale}, autotune_custom={autotune_custom}")
    print(f"Warmup: {num_warmup} repeats, Main: {num_repeats} repeats.")

    q, k, v, lens_custom, attn_mask_sdpa = generate_inputs(batch_size, num_heads, seq_len, head_dim, dtype, device, requires_grad=False, use_lens=use_lens, is_causal=is_causal) # requires_grad=False
    sm_scale = head_dim ** -0.5

    # --- Custom Flash Attention (Forward Only) ---
    print("\nBenchmarking Custom Flash Attention (Forward):")
    try:
        fwd_times_custom = []
        for i in range(num_warmup + num_repeats):
            if i == num_warmup and device == 'cuda': torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = self_attention(q, k, v, lens_custom, sm_scale=sm_scale, autotune=autotune_custom, prescale=prescale, is_causal=is_causal)
            if device == 'cuda': torch.cuda.synchronize()
            end_time = time.perf_counter()
            if i >= num_warmup:
                fwd_times_custom.append(end_time - start_time)

        fwd_avg_custom = (sum(fwd_times_custom) / len(fwd_times_custom)) * 1000 if fwd_times_custom else float('nan')
        print(f"Custom Flash Attention: Forward: {fwd_avg_custom:.3f} ms")

    except Exception as e:
        print(f"Error benchmarking custom self_attention: {e}")
        import traceback
        traceback.print_exc()
        fwd_avg_custom = float('nan')

    # --- PyTorch SDPA (Forward Only) ---
    print("\nBenchmarking PyTorch SDPA (Forward):")
    try:
        fwd_times_ref = []
        for i in range(num_warmup + num_repeats):
            if i == num_warmup and device == 'cuda': torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_sdpa, is_causal=is_causal, scale=sm_scale)
            if device == 'cuda': torch.cuda.synchronize()
            end_time = time.perf_counter()
            if i >= num_warmup:
                fwd_times_ref.append(end_time - start_time)

        fwd_avg_ref = (sum(fwd_times_ref) / len(fwd_times_ref)) * 1000 if fwd_times_ref else float('nan')
        print(f"PyTorch SDPA:       Forward: {fwd_avg_ref:.3f} ms")

        if not (fwd_avg_custom == float('nan') or fwd_avg_ref == float('nan') or fwd_avg_ref == 0 or fwd_avg_custom == 0) :
            print(f"Custom Speedup vs SDPA (Fwd): {fwd_avg_ref/fwd_avg_custom:.2f}x")
    except Exception as e:
        print(f"Error benchmarking PyTorch SDPA: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Flash Attention Implementation Test and Benchmark")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--head_dim", type=int, default=64, help="Dimension of each attention head")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Data type")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on")
    parser.add_argument("--use_lens", action="store_true", help="Use variable sequence lengths (padding)")
    parser.add_argument("--is_causal", action="store_true", help="Enable causal masking")
    parser.add_argument("--prescale", action="store_true", help="Enable prescaling in custom attention (Q_scaled = Q * SM_SCALE)")
    parser.add_argument("--autotune_custom", action="store_true", help="Enable autotuning for custom Triton kernels")
    parser.add_argument("--skip_accuracy", action="store_true", help="Skip accuracy tests")
    parser.add_argument("--skip_benchmark", action="store_true", help="Skip speed benchmarks")
    parser.add_argument("--benchmark_repeats", type=int, default=20, help="Number of repeats for benchmark timing")
    parser.add_argument("--benchmark_warmup", type=int, default=5, help="Number of warmup repeats for benchmark timing")


    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Please run on CPU or install CUDA.")
        return
    if args.device == "cpu" and args.dtype in ["float16", "bfloat16"]:
        print(f"{args.dtype} is not well supported on CPU for this script. Please use float32 for CPU or run on CUDA.")
        # return # Or force float32

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    if not args.skip_accuracy:
        test_accuracy(args.batch_size, args.num_heads, args.seq_len, args.head_dim, torch_dtype, args.device, args.use_lens, args.is_causal, args.prescale, args.autotune_custom)

    if not args.skip_benchmark:
        benchmark_speed(args.batch_size, args.num_heads, args.seq_len, args.head_dim, torch_dtype, args.device, args.use_lens, args.is_causal, args.prescale, args.autotune_custom, num_repeats=args.benchmark_repeats, num_warmup=args.benchmark_warmup)

if __name__ == "__main__":
    main()
