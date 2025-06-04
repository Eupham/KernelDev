import torch
import torch.nn.functional as F
import time
import argparse
from typing import Optional

# Assuming flash_attention.py is in the same directory or accessible in PYTHONPATH
from flash_attention import self_attention

# Register a fake op for the case where flash_attention.py might not be fully built/imported in some CI test environments
# This is a fallback and should ideally not be needed if the environment is set up for custom ops.
try:
    torch.ops.alexdremov_flash_attention.forward
    torch.ops.alexdremov_flash_attention.backward
except AttributeError:
    print("Warning: Custom ops 'alexdremov_flash_attention::forward/backward' not found. Registering dummy ops for basic script execution.")
    # This part is tricky because the actual registration is in flash_attention.py
    # If that file itself fails to import due to triton/cuda issues, this script won't run far.
    # This is more of a placeholder for thought. A real setup would ensure flash_attention.py is importable.
    if not hasattr(torch.ops, 'alexdremov_flash_attention'):
        torch.ops.load_library("") # This would need the path to the compiled .so if it were a C++ op.
                                   # For Triton JIT, it's about availability of Triton and CUDA.
        # This dynamic registration here is complex if flash_attention.py itself is the source of registration.
        # For now, we'll assume flash_attention.py handles its own op registration when imported.
        pass


def generate_inputs(batch_size, num_heads, seq_len, head_dim, dtype, device, requires_grad=True, use_lens=False, is_causal=False):
    q = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device, requires_grad=requires_grad)
    k = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device, requires_grad=requires_grad)
    v = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device, requires_grad=requires_grad)

    lens = None
    if use_lens:
        # Create somewhat realistic lengths, e.g., between seq_len/2 and seq_len
        min_len = seq_len // 2 if seq_len > 1 else 1
        lens = torch.randint(min_len, seq_len + 1, (batch_size,), device=device, dtype=torch.int32)
        # Ensure at least one element has max length for certain tests if needed, or ensure all are valid.
        if batch_size > 0 and not (lens == seq_len).any():
             lens[0] = seq_len # Ensure full context for at least one if seq_len is small

    # PyTorch SDPA requires attn_mask for causal or padding.
    attn_mask = None
    if is_causal and use_lens: # PyTorch SDPA needs combined mask
        # Create a causal mask: (T, T)
        causal_mask_matrix = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        # Create a padding mask: (B, 1, 1, T)
        padding_mask_matrix = (torch.arange(seq_len, device=device)[None, :] < lens[:, None]).unsqueeze(1).unsqueeze(2)
        # Combine: causal is True where attention is NOT allowed. SDPA wants True where allowed for mask.
        # So, causal_mask_matrix should be True for elements to be masked out.
        # Padding mask: True where valid.
        # SDPA attn_mask: True means value is NOT masked, False means value IS masked.
        # So, if causal_mask_matrix_sdpa = ~causal_mask_matrix
        # And padding_mask_matrix_sdpa = padding_mask_matrix
        # attn_mask = causal_mask_matrix_sdpa & padding_mask_matrix_sdpa (broadcasted)
        # This gets complicated. SDPA takes bool mask where True = NOT MASK.
        # Let's use SDPA's is_causal flag and generate a key_padding_mask if use_lens.

        # For SDPA: is_causal=True handles the causal part.
        # For padding, SDPA expects a boolean mask (B, H, T_q, T_kv) or (B, 1, T_q, T_kv)
        # where True means "do not mask" and False means "mask".
        # Or, a float mask where -inf means "mask".
        # Let's use boolean key padding mask.
        # (B, T_kv) -> (B, 1, 1, T_kv)
        attn_mask = (torch.arange(seq_len, device=device)[None, :] < lens[:, None]).unsqueeze(1).unsqueeze(2)
        # This attn_mask is for key padding. SDPA's is_causal handles the causal part separately.
        # If only causal, is_causal=True for SDPA, lens=None for custom.
        # If only padding, is_causal=False for SDPA, pass lens for custom, pass attn_mask for SDPA.
        # If both, is_causal=True for SDPA, pass lens for custom, pass attn_mask for SDPA.

    elif use_lens: # Only padding
        attn_mask = (torch.arange(seq_len, device=device)[None, :] < lens[:, None]).unsqueeze(1).unsqueeze(2)

    # If only causal, is_causal=True for our op. For SDPA, set is_causal=True. attn_mask=None.
    # If no lens and not causal, attn_mask=None for SDPA.

    return q, k, v, lens, attn_mask

def test_accuracy(batch_size, num_heads, seq_len, head_dim, dtype, device, use_lens, is_causal, prescale, autotune_custom):
    print(f"\n--- Accuracy Test ---")
    print(f"Params: B={batch_size}, H={num_heads}, T={seq_len}, D={head_dim}, dtype={dtype}, device={device}, use_lens={use_lens}, is_causal={is_causal}, prescale={prescale}, autotune_custom={autotune_custom}")

    q, k, v, lens_custom, attn_mask_sdpa = generate_inputs(batch_size, num_heads, seq_len, head_dim, dtype, device, True, use_lens, is_causal)

    # Our custom implementation
    # Detach inputs for separate backward if comparing intermediate grads too.
    q_cust, k_cust, v_cust = q.clone().requires_grad_(), k.clone().requires_grad_(), v.clone().requires_grad_()

    try:
        sm_scale = head_dim ** -0.5
        output_custom = self_attention(q_cust, k_cust, v_cust, lens_custom, sm_scale=sm_scale, autotune=autotune_custom, prescale=prescale, is_causal=is_causal)

        # Backward pass for custom
        # Create a dummy gradient for the output
        do = torch.randn_like(output_custom)
        output_custom.backward(do)

        dq_custom, dk_custom, dv_custom = q_cust.grad, k_cust.grad, v_cust.grad
    except Exception as e:
        print(f"Error in custom self_attention: {e}")
        # Print full traceback
        import traceback
        traceback.print_exc()
        return False

    # PyTorch's scaled_dot_product_attention
    q_ref, k_ref, v_ref = q.clone().requires_grad_(), k.clone().requires_grad_(), v.clone().requires_grad_()

    # For SDPA:
    # - if use_lens is True, attn_mask_sdpa is the key_padding_mask.
    # - if is_causal is True, SDPA's is_causal flag handles it.
    # - sm_scale is handled by default or can be passed.
    try:
        output_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, attn_mask=attn_mask_sdpa if use_lens else None, is_causal=is_causal and not use_lens, scale=sm_scale if not prescale else None)
        # If prescale is True for custom, it means Q was prescaled. SDPA doesn't have direct prescale.
        # If custom prescales Q, then SDPA should use scale=1.0.
        # For now, assume sm_scale is the common scaling factor. If custom prescales, this comparison needs adjustment.
        # The prescale in custom op means Q is multiplied by SM_SCALE * RCP_LN2.
        # SDPA applies SM_SCALE. If PRESCALE is true in custom, then custom's SM_SCALE input should be 1.0.
        # Let's adjust: if prescale is true for custom, then the `sm_scale` fed to it should be 1.0 for apples-to-apples.
        # OR, ensure the sm_scale factor is correctly interpreted by both.
        # The current `self_attention` wrapper calculates sm_scale if None. If prescale=True, this calculated sm_scale is still passed.
        # The Triton kernel then does `q_tile *= softmax_scale` if PRESCALE. `softmax_scale` is `SM_SCALE * RCP_LN2`.
        # This means if prescale=True, the effective scale is (SM_SCALE*RCP_LN2) * (SM_SCALE*RCP_LN2) if not careful.
        # Let's assume `prescale=True` means the `sm_scale` argument to `self_attention` is effectively `1.0 / RCP_LN2`
        # and the kernel only applies `q_tile *= RCP_LN2`.
        # This part is tricky. For now, we use the same `sm_scale` for both, and `prescale` is a custom op feature.
        # The comparison might show differences if `prescale` significantly alters scaling logic vs SDPA.

        # Backward pass for reference
        output_ref.backward(do) # Use same dO
        dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad
    except Exception as e:
        print(f"Error in PyTorch SDPA: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Comparison
    # Tolerances might need adjustment based on dtype
    fwd_atol = 1e-5 if dtype == torch.float32 else 1e-2
    fwd_rtol = 1e-4 if dtype == torch.float32 else 1e-1
    bwd_atol = 1e-5 if dtype == torch.float32 else 1e-2
    bwd_rtol = 1e-4 if dtype == torch.float32 else 1e-1

    # Mask outputs if lens were used for custom op, as SDPA might handle padding differently (e.g. explicit zeroing vs not computing)
    # Our custom op should correctly handle padding (output zero for padded tokens if q_mask is applied).
    # SDPA also zeros out padded query token outputs if key_padding_mask is correctly applied.

    try:
        fwd_match = torch.allclose(output_custom, output_ref, atol=fwd_atol, rtol=fwd_rtol)
        dq_match = torch.allclose(dq_custom, dq_ref, atol=bwd_atol, rtol=bwd_rtol)
        dk_match = torch.allclose(dk_custom, dk_ref, atol=bwd_atol, rtol=bwd_rtol)
        dv_match = torch.allclose(dv_custom, dv_ref, atol=bwd_atol, rtol=bwd_rtol)
    except Exception as e:
        print(f"Error during torch.allclose: {e}")
        return False

    print(f"Forward output match: {fwd_match}")
    print(f"dQ gradient match: {dq_match}")
    print(f"dK gradient match: {dk_match}")
    print(f"dV gradient match: {dv_match}")

    if not (fwd_match and dq_match and dk_match and dv_match):
        print("One or more accuracy checks failed.")
        # Optional: print parts of the tensors that don't match.
        # print("Custom output sample:", output_custom[0,0,0,:5])
        # print("Ref output sample:", output_ref[0,0,0,:5])
        # print("Difference output sample:", (output_custom - output_ref).abs().max())
        return False
    return True


def benchmark_speed(batch_size, num_heads, seq_len, head_dim, dtype, device, use_lens, is_causal, prescale, autotune_custom, num_repeats=10, num_warmup=3):
    print(f"\n--- Speed Benchmark ---")
    print(f"Params: B={batch_size}, H={num_heads}, T={seq_len}, D={head_dim}, dtype={dtype}, device={device}, use_lens={use_lens}, is_causal={is_causal}, prescale={prescale}, autotune_custom={autotune_custom}")
    print(f"Warmup: {num_warmup} repeats, Main: {num_repeats} repeats.")

    q, k, v, lens_custom, attn_mask_sdpa = generate_inputs(batch_size, num_heads, seq_len, head_dim, dtype, device, True, use_lens, is_causal)
    do = torch.randn_like(q) # Assuming dO has same shape as Q for this example, should be like V.
    if v.shape[-1] != q.shape[-1]: # If head_dim_v is different
        do = torch.randn((batch_size, num_heads, seq_len, v.shape[-1]), dtype=dtype, device=device)
    else:
        do = torch.randn_like(v)


    sm_scale = head_dim ** -0.5

    # --- Custom Flash Attention ---
    # Forward
    stmt_fwd_custom = "self_attention(q, k, v, lens_custom, sm_scale=sm_scale, autotune=autotune_custom, prescale=prescale, is_causal=is_causal)"
    # Backward
    # Setup for backward: run forward first
    setup_bwd_custom = "output_custom = self_attention(q, k, v, lens_custom, sm_scale=sm_scale, autotune=autotune_custom, prescale=prescale, is_causal=is_causal)"
    stmt_bwd_custom = "output_custom.backward(do, retain_graph=True)" # Use retain_graph=True if benchmarking repeatedly

    custom_globals = {'self_attention': self_attention, 'q': q, 'k': k, 'v': v, 'lens_custom': lens_custom, 'sm_scale': sm_scale, 'autotune_custom': autotune_custom, 'prescale': prescale, 'is_causal': is_causal, 'do': do}

    try:
        # Warmup for custom forward
        for _ in range(num_warmup):
            eval(stmt_fwd_custom, custom_globals)
        torch.cuda.synchronize()
        # Benchmark custom forward
        fwd_timer_custom = torch.utils.benchmark.Timer(stmt=stmt_fwd_custom, globals=custom_globals, num_threads=1)
        fwd_result_custom = fwd_timer_custom.timeit(num_repeats)
        fwd_avg_custom = fwd_result_custom.mean * 1000 # ms

        # Warmup for custom backward
        for _ in range(num_warmup):
            eval(setup_bwd_custom, custom_globals) # output_custom is created here
            eval(stmt_bwd_custom, {**custom_globals, 'output_custom': custom_globals['self_attention'](q,k,v,lens_custom,sm_scale=sm_scale,autotune=autotune_custom,prescale=prescale,is_causal=is_causal)}) # output_custom needs to be from the current scope
            # Clear grads for next warmup iter
            if q.grad is not None: q.grad.zero_()
            if k.grad is not None: k.grad.zero_()
            if v.grad is not None: v.grad.zero_()

        torch.cuda.synchronize()
        # Benchmark custom backward
        # Need to ensure 'output_custom' is correctly set up for each timing loop
        # Timer does its own looping, so setup needs to be part of it or done once if state is not changing.
        # For autograd, state (grads) does change.
        # A more robust way for backward is to wrap in a function.
        def custom_fwd_bwd_once():
            # Clear grads before each run
            if q.grad is not None: q.grad.zero_()
            if k.grad is not None: k.grad.zero_()
            if v.grad is not None: v.grad.zero_()

            output_custom = self_attention(q, k, v, lens_custom, sm_scale=sm_scale, autotune=autotune_custom, prescale=prescale, is_causal=is_causal)
            output_custom.backward(do)
            torch.cuda.synchronize() # Ensure backward is done

        def custom_bwd_only_after_fwd():
            # Assumes fwd has run and q,k,v grads are None or zeroed
            # Clear grads before each run
            if q.grad is not None: q.grad.zero_()
            if k.grad is not None: k.grad.zero_()
            if v.grad is not None: v.grad.zero_()
            # Re-run forward to get output_custom in the current grad tape context for THIS iteration
            output_custom = self_attention(q, k, v, lens_custom, sm_scale=sm_scale, autotune=autotune_custom, prescale=prescale, is_causal=is_causal)
            output_custom.backward(do)
            torch.cuda.synchronize()

        # Benchmark custom fwd+bwd
        timer_fwd_bwd_custom = torch.utils.benchmark.Timer(stmt="custom_fwd_bwd_once()", globals={'custom_fwd_bwd_once': custom_fwd_bwd_once})
        fwd_bwd_result_custom = timer_fwd_bwd_custom.timeit(num_repeats)
        fwd_bwd_avg_custom = fwd_bwd_result_custom.mean * 1000 # ms

        # Estimate backward time (less precise than dedicated bwd timer)
        bwd_avg_custom = fwd_bwd_avg_custom - fwd_avg_custom

        print(f"Custom Flash Attention: Forward: {fwd_avg_custom:.3f} ms | Backward (estimated): {bwd_avg_custom:.3f} ms | Fwd+Bwd: {fwd_bwd_avg_custom:.3f} ms")

    except Exception as e:
        print(f"Error benchmarking custom self_attention: {e}")
        import traceback
        traceback.print_exc()
        fwd_avg_custom, bwd_avg_custom, fwd_bwd_avg_custom = float('nan'), float('nan'), float('nan')


    # --- PyTorch SDPA ---
    # Forward
    stmt_fwd_ref = "F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_sdpa, is_causal=is_causal_sdpa, scale=sm_scale if not prescale else None)"
    # Backward
    setup_bwd_ref = "output_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_sdpa, is_causal=is_causal_sdpa, scale=sm_scale if not prescale else None)"
    stmt_bwd_ref = "output_ref.backward(do, retain_graph=True)"

    # SDPA's is_causal flag should be True if we want causal and there's no attn_mask for padding.
    # If attn_mask_sdpa is present (due to use_lens), SDPA's is_causal should be False if the mask already handles causality,
    # or True if the mask is only for padding and SDPA should still add causality.
    # The generate_inputs creates attn_mask_sdpa only for padding. So, if is_causal is true, SDPA's is_causal should also be true.
    is_causal_sdpa = is_causal

    ref_globals = {'F': F, 'q': q, 'k': k, 'v': v, 'attn_mask_sdpa': attn_mask_sdpa, 'is_causal_sdpa': is_causal_sdpa, 'sm_scale': sm_scale, 'prescale': prescale, 'do': do}

    try:
        # Warmup for ref forward
        for _ in range(num_warmup):
            eval(stmt_fwd_ref, ref_globals)
        torch.cuda.synchronize()
        # Benchmark ref forward
        fwd_timer_ref = torch.utils.benchmark.Timer(stmt=stmt_fwd_ref, globals=ref_globals, num_threads=1)
        fwd_result_ref = fwd_timer_ref.timeit(num_repeats)
        fwd_avg_ref = fwd_result_ref.mean * 1000 # ms

        def ref_fwd_bwd_once():
            if q.grad is not None: q.grad.zero_()
            if k.grad is not None: k.grad.zero_()
            if v.grad is not None: v.grad.zero_()
            output_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_sdpa, is_causal=is_causal_sdpa, scale=sm_scale if not prescale else None)
            output_ref.backward(do)
            torch.cuda.synchronize()

        # Benchmark ref fwd+bwd
        timer_fwd_bwd_ref = torch.utils.benchmark.Timer(stmt="ref_fwd_bwd_once()", globals={'ref_fwd_bwd_once': ref_fwd_bwd_once, 'q':q, 'k':k, 'v':v, 'attn_mask_sdpa':attn_mask_sdpa, 'is_causal_sdpa':is_causal_sdpa, 'sm_scale':sm_scale, 'prescale':prescale, 'do':do, 'F':F})
        fwd_bwd_result_ref = timer_fwd_bwd_ref.timeit(num_repeats)
        fwd_bwd_avg_ref = fwd_bwd_result_ref.mean * 1000 # ms
        bwd_avg_ref = fwd_bwd_avg_ref - fwd_avg_ref

        print(f"PyTorch SDPA:       Forward: {fwd_avg_ref:.3f} ms | Backward (estimated): {bwd_avg_ref:.3f} ms | Fwd+Bwd: {fwd_bwd_avg_ref:.3f} ms")

        if fwd_avg_custom is not float('nan') and fwd_avg_ref is not float('nan') and fwd_avg_ref > 0:
            print(f"Custom Speedup vs SDPA (Fwd): {fwd_avg_ref/fwd_avg_custom:.2f}x")
        if fwd_bwd_avg_custom is not float('nan') and fwd_bwd_avg_ref is not float('nan') and fwd_bwd_avg_ref > 0 :
            print(f"Custom Speedup vs SDPA (Fwd+Bwd): {fwd_bwd_avg_ref/fwd_bwd_avg_custom:.2f}x")

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
