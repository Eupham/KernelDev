import unittest
import torch
import torch.nn.functional as F
import os

# Add project root to sys.path to allow importing original_kernel
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from original_kernel import flash_attention

# Determine if CUDA is available for tests
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

class TestOriginalKernel(unittest.TestCase):
    def helper_create_prefix_mask(self, seq_len, num_prefix_tokens=1, device=DEVICE):
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if num_prefix_tokens > 0:
            mask[:num_prefix_tokens] = True
        return mask

    def helper_construct_manual_sdpa_mask(self, seq_len, num_prefix_tokens=0, is_causal=True, device=DEVICE):
        attn_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        if is_causal:
            # Standard causal mask (upper triangle is masked)
            causal_mask_part = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            attn_mask.copy_(causal_mask_part)

        if num_prefix_tokens > 0:
            # For prefix tokens (queries), they can attend to all keys.
            # So, for rows corresponding to prefix queries, the mask should be all False.
            attn_mask[:num_prefix_tokens, :] = False

        # SDPA expects mask where True means "skip attention"
        return attn_mask


    def run_comparison(self, q, k, v, flash_causal, flash_is_prefix_mask_tensor,
                       sdpa_attn_mask, test_grads=True, atol=1e-6, rtol=1e-5):

        q_ref, k_ref, v_ref = q.clone().requires_grad_(test_grads), k.clone().requires_grad_(test_grads), v.clone().requires_grad_(test_grads)
        q_flash, k_flash, v_flash = q.clone().requires_grad_(test_grads), k.clone().requires_grad_(test_grads), v.clone().requires_grad_(test_grads)

        # Flash Attention
        # Assuming sm_scale is implicitly 1.0/sqrt(head_dim) in flash_attention if not provided
        # The flash_attention in original_kernel.py uses sm_scale = HEAD_DIM**-0.5 by default if None
        head_dim = q.shape[-1]
        sm_scale_flash = head_dim ** -0.5

        out_flash, _ = flash_attention(
            q_flash, k_flash, v_flash,
            causal=flash_causal,
            sm_scale=sm_scale_flash, # Pass the scale explicitly
            is_prefix_token_mask=flash_is_prefix_mask_tensor
        )

        # Reference SDPA
        # sm_scale for SDPA is passed directly. If flash_attention applies it internally, ensure consistency.
        # sdpa by default applies 1/sqrt(d_k) if scale is None.
        out_ref = F.scaled_dot_product_attention(
            q_ref, k_ref, v_ref,
            attn_mask=sdpa_attn_mask,
            is_causal=False, # We provide explicit mask, so is_causal should be False for SDPA
            scale=sm_scale_flash # Pass the same scale
        )

        self.assertTrue(torch.allclose(out_flash, out_ref, atol=atol, rtol=rtol),
                        f"Outputs differ. Max diff: {torch.max(torch.abs(out_flash - out_ref))}")

        if test_grads:
            # Test gradients
            grad_out = torch.randn_like(out_flash)
            out_flash.backward(grad_out)
            out_ref.backward(grad_out)

            self.assertTrue(torch.allclose(q_flash.grad, q_ref.grad, atol=atol, rtol=rtol), "dQ differ")
            self.assertTrue(torch.allclose(k_flash.grad, k_ref.grad, atol=atol, rtol=rtol), "dK differ")
            self.assertTrue(torch.allclose(v_flash.grad, v_ref.grad, atol=atol, rtol=rtol), "dV differ")

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA not available")
    def test_prefix_attention_causal_true(self):
        print("\nRunning test_prefix_attention_causal_true (CUDA)")
        B, H, T, D = 2, 4, 64, 64
        q = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)
        v = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)

        num_prefix = 1
        flash_prefix_mask = self.helper_create_prefix_mask(T, num_prefix_tokens=num_prefix)
        # SDPA mask: True means "skip". For prefix query, all keys are attended (mask is False).
        # For non-prefix queries, causal masking applies (upper triangle is True).
        sdpa_manual_mask = self.helper_construct_manual_sdpa_mask(T, num_prefix_tokens=num_prefix, is_causal=True)

        self.run_comparison(q, k, v, flash_causal=True, flash_is_prefix_mask_tensor=flash_prefix_mask,
                            sdpa_attn_mask=sdpa_manual_mask, atol=1e-2, rtol=1e-2) # Looser tolerance for fp16

        # Test with non-multiple of tile size
        T_odd = 70
        q_odd = torch.randn(B, H, T_odd, D, device=DEVICE, dtype=torch.float16)
        k_odd = torch.randn(B, H, T_odd, D, device=DEVICE, dtype=torch.float16)
        v_odd = torch.randn(B, H, T_odd, D, device=DEVICE, dtype=torch.float16)
        flash_prefix_mask_odd = self.helper_create_prefix_mask(T_odd, num_prefix_tokens=num_prefix)
        sdpa_manual_mask_odd = self.helper_construct_manual_sdpa_mask(T_odd, num_prefix_tokens=num_prefix, is_causal=True)
        self.run_comparison(q_odd, k_odd, v_odd, flash_causal=True, flash_is_prefix_mask_tensor=flash_prefix_mask_odd,
                            sdpa_attn_mask=sdpa_manual_mask_odd, atol=1e-2, rtol=1e-2)


    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA not available")
    def test_standard_causal_attention(self):
        print("\nRunning test_standard_causal_attention (CUDA)")
        B, H, T, D = 2, 4, 64, 64
        q = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)
        v = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)

        # flash_is_prefix_mask_tensor=None or all False
        flash_prefix_mask_all_false = self.helper_create_prefix_mask(T, num_prefix_tokens=0)

        # For SDPA with is_causal=True, attn_mask should be None
        sdpa_native_causal_mask = None

        # Run with flash_prefix_mask_all_false
        # Reference uses SDPA's built-in causal
        q_ref, k_ref, v_ref = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)
        q_flash, k_flash, v_flash = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)

        head_dim = q.shape[-1]
        sm_scale_val = head_dim ** -0.5

        out_flash, _ = flash_attention(q_flash, k_flash, v_flash, causal=True, sm_scale=sm_scale_val, is_prefix_token_mask=flash_prefix_mask_all_false)
        out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, attn_mask=None, is_causal=True, scale=sm_scale_val) # is_causal=True for SDPA

        self.assertTrue(torch.allclose(out_flash, out_ref, atol=1e-2, rtol=1e-2), "Outputs differ for all_false prefix mask")
        grad_out = torch.randn_like(out_flash)
        out_flash.backward(grad_out)
        out_ref.backward(grad_out)
        self.assertTrue(torch.allclose(q_flash.grad, q_ref.grad, atol=1e-2, rtol=1e-2), "dQ differ for all_false prefix mask")

        # Run with flash_is_prefix_mask_tensor=None
        q_flash, k_flash, v_flash = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True) # Reset grads
        out_flash_none, _ = flash_attention(q_flash, k_flash, v_flash, causal=True, sm_scale=sm_scale_val, is_prefix_token_mask=None)
        self.assertTrue(torch.allclose(out_flash_none, out_ref, atol=1e-2, rtol=1e-2), "Outputs differ for None prefix mask")
        # Grads for out_flash_none
        grad_out_none = torch.randn_like(out_flash_none)
        out_flash_none.backward(grad_out_none) # q_flash.grad will be updated
        # We need new q_ref, k_ref, v_ref for this grad comparison if we want to be super clean, or ensure out_ref backward is called with same grad_out_none
        # For simplicity, we assume the reference grads are already computed correctly with grad_out. Here we check consistency.
        # This part of the test is mainly about the forward pass consistency of None vs all_false mask.

    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA not available")
    def test_non_causal_attention(self):
        print("\nRunning test_non_causal_attention (CUDA)")
        B, H, T, D = 2, 4, 64, 64
        q = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)
        v = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)

        # For non-causal, prefix mask should ideally not change behavior if causal=False in flash_attention
        # as the base mask logic in kernel becomes `True` (attend all) before prefix logic.
        flash_prefix_mask = self.helper_create_prefix_mask(T, num_prefix_tokens=1)

        # SDPA: attn_mask=None and is_causal=False means dense attention.
        sdpa_no_mask = None

        # Run with flash_prefix_mask (it should be ignored by flash_attention if causal=False)
        # Reference uses SDPA's non-causal (dense attention)
        q_ref, k_ref, v_ref = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)
        q_flash, k_flash, v_flash = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)

        head_dim = q.shape[-1]
        sm_scale_val = head_dim ** -0.5

        out_flash, _ = flash_attention(q_flash, k_flash, v_flash, causal=False, sm_scale=sm_scale_val, is_prefix_token_mask=flash_prefix_mask)
        out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, attn_mask=None, is_causal=False, scale=sm_scale_val)

        self.assertTrue(torch.allclose(out_flash, out_ref, atol=1e-2, rtol=1e-2), "Outputs differ for non-causal")
        grad_out = torch.randn_like(out_flash)
        out_flash.backward(grad_out)
        out_ref.backward(grad_out)
        self.assertTrue(torch.allclose(q_flash.grad, q_ref.grad, atol=1e-2, rtol=1e-2), "dQ differ for non-causal")


if __name__ == '__main__':
    unittest.main()
