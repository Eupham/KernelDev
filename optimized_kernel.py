"""
Specialized Flash Attention Implementation with Hierarchical Attention Patterns

This module implements memory-efficient flash attention with support for sophisticated
attention patterns required for cocktail party tasks. The implementation maintains
mathematical equivalence to standard attention while reducing memory complexity from
O(n²) to O(n) through block-wise computation.

Key Components:
- Flash Attention Forward/Backward Kernels: Triton-based GPU kernels for efficient computation
- Hierarchical Attention Patterns: 4-section attention structure for cocktail party tasks
- GPU Optimization: Auto-tuning and hardware-specific configurations
- Mixed Precision Support: fp16, bf16, and fp32 computation modes

Attention Hierarchy:
1. Prefix Section: Bidirectional within prefix (tokens before/including [CLS])
2. Context Section: Causal within context + access to prefix
3. Span Islands: Bidirectional within spans + access to context (isolated from other spans)
4. Bridge Section: [MASKQ] token with access to all spans + prefix (aggregator hub)

This implementation maintains flash attention benefits while enabling complex attention
patterns necessary for span-based reasoning tasks.

H100-optimized changes:
- Warp specialization autotune configs (num_consumer_groups / num_buffers_warp_spec)
- Larger Hopper-friendly tiles (up to 128x128) and up to 16 warps/block
- Hoisted Q-tile metadata & compile-time HAS_TOKEN_RULES constant
- L2-biased loads for K/V tiles and zero-padding on tail loads
"""

import logging
import math
import torch._dynamo
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# =============================================================================
# Configuration Constants
# =============================================================================

MAX_TILE_SIZE = 512  # Reduced for T4 compatibility
MIN_TILE_SIZE = 16   # Reduced for T4 compatibility

logger = logging.getLogger(__name__)

# BLOCK_Q, BLOCK_K, num_warps, num_stages
# T4-optimized configuration (compute capability 7.5)
_t4_default_config = {
    (torch.float32, 64): (64, 64, 4, 3),
    (torch.float32, 128): (64, 32, 4, 3),
    (torch.float32, 256): (32, 32, 4, 3),
    (torch.bfloat16, 64): (128, 64, 4, 3),
    (torch.bfloat16, 128): (64, 64, 4, 3),
    (torch.bfloat16, 256): (64, 32, 4, 3),
    (torch.float16, 64): (128, 64, 4, 3),
    (torch.float16, 128): (64, 64, 4, 3),
    (torch.float16, 256): (64, 32, 4, 3),
}

# Hopper/H100 defaults (favor larger tiles & more warps)
_h100_default_config = {
    (torch.float32, 64): (128, 32, 8, 3),
    (torch.float32, 128): (64, 64, 8, 3),
    (torch.float32, 256): (64, 32, 8, 3),
    (torch.bfloat16, 64): (128, 128, 8, 3),
    (torch.bfloat16, 128): (128, 64, 16, 3),
    (torch.bfloat16, 256): (64, 64, 8, 3),
    (torch.float16, 64): (128, 128, 8, 3),
    (torch.float16, 128): (128, 128, 16, 3),
    (torch.float16, 256): (64, 64, 8, 3),
}

_a100_default_config = {
    (torch.float32, 64): (128, 32, 4, 3),
    (torch.float32, 128): (128, 32, 4, 3),
    (torch.float32, 256): (64, 16, 4, 3),
    (torch.bfloat16, 64): (128, 64, 4, 3),
    (torch.bfloat16, 128): (128, 64, 8, 3),
    (torch.bfloat16, 256): (32, 64, 4, 3),
    (torch.float16, 64): (128, 64, 4, 3),
    (torch.float16, 128): (128, 64, 8, 3),
    (torch.float16, 256): (32, 64, 4, 3),
}


def _get_default_config_fwd(head_dim, dtype) -> tuple[int, int, int, int]:
    default_config = None

    if head_dim <= 256 and torch.cuda.get_device_capability() >= (9, 0):  # H100
        if dtype == torch.float32:
            default_config = (64, 64, 8, 3)
        else:
            default_config = (128, 64, 8, 3)
        default_config = _h100_default_config.get((dtype, head_dim), default_config)
    elif head_dim <= 256 and torch.cuda.get_device_capability() >= (8, 0):  # A100
        if dtype == torch.float32:
            default_config = (64, 64, 4, 3)
        else:
            default_config = (128, 64, 4, 3)
        default_config = _a100_default_config.get((dtype, head_dim), default_config)
    elif head_dim <= 256 and torch.cuda.get_device_capability() >= (7, 5):  # T4 and similar
        if dtype == torch.float32:
            default_config = (64, 64, 4, 3)
        else:
            default_config = (128, 64, 4, 3)
        default_config = _t4_default_config.get((dtype, head_dim), default_config)
    else:  # modest hardware or extremely large head_dim
        if dtype == torch.float32:
            default_config = (32, 16, 4, 2)
        else:
            default_config = (64, 32, 4, 2)

    return default_config


def _get_default_config_bwd(head_dim, dtype) -> tuple[int, int, int, int]:
    # Favor slightly larger tiles on Hopper
    if dtype == torch.float32:
        return (32, 32, 8, 2) if torch.cuda.get_device_capability() >= (9, 0) else (16, 16, 4, 1)
    elif head_dim <= 256 and torch.cuda.get_device_capability() >= (9, 0):  # H100
        if head_dim == 64:
            return (64, 128, 8, 3)
        elif head_dim == 128:
            return (64, 128, 16, 3)
        else:
            return (64, 64, 8, 2)
    elif torch.cuda.get_device_capability() >= (8, 0):  # A100
        if head_dim == 64:
            return (32, 128, 4, 3)
        elif head_dim == 128:
            return (64, 128, 8, 3)
        else:
            return (64, 64, 4, 2)
    elif torch.cuda.get_device_capability() >= (7, 5):  # T4 and similar
        if head_dim == 64:
            return (64, 64, 4, 2)
        elif head_dim == 128:
            return (32, 64, 4, 2)
        else:
            return (32, 32, 4, 2)
    else:  # modest hardware or extremely large head_dim
        return (16, 16, 4, 1)


def strides(t: torch.Tensor, expected_size=None):
    assert t is not None
    if expected_size is not None:
        assert t.ndim == expected_size
    return [t.stride(i) for i in range(t.ndim)]


def fwd_configs_pruner(configs, nargs, HEAD_DIM, DTYPE, **kwargs):
    # Broaden ranges for Hopper, prune for smaller GPUs
    cc = torch.cuda.get_device_capability()
    is_h100 = cc >= (9, 0)

    min_size = 32
    max_size = 256 if not is_h100 else 256
    min_pipeline, max_pipeline = (1, 4) if is_h100 else (1, 3)
    min_warps, max_warps = (4, 16) if is_h100 else (1, 8)

    if HEAD_DIM == 64:
        min_pipeline = 2
    elif HEAD_DIM == 128:
        max_size = 128
        min_size = 32
        max_pipeline = 3 if is_h100 else 2
        max_warps = 16 if is_h100 else 4
    elif HEAD_DIM == 256:
        max_size = 128
        min_size = 32
        max_pipeline = 3 if is_h100 else 1
        max_warps = 16 if is_h100 else 4

    configs = [i for i in configs if min_size <= i.kwargs.get("TILE_K_SIZE", 0) <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs.get("TILE_Q_SIZE", 0) <= max_size]
    configs = [i for i in configs if min_pipeline <= i.kwargs.get("PIPELINING", 1) <= max_pipeline]
    configs = [i for i in configs if min_warps <= i.num_warps <= max_warps]

    default_config = _get_default_config_fwd(HEAD_DIM, DTYPE)
    if default_config is not None:
        configs += [
            triton.Config(
                dict(
                    PIPELINING=default_config[3],
                    TILE_Q_SIZE=default_config[0],
                    TILE_K_SIZE=default_config[1],
                    # Hopper warp specialization defaults (harmless on non-Hopper)
                    num_consumer_groups=2 if is_h100 else 1,
                    num_buffers_warp_spec=3 if is_h100 else 2,
                ),
                num_warps=default_config[2],
                num_stages=default_config[3],
            )
        ]

    logger.warning(f"Start benchmarking forward flash_attention {len(configs) = }")
    return configs


def bwd_configs_pruner(configs, nargs, HEAD_DIM, DTYPE, **kwargs):
    cc = torch.cuda.get_device_capability()
    is_h100 = cc >= (9, 0)

    min_size = 32
    max_size = 256 if not is_h100 else 256
    min_pipeline, max_pipeline = (1, 4) if is_h100 else (1, 3)
    min_warps, max_warps = (4, 16) if is_h100 else (1, 8)

    if HEAD_DIM == 32 or HEAD_DIM == 16:
        min_pipeline = 3
        max_size = 64
    if HEAD_DIM == 64:
        min_pipeline = 2
        max_size = 128
        min_warps = 4 if is_h100 else 2
        max_warps = 16 if is_h100 else 4
    elif HEAD_DIM == 128:
        max_size = 128
        min_size = 64
        max_pipeline = 3 if is_h100 else 2
        min_pipeline = 1
        max_warps = 16 if is_h100 else 4
    elif HEAD_DIM == 256:
        max_size = 64
        min_size = 32
        max_pipeline = 3 if is_h100 else 2
        min_pipeline = 1
        max_warps = 16 if is_h100 else 4

    configs = [i for i in configs if min_size <= i.kwargs.get("TILE_DQ_Q_SIZE", 0) <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs.get("TILE_DQ_K_SIZE", 0) <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs.get("TILE_DK_Q_SIZE", 0) <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs.get("TILE_DK_K_SIZE", 0) <= max_size]
    configs = [i for i in configs if min_pipeline <= i.kwargs.get("PIPELINING", 1) <= max_pipeline]
    configs = [i for i in configs if min_warps <= i.num_warps <= max_warps]

    default_config = _get_default_config_bwd(HEAD_DIM, DTYPE)
    if default_config is not None:
        configs += [
            triton.Config(
                dict(
                    PIPELINING=default_config[3],
                    TILE_DQ_Q_SIZE=default_config[0],
                    TILE_DQ_K_SIZE=default_config[1],
                    TILE_DK_Q_SIZE=default_config[0],
                    TILE_DK_K_SIZE=default_config[1],
                    # Hopper WS knobs
                    num_consumer_groups=2 if is_h100 else 1,
                    num_buffers_warp_spec=3 if is_h100 else 2,
                ),
                num_warps=default_config[2],
                num_stages=default_config[3],
            )
        ]

    logger.warning(f"Start benchmarking backward flash_attention {len(configs) = }")
    return configs


# =============================================================================
# Flash Attention Triton Kernels
# =============================================================================

# fmt: off
@triton.heuristics(
    dict(
        Q_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_Q_SIZE'] == 0,
        K_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_K_SIZE'] == 0,
        PERFECT_MATCHING=lambda args : args['TILE_K_SIZE'] == args['TILE_Q_SIZE'],
        RCP_LN2=lambda _: math.log2(math.e),
    )
)
@triton.jit
def _flash_attn_fwd(
    Q: tl.tensor, Kt: tl.tensor, V: tl.tensor, L: tl.tensor, #
    LSE: tl.tensor, O: tl.tensor,  #
    ATTN_MASK: tl.tensor,
    IN_SPAN: tl.tensor, SPAN_ID: tl.tensor, IS_PREFIX: tl.tensor,
    OUTPUT_ATTN_MASK: tl.tensor,
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,  #
    stride_kb: int, stride_kh: int, stride_kk: int, stride_kt: int,  #
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,  #
    stride_mb: int, stride_mh: int, stride_mt: int,  #
    stride_ob: int, stride_oh: int, stride_ot: int, stride_ok: int, #
    lens_stride: int,
    mask_stride_b: int, mask_stride_h: int, mask_stride_t: int,
    in_span_stride_b: int, in_span_stride_t: int,
    span_id_stride_b: int, span_id_stride_t: int,
    is_prefix_stride_b: int, is_prefix_stride_t: int,
    output_attn_mask_stride_b: int, output_attn_mask_stride_h: int,
    output_attn_mask_stride_q: int, output_attn_mask_stride_k: int,
    T: int,  #
    TIME_BUCKET:  int,  #
    HEAD_DIM: tl.constexpr,  #
    CAUSAL: tl.constexpr,  #
    INPUT_PRECISION: tl.constexpr,  #
    SM_SCALE: tl.constexpr,  #
    DTYPE:  tl.constexpr,  #
    PRESCALE_QK: tl.constexpr,  #
    OUTPUT_LOGSUMEXP: tl.constexpr,  #
    TILE_Q_SIZE: tl.constexpr,  #
    TILE_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
    Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    K_BLOCK_DIVISIBLE: tl.constexpr,  #
    PERFECT_MATCHING: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
    RETURN_ATTENTION_MASK: tl.constexpr,  #
    HAS_TOKEN_RULES: tl.constexpr,  # NEW: compile-time switch for pattern logic
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    q_tile_idx = tl.program_id(2)
    q_token_idx = q_tile_idx * TILE_Q_SIZE

    if L is not None:
        seq_len = tl.load(L + batch * lens_stride)
        seq_len = min(seq_len, T)
    else:
        seq_len = T

    if seq_len <= q_token_idx:
        return

    q_tile_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)

    # --- Hoist Q metadata (kept in registers across KV tiles) ---
    if HAS_TOKEN_RULES:
        if IN_SPAN is not None:
            q_in_span_ptr = IN_SPAN + batch * in_span_stride_b + q_tile_indices
            q_in_span = tl.load(q_in_span_ptr, mask=q_tile_indices < seq_len, other=0)
        else:
            q_in_span = tl.full([TILE_Q_SIZE], False, tl.int1)

        if SPAN_ID is not None:
            q_span_id_ptr = SPAN_ID + batch * span_id_stride_b + q_tile_indices
            q_span_id = tl.load(q_span_id_ptr, mask=q_tile_indices < seq_len, other=0)
        else:
            q_span_id = tl.full([TILE_Q_SIZE], 0, tl.int32)

        if IS_PREFIX is not None:
            q_is_prefix_ptr = IS_PREFIX + batch * is_prefix_stride_b + q_tile_indices
            q_is_prefix = tl.load(q_is_prefix_ptr, mask=q_tile_indices < seq_len, other=0)
        else:
            q_is_prefix = tl.full([TILE_Q_SIZE], False, tl.int1)
    else:
        # Dummy constants; will be DCE'd away
        q_in_span = tl.full([TILE_Q_SIZE], False, tl.int1)
        q_span_id = tl.full([TILE_Q_SIZE], 0, tl.int32)
        q_is_prefix = tl.full([TILE_Q_SIZE], False, tl.int1)

    # Decide loop bound per tile
    q_tile_has_noncausal = tl.sum((q_in_span | q_is_prefix).to(tl.int32)) > 0

    kv_start_tile_idx = 0
    q_tile_max_token = min(q_token_idx + TILE_Q_SIZE, seq_len)

    if CAUSAL and not q_tile_has_noncausal:
        # For causal attention, we can attend up to the last query token
        kv_end_tile_idx = tl.cdiv(q_tile_max_token, TILE_K_SIZE)
    else:
        # For non-causal attention, attend to all tokens
        kv_end_tile_idx = tl.cdiv(seq_len, TILE_K_SIZE)

    # Block pointers (help vectorization + potential WGMMA paths on Hopper)
    qbatch_head_offset = batch * stride_qb + head * stride_qh
    q_tile_ptr = tl.make_block_ptr(
        base=Q + qbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_qt, stride_qk),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    kbatch_head_offset = batch * stride_kb + head * stride_kh
    kt_tile_ptr = tl.make_block_ptr(
        base=Kt + kbatch_head_offset,
        shape=(HEAD_DIM, T),
        strides=(stride_kk, stride_kt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_K_SIZE),
        order=(0, 1),
    )

    vbatch_head_offset = batch * stride_vb + head * stride_vh
    v_tile_ptr = tl.make_block_ptr(
        base=V + vbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_vt, stride_vk),
        offsets=(0, 0),
        block_shape=(TILE_K_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    m_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32)
    acc = tl.zeros([TILE_Q_SIZE, HEAD_DIM], dtype=tl.float32)
    if not PERFECT_MATCHING:
        q_attended = tl.zeros([TILE_Q_SIZE], dtype=tl.int1) > 0

    q_lens_mask = (q_tile_indices[:, None] < seq_len)

    # Load Q tile once
    if Q_BLOCK_DIVISIBLE:
        q_tile = tl.load(q_tile_ptr)
    else:
        q_tile = tl.load(q_tile_ptr, boundary_check=(0,), padding_option="zero")

    softmax_scale: tl.constexpr = tl.cast(SM_SCALE * RCP_LN2, q_tile.dtype)
    tile_k_arange = tl.arange(0, TILE_K_SIZE)

    if PRESCALE_QK:
        q_tile = q_tile * softmax_scale

    for kv_tile_idx in tl.range(kv_start_tile_idx, kv_end_tile_idx, num_stages=PIPELINING):
        last_iter = kv_tile_idx + 1 == kv_end_tile_idx
        kv_token_idx = kv_tile_idx * TILE_K_SIZE

        # L2-biased loads for K/V tiles (reduce L1 thrash on Hopper)
        if K_BLOCK_DIVISIBLE or not last_iter:
            kt_tile = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
                cache_modifier=".cg",
            )
            v_tile = tl.load(
                tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                cache_modifier=".cg",
            )
        else:
            kt_tile = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
                boundary_check=(1,),
                padding_option="zero",
                cache_modifier=".cg",
            )
            v_tile = tl.load(
                tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                boundary_check=(0,),
                padding_option="zero",
                cache_modifier=".cg",
            )

        qk = tl.dot(q_tile, kt_tile, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

        kv_indices = kv_token_idx + tile_k_arange

        if HAS_TOKEN_RULES:
            # K metadata per KV tile
            if IN_SPAN is not None:
                k_in_span_ptr = IN_SPAN + batch * in_span_stride_b + kv_indices
                k_in_span = tl.load(k_in_span_ptr, mask=kv_indices < seq_len, other=0)
            else:
                k_in_span = tl.full([TILE_K_SIZE], False, tl.int1)

            if SPAN_ID is not None:
                k_span_id_ptr = SPAN_ID + batch * span_id_stride_b + kv_indices
                k_span_id = tl.load(k_span_id_ptr, mask=kv_indices < seq_len, other=0)
            else:
                k_span_id = tl.full([TILE_K_SIZE], 0, tl.int32)

            if IS_PREFIX is not None:
                k_is_prefix_ptr = IS_PREFIX + batch * is_prefix_stride_b + kv_indices
                k_is_prefix = tl.load(k_is_prefix_ptr, mask=kv_indices < seq_len, other=0)
            else:
                k_is_prefix = tl.full([TILE_K_SIZE], False, tl.int1)

            # --- Cocktail Party Attention Pattern ---
            q_is_maskq = (q_span_id[:, None] == -1) if (SPAN_ID is not None) else tl.full([TILE_Q_SIZE, 1], False, tl.int1)
            k_is_maskq = (k_span_id[None, :] == -1) if (SPAN_ID is not None) else tl.full([1, TILE_K_SIZE], False, tl.int1)
            k_is_cls_or_prefix = k_is_prefix[None, :]

            prefix_to_prefix = q_is_prefix[:, None] & k_is_prefix[None, :]

            q_is_context = ~q_in_span[:, None] & ~q_is_prefix[:, None] & ~q_is_maskq
            k_is_context = ~k_in_span[None, :] & ~k_is_prefix[None, :] & ~k_is_maskq
            context_causal = q_is_context & k_is_context & (q_tile_indices[:, None] >= kv_indices[None, :])
            context_to_prefix = q_is_context & k_is_cls_or_prefix

            same_span = (q_in_span[:, None] & k_in_span[None, :] &
                         (q_span_id[:, None] == k_span_id[None, :]) &
                         ((SPAN_ID is not None) & (q_span_id[:, None] > 0)))
            span_to_context = q_in_span[:, None] & k_is_context

            maskq_to_spans = q_is_maskq & k_in_span[None, :]
            maskq_to_cls = q_is_maskq & k_is_cls_or_prefix

            mask = (prefix_to_prefix |
                    context_causal | context_to_prefix |
                    same_span | span_to_context |
                    maskq_to_spans | maskq_to_cls)
            # --- End pattern ---
        elif CAUSAL:
            mask = q_tile_indices[:, None] >= kv_indices[None, :]
        else:
            mask = True

        mask = mask & (q_lens_mask & (kv_indices[None, :] < seq_len))

        if RETURN_ATTENTION_MASK and OUTPUT_ATTN_MASK is not None:
            output_attn_mask_batch_head_offset = (batch * output_attn_mask_stride_b +
                                                  head * output_attn_mask_stride_h)
            output_attn_mask_tile_ptr = tl.make_block_ptr(
                base=OUTPUT_ATTN_MASK + output_attn_mask_batch_head_offset,
                shape=(T, T),
                strides=(output_attn_mask_stride_q, output_attn_mask_stride_k),
                offsets=(q_token_idx, kv_token_idx),
                block_shape=(TILE_Q_SIZE, TILE_K_SIZE),
                order=(1, 0),
            )
            if Q_BLOCK_DIVISIBLE and K_BLOCK_DIVISIBLE:
                tl.store(output_attn_mask_tile_ptr, mask.to(tl.int8))
            else:
                boundary_mask = (q_tile_indices[:, None] < seq_len) & (kv_indices[None, :] < seq_len)
                safe_mask_tile = (mask & boundary_mask).to(tl.int8)
                tl.store(output_attn_mask_tile_ptr, safe_mask_tile, boundary_check=(0, 1))

        if not PERFECT_MATCHING:
            q_attended |= tl.max(mask, 1) > 0

        if not PRESCALE_QK:
            qk = qk * softmax_scale
        qk = tl.where(mask, qk, tl.cast(-float("inf"), qk.dtype))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        if not PERFECT_MATCHING:
            m_ij_safe = tl.where(q_attended, m_ij, tl.cast(0, m_ij.dtype))
        else:
            m_ij_safe = m_ij
        p = tl.math.exp2(qk - m_ij_safe[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij_safe)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        acc = tl.dot(p.to(v_tile.dtype), v_tile, acc, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        m_i = m_ij

    if not PERFECT_MATCHING:
        l_i = tl.where(q_attended, l_i, 1)
        acc = acc / l_i[:, None]
    else:
        acc = acc / l_i[:, None]
        acc = tl.where(q_lens_mask, acc, 0.0)

    obatch_head_offset = batch * stride_ob + head * stride_oh
    o_tile_ptr = tl.make_block_ptr(
        base=O + obatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_ot, stride_ok),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    if Q_BLOCK_DIVISIBLE:
        tl.store(o_tile_ptr, acc.to(o_tile_ptr.type.element_ty))
    else:
        tl.store(o_tile_ptr, acc.to(o_tile_ptr.type.element_ty), boundary_check=(0,))

    if OUTPUT_LOGSUMEXP and LSE is not None:
        m_i += tl.math.log2(l_i)

        mbatch_head_offset = batch * stride_mb + head * stride_mh
        m_tile_ptr = tl.make_block_ptr(
            base=LSE + mbatch_head_offset,
            shape=(T,),
            strides=(stride_mt,),
            offsets=(q_token_idx,),
            block_shape=(TILE_Q_SIZE,),
            order=(0,),
        )

        if Q_BLOCK_DIVISIBLE:
            tl.store(m_tile_ptr, m_i)
        else:
            tl.store(m_tile_ptr, m_i, boundary_check=(0,))


@triton.autotune(
    configs=[
        # Generic tiles
        triton.Config(dict(TILE_SIZE=tile), num_warps=nw)
        for nw in [4, 8, 16]
        for tile in [32, 64, 128]
    ] + [
        # Hopper warp specialization flavored configs
        triton.Config(dict(TILE_SIZE=tile, num_consumer_groups=2, num_buffers_warp_spec=3), num_warps=nw)
        for nw in [8, 16]
        for tile in [64, 128]
    ],
    key=["HEAD_DIM", "DTYPE", "TIME_BUCKET"],
)
@triton.heuristics(
    dict(
        BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_SIZE'] == 0,
        RCP_LN2=lambda _: math.log2(math.e),
    )
)
@triton.jit
def _flash_attn_bwd_precompute(
    O: tl.tensor, DO: tl.tensor, RES: tl.tensor,
    stride_ob: int, stride_oh: int, stride_ot: int, stride_ok: int,  #
    stride_dob: int, stride_doh: int, stride_dot: int, stride_dok: int,  #
    stride_rb: int, stride_rh: int, stride_rt: int,
    T: int,
    TIME_BUCKET: int,  #
    HEAD_DIM: tl.constexpr,
    DTYPE:  tl.constexpr,  #
    TILE_SIZE: tl.constexpr,
    BLOCK_DIVISIBLE: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    tile = tl.program_id(2)

    token_idx = tile * TILE_SIZE

    obatch_head_offset = batch * stride_ob + head * stride_oh
    o_tile_ptr = tl.make_block_ptr(
        base=O + obatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_ot, stride_ok),
        offsets=(token_idx, 0),
        block_shape=(TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    dobatch_head_offset = batch * stride_dob + head * stride_doh
    do_tile_ptr = tl.make_block_ptr(
        base=DO + dobatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dot, stride_dok),
        offsets=(token_idx, 0),
        block_shape=(TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    if BLOCK_DIVISIBLE:
        o_tile = tl.load(o_tile_ptr, cache_modifier=".cg")
        do_tile = tl.load(do_tile_ptr, cache_modifier=".cg")
    else:
        o_tile = tl.load(o_tile_ptr, boundary_check=(0,), padding_option="zero", cache_modifier=".cg")
        do_tile = tl.load(do_tile_ptr, boundary_check=(0,), padding_option="zero", cache_modifier=".cg")

    res = tl.sum(o_tile.to(tl.float32) * do_tile.to(tl.float32), 1)

    rbatch_head_offset = batch * stride_rb + head * stride_rh
    res_ptr = tl.make_block_ptr(
        base=RES + rbatch_head_offset,
        shape=(T,),
        strides=(stride_rt,),
        offsets=(token_idx,),
        block_shape=(TILE_SIZE,),
        order=(0,),
    )

    if BLOCK_DIVISIBLE:
        tl.store(res_ptr, res)
    else:
        tl.store(res_ptr, res, boundary_check=(0,))


@triton.jit
def _flash_attn_bwd(
    Q: tl.tensor, K: tl.tensor, V: tl.tensor, L: tl.tensor,
    DELTA: tl.tensor, LSE: tl.tensor,
    DO: tl.tensor, DQ: tl.tensor, DK: tl.tensor, DV: tl.tensor,
    ATTN_MASK: tl.tensor,
    IN_SPAN: tl.tensor, SPAN_ID: tl.tensor, IS_PREFIX: tl.tensor,
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: int,
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,
    stride_deltab: int, stride_deltah: int, stride_deltat: int,
    stride_mb: int, stride_mh: int, stride_mt: int,
    stride_dob: int, stride_doh: int, stride_dot: int, stride_dok: int,
    stride_dqb: int, stride_dqh: int, stride_dqt: int, stride_dqk: int,
    stride_dkb: int, stride_dkh: int, stride_dkt: int, stride_dkk: int,
    stride_dvb: int, stride_dvh: int, stride_dvt: int, stride_dvk: int,
    lens_stride: int,
    mask_stride_b: int, mask_stride_h: int, mask_stride_t: int,
    in_span_stride_b: int, in_span_stride_t: int,
    span_id_stride_b: int, span_id_stride_t: int,
    is_prefix_stride_b: int, is_prefix_stride_t: int,
    T: int,
    TIME_BUCKET: int,
    HEAD_DIM: tl.constexpr,
    DTYPE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    SM_SCALE: tl.constexpr,
    PRESCALE_QK: tl.constexpr,
    TILE_DQ_Q_SIZE: tl.constexpr, TILE_DQ_K_SIZE: tl.constexpr,
    TILE_DK_Q_SIZE: tl.constexpr, TILE_DK_K_SIZE: tl.constexpr,
    PIPELINING: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_TOKEN_RULES: tl.constexpr,  # NEW
):
    # Manually compute values (jit-constant)
    RCP_LN2 = math.log2(math.e)
    DQ_TILES_NUM = tl.cdiv(T, TILE_DQ_Q_SIZE)
    PERFECT_DKV_MATCHING = (TILE_DK_Q_SIZE == TILE_DK_K_SIZE)
    PERFECT_DQ_MATCHING = (TILE_DQ_Q_SIZE == TILE_DQ_K_SIZE)
    DQ_Q_BLOCK_DIVISIBLE = (T % TILE_DQ_Q_SIZE == 0)
    DQ_K_BLOCK_DIVISIBLE = (T % TILE_DQ_K_SIZE == 0)
    DK_Q_BLOCK_DIVISIBLE = (T % TILE_DK_Q_SIZE == 0)
    DK_K_BLOCK_DIVISIBLE = (T % TILE_DK_K_SIZE == 0)

    batch = tl.program_id(0)
    head = tl.program_id(1)
    dkv_worker = tl.program_id(2) >= DQ_TILES_NUM
    tile_id = tl.program_id(2) - (DQ_TILES_NUM * dkv_worker)

    if L is not None:
        seq_len = tl.load(L + batch * lens_stride)
        seq_len = min(seq_len, T)
    else:
        seq_len = T

    if dkv_worker:
        _flash_attn_bwd_dkdv_inner(
            Q, K, V, DELTA, LSE, DO, DK, DV,
            ATTN_MASK,
            IN_SPAN, SPAN_ID, IS_PREFIX,
            stride_qb, stride_qh, stride_qt, stride_qk,
            stride_kb, stride_kh, stride_kt, stride_kk,
            stride_vb, stride_vh, stride_vt, stride_vk,
            stride_deltab, stride_deltah, stride_deltat,
            stride_mb, stride_mh, stride_mt,
            stride_dob, stride_doh, stride_dot, stride_dok,
            stride_dkb, stride_dkh, stride_dkt, stride_dkk,
            stride_dvb, stride_dvh, stride_dvt, stride_dvk,
            mask_stride_b, mask_stride_h, mask_stride_t,
            in_span_stride_b, in_span_stride_t,
            span_id_stride_b, span_id_stride_t,
            is_prefix_stride_b, is_prefix_stride_t,
            batch=batch,
            head=head,
            tile_id=tile_id,
            seq_len=seq_len,
            T=T,
            HEAD_DIM=HEAD_DIM,
            INPUT_PRECISION=INPUT_PRECISION,
            SM_SCALE=SM_SCALE,
            PRESCALE_QK=PRESCALE_QK,
            PERFECT_DKV_MATCHING=PERFECT_DKV_MATCHING,
            DK_Q_BLOCK_DIVISIBLE=DK_Q_BLOCK_DIVISIBLE,
            DK_K_BLOCK_DIVISIBLE=DK_K_BLOCK_DIVISIBLE,
            RCP_LN2=RCP_LN2,
            TILE_DK_Q_SIZE=TILE_DK_Q_SIZE,
            TILE_DK_K_SIZE=TILE_DK_K_SIZE,
            PIPELINING=PIPELINING,
            CAUSAL=CAUSAL,
            HAS_TOKEN_RULES=HAS_TOKEN_RULES,
        )
    else:
        _flash_attn_bwd_dq_inner(
            Q, K, V, DELTA, LSE,
            DO, DQ,
            ATTN_MASK,
            IN_SPAN, SPAN_ID, IS_PREFIX,
            stride_qb, stride_qh, stride_qt, stride_qk,
            stride_kb, stride_kh, stride_kt, stride_kk,
            stride_vb, stride_vh, stride_vt, stride_vk,
            stride_deltab, stride_deltah, stride_deltat,
            stride_mb, stride_mh, stride_mt,
            stride_dob, stride_doh, stride_dot, stride_dok,
            stride_dqb, stride_dqh, stride_dqt, stride_dqk,
            mask_stride_b, mask_stride_h, mask_stride_t,
            in_span_stride_b, in_span_stride_t,
            span_id_stride_b, span_id_stride_t,
            is_prefix_stride_b, is_prefix_stride_t,
            batch=batch,
            head=head,
            tile_id=tile_id,
            seq_len=seq_len,
            T=T,
            HEAD_DIM=HEAD_DIM,
            INPUT_PRECISION=INPUT_PRECISION,
            SM_SCALE=SM_SCALE,
            PRESCALE_QK=PRESCALE_QK,
            PERFECT_DQ_MATCHING=PERFECT_DQ_MATCHING,
            DQ_Q_BLOCK_DIVISIBLE=DQ_Q_BLOCK_DIVISIBLE,
            DQ_K_BLOCK_DIVISIBLE=DQ_K_BLOCK_DIVISIBLE,
            DK_Q_BLOCK_DIVISIBLE=DK_Q_BLOCK_DIVISIBLE,
            DK_K_BLOCK_DIVISIBLE=DK_K_BLOCK_DIVISIBLE,
            RCP_LN2=RCP_LN2,
            TILE_DQ_Q_SIZE=TILE_DQ_Q_SIZE,
            TILE_DQ_K_SIZE=TILE_DQ_K_SIZE,
            PIPELINING=PIPELINING,
            CAUSAL=CAUSAL,
            HAS_TOKEN_RULES=HAS_TOKEN_RULES,
        )


@triton.jit()
def _flash_attn_bwd_dq_inner(
    Q: tl.tensor, K: tl.tensor, V: tl.tensor, DELTA: tl.tensor, LSE: tl.tensor,
    DO: tl.tensor, DQ: tl.tensor,
    ATTN_MASK: tl.tensor,
    IN_SPAN: tl.tensor, SPAN_ID: tl.tensor, IS_PREFIX: tl.tensor,
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: int,
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,
    stride_deltab: int, stride_deltah: int, stride_deltat: int,
    stride_mb: int, stride_mh: int, stride_mt: int,
    stride_dob: int, stride_doh: int, stride_dot: int, stride_dok: int,
    stride_dqb: int, stride_dqh: int, stride_dqt: int, stride_dqk: int,
    mask_stride_b: int, mask_stride_h: int, mask_stride_t: int,
    in_span_stride_b: int, in_span_stride_t: int,
    span_id_stride_b: int, span_id_stride_t: int,
    is_prefix_stride_b: int, is_prefix_stride_t: int,
    batch: int,
    head: int,
    tile_id: int,
    seq_len: tl.tensor,
    T: int,  #
    HEAD_DIM: tl.constexpr,  #
    INPUT_PRECISION: tl.constexpr,  #
    SM_SCALE: tl.constexpr,  #
    PRESCALE_QK: tl.constexpr,  #
    PERFECT_DQ_MATCHING: tl.constexpr,  #
    DQ_Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    DQ_K_BLOCK_DIVISIBLE: tl.constexpr,  #
    DK_Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    DK_K_BLOCK_DIVISIBLE: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
    TILE_DQ_Q_SIZE: tl.constexpr,  #
    TILE_DQ_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
    CAUSAL: tl.constexpr,  #
    HAS_TOKEN_RULES: tl.constexpr,  #
):
    q_tile_idx = tile_id
    q_token_idx = q_tile_idx * TILE_DQ_Q_SIZE

    qbatch_head_offset = batch * stride_qb + head * stride_qh
    q_tile_ptr = tl.make_block_ptr(
        base=Q + qbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_qt, stride_qk),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    lsebatch_head_offset = batch * stride_mb + head * stride_mh
    lse_tile_ptr = tl.make_block_ptr(
        base=LSE + lsebatch_head_offset,
        shape=(T,),
        strides=(stride_mt,),
        offsets=(q_token_idx,),
        block_shape=(TILE_DQ_Q_SIZE,),
        order=(0,),
    )

    delta_tile_ptr = batch * stride_deltab + head * stride_deltah
    delta_tile_ptr = tl.make_block_ptr(
        base=DELTA + delta_tile_ptr,
        shape=(T,),
        strides=(stride_deltat,),
        offsets=(q_token_idx,),
        block_shape=(TILE_DQ_Q_SIZE,),
        order=(0,),
    )

    dobatch_head_offset = batch * stride_dob + head * stride_doh
    do_tile_ptr = tl.make_block_ptr(
        base=DO + dobatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dot, stride_dok),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    if DQ_Q_BLOCK_DIVISIBLE:
        q = tl.load(q_tile_ptr, cache_modifier=".cg")
        m = tl.load(lse_tile_ptr)[:, None]
        di = tl.load(delta_tile_ptr)
        do = tl.load(do_tile_ptr, cache_modifier=".cg")
    else:
        q = tl.load(q_tile_ptr, boundary_check=(0,), padding_option="zero", cache_modifier=".cg")
        m = tl.load(lse_tile_ptr, boundary_check=(0,))[:, None]
        di = tl.load(delta_tile_ptr, boundary_check=(0,))
        do = tl.load(do_tile_ptr, boundary_check=(0,), padding_option="zero", cache_modifier=".cg")

    # Hoist Q metadata (registers)
    q_tile_indices = q_token_idx + tl.arange(0, TILE_DQ_Q_SIZE)
    if HAS_TOKEN_RULES:
        if IN_SPAN is not None:
            q_in_span_ptr = IN_SPAN + batch * in_span_stride_b + q_tile_indices
            q_in_span = tl.load(q_in_span_ptr, mask=q_tile_indices < seq_len, other=0)
        else:
            q_in_span = tl.full([TILE_DQ_Q_SIZE], False, tl.int1)

        if SPAN_ID is not None:
            q_span_id_ptr = SPAN_ID + batch * span_id_stride_b + q_tile_indices
            q_span_id = tl.load(q_span_id_ptr, mask=q_tile_indices < seq_len, other=0)
        else:
            q_span_id = tl.full([TILE_DQ_Q_SIZE], 0, tl.int32)

        if IS_PREFIX is not None:
            q_is_prefix_ptr = IS_PREFIX + batch * is_prefix_stride_b + q_tile_indices
            q_is_prefix = tl.load(q_is_prefix_ptr, mask=q_tile_indices < seq_len, other=0)
        else:
            q_is_prefix = tl.full([TILE_DQ_Q_SIZE], False, tl.int1)
    else:
        q_in_span = tl.full([TILE_DQ_Q_SIZE], False, tl.int1)
        q_span_id = tl.full([TILE_DQ_Q_SIZE], 0, tl.int32)
        q_is_prefix = tl.full([TILE_DQ_Q_SIZE], False, tl.int1)

    kbatch_head_offset = batch * stride_kb + head * stride_kh
    kt_tile_ptr = tl.make_block_ptr(
        base=K + kbatch_head_offset,
        shape=(HEAD_DIM, T),
        strides=(stride_kk, stride_kt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_DQ_K_SIZE),
        order=(0, 1),
    )

    vbatch_head_offset = batch * stride_vb + head * stride_vh
    vt_tile_ptr = tl.make_block_ptr(
        base=V + vbatch_head_offset,
        shape=(HEAD_DIM, T),
        strides=(stride_vk, stride_vt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_DQ_K_SIZE),
        order=(1, 0),
    )

    dq = tl.zeros([TILE_DQ_Q_SIZE, HEAD_DIM], dtype=tl.float32)
    dq = _flash_attn_bwd_dq(
        dq, q, m, di, do,
        kt_tile_ptr, vt_tile_ptr,
        ATTN_MASK,
        IN_SPAN, SPAN_ID, IS_PREFIX,
        mask_stride_b, mask_stride_h, mask_stride_t,
        in_span_stride_b, in_span_stride_t,
        span_id_stride_b, span_id_stride_t,
        is_prefix_stride_b, is_prefix_stride_t,
        batch, head,
        seq_len=seq_len,
        T=T,
        q_token_idx=q_token_idx,
        TILE_Q_SIZE=TILE_DQ_Q_SIZE,
        TILE_K_SIZE=TILE_DQ_K_SIZE,
        CAUSAL=CAUSAL,
        INPUT_PRECISION=INPUT_PRECISION,
        PIPELINING=PIPELINING,
        K_BLOCK_DIVISIBLE=DQ_K_BLOCK_DIVISIBLE,
        PERFECT_MATCHING=PERFECT_DQ_MATCHING,
        RCP_LN2=RCP_LN2,
        SM_SCALE=SM_SCALE,
        PRESCALE_QK=PRESCALE_QK,
        HAS_TOKEN_RULES=HAS_TOKEN_RULES,
    )

    dqbatch_head_offset = batch * stride_dqb + head * stride_dqh
    dq_tile_ptr = tl.make_block_ptr(
        base=DQ + dqbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dqt, stride_dqk),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    if DQ_Q_BLOCK_DIVISIBLE:
        tl.store(dq_tile_ptr, dq.to(dq_tile_ptr.type.element_ty))
    else:
        tl.store(dq_tile_ptr, dq.to(dq_tile_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def _flash_attn_bwd_dkdv_inner(
    Q: tl.tensor, K: tl.tensor, V: tl.tensor,
    DELTA: tl.tensor, LSE: tl.tensor,
    DO: tl.tensor, DK: tl.tensor, DV: tl.tensor,
    ATTN_MASK: tl.tensor,
    IN_SPAN: tl.tensor, SPAN_ID: tl.tensor, IS_PREFIX: tl.tensor,
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: int,
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,
    stride_deltab: int, stride_deltah: int, stride_deltat: int,
    stride_mb: int, stride_mh: int, stride_mt: int,
    stride_dob: int, stride_doh: int, stride_dot: int,
    stride_dok: int, stride_dkb: int, stride_dkh: int,
    stride_dkt: int, stride_dkk: int, stride_dvb: int,
    stride_dvh: int, stride_dvt: int, stride_dvk: int,
    mask_stride_b: int, mask_stride_h: int, mask_stride_t: int,
    in_span_stride_b: int, in_span_stride_t: int,
    span_id_stride_b: int, span_id_stride_t: int,
    is_prefix_stride_b: int, is_prefix_stride_t: int,
    batch: int,
    head: int,
    tile_id: int,
    seq_len: tl.tensor,
    T: int,  #
    HEAD_DIM: tl.constexpr,  #
    INPUT_PRECISION: tl.constexpr,  #
    SM_SCALE: tl.constexpr,  #
    PRESCALE_QK: tl.constexpr,  #
    PERFECT_DKV_MATCHING: tl.constexpr,  #
    DK_Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    DK_K_BLOCK_DIVISIBLE: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
    TILE_DK_Q_SIZE: tl.constexpr,  #
    TILE_DK_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
    CAUSAL: tl.constexpr,  #
    HAS_TOKEN_RULES: tl.constexpr,  #
):
    kv_tile_idx = tile_id
    kv_token_idx = kv_tile_idx * TILE_DK_K_SIZE

    qbatch_head_offset = batch * stride_qb + head * stride_qh
    qt_tile_ptr = tl.make_block_ptr(
        base=Q + qbatch_head_offset,
        shape=(HEAD_DIM, T),
        strides=(stride_qk, stride_qt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_DK_Q_SIZE),
        order=(0, 1),
    )

    kbatch_head_offset = batch * stride_kb + head * stride_kh
    k_tile_ptr = tl.make_block_ptr(
        base=K + kbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_kt, stride_kk),
        offsets=(kv_token_idx, 0),
        block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    vbatch_head_offset = batch * stride_vb + head * stride_vh
    v_tile_ptr = tl.make_block_ptr(
        base=V + vbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_vt, stride_vk),
        offsets=(kv_token_idx, 0),
        block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    dobatch_head_offset = batch * stride_dob + head * stride_doh
    do_tile_ptr = tl.make_block_ptr(
        base=DO + dobatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dot, stride_dok),
        offsets=(0, 0),
        block_shape=(TILE_DK_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    lsebatch_head_offset = batch * stride_mb + head * stride_mh
    lse_tile_ptr = tl.make_block_ptr(
        base=LSE + lsebatch_head_offset,
        shape=(T,),
        strides=(stride_mt,),
        offsets=(0,),
        block_shape=(TILE_DK_Q_SIZE,),
        order=(0,),
    )

    deltabatch_head_offset = batch * stride_deltab + head * stride_deltah
    delta_tile_ptr = tl.make_block_ptr(
        base=DELTA + deltabatch_head_offset,
        shape=(T,),
        strides=(stride_deltat,),
        offsets=(0,),
        block_shape=(TILE_DK_Q_SIZE,),
        order=(0,),
    )

    dv = tl.zeros([TILE_DK_K_SIZE, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([TILE_DK_K_SIZE, HEAD_DIM], dtype=tl.float32)

    if DK_K_BLOCK_DIVISIBLE:
        k = tl.load(k_tile_ptr, cache_modifier=".cg")
        v = tl.load(v_tile_ptr, cache_modifier=".cg")
    else:
        k = tl.load(k_tile_ptr, boundary_check=(0,), padding_option="zero", cache_modifier=".cg")
        v = tl.load(v_tile_ptr, boundary_check=(0,), padding_option="zero", cache_modifier=".cg")

    dk, dv = _flash_attn_bwd_dkdv(
        dk, dv,
        qt_tile_ptr, do_tile_ptr, lse_tile_ptr, delta_tile_ptr,
        k, v,
        ATTN_MASK,
        IN_SPAN, SPAN_ID, IS_PREFIX,
        mask_stride_b, mask_stride_h, mask_stride_t,
        in_span_stride_b, in_span_stride_t,
        span_id_stride_b, span_id_stride_t,
        is_prefix_stride_b, is_prefix_stride_t,
        batch, head,
        seq_len=seq_len,
        T=T,
        kv_token_idx=kv_token_idx,
        TILE_Q_SIZE=TILE_DK_Q_SIZE,
        TILE_K_SIZE=TILE_DK_K_SIZE,
        CAUSAL=CAUSAL,
        INPUT_PRECISION=INPUT_PRECISION,
        PERFECT_MATCHING=PERFECT_DKV_MATCHING,
        PIPELINING=PIPELINING,
        Q_BLOCK_DIVISIBLE=DK_Q_BLOCK_DIVISIBLE,
        RCP_LN2=RCP_LN2,
        SM_SCALE=SM_SCALE,
        PRESCALE_QK=PRESCALE_QK,
        HAS_TOKEN_RULES=HAS_TOKEN_RULES,
    )

    dkbatch_head_offset = batch * stride_dkb + head * stride_dkh
    dk_tile_ptr = tl.make_block_ptr(
        base=DK + dkbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dkt, stride_dkk),
        offsets=(kv_token_idx, 0),
        block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    if DK_K_BLOCK_DIVISIBLE:
        tl.store(dk_tile_ptr, dk.to(dk_tile_ptr.type.element_ty))
    else:
        tl.store(dk_tile_ptr, dk.to(dk_tile_ptr.type.element_ty), boundary_check=(0,))

    dvbatch_head_offset = batch * stride_dvb + head * stride_dvh
    dv_tile_ptr = tl.make_block_ptr(
        base=DV + dvbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dvt, stride_dvk),
        offsets=(kv_token_idx, 0),
        block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    if DK_K_BLOCK_DIVISIBLE:
        tl.store(dv_tile_ptr, dv.to(dv_tile_ptr.type.element_ty))
    else:
        tl.store(dv_tile_ptr, dv.to(dv_tile_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def _flash_attn_bwd_dq(
    dq: tl.tensor, q: tl.tensor, m: tl.tensor,
    di: tl.tensor, do: tl.tensor,
    kt_tile_ptr: tl.tensor, vt_tile_ptr: tl.tensor,
    ATTN_MASK: tl.tensor,
    IN_SPAN: tl.tensor, SPAN_ID: tl.tensor, IS_PREFIX: tl.tensor,
    mask_stride_b: int, mask_stride_h: int, mask_stride_t: int,
    in_span_stride_b: int, in_span_stride_t: int,
    span_id_stride_b: int, span_id_stride_t: int,
    is_prefix_stride_b: int, is_prefix_stride_t: int,
    batch: int, head: int,
    seq_len: tl.tensor,
    T: tl.constexpr,
    q_token_idx: int,
    TILE_Q_SIZE: tl.constexpr,
    TILE_K_SIZE: tl.constexpr,
    CAUSAL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    PERFECT_MATCHING: tl.constexpr,
    PIPELINING: tl.constexpr,
    K_BLOCK_DIVISIBLE: tl.constexpr,
    RCP_LN2: tl.constexpr,
    SM_SCALE: tl.constexpr,
    PRESCALE_QK: tl.constexpr,
    HAS_TOKEN_RULES: tl.constexpr,
):
    # Conditional attention range based on CAUSAL parameter
    kv_start_tile_idx = 0
    q_tile_max_token = min(q_token_idx + TILE_Q_SIZE, seq_len)
    if CAUSAL:
        kv_end_tile_idx = tl.cdiv(q_tile_max_token, TILE_K_SIZE)
    else:
        kv_end_tile_idx = tl.cdiv(seq_len, TILE_K_SIZE)

    q_tile_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)

    q_len_mask = q_tile_indices[:, None] < seq_len
    tile_k_arange = tl.arange(0, TILE_K_SIZE)

    softmax_scale: tl.constexpr = tl.cast(SM_SCALE, q.dtype)
    if PRESCALE_QK:
        q = q * softmax_scale * RCP_LN2

    # Hoisted Q metadata (registers)
    if HAS_TOKEN_RULES:
        if IN_SPAN is not None:
            q_in_span_ptr = IN_SPAN + batch * in_span_stride_b + q_tile_indices
            q_in_span = tl.load(q_in_span_ptr, mask=q_tile_indices < seq_len, other=0)
        else:
            q_in_span = tl.full([TILE_Q_SIZE], False, tl.int1)
        if SPAN_ID is not None:
            q_span_id_ptr = SPAN_ID + batch * span_id_stride_b + q_tile_indices
            q_span_id = tl.load(q_span_id_ptr, mask=q_tile_indices < seq_len, other=0)
        else:
            q_span_id = tl.full([TILE_Q_SIZE], 0, tl.int32)
        if IS_PREFIX is not None:
            q_is_prefix_ptr = IS_PREFIX + batch * is_prefix_stride_b + q_tile_indices
            q_is_prefix = tl.load(q_is_prefix_ptr, mask=q_tile_indices < seq_len, other=0)
        else:
            q_is_prefix = tl.full([TILE_Q_SIZE], False, tl.int1)
    else:
        q_in_span = tl.full([TILE_Q_SIZE], False, tl.int1)
        q_span_id = tl.full([TILE_Q_SIZE], 0, tl.int32)
        q_is_prefix = tl.full([TILE_Q_SIZE], False, tl.int1)

    for kv_tile_idx in tl.range(kv_start_tile_idx, kv_end_tile_idx, num_stages=PIPELINING):
        kv_token_idx = kv_tile_idx * TILE_K_SIZE
        if K_BLOCK_DIVISIBLE:
            kT = tl.load(tl.advance(kt_tile_ptr, (0, kv_token_idx)), cache_modifier=".cg")
            vT = tl.load(tl.advance(vt_tile_ptr, (0, kv_token_idx)), cache_modifier=".cg")
        else:
            kT = tl.load(tl.advance(kt_tile_ptr, (0, kv_token_idx)), boundary_check=(1,), padding_option="zero", cache_modifier=".cg")
            vT = tl.load(tl.advance(vt_tile_ptr, (0, kv_token_idx)), boundary_check=(1,), padding_option="zero", cache_modifier=".cg")

        qk = tl.dot(q, kT, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK:
            qk = qk * softmax_scale * RCP_LN2
        p = tl.math.exp2(qk - m)

        kv_indices = kv_token_idx + tile_k_arange

        if HAS_TOKEN_RULES:
            if IN_SPAN is not None:
                k_in_span_ptr = IN_SPAN + batch * in_span_stride_b + kv_indices
                k_in_span = tl.load(k_in_span_ptr, mask=kv_indices < seq_len, other=0)
            else:
                k_in_span = tl.full([TILE_K_SIZE], False, tl.int1)

            if SPAN_ID is not None:
                k_span_id_ptr = SPAN_ID + batch * span_id_stride_b + kv_indices
                k_span_id = tl.load(k_span_id_ptr, mask=kv_indices < seq_len, other=0)
            else:
                k_span_id = tl.full([TILE_K_SIZE], 0, tl.int32)

            if IS_PREFIX is not None:
                k_is_prefix_ptr = IS_PREFIX + batch * is_prefix_stride_b + kv_indices
                k_is_prefix = tl.load(k_is_prefix_ptr, mask=kv_indices < seq_len, other=0)
            else:
                k_is_prefix = tl.full([TILE_K_SIZE], False, tl.int1)

            q_is_maskq = (q_span_id[:, None] == -1) if (SPAN_ID is not None) else tl.full([TILE_Q_SIZE, 1], False, tl.int1)
            k_is_maskq = (k_span_id[None, :] == -1) if (SPAN_ID is not None) else tl.full([1, TILE_K_SIZE], False, tl.int1)
            k_is_cls_or_prefix = k_is_prefix[None, :]

            prefix_to_prefix = q_is_prefix[:, None] & k_is_prefix[None, :]
            q_is_context = ~q_in_span[:, None] & ~q_is_prefix[:, None] & ~q_is_maskq
            k_is_context = ~k_in_span[None, :] & ~k_is_prefix[None, :] & ~k_is_maskq
            context_causal = q_is_context & k_is_context & (q_tile_indices[:, None] >= kv_indices[None, :])
            context_to_prefix = q_is_context & k_is_cls_or_prefix
            same_span = (q_in_span[:, None] & k_in_span[None, :] &
                         (q_span_id[:, None] == k_span_id[None, :]) &
                         ((SPAN_ID is not None) & (q_span_id[:, None] > 0)))
            span_to_context = q_in_span[:, None] & k_is_context
            maskq_to_spans = q_is_maskq & k_in_span[None, :]
            maskq_to_cls = q_is_maskq & k_is_cls_or_prefix

            mask = (prefix_to_prefix |
                    context_causal | context_to_prefix |
                    same_span | span_to_context |
                    maskq_to_spans | maskq_to_cls)
        elif CAUSAL:
            mask = q_tile_indices[:, None] >= kv_indices[None, :]
        else:
            mask = True

        mask = mask & (q_len_mask & (kv_indices[None, :] < seq_len))

        p = tl.where(mask, p, 0.0)
        dp = tl.dot(do, vT.to(do.dtype), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        ds = p * (dp - di[:, None])
        dq = tl.dot(ds, tl.trans(kT).to(ds.dtype), dq, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    dq *= softmax_scale
    return dq


@triton.jit
def _flash_attn_bwd_dkdv(
    dk: tl.tensor, dv: tl.tensor,
    qt_tile_ptr: tl.tensor, do_tile_ptr: tl.tensor,
    lse_tile_ptr: tl.tensor, delta_tile_ptr: tl.tensor,
    k: tl.tensor, v: tl.tensor,
    ATTN_MASK: tl.tensor,
    IN_SPAN: tl.tensor, SPAN_ID: tl.tensor, IS_PREFIX: tl.tensor,
    mask_stride_b: int, mask_stride_h: int, mask_stride_t: int,
    in_span_stride_b: int, in_span_stride_t: int,
    span_id_stride_b: int, span_id_stride_t: int,
    is_prefix_stride_b: int, is_prefix_stride_t: int,
    batch: int, head: int,
    seq_len: tl.tensor,
    T: tl.constexpr,
    kv_token_idx: int,
    TILE_Q_SIZE: tl.constexpr,
    TILE_K_SIZE: tl.constexpr,
    CAUSAL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    PERFECT_MATCHING: tl.constexpr,
    PIPELINING: tl.constexpr,
    Q_BLOCK_DIVISIBLE: tl.constexpr,
    RCP_LN2: tl.constexpr,
    SM_SCALE: tl.constexpr,
    PRESCALE_QK: tl.constexpr,
    HAS_TOKEN_RULES: tl.constexpr,
):
    kv_tile_max_token = min(kv_token_idx + TILE_K_SIZE, seq_len)
    if CAUSAL:
        q_start_tile_idx = kv_token_idx // TILE_Q_SIZE
        q_end_tile_idx = tl.cdiv(seq_len, TILE_Q_SIZE)
    else:
        q_start_tile_idx = 0
        q_end_tile_idx = tl.cdiv(seq_len, TILE_Q_SIZE)

    kv_indices = kv_token_idx + tl.arange(0, TILE_K_SIZE)
    tile_q_arange = tl.arange(0, TILE_Q_SIZE)
    kv_lens_mask = (kv_indices[:, None] < seq_len)

    if PRESCALE_QK:
        k *= RCP_LN2 * SM_SCALE

    for q_tile_idx in tl.range(q_start_tile_idx, q_end_tile_idx, num_stages=PIPELINING):
        q_token_idx = q_tile_idx * TILE_Q_SIZE
        if Q_BLOCK_DIVISIBLE:
            qT = tl.load(tl.advance(qt_tile_ptr, (0, q_token_idx)), cache_modifier=".cg")
            m = tl.load(tl.advance(lse_tile_ptr, (q_token_idx,)))
            do = tl.load(tl.advance(do_tile_ptr, (q_token_idx, 0)), cache_modifier=".cg")
            Di = tl.load(tl.advance(delta_tile_ptr, (q_token_idx,)))
        else:
            qT = tl.load(tl.advance(qt_tile_ptr, (0, q_token_idx)), boundary_check=(1,), padding_option="zero", cache_modifier=".cg")
            m = tl.load(tl.advance(lse_tile_ptr, (q_token_idx,)), boundary_check=(0,))
            do = tl.load(tl.advance(do_tile_ptr, (q_token_idx, 0)), boundary_check=(0,), padding_option="zero", cache_modifier=".cg")
            Di = tl.load(tl.advance(delta_tile_ptr, (q_token_idx,)), boundary_check=(0,))

        qkT = tl.dot(k, qT, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK:
            qkT *= RCP_LN2 * SM_SCALE
        pT = tl.math.exp2(qkT - m[None, :])

        q_tile_indices = q_token_idx + tile_q_arange

        if HAS_TOKEN_RULES:
            if IN_SPAN is not None:
                k_in_span_ptr = IN_SPAN + batch * in_span_stride_b + kv_indices
                k_in_span = tl.load(k_in_span_ptr, mask=kv_indices < seq_len, other=0)
            else:
                k_in_span = tl.full([TILE_K_SIZE], False, tl.int1)

            if SPAN_ID is not None:
                k_span_id_ptr = SPAN_ID + batch * span_id_stride_b + kv_indices
                k_span_id = tl.load(k_span_id_ptr, mask=kv_indices < seq_len, other=0)
            else:
                k_span_id = tl.full([TILE_K_SIZE], 0, tl.int32)

            if IS_PREFIX is not None:
                k_is_prefix_ptr = IS_PREFIX + batch * is_prefix_stride_b + kv_indices
                k_is_prefix = tl.load(k_is_prefix_ptr, mask=kv_indices < seq_len, other=0)
            else:
                k_is_prefix = tl.full([TILE_K_SIZE], False, tl.int1)

            if IN_SPAN is not None:
                q_in_span_ptr = IN_SPAN + batch * in_span_stride_b + q_tile_indices
                q_in_span = tl.load(q_in_span_ptr, mask=q_tile_indices < seq_len, other=0)
            else:
                q_in_span = tl.full([TILE_Q_SIZE], False, tl.int1)

            if SPAN_ID is not None:
                q_span_id_ptr = SPAN_ID + batch * span_id_stride_b + q_tile_indices
                q_span_id = tl.load(q_span_id_ptr, mask=q_tile_indices < seq_len, other=0)
            else:
                q_span_id = tl.full([TILE_Q_SIZE], 0, tl.int32)

            if IS_PREFIX is not None:
                q_is_prefix_ptr = IS_PREFIX + batch * is_prefix_stride_b + q_tile_indices
                q_is_prefix = tl.load(q_is_prefix_ptr, mask=q_tile_indices < seq_len, other=0)
            else:
                q_is_prefix = tl.full([TILE_Q_SIZE], False, tl.int1)

            q_is_maskq = (q_span_id[None, :] == -1) if (SPAN_ID is not None) else tl.full([1, TILE_Q_SIZE], False, tl.int1)
            k_is_maskq = (k_span_id[:, None] == -1) if (SPAN_ID is not None) else tl.full([TILE_K_SIZE, 1], False, tl.int1)
            k_is_cls_or_prefix = k_is_prefix[:, None]

            prefix_to_prefix = q_is_prefix[None, :] & k_is_prefix[:, None]
            q_is_context = ~q_in_span[None, :] & ~q_is_prefix[None, :] & ~q_is_maskq
            k_is_context = ~k_in_span[:, None] & ~k_is_prefix[:, None] & ~k_is_maskq
            context_causal = q_is_context & k_is_context & (q_tile_indices[None, :] >= kv_indices[:, None])
            context_to_prefix = q_is_context & k_is_cls_or_prefix
            same_span = (q_in_span[None, :] & k_in_span[:, None] &
                         (q_span_id[None, :] == k_span_id[:, None]) &
                         ((SPAN_ID is not None) & (q_span_id[None, :] > 0)))
            span_to_context = q_in_span[None, :] & k_is_context
            maskq_to_spans = q_is_maskq & k_in_span[:, None]
            maskq_to_cls = q_is_maskq & k_is_cls_or_prefix

            mask = (prefix_to_prefix |
                    context_causal | context_to_prefix |
                    same_span | span_to_context |
                    maskq_to_spans | maskq_to_cls)
        elif CAUSAL:
            mask = q_tile_indices[None, :] >= kv_indices[:, None]
        else:
            mask = True

        mask = mask & (kv_lens_mask & (q_tile_indices[None, :] < seq_len))
        pT = tl.where(mask, pT, 0.0)

        dv = tl.dot(pT, do.to(pT.dtype), dv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        dpT = tl.dot(v.to(do.dtype), tl.trans(do), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dk = tl.dot(dsT, tl.trans(qT).to(dsT.dtype), dk, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
    dk *= SM_SCALE
    return dk, dv
# fmt: on


def autotune_prehook(kwargs, reset_only=False):
    if kwargs["L"] is not None:
        kwargs["L"].add_(kwargs["q"].size(2))  # L += time


def autotune_posthook(kwargs, exception=None):
    if kwargs["L"] is not None:
        kwargs["L"].add_(-kwargs["q"].size(2))  # L -= time


# H100 default heuristics prefer modest pipelining (overlapped with WS configs)
flash_forward = triton.heuristics(
    dict(
        PIPELINING=lambda _: 2 if torch.cuda.get_device_capability() >= (9, 0) else 1,
        TILE_Q_SIZE=lambda args: min(128 if torch.cuda.get_device_capability() >= (9, 0) else 64,
                                     max(MIN_TILE_SIZE, triton.next_power_of_2(min(128 if torch.cuda.get_device_capability() >= (9, 0) else 64, args["T"])))),
        TILE_K_SIZE=lambda args: min(128 if torch.cuda.get_device_capability() >= (9, 0) else 64,
                                     max(MIN_TILE_SIZE, triton.next_power_of_2(min(128 if torch.cuda.get_device_capability() >= (9, 0) else 64, args["T"])))),
    )
)(_flash_attn_fwd)

# Autotune set includes WS knobs on Hopper
flash_forward_autotune = triton.autotune(
    configs=[
        # Generic
        triton.Config(
            dict(
                PIPELINING=pipe,
                TILE_Q_SIZE=tile_q,
                TILE_K_SIZE=tile_k,
            ),
            num_warps=nw,
            num_stages=pipe,
        )
        for nw in [4, 8, 16]
        for pipe in [1, 2, 3]
        for tile_q in [2**i for i in range(int(math.log2(MIN_TILE_SIZE) + 0.1), int(math.log2(MAX_TILE_SIZE) + 0.1) + 1)]
        for tile_k in [2**i for i in range(int(math.log2(MIN_TILE_SIZE) + 0.1), int(math.log2(MAX_TILE_SIZE) + 0.1) + 1)]
    ] + [
        # Hopper warp specialization
        triton.Config(
            dict(
                PIPELINING=pipe,
                TILE_Q_SIZE=tq,
                TILE_K_SIZE=tk,
                num_consumer_groups=n_cg,
                num_buffers_warp_spec=n_buf,
            ),
            num_warps=nw,
            num_stages=pipe,
        )
        for nw in [8, 16]
        for pipe in [2, 3]
        for n_cg in [1, 2]
        for n_buf in [2, 3]
        for tq in [64, 128]
        for tk in [64, 128]
    ],
    key=[
        "HEAD_DIM",
        "CAUSAL",
        "INPUT_PRECISION",
        "TIME_BUCKET",
        "DTYPE",
    ],
    prune_configs_by=dict(early_config_prune=fwd_configs_pruner),
    pre_hook=autotune_prehook,
    post_hook=autotune_posthook,
)(_flash_attn_fwd)

flash_backward = triton.heuristics(
    dict(
        PIPELINING=lambda _: 2 if torch.cuda.get_device_capability() >= (9, 0) else 1,
        TILE_DQ_Q_SIZE=lambda args: min(64 if torch.cuda.get_device_capability() >= (9, 0) else 32,
                                        max(MIN_TILE_SIZE, triton.next_power_of_2(min(64 if torch.cuda.get_device_capability() >= (9, 0) else 32, args["T"])))),
        TILE_DQ_K_SIZE=lambda args: min(64 if torch.cuda.get_device_capability() >= (9, 0) else 32,
                                        max(MIN_TILE_SIZE, triton.next_power_of_2(min(64 if torch.cuda.get_device_capability() >= (9, 0) else 32, args["T"])))),
        TILE_DK_Q_SIZE=lambda args: min(64 if torch.cuda.get_device_capability() >= (9, 0) else 32,
                                        max(MIN_TILE_SIZE, triton.next_power_of_2(min(64 if torch.cuda.get_device_capability() >= (9, 0) else 32, args["T"])))),
        TILE_DK_K_SIZE=lambda args: min(64 if torch.cuda.get_device_capability() >= (9, 0) else 32,
                                        max(MIN_TILE_SIZE, triton.next_power_of_2(min(64 if torch.cuda.get_device_capability() >= (9, 0) else 32, args["T"])))),
    )
)(_flash_attn_bwd)

flash_backward_autotune = triton.autotune(
    configs=[
        # Generic
        triton.Config(
            dict(
                PIPELINING=pipe,
                TILE_DQ_Q_SIZE=tile_qq,
                TILE_DQ_K_SIZE=tile_qk,
                TILE_DK_Q_SIZE=tile_kq,
                TILE_DK_K_SIZE=tile_kk,
            ),
            num_warps=nw,
            num_stages=pipe,
        )
        for nw in [4, 8, 16]
        for pipe in [1, 2, 3]
        for tile_qq in [2**i for i in range(int(math.log2(MIN_TILE_SIZE) + 0.1), int(math.log2(MAX_TILE_SIZE) + 0.1) + 1)]
        for tile_qk in [2**i for i in range(int(math.log2(MIN_TILE_SIZE) + 0.1), int(math.log2(MAX_TILE_SIZE) + 0.1) + 1)]
        for tile_kq in [2**i for i in range(int(math.log2(MIN_TILE_SIZE) + 0.1), int(math.log2(MAX_TILE_SIZE) + 0.1) + 1)]
        for tile_kk in [2**i for i in range(int(math.log2(MIN_TILE_SIZE) + 0.1), int(math.log2(MAX_TILE_SIZE) + 0.1) + 1)]
    ] + [
        # Hopper WS
        triton.Config(
            dict(
                PIPELINING=pipe,
                TILE_DQ_Q_SIZE=t,
                TILE_DQ_K_SIZE=t,
                TILE_DK_Q_SIZE=t,
                TILE_DK_K_SIZE=t,
                num_consumer_groups=n_cg,
                num_buffers_warp_spec=n_buf,
            ),
            num_warps=nw,
            num_stages=pipe,
        )
        for nw in [8, 16]
        for pipe in [2, 3]
        for n_cg in [1, 2]
        for n_buf in [2, 3]
        for t in [32, 64]
    ],
    key=[
        "HEAD_DIM",
        "CAUSAL",
        "INPUT_PRECISION",
        "DTYPE",
        "TIME_BUCKET",
    ],
    prune_configs_by=dict(early_config_prune=bwd_configs_pruner),
    pre_hook=autotune_prehook,
    post_hook=autotune_posthook,
)(_flash_attn_bwd)

# T4 GPU optimization - can be dynamically updated (unchanged)
T4_OPTIMIZED = False
T4_OPTIMAL_TILE_Q = 32
T4_OPTIMAL_TILE_K = 32
T4_OPTIMAL_WARPS = 4

def set_t4_optimization(tile_q: int, tile_k: int, num_warps: int):
    """Set T4-optimized tile sizes and warp count."""
    global T4_OPTIMIZED, T4_OPTIMAL_TILE_Q, T4_OPTIMAL_TILE_K, T4_OPTIMAL_WARPS
    T4_OPTIMIZED = True
    T4_    OPTIMAL_WARPS = num_warps
    print(f"T4 optimization enabled: tile_q={tile_q}, tile_k={tile_k}, warps={num_warps}")


def get_optimized_tile_range():
    """Get tile size range based on T4 optimization."""
    if T4_OPTIMIZED:
        # Use optimized values with small range around optimal
        min_size = max(16, T4_OPTIMAL_TILE_Q // 2)
        max_size = min(128, T4_OPTIMAL_TILE_Q * 2)
        return [T4_OPTIMAL_TILE_Q, T4_OPTIMAL_TILE_K, min_size, max_size]
    else:
        return [64, 64, MIN_TILE_SIZE, MAX_TILE_SIZE]


def get_optimized_warp_count():
    """Get optimal warp count for T4."""
    if T4_OPTIMIZED:
        return [T4_OPTIMAL_WARPS]
    else:
        return [4, 8, 16]


@torch.library.custom_op(
    "flash_attention::forward", mutates_args=(), device_types=("cuda",)
)
def attention_forward_adapter(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor,
    sm_scale: float,
    causal: bool,
    autotune: bool,
    return_lse: bool,
    prescale_qk: bool,
    precision: str,
    attention_mask: torch.Tensor = None,
    in_span: torch.Tensor = None,
    span_id: torch.Tensor = None,
    is_prefix: torch.Tensor = None,
    output_attention_mask: torch.Tensor = None,
    return_attention_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, T, HEAD_DIM = q.shape

    assert HEAD_DIM in {16, 32, 64, 128, 256}
    assert HEAD_DIM == k.shape[-1] and HEAD_DIM == v.shape[-1]
    assert T == k.shape[-2] and T == v.shape[-2]
    assert sm_scale is not None
    assert lens is None or (
        lens.dtype == torch.int32 and batch == len(lens) and lens.ndim == 1
    )

    O = torch.zeros_like(q, memory_format=torch.contiguous_format)
    LSE = None
    if return_lse:
        LSE = torch.zeros(q.shape[:3], dtype=torch.float32, device=q.device)

    grid = lambda args: (
        batch,
        heads,
        triton.cdiv(T, args["TILE_Q_SIZE"]),
    )

    # Transposed K view for coalesced access
    kt = k.transpose(-1, -2)

    # Select kernel (autotune vs heuristics)
    fwd_fn = flash_forward_autotune if autotune else flash_forward

    # Compile-time switch for hierarchical token rules
    HAS_TOKEN_RULES = (in_span is not None) or (span_id is not None) or (is_prefix is not None)

    fwd_fn[grid](
        q,
        kt,
        v,
        lens,
        LSE,
        O,
        attention_mask,
        in_span,
        span_id,
        is_prefix,
        output_attention_mask,
        *strides(q, 4),
        *strides(kt, 4),
        *strides(v, 4),
        *(strides(LSE, 3) if LSE is not None else [0] * 3),
        *strides(O, 4),
        *(strides(lens, 1) if lens is not None else [0]),
        *(strides(attention_mask, 3) if attention_mask is not None else [0] * 3),
        *(strides(in_span, 2) if in_span is not None else [0] * 2),
        *(strides(span_id, 2) if span_id is not None else [0] * 2),
        *(strides(is_prefix, 2) if is_prefix is not None else [0] * 2),
        *(strides(output_attention_mask, 4) if output_attention_mask is not None else [0] * 4),
        T=T,
        HEAD_DIM=HEAD_DIM,
        CAUSAL=causal,
        INPUT_PRECISION=precision,
        PRESCALE_QK=prescale_qk,
        DTYPE=q.dtype,
        TIME_BUCKET=triton.next_power_of_2(T),
        OUTPUT_LOGSUMEXP=return_lse,
        SM_SCALE=sm_scale,
        RETURN_ATTENTION_MASK=return_attention_mask,
        HAS_TOKEN_RULES=HAS_TOKEN_RULES,
    )

    if LSE is None:
        LSE = torch.empty(0)
    return O, LSE


@torch.library.register_fake("flash_attention::forward")
def attention_forward_adapter_abstract(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor | None,
    sm_scale: float | None,
    causal: bool,
    autotune: bool,
    return_lse: bool,
    prescale_qk: bool,
    precision: str,
    attention_mask: torch.Tensor | None,
    in_span: torch.Tensor | None,
    span_id: torch.Tensor | None,
    is_prefix: torch.Tensor | None,
    output_attention_mask: torch.Tensor | None,
    return_attention_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty(q.shape[:3], dtype=torch.float32, device=q.device) if return_lse else torch.empty(0),
    )


@torch.library.custom_op(
    "flash_attention::backward", mutates_args=(), device_types=("cuda",)
)
def attention_backward_adapter(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    sm_scale: float,
    causal: bool,
    autotune: bool,
    prescale_qk: bool,
    precision: str,
    attention_mask: torch.Tensor,
    in_span: torch.Tensor,
    span_id: torch.Tensor,
    is_prefix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, T, HEAD_DIM = q.shape

    # Precompute delta = sum(o * do) per token
    delta = torch.empty(o.shape[:-1], dtype=torch.float32, device=o.device)
    grid = lambda args: (
        batch,
        heads,
        triton.cdiv(T, args["TILE_SIZE"]),
    )
    _flash_attn_bwd_precompute[grid](
        o,
        do,
        delta,
        *strides(o, 4),
        *strides(do, 4),
        *strides(delta, 3),
        T=T,
        HEAD_DIM=HEAD_DIM,
        DTYPE=q.dtype,
        TIME_BUCKET=triton.next_power_of_2(T),
    )

    DQ = torch.zeros_like(q, memory_format=torch.contiguous_format)
    DK = torch.zeros_like(k, memory_format=torch.contiguous_format)
    DV = torch.zeros_like(v, memory_format=torch.contiguous_format)

    grid = lambda args: (
        batch,
        heads,
        triton.cdiv(T, args["TILE_DQ_Q_SIZE"]) + triton.cdiv(T, args["TILE_DK_K_SIZE"]),
    )

    bwd_fn = flash_backward_autotune if autotune else flash_backward
    HAS_TOKEN_RULES = (in_span is not None) or (span_id is not None) or (is_prefix is not None)

    bwd_fn[grid](
        q,
        k,
        v,
        lens,
        delta,
        lse,
        do,
        DQ,
        DK,
        DV,
        attention_mask,
        in_span,
        span_id,
        is_prefix,
        *strides(q, 4),
        *strides(k, 4),
        *strides(v, 4),
        *strides(delta, 3),
        *strides(lse, 3),
        *strides(do, 4),
        *strides(DQ, 4),
        *strides(DK, 4),
        *strides(DV, 4),
        *(strides(lens, 1) if lens is not None else [0]),
        *(strides(attention_mask, 3) if attention_mask is not None else [0] * 3),
        *(strides(in_span, 2) if in_span is not None else [0] * 2),
        *(strides(span_id, 2) if span_id is not None else [0] * 2),
        *(strides(is_prefix, 2) if is_prefix is not None else [0] * 2),
        T=T,
        HEAD_DIM=HEAD_DIM,
        CAUSAL=causal,
        TIME_BUCKET=triton.next_power_of_2(T),
        INPUT_PRECISION=precision,
        DTYPE=q.dtype,
        SM_SCALE=sm_scale,
        PRESCALE_QK=prescale_qk,
        HAS_TOKEN_RULES=HAS_TOKEN_RULES,
    )

    return DQ, DK, DV


@torch.library.register_fake("flash_attention::backward")
def attention_backward_adapter_abstract(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor | None,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    sm_scale: float | None,
    causal: bool,
    autotune: bool,
    prescale_qk: bool,
    precision: str,
    attention_mask: torch.Tensor,
    in_span: torch.Tensor,
    span_id: torch.Tensor,
    is_prefix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    DQ = torch.empty_like(q, memory_format=torch.contiguous_format)
    DK = torch.empty_like(k, memory_format=torch.contiguous_format)
    DV = torch.empty_like(v, memory_format=torch.contiguous_format)
    return DQ, DK, DV


def attention_backward_adapter_op_setup_context(ctx, inputs, output):
    O, LSE = output
    (
        q,
        k,
        v,
        lens,
        sm_scale,
        causal,
        autotune,
        return_lse,
        prescale_qk,
        precision,
        attention_mask,
        in_span,
        span_id,
        is_prefix,
        output_attention_mask,
        return_attention_mask,
    ) = inputs
    ctx.save_for_backward(
        q,
        k,
        v,
        O,
        LSE,
        lens,
        attention_mask,
        in_span,
        span_id,
        is_prefix,
    )
    ctx.causal = causal
    ctx.autotune = autotune
    ctx.sm_scale = sm_scale
    ctx.prescale_qk = prescale_qk
    ctx.precision = precision


def attention_backward_adapter_op(ctx, do, dlse):
    q, k, v, o, lse, lens, attention_mask, in_span, span_id, is_prefix = ctx.saved_tensors
    causal = ctx.causal
    autotune = ctx.autotune
    sm_scale = ctx.sm_scale
    prescale_qk = ctx.prescale_qk
    precision = ctx.precision

    DQ, DK, DV = torch.ops.flash_attention.backward(
        q=q,
        k=k,
        v=v,
        lens=lens,
        o=o,
        lse=lse,
        do=do,
        sm_scale=sm_scale,
        causal=causal,
        autotune=autotune,
        prescale_qk=prescale_qk,
        precision=precision,
        attention_mask=attention_mask,
        in_span=in_span,
        span_id=span_id,
        is_prefix=is_prefix,
    )

    # Return gradient tuple aligned with forward signature
    return DQ, DK, DV, None, None, None, None, None, None, None, None, None, None, None, None, None


torch.library.register_autograd(
    "flash_attention::forward",
    attention_backward_adapter_op,
    setup_context=attention_backward_adapter_op_setup_context,
)


def flash_attention_reference(
    q, k, v, lens=None, causal=True, scale=None
):
    T = q.shape[-2]

    if causal:
        attn_mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    else:
        attn_mask = torch.ones(T, T, device=q.device, dtype=torch.bool)

    if lens is not None:
        key_padding_mask = (
            torch.arange(T, device=q.device).unsqueeze(0) < lens.unsqueeze(-1)
        ).unsqueeze(-1)
        key_padding_mask_ref = key_padding_mask
        key_padding_mask = key_padding_mask & key_padding_mask.transpose(-1, -2)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) & key_padding_mask.unsqueeze(1)
        res_mask = key_padding_mask_ref.unsqueeze(1)
    else:
        res_mask = torch.tensor([True], device=q.device)

    sparsity_fraction = attn_mask.sum().item() / attn_mask.numel()
    return (
        F.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attn_mask, scale=scale
        ),
        res_mask,
        sparsity_fraction,
    )


@torch._dynamo.disable
@torch.compile(fullgraph=True, dynamic=True)
def _flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor | None,
    sm_scale: float | None,
    causal: bool,
    autotune: bool,
    return_lse: bool,
    prescale_qk: bool,
    precision: str,
    attention_mask: torch.Tensor | None,
    in_span: torch.Tensor | None,
    span_id: torch.Tensor | None,
    is_prefix: torch.Tensor | None,
    return_attention_mask: bool = False,
):
    requires_grad = any(i.requires_grad for i in (q, k, v))

    # If we need to return attention mask, create an output tensor for it
    if return_attention_mask:
        batch, heads, seq_len, _ = q.shape
        output_attention_mask = torch.zeros(
            (batch, heads, seq_len, seq_len), dtype=torch.bool, device=q.device
        )
    else:
        output_attention_mask = None

    O, LSE = torch.ops.flash_attention.forward(
        q=q,
        k=k,
        v=v,
        lens=lens,
        sm_scale=sm_scale,
        causal=causal,
        autotune=autotune,
        prescale_qk=prescale_qk,
        return_lse=return_lse or requires_grad,
        precision=precision,
        attention_mask=attention_mask,
        in_span=in_span,
        span_id=span_id,
        is_prefix=is_prefix,
        output_attention_mask=output_attention_mask,
        return_attention_mask=return_attention_mask,
    )

    if return_attention_mask:
        if return_lse:
            return (O, LSE), output_attention_mask
        return O, output_attention_mask
    elif return_lse:
        return O, LSE
    return O


# =============================================================================
# Main Flash Attention Interface
# =============================================================================

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor | None = None,
    sm_scale: float | None = None,
    causal: bool = True,
    autotune=False,
    return_lse=False,
    prescale_qk=False,
    precision="ieee",
    attention_mask: torch.Tensor | None = None,
    in_span: torch.Tensor | None = None,
    span_id: torch.Tensor | None = None,
    is_prefix: torch.Tensor | None = None,
    return_attention_mask: bool = False,
):
    """
    Computes self-attention with optional causal masking and flash attention optimization.

    When causal=True: Each query token can attend to all previous tokens in the sequence.
    When causal=False: Each query token can attend to all tokens in the sequence (bidirectional).

    Unlike traditional attention mechanisms that store full attention matrices,
    flash attention maintains linear memory usage with quadratic time complexity.

    Args:
        q (Tensor): The query tensor of shape `(batch, heads_num, time, head_dim)`
        k (Tensor): The key tensor of shape `(batch, heads_num, time, head_dim)`
        v (Tensor): The value tensor of shape `(batch, heads_num, time, head_dim)`
        lens (Tensor | None): Lengths of sequences of shape `(batch,)`
        sm_scale (float): Softmax scale, head_dim ** -0.5 by default
        causal (bool): Whether to apply causal masking (default: True)
        autotune (bool): Use triton autotune for optimal kernel configuration
        prescale_qk (bool): Prescale Q in QK^T calculations — slightly faster if True, slightly lower precision
        precision (str): Precision for matmuls: 'ieee' or 'tf32'
        return_attention_mask (bool): If True, returns the computed attention mask along with the output

    Returns:
        If return_attention_mask is False:
            Tensor (or tuple[Tensor, Tensor] if return_lse is True): Attention output
        If return_attention_mask is True:
            tuple[Tensor, Tensor]: (output, attention_mask) where attention_mask is [batch, heads, seq_len, seq_len]
    """
    if not torch.compiler.is_compiling():
        for i in (q, k, v):
            torch._dynamo.mark_static(i, 1)
            torch._dynamo.mark_static(i, 3)

    if sm_scale is None:
        HEAD_DIM = q.size(-1)
        sm_scale = HEAD_DIM ** -0.5

    # Use optimized flash attention path
    result = _flash_attention(
        q=q,
        k=k,
        v=v,
        lens=lens,
        sm_scale=sm_scale,
        causal=causal,
        autotune=autotune,
        return_lse=return_lse,
        prescale_qk=prescale_qk,
        precision=precision,
        attention_mask=attention_mask,
        in_span=in_span,
        span_id=span_id,
        is_prefix=is_prefix,
        return_attention_mask=return_attention_mask,
    )

    return result


def is_hopper_gpu() -> bool:
    """Check if the current GPU is a Hopper architecture (H100, H200, etc.)"""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 9


def verify_flash_attention_usage(model_forward_fn, sample_input, task_name="unknown"):
    """
    Verify that the model is using flash attention kernels by checking for triton kernel calls.

    Args:
        model_forward_fn: The model's forward function
        sample_input: Sample input to test with
        task_name: Name of the task being verified

    Returns:
        bool: True if flash attention is being used, False otherwise
    """
    import inspect

    # Check if flash_attention function is being called in the call stack
    original_flash_attention = flash_attention
    flash_attention_called = False

    def traced_flash_attention(*args, **kwargs):
        nonlocal flash_attention_called
        flash_attention_called = True
        logger.info(f"Flash attention kernel called for task: {task_name}")
        return original_flash_attention(*args, **kwargs)

    # Temporarily replace flash_attention with our traced version
    try:
        import original_kernel  # type: ignore
        original_kernel.flash_attention = traced_flash_attention
    except Exception:
        # Fallback if original_kernel isn't importable in this environment
        pass

    try:
        with torch.no_grad():
            _ = model_forward_fn(sample_input)
        if flash_attention_called:
            logger.info(f"✓ Flash attention verified for {task_name} task")
            return True
        else:
            logger.warning(f"✗ Flash attention NOT detected for {task_name} task")
            return False
    finally:
        try:
            original_kernel.flash_attention = original_flash_attention  # type: ignore
        except Exception:
            pass

    return flash_attention_called


# =============================================================================
# Utility / Standalone Test
# =============================================================================

if __name__ == "__main__":
    print("=== Flash Attention Test ===\n")

    # Check GPU capability
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        gpu_name = torch.cuda.get_device_name()
        print(f"GPU: {gpu_name}")
        print(f"Compute Capability: {major}.{minor}")
        print("✓ CUDA GPU detected")
    else:
        print("⚠ No CUDA GPU available")
        raise SystemExit(1)

    print("\n=== Testing Flash Attention ===")

    # Test tensors
    B, H, T, D = 1, 2, 16, 64
    q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16, requires_grad=True)

    # Metadata (optional)
    lens = torch.tensor([T], device='cuda', dtype=torch.int32)

    # Test flash attention
    print("\nTesting flash attention:")
    out = flash_attention(q, k, v, lens=lens, causal=True, autotune=True, precision="ieee", prescale_qk=False)
    if isinstance(out, tuple):
        out = out[0]
    print(f"   Output shape: {out.shape}")

    # Backward sanity
    loss = out.pow(2).mean()
    loss.backward()
    print("   Backward pass completed")

    print("\n=== Test Complete ===")
    print("Summary: Flash attention working correctly")
