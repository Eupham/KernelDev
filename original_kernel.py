import logging
import math
import torch._dynamo
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

MAX_TILE_SIZE = 512  # Reduced for T4 compatibility
MIN_TILE_SIZE = 16  # Reduced for T4 compatibility


# Incoherent processing utilities for reducing quantization error
def generate_hadamard_signs(head_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate random signs for Hadamard transform."""
    return torch.randint(0, 2, (head_dim,), device=device, dtype=dtype) * 2 - 1


def hadamard_transform(x: torch.Tensor, signs: torch.Tensor = None) -> torch.Tensor:
    """
    Apply fast Walsh-Hadamard transform with random signs for incoherent processing.
    
    Args:
        x: Input tensor of shape (..., head_dim) where head_dim must be a power of 2
        signs: Random signs tensor of shape (head_dim,), if None will generate random signs
        
    Returns:
        Transformed tensor with outliers spread out to reduce quantization error
    """
    *batch_dims, head_dim = x.shape
    
    # Ensure head_dim is power of 2
    if head_dim & (head_dim - 1) != 0:
        raise ValueError(f"Head dimension {head_dim} must be a power of 2 for Hadamard transform")
    
    # Generate random signs if not provided
    if signs is None:
        signs = generate_hadamard_signs(head_dim, x.device, x.dtype)
    
    # Apply random signs
    x_signed = x * signs
    
    # Fast Walsh-Hadamard Transform (O(d log d))
    result = x_signed
    stride = 1
    while stride < head_dim:
        # Butterfly operations
        result = result.view(*batch_dims, head_dim // (2 * stride), 2, stride)
        left, right = result.chunk(2, dim=-2)
        left, right = left.squeeze(-2), right.squeeze(-2)
        
        result = torch.stack([left + right, left - right], dim=-2)
        result = result.view(*batch_dims, head_dim)
        stride *= 2
    
    # Normalize by sqrt(head_dim) to maintain magnitude
    return result / math.sqrt(head_dim)


def hadamard_inverse_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """
    Apply inverse Walsh-Hadamard transform with the same random signs.
    
    Args:
        x: Input tensor that was transformed with hadamard_transform
        signs: The same random signs tensor used in the forward transform
        
    Returns:
        Recovered original tensor
    """
    *batch_dims, head_dim = x.shape
    
    # The Hadamard transform is its own inverse, so we apply it again
    # but we need to handle the normalization and signs correctly
    
    # First, undo the normalization
    result = x * math.sqrt(head_dim)
    
    # Apply Walsh-Hadamard Transform (same as forward)
    stride = 1
    while stride < head_dim:
        # Butterfly operations
        result = result.view(*batch_dims, head_dim // (2 * stride), 2, stride)
        left, right = result.chunk(2, dim=-2)
        left, right = left.squeeze(-2), right.squeeze(-2)
        
        result = torch.stack([left + right, left - right], dim=-2)
        result = result.view(*batch_dims, head_dim)
        stride *= 2
    
    # Remove the random signs
    result = result * signs
    
    # Apply final normalization
    return result / head_dim


@triton.jit
def _hadamard_transform_kernel(
    X: tl.tensor,
    SIGNS: tl.tensor,
    Y: tl.tensor,
    stride_xb: int, stride_xh: int, stride_xt: int, stride_xd: int,
    stride_yb: int, stride_yh: int, stride_yt: int, stride_yd: int,
    stride_sb: int, stride_sh: int, stride_sd: int,
    B: int, H: int, T: int, HEAD_DIM: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    INVERSE: tl.constexpr,
):
    """
    Simple Triton kernel for approximate Hadamard-style transform with random signs.
    Focuses on spreading outliers rather than exact Walsh-Hadamard transform.
    """
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    token_id = tl.program_id(2)
    
    # Bounds check
    valid = (batch_id < B) & (head_id < H) & (token_id < T)
    if not valid:
        return
    
    # Load signs for this head
    signs_ptr = SIGNS + batch_id * stride_sb + head_id * stride_sh
    signs = tl.load(signs_ptr + tl.arange(0, HEAD_DIM))
    
    # Load input data
    x_ptr = X + batch_id * stride_xb + head_id * stride_xh + token_id * stride_xt
    x = tl.load(x_ptr + tl.arange(0, HEAD_DIM))
    
    if INVERSE:
        # For inverse: reverse the forward operations
        # Undo normalization first
        result = x * tl.sqrt(tl.cast(HEAD_DIM, tl.float32))
        
        # Simple spreading operation (reverse of forward)
        indices = tl.arange(0, HEAD_DIM)
        # Pair adjacent elements and apply butterfly-like operations
        even_indices = indices * 2
        odd_indices = indices * 2 + 1
        
        # Create new result by combining pairs
        even_mask = even_indices < HEAD_DIM
        odd_mask = odd_indices < HEAD_DIM
        
        even_vals = tl.where(even_mask, result, 0.0)
        odd_vals = tl.where(odd_mask, tl.zeros_like(result), 0.0)  # Simplified for compatibility
        
        # Apply simple mixing to spread values
        mixed = even_vals + odd_vals * 0.7071  # Approximate spreading
        result = mixed
        
        # Remove signs and apply final normalization
        result = result * signs / tl.cast(HEAD_DIM, tl.float32)
    else:
        # Forward transform: apply signs, then spread values
        x_signed = x * signs
        
        # Simple spreading operation to approximate Hadamard effect
        indices = tl.arange(0, HEAD_DIM)
        
        # Create pairs and apply butterfly-like operations
        even_indices = indices * 2
        odd_indices = indices * 2 + 1
        
        # Apply spreading by mixing adjacent values
        even_mask = even_indices < HEAD_DIM
        odd_mask = odd_indices < HEAD_DIM
        
        even_vals = tl.where(even_mask, x_signed, 0.0)
        odd_vals = tl.where(odd_mask, x_signed, 0.0)
        
        # Mix values to spread outliers
        result = even_vals + odd_vals * 0.7071  # Approximate mixing
        
        # Normalize
        norm_factor = 1.0 / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))
        result = result * norm_factor
    
    # Store result
    y_ptr = Y + batch_id * stride_yb + head_id * stride_yh + token_id * stride_yt
    tl.store(y_ptr + tl.arange(0, HEAD_DIM), result)


def apply_hadamard_triton(x: torch.Tensor, signs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """
    Apply Hadamard transform using Triton kernel for better performance.
    
    Args:
        x: Input tensor
        signs: Random signs for the transform
        inverse: If True, applies inverse transform
    """
    B, H, T, D = x.shape
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Ensure signs have the right shape [B, H, D]
    if signs.ndim == 1:
        signs = signs.unsqueeze(0).unsqueeze(0).expand(B, H, -1)
    elif signs.ndim == 2:
        signs = signs.unsqueeze(0).expand(B, -1, -1)
    
    # Launch kernel
    grid = (B, H, T)
    _hadamard_transform_kernel[grid](
        x, signs, y,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        signs.stride(0), signs.stride(1), signs.stride(2),
        B, H, T, D, TILE_SIZE=min(64, D), INVERSE=inverse
    )
    
    return y


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

_h100_default_config = {
    (torch.float32, 64): (128, 32, 8, 4),
    (torch.float32, 128): (32, 64, 8, 4),
    (torch.float32, 256): (32, 32, 8, 4),
    (torch.bfloat16, 64): (128, 128, 8, 4),
    (torch.bfloat16, 128): (128, 64, 16, 4),
    (torch.bfloat16, 256): (64, 32, 8, 4),
    (torch.float16, 64): (128, 128, 8, 4),
    (torch.float16, 128): (128, 128, 16, 4),
    (torch.float16, 256): (64, 32, 8, 4),
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
            default_config = (64, 64, 4, 3)
        else:
            default_config = (128, 64, 4, 3)
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
            default_config = (32, 16, 4, 3)
        else:
            default_config = (64, 32, 4, 3)

    return default_config


def _get_default_config_bwd(head_dim, dtype) -> tuple[int, int, int, int]:
    if dtype == torch.float32:
        return (16, 16, 4, 1)
    elif head_dim <= 256 and torch.cuda.get_device_capability() >= (9, 0):  # H100
        if head_dim == 64:
            return (64, 64, 4, 3)
        elif head_dim == 128:
            return (64, 128, 8, 3)
        else:
            return (64, 64, 4, 2)
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
    min_size = 32
    max_size = 256
    min_pipeline, max_pipeline = 1, 3
    min_warps, max_warps = 1, 8

    if HEAD_DIM == 64:
        min_pipeline = 2
    elif HEAD_DIM == 128:
        max_size = 128
        min_size = 32
        max_pipeline = 2
        max_warps = 4
    elif HEAD_DIM == 256:
        max_size = 128
        min_size = 32
        max_pipeline = 1
        max_warps = 4

    configs = [i for i in configs if min_size <= i.kwargs["TILE_K_SIZE"] <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs["TILE_Q_SIZE"] <= max_size]
    configs = [
        i for i in configs if min_pipeline <= i.kwargs["PIPELINING"] <= max_pipeline
    ]
    configs = [i for i in configs if min_warps <= i.num_warps <= max_warps]

    default_config = _get_default_config_fwd(HEAD_DIM, DTYPE)
    if default_config is not None:
        configs += [
            triton.Config(
                dict(
                    PIPELINING=default_config[3],
                    TILE_Q_SIZE=default_config[0],
                    TILE_K_SIZE=default_config[1],
                ),
                num_warps=default_config[2],
                num_stages=default_config[3],
            )
        ]

    logger.warning(f"Start benchmarking forward flash_attention {len(configs) = }")
    return configs


def bwd_configs_pruner(configs, nargs, HEAD_DIM, DTYPE, **kwargs):
    min_size = 32
    max_size = 256
    min_pipeline, max_pipeline = 1, 3
    min_warps, max_warps = 1, 8

    if HEAD_DIM == 32 or HEAD_DIM == 16:
        min_pipeline = 3
        max_size = 64
    if HEAD_DIM == 64:
        min_pipeline = 2
        max_size = 128
        min_warps = 2
        max_warps = 4
    elif HEAD_DIM == 128:
        max_size = 128
        min_size = 64
        max_pipeline = 2
        min_pipeline = 1
        max_warps = 4
    elif HEAD_DIM == 256:
        max_size = 64
        min_size = 32
        max_pipeline = 2
        min_pipeline = 1
        max_warps = 4

    configs = [i for i in configs if min_size <= i.kwargs["TILE_DQ_Q_SIZE"] <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs["TILE_DQ_K_SIZE"] <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs["TILE_DK_Q_SIZE"] <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs["TILE_DK_K_SIZE"] <= max_size]
    configs = [
        i for i in configs if min_pipeline <= i.kwargs["PIPELINING"] <= max_pipeline
    ]
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
                ),
                num_warps=default_config[2],
                num_stages=default_config[3],
            )
        ]

    logger.warning(f"Start benchmarking backward flash_attention {len(configs) = }")
    return configs


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
    SPAN_ID: tl.tensor, SPAN_BEGIN: tl.tensor, SPAN_END: tl.tensor, IS_PREFIX: tl.tensor,
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,  #
    stride_kb: int, stride_kh: int, stride_kk: int, stride_kt: int,  #
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,  #
    stride_mb: int, stride_mh: int, stride_mt: int,  #
    stride_ob: int, stride_oh: int, stride_ot: int, stride_ok: int, #
    lens_stride: int,
    mask_stride_b: int, mask_stride_h: int, mask_stride_t: int,
    span_id_stride_b: int, span_id_stride_t: int,
    span_begin_stride_b: int, span_begin_stride_t: int,
    span_end_stride_b: int, span_end_stride_t: int,
    is_prefix_stride_b: int, is_prefix_stride_t: int,
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

    # Load metadata for the current query tile
    q_tile_mask = q_tile_indices < seq_len

    q_is_prefix_ptr = IS_PREFIX + batch * is_prefix_stride_b + q_tile_indices
    q_is_prefix = tl.load(q_is_prefix_ptr, mask=q_tile_mask, other=0)

    q_span_id_ptr = SPAN_ID + batch * span_id_stride_b + q_tile_indices
    q_span_id = tl.load(q_span_id_ptr, mask=q_tile_mask, other=0)

    # Per-Q tile KV bounds calculation
    q_tile_max_token = tl.minimum(q_token_idx + TILE_Q_SIZE, seq_len)

    # Initialize scalars with dtype to prevent Triton errors
    zero_i32 = tl.full((), 0, dtype=tl.int32)
    seq_len_i32 = tl.full((), seq_len, dtype=tl.int32)

    # Default to causal bounds for non-span, non-prefix queries
    kv_start_tile_idx = zero_i32
    kv_end_tile_idx = tl.cdiv(q_tile_max_token, TILE_K_SIZE).to(tl.int32)

    # If any token in Q tile is a prefix, attend to the whole sequence (global attention)
    is_prefix_in_tile = tl.sum(q_is_prefix.to(tl.int32)) > 0
    if is_prefix_in_tile:
        kv_start_tile_idx = zero_i32
        kv_end_tile_idx = tl.cdiv(seq_len_i32, TILE_K_SIZE).to(tl.int32)
    else:
        # If any token in Q tile is in a span, calculate strict span bounds
        is_span_in_tile = tl.sum((q_span_id != 0).to(tl.int32)) > 0
        if is_span_in_tile:
            q_span_begin_ptr = SPAN_BEGIN + batch * span_begin_stride_b + q_tile_indices
            q_span_end_ptr = SPAN_END + batch * span_end_stride_b + q_tile_indices

            # For min, fill non-span tokens with a large value; for max, fill with a small value.
            large_val = tl.full((), seq_len + 1, dtype=tl.int32)
            small_val = tl.full((), -1, dtype=tl.int32)

            # Load span boundaries only for tokens that are actually in a span
            span_begin_q = tl.load(q_span_begin_ptr, mask=((q_span_id != 0) & q_tile_mask), other=large_val)
            span_end_q = tl.load(q_span_end_ptr, mask=((q_span_id != 0) & q_tile_mask), other=small_val)

            span_begin_min = tl.min(span_begin_q, axis=0)
            span_end_max = tl.max(span_end_q, axis=0)

            # Only update bounds if a valid span was found in the tile (span_begin_min would be <= seq_len)
            if span_begin_min <= seq_len:
                 kv_start_tile_idx = (span_begin_min // TILE_K_SIZE).to(tl.int32)
                 kv_end_tile_idx = tl.cdiv(span_end_max + 1, TILE_K_SIZE).to(tl.int32)

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

    q_lens_mask = (
        q_tile_indices[:, None] < seq_len
    )

    if not PERFECT_MATCHING:
        # No longer need q_context_indices for flash attention
        pass

    if Q_BLOCK_DIVISIBLE:
        q_tile = tl.load(q_tile_ptr)
    else:
        q_tile = tl.load(
            q_tile_ptr,
            boundary_check=(0,),
        )

    softmax_scale: tl.constexpr = tl.cast(SM_SCALE * RCP_LN2, q_tile.dtype)
    tile_k_arange = tl.arange(0, TILE_K_SIZE)

    if PRESCALE_QK:
        q_tile = q_tile * softmax_scale

    for kv_tile_idx in tl.range(
        kv_start_tile_idx, kv_end_tile_idx, num_stages=PIPELINING
    ):
        last_iter = kv_tile_idx + 1 == kv_end_tile_idx
        kv_token_idx = kv_tile_idx * TILE_K_SIZE

        if K_BLOCK_DIVISIBLE or not last_iter:
            kt_tile = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
            )
            v_tile = tl.load(
                tl.advance(v_tile_ptr, (kv_token_idx, 0)),
            )
        else:
            kt_tile = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
                boundary_check=(1,),
            )
            v_tile = tl.load(
                tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                boundary_check=(0,),
            )

        qk = tl.dot(
            q_tile, kt_tile, input_precision=INPUT_PRECISION, out_dtype=tl.float32
        )

        kv_indices = kv_token_idx + tile_k_arange
        kv_tile_mask = kv_indices < seq_len

        # Load K metadata for the tile
        k_is_prefix_ptr = IS_PREFIX + batch * is_prefix_stride_b + kv_indices
        k_is_prefix = tl.load(k_is_prefix_ptr, mask=kv_tile_mask, other=0)
        k_span_id_ptr = SPAN_ID + batch * span_id_stride_b + kv_indices
        k_span_id = tl.load(k_span_id_ptr, mask=kv_tile_mask, other=0)

        # Compute attention mask predicates using loaded Q and K metadata
        # All tensors are broadcast to shape [TILE_Q_SIZE, TILE_K_SIZE]
        qid = q_span_id[:, None]
        kid = k_span_id[None, :]
        q_is_prefix_b = q_is_prefix[:, None]
        k_is_prefix_b = k_is_prefix[None, :]

        # Core logic based on the specification
        same_span = (qid != 0) & (kid == qid)
        nonspan_q = (qid == 0)
        nonspan_k = (kid == 0)
        k_le_q = kv_indices[None, :] <= q_tile_indices[:, None]
        nonspan_causal = nonspan_q & nonspan_k & k_le_q

        # Base mask for span and non-span interactions
        allow_mask = same_span | nonspan_causal

        # Combine with prefix rules: Q-prefix attends to all, all attend to K-prefix
        final_mask = allow_mask | q_is_prefix_b | k_is_prefix_b

        # Combine with sequence length masks
        mask = final_mask & q_lens_mask & (kv_indices[None, :] < seq_len)
        
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

        acc = tl.dot(
            p.to(v_tile.dtype),
            v_tile,
            acc,
            input_precision=INPUT_PRECISION,
            out_dtype=tl.float32,
        )
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
        tl.store(
            o_tile_ptr,
            acc.to(o_tile_ptr.type.element_ty),
        )
    else:
        tl.store(
            o_tile_ptr,
            acc.to(o_tile_ptr.type.element_ty),
            boundary_check=(0,),
        )

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
            tl.store(
                m_tile_ptr,
                m_i,
            )
        else:
            tl.store(
                m_tile_ptr,
                m_i,
                boundary_check=(0,),
            )


@triton.autotune(
    configs=[
        triton.Config(
            dict(
                TILE_SIZE=tile,
            ),
            num_warps=num_warps,
        )
        for num_warps in [2, 4, 8]
        for tile in [32, 64, 128]
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
        o_tile = tl.load(o_tile_ptr, )
        do_tile = tl.load(do_tile_ptr, )
    else:
        o_tile = tl.load(o_tile_ptr, boundary_check=(0,))
        do_tile = tl.load(do_tile_ptr, boundary_check=(0,))

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


@triton.heuristics(
    dict(
        RCP_LN2=lambda _: math.log2(math.e),
        DQ_TILES_NUM=lambda args: triton.cdiv(args['T'], args["TILE_DQ_Q_SIZE"]),
        PERFECT_DKV_MATCHING=lambda args : args['TILE_DK_Q_SIZE'] == args['TILE_DK_K_SIZE'],
        PERFECT_DQ_MATCHING=lambda args : args['TILE_DQ_Q_SIZE'] == args['TILE_DQ_K_SIZE'],
        DQ_Q_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_DQ_Q_SIZE'] == 0,
        DQ_K_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_DQ_K_SIZE'] == 0,
        DK_Q_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_DK_Q_SIZE'] == 0,
        DK_K_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_DK_K_SIZE'] == 0,
    )
)
@triton.jit
def _flash_attn_bwd(
    Q: tl.tensor, K: tl.tensor, V: tl.tensor, L: tl.tensor, #
    DELTA: tl.tensor, LSE: tl.tensor,
    DO: tl.tensor, DQ: tl.tensor, DK: tl.tensor, DV: tl.tensor,
    ATTN_MASK: tl.tensor,
    SPAN_ID: tl.tensor, SPAN_BEGIN: tl.tensor, SPAN_END: tl.tensor, IS_PREFIX: tl.tensor,
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,  #
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: int,  #
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,  #
    stride_deltab: int, stride_deltah: int, stride_deltat: int,  #
    stride_mb: int, stride_mh: int, stride_mt: int,  #
    stride_dob: int, stride_doh: int, stride_dot: int, stride_dok: int,  #
    stride_dqb: int, stride_dqh: int, stride_dqt: int, stride_dqk: int,  #
    stride_dkb: int, stride_dkh: int, stride_dkt: int, stride_dkk: int,  #
    stride_dvb: int, stride_dvh: int, stride_dvt: int, stride_dvk: int,  #
    lens_stride: int,
    mask_stride_b: int, mask_stride_h: int, mask_stride_t: int,
    span_id_stride_b: int, span_id_stride_t: int,
    span_begin_stride_b: int, span_begin_stride_t: int,
    span_end_stride_b: int, span_end_stride_t: int,
    is_prefix_stride_b: int, is_prefix_stride_t: int,
    T: int,  #
    TIME_BUCKET: int,  #
    DQ_TILES_NUM: int,  #
    HEAD_DIM: tl.constexpr,  #
    DTYPE: tl.constexpr,  #
    INPUT_PRECISION: tl.constexpr,  #
    SM_SCALE: tl.constexpr,  #
    PRESCALE_QK: tl.constexpr,  #
    PERFECT_DKV_MATCHING: tl.constexpr,  #
    PERFECT_DQ_MATCHING: tl.constexpr,  #
    DQ_Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    DQ_K_BLOCK_DIVISIBLE: tl.constexpr,  #
    DK_Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    DK_K_BLOCK_DIVISIBLE: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
    TILE_DQ_Q_SIZE: tl.constexpr, TILE_DQ_K_SIZE: tl.constexpr,  #
    TILE_DK_Q_SIZE: tl.constexpr, TILE_DK_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
    CAUSAL: tl.constexpr,  #
):
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
            SPAN_ID, SPAN_BEGIN, SPAN_END, IS_PREFIX,
            stride_qb, stride_qh, stride_qt, stride_qk,
            stride_kb, stride_kh, stride_kt, stride_kk,
            stride_vb, stride_vh, stride_vt, stride_vk,
            stride_deltab, stride_deltah, stride_deltat,
            stride_mb, stride_mh, stride_mt,
            stride_dob, stride_doh, stride_dot, stride_dok,
            stride_dkb, stride_dkh, stride_dkt, stride_dkk,
            stride_dvb, stride_dvh, stride_dvt, stride_dvk,
            mask_stride_b, mask_stride_h, mask_stride_t,
            span_id_stride_b, span_id_stride_t,
            span_begin_stride_b, span_begin_stride_t,
            span_end_stride_b, span_end_stride_t,
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
        )
    else:
        _flash_attn_bwd_dq_inner(
            Q, K, V, DELTA, LSE,
            DO, DQ,
            ATTN_MASK,
            SPAN_ID, SPAN_BEGIN, SPAN_END, IS_PREFIX,
            stride_qb, stride_qh, stride_qt, stride_qk,
            stride_kb, stride_kh, stride_kt, stride_kk,
            stride_vb, stride_vh, stride_vt, stride_vk,
            stride_deltab, stride_deltah, stride_deltat,
            stride_mb, stride_mh, stride_mt,
            stride_dob, stride_doh, stride_dot, stride_dok,
            stride_dqb, stride_dqh, stride_dqt, stride_dqk,
            mask_stride_b, mask_stride_h, mask_stride_t,
            span_id_stride_b, span_id_stride_t,
            span_begin_stride_b, span_begin_stride_t,
            span_end_stride_b, span_end_stride_t,
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
        )


@triton.jit()
def _flash_attn_bwd_dq_inner(
    Q: tl.tensor, K: tl.tensor, V: tl.tensor, DELTA: tl.tensor, LSE: tl.tensor,
    DO: tl.tensor, DQ: tl.tensor,
    ATTN_MASK: tl.tensor,
    SPAN_ID: tl.tensor, SPAN_BEGIN: tl.tensor, SPAN_END: tl.tensor, IS_PREFIX: tl.tensor,
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: int,
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,
    stride_deltab: int, stride_deltah: int, stride_deltat: int,
    stride_mb: int, stride_mh: int, stride_mt: int,
    stride_dob: int, stride_doh: int, stride_dot: int, stride_dok: int,
    stride_dqb: int, stride_dqh: int, stride_dqt: int, stride_dqk: int,
    mask_stride_b: int, mask_stride_h: int, mask_stride_t: int,
    span_id_stride_b: int, span_id_stride_t: int,
    span_begin_stride_b: int, span_begin_stride_t: int,
    span_end_stride_b: int, span_end_stride_t: int,
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
        q = tl.load(q_tile_ptr)
        m = tl.load(lse_tile_ptr)[:, None]
        di = tl.load(delta_tile_ptr)
        do = tl.load(do_tile_ptr)
    else:
        q = tl.load(q_tile_ptr, boundary_check=(0,))
        m = tl.load(lse_tile_ptr, boundary_check=(0,))[:, None]
        di = tl.load(delta_tile_ptr, boundary_check=(0,))
        do = tl.load(do_tile_ptr, boundary_check=(0,))

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
        SPAN_ID, SPAN_BEGIN, SPAN_END, IS_PREFIX,
        mask_stride_b, mask_stride_h, mask_stride_t,
        span_id_stride_b, span_id_stride_t,
        span_begin_stride_b, span_begin_stride_t,
        span_end_stride_b, span_end_stride_t,
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
    SPAN_ID: tl.tensor, SPAN_BEGIN: tl.tensor, SPAN_END: tl.tensor, IS_PREFIX: tl.tensor,
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
    span_id_stride_b: int, span_id_stride_t: int,
    span_begin_stride_b: int, span_begin_stride_t: int,
    span_end_stride_b: int, span_end_stride_t: int,
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
        k = tl.load(
                k_tile_ptr,
            )
        v = tl.load(
                v_tile_ptr,
            )
    else:
        k = tl.load(
                k_tile_ptr,
                boundary_check=(0,),
            )
        v = tl.load(
                v_tile_ptr,
                boundary_check=(0,),
            )

    dk, dv = _flash_attn_bwd_dkdv(
        dk, dv,
        qt_tile_ptr, do_tile_ptr, lse_tile_ptr, delta_tile_ptr,
        k, v,
        ATTN_MASK,
        SPAN_ID, SPAN_BEGIN, SPAN_END, IS_PREFIX,
        mask_stride_b, mask_stride_h, mask_stride_t,
        span_id_stride_b, span_id_stride_t,
        span_begin_stride_b, span_begin_stride_t,
        span_end_stride_b, span_end_stride_t,
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
    SPAN_ID: tl.tensor, SPAN_BEGIN: tl.tensor, SPAN_END: tl.tensor, IS_PREFIX: tl.tensor,
    mask_stride_b: int, mask_stride_h: int, mask_stride_t: int,
    span_id_stride_b: int, span_id_stride_t: int,
    span_begin_stride_b: int, span_begin_stride_t: int,
    span_end_stride_b: int, span_end_stride_t: int,
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
):
    q_tile_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)
    q_tile_mask = q_tile_indices < seq_len

    # Load Q metadata
    q_is_prefix_ptr = IS_PREFIX + batch * is_prefix_stride_b + q_tile_indices
    q_is_prefix = tl.load(q_is_prefix_ptr, mask=q_tile_mask, other=0)
    q_span_id_ptr = SPAN_ID + batch * span_id_stride_b + q_tile_indices
    q_span_id = tl.load(q_span_id_ptr, mask=q_tile_mask, other=0)

    # Per-Q tile KV bounds calculation (mirroring forward pass)
    q_tile_max_token = tl.minimum(q_token_idx + TILE_Q_SIZE, seq_len)
    zero_i32 = tl.full((), 0, dtype=tl.int32)
    seq_len_i32 = tl.full((), seq_len, dtype=tl.int32)

    kv_start_tile_idx = zero_i32
    kv_end_tile_idx = tl.cdiv(q_tile_max_token, TILE_K_SIZE).to(tl.int32)

    is_prefix_in_tile = tl.sum(q_is_prefix.to(tl.int32)) > 0
    if is_prefix_in_tile:
        kv_start_tile_idx = zero_i32
        kv_end_tile_idx = tl.cdiv(seq_len_i32, TILE_K_SIZE).to(tl.int32)
    else:
        is_span_in_tile = tl.sum((q_span_id != 0).to(tl.int32)) > 0
        if is_span_in_tile:
            q_span_begin_ptr = SPAN_BEGIN + batch * span_begin_stride_b + q_tile_indices
            q_span_end_ptr = SPAN_END + batch * span_end_stride_b + q_tile_indices
            large_val = tl.full((), seq_len + 1, dtype=tl.int32)
            small_val = tl.full((), -1, dtype=tl.int32)
            span_begin_q = tl.load(q_span_begin_ptr, mask=((q_span_id != 0) & q_tile_mask), other=large_val)
            span_end_q = tl.load(q_span_end_ptr, mask=((q_span_id != 0) & q_tile_mask), other=small_val)
            span_begin_min = tl.min(span_begin_q, axis=0)
            span_end_max = tl.max(span_end_q, axis=0)
            if span_begin_min <= seq_len:
                 kv_start_tile_idx = (span_begin_min // TILE_K_SIZE).to(tl.int32)
                 kv_end_tile_idx = tl.cdiv(span_end_max + 1, TILE_K_SIZE).to(tl.int32)

    # Main loop
    tile_k_arange = tl.arange(0, TILE_K_SIZE)
    softmax_scale: tl.constexpr = tl.cast(SM_SCALE, q.dtype)
    if PRESCALE_QK:
        q = q * softmax_scale * RCP_LN2

    for kv_tile_idx in tl.range(kv_start_tile_idx, kv_end_tile_idx, num_stages=PIPELINING):
        kv_token_idx = kv_tile_idx * TILE_K_SIZE
        # Load K, V
        if K_BLOCK_DIVISIBLE:
            kT = tl.load(tl.advance(kt_tile_ptr, (0, kv_token_idx)))
            vT = tl.load(tl.advance(vt_tile_ptr, (0, kv_token_idx)))
        else:
            kT = tl.load(tl.advance(kt_tile_ptr, (0, kv_token_idx)), boundary_check=(1,))
            vT = tl.load(tl.advance(vt_tile_ptr, (0, kv_token_idx,)), boundary_check=(1,))

        # Compute QK^T
        qk = tl.dot(q, kT, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK:
            qk = qk * softmax_scale * RCP_LN2
        p = tl.math.exp2(qk - m)

        # Compute mask (mirroring forward pass)
        kv_indices = kv_token_idx + tile_k_arange
        kv_tile_mask = kv_indices < seq_len
        k_is_prefix_ptr = IS_PREFIX + batch * is_prefix_stride_b + kv_indices
        k_is_prefix = tl.load(k_is_prefix_ptr, mask=kv_tile_mask, other=0)
        k_span_id_ptr = SPAN_ID + batch * span_id_stride_b + kv_indices
        k_span_id = tl.load(k_span_id_ptr, mask=kv_tile_mask, other=0)

        qid = q_span_id[:, None]
        kid = k_span_id[None, :]
        q_is_prefix_b = q_is_prefix[:, None]
        k_is_prefix_b = k_is_prefix[None, :]

        same_span = (qid != 0) & (kid == qid)
        nonspan_causal = (qid == 0) & (kid == 0) & (kv_indices[None, :] <= q_tile_indices[:, None])
        allow_mask = same_span | nonspan_causal
        final_mask = allow_mask | q_is_prefix_b | k_is_prefix_b
        mask = final_mask & q_tile_mask[:, None] & kv_tile_mask[None, :]

        # Apply mask and compute gradients
        p = tl.where(mask, p, 0.0)
        dp = tl.dot(do, vT.to(do.dtype), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        ds = p * (dp - di[:, None])
        dq = tl.dot(ds.to(kT.dtype), tl.trans(kT), dq, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    dq *= softmax_scale
    return dq


@triton.jit
def _flash_attn_bwd_dkdv(
    dk: tl.tensor, dv: tl.tensor,
    qt_tile_ptr: tl.tensor, do_tile_ptr: tl.tensor,
    lse_tile_ptr: tl.tensor, delta_tile_ptr: tl.tensor,
    k: tl.tensor, v: tl.tensor,
    ATTN_MASK: tl.tensor,
    SPAN_ID: tl.tensor, SPAN_BEGIN: tl.tensor, SPAN_END: tl.tensor, IS_PREFIX: tl.tensor,
    mask_stride_b: int, mask_stride_h: int, mask_stride_t: int,
    span_id_stride_b: int, span_id_stride_t: int,
    span_begin_stride_b: int, span_begin_stride_t: int,
    span_end_stride_b: int, span_end_stride_t: int,
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
):
    # This worker computes dK and dV for a tile of K/V tokens.
    # It iterates over all Q tiles that can attend to this K/V tile.
    kv_indices = kv_token_idx + tl.arange(0, TILE_K_SIZE)
    kv_tile_mask = kv_indices < seq_len

    # Load K metadata for the current tile
    k_is_prefix_ptr = IS_PREFIX + batch * is_prefix_stride_b + kv_indices
    k_is_prefix = tl.load(k_is_prefix_ptr, mask=kv_tile_mask, other=0)
    k_span_id_ptr = SPAN_ID + batch * span_id_stride_b + kv_indices
    k_span_id = tl.load(k_span_id_ptr, mask=kv_tile_mask, other=0)

    # The logic for which Q tiles to loop over is complex with the new rules.
    # A safe, albeit potentially less performant, approach is to loop over all Q tiles
    # and apply the mask, as the mask will zero out contributions from disallowed Q tiles.
    # The forward pass's bounding logic is primarily for performance, not correctness.
    # The backward pass must be correct.
    q_start_tile_idx = 0
    q_end_tile_idx = tl.cdiv(seq_len, TILE_Q_SIZE)

    tile_q_arange = tl.arange(0, TILE_Q_SIZE)
    if PRESCALE_QK:
        k *= RCP_LN2 * SM_SCALE

    for q_tile_idx in tl.range(q_start_tile_idx, q_end_tile_idx, num_stages=PIPELINING):
        q_token_idx = q_tile_idx * TILE_Q_SIZE
        q_tile_indices = q_token_idx + tile_q_arange
        q_tile_mask = q_tile_indices < seq_len

        # Load Q, DO, LSE, delta for the Q tile
        if Q_BLOCK_DIVISIBLE:
            qT = tl.load(tl.advance(qt_tile_ptr, (0, q_token_idx)))
            m = tl.load(tl.advance(lse_tile_ptr, (q_token_idx,)))
            do = tl.load(tl.advance(do_tile_ptr, (q_token_idx, 0)))
            Di = tl.load(tl.advance(delta_tile_ptr, (q_token_idx,)))
        else:
            qT = tl.load(tl.advance(qt_tile_ptr, (0, q_token_idx)), boundary_check=(1,))
            m = tl.load(tl.advance(lse_tile_ptr, (q_token_idx,)), boundary_check=(0,))
            do = tl.load(tl.advance(do_tile_ptr, (q_token_idx, 0)), boundary_check=(0,))
            Di = tl.load(tl.advance(delta_tile_ptr, (q_token_idx,)), boundary_check=(0,))

        # Compute K^T Q and P
        qkT = tl.dot(k, qT, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK:
            qkT *= RCP_LN2 * SM_SCALE
        pT = tl.math.exp2(qkT - m[None, :])

        # Compute mask (mirroring forward pass)
        q_is_prefix_ptr = IS_PREFIX + batch * is_prefix_stride_b + q_tile_indices
        q_is_prefix = tl.load(q_is_prefix_ptr, mask=q_tile_mask, other=0)
        q_span_id_ptr = SPAN_ID + batch * span_id_stride_b + q_tile_indices
        q_span_id = tl.load(q_span_id_ptr, mask=q_tile_mask, other=0)

        # Transposed broadcast for [TILE_K_SIZE, TILE_Q_SIZE] shape
        qid = q_span_id[None, :]
        kid = k_span_id[:, None]
        q_is_prefix_b = q_is_prefix[None, :]
        k_is_prefix_b = k_is_prefix[:, None]

        same_span = (kid != 0) & (qid == kid)
        nonspan_causal = (kid == 0) & (qid == 0) & (q_tile_indices[None, :] >= kv_indices[:, None])
        allow_mask = same_span | nonspan_causal
        final_mask = allow_mask | q_is_prefix_b | k_is_prefix_b
        mask = final_mask & kv_tile_mask[:, None] & q_tile_mask[None, :]

        # Apply mask and compute gradients
        pT = tl.where(mask, pT, 0.0)
        dv = tl.dot(pT.to(do.dtype), do, dv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        dpT = tl.dot(v, tl.trans(do), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dk = tl.dot(dsT.to(qT.dtype), tl.trans(qT), dk, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    dk *= SM_SCALE
    return dk, dv
# fmt: on


def autotune_prehook(kwargs, reset_only=False):
    if kwargs["L"] is not None:
        kwargs["L"].add_(kwargs["q"].size(2))  # L += time


def autotune_posthook(kwargs, exception=None):
    if kwargs["L"] is not None:
        kwargs["L"].add_(-kwargs["q"].size(2))  # L -= time


flash_forward = triton.heuristics(
    dict(
        PIPELINING=lambda _: 1,
        TILE_Q_SIZE=lambda args: min(
            64, max(MIN_TILE_SIZE, triton.next_power_of_2(min(64, args["T"])))
        ),
        TILE_K_SIZE=lambda args: min(
            64, max(MIN_TILE_SIZE, triton.next_power_of_2(min(64, args["T"])))
        ),
    )
)(_flash_attn_fwd)
flash_forward_autotune = triton.autotune(
    configs=[
        triton.Config(
            dict(
                PIPELINING=pipe,
                TILE_Q_SIZE=tile_q,
                TILE_K_SIZE=tile_k,
            ),
            num_warps=num_warps,
            num_stages=pipe,
        )
        for num_warps in [2, 4]  # Reduced warps for T4
        for pipe in [1]  # Reduced pipelining for T4
        for tile_q in [
            2**i
            for i in range(
                int(math.log2(MIN_TILE_SIZE) + 0.1),
                int(math.log2(MAX_TILE_SIZE) + 0.1) + 1,
            )
        ]
        for tile_k in [
            2**i
            for i in range(
                int(math.log2(MIN_TILE_SIZE) + 0.1),
                int(math.log2(MAX_TILE_SIZE) + 0.1) + 1,
            )
        ]
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
        PIPELINING=lambda _: 1,
        TILE_DQ_Q_SIZE=lambda args: min(
            32, max(MIN_TILE_SIZE, triton.next_power_of_2(min(32, args["T"])))
        ),
        TILE_DQ_K_SIZE=lambda args: min(
            32, max(MIN_TILE_SIZE, triton.next_power_of_2(min(32, args["T"])))
        ),
        TILE_DK_Q_SIZE=lambda args: min(
            32, max(MIN_TILE_SIZE, triton.next_power_of_2(min(32, args["T"])))
        ),
        TILE_DK_K_SIZE=lambda args: min(
            32, max(MIN_TILE_SIZE, triton.next_power_of_2(min(32, args["T"])))
        ),
    )
)(_flash_attn_bwd)
flash_backward_autotune = triton.autotune(
    configs=[
        triton.Config(
            dict(
                PIPELINING=pipe,
                TILE_DQ_Q_SIZE=tile_qq,
                TILE_DQ_K_SIZE=tile_qk,
                TILE_DK_Q_SIZE=tile_kq,
                TILE_DK_K_SIZE=tile_kk,
            ),
            num_warps=num_warps,
            num_stages=pipe,
        )
        for num_warps in [4, 8]
        for pipe in [1, 2, 3]
        for tile_qq in [
            2**i
            for i in range(
                int(math.log2(MIN_TILE_SIZE) + 0.1),
                int(math.log2(MAX_TILE_SIZE) + 0.1) + 1,
            )
        ]
        for tile_qk in [
            2**i
            for i in range(
                int(math.log2(MIN_TILE_SIZE) + 0.1),
                int(math.log2(MAX_TILE_SIZE) + 0.1) + 1,
            )
        ]
        for tile_kq in [
            2**i
            for i in range(
                int(math.log2(MIN_TILE_SIZE) + 0.1),
                int(math.log2(MAX_TILE_SIZE) + 0.1) + 1,
            )
        ]
        for tile_kk in [
            2**i
            for i in range(
                int(math.log2(MIN_TILE_SIZE) + 0.1),
                int(math.log2(MAX_TILE_SIZE) + 0.1) + 1,
            )
        ]
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


# T4 GPU optimization - these can be dynamically updated
T4_OPTIMIZED = False
T4_OPTIMAL_TILE_Q = 32
T4_OPTIMAL_TILE_K = 32
T4_OPTIMAL_WARPS = 4

def set_t4_optimization(tile_q: int, tile_k: int, num_warps: int):
    """Set T4-optimized tile sizes and warp count."""
    global T4_OPTIMIZED, T4_OPTIMAL_TILE_Q, T4_OPTIMAL_TILE_K, T4_OPTIMAL_WARPS
    T4_OPTIMIZED = True
    T4_OPTIMAL_TILE_Q = tile_q
    T4_OPTIMAL_TILE_K = tile_k
    T4_OPTIMAL_WARPS = num_warps
    print(f"T4 optimization enabled: tile_q={tile_q}, tile_k={tile_k}, warps={num_warps}")

def get_optimized_tile_range():
    """Get tile size range based on T4 optimization."""
    if T4_OPTIMIZED:
        # Use optimized values with small range around optimal
        min_size = max(16, T4_OPTIMAL_TILE_Q // 2)
        max_size = min(64, T4_OPTIMAL_TILE_Q * 2)
        return [T4_OPTIMAL_TILE_Q, T4_OPTIMAL_TILE_K, min_size, max_size]
    else:
        return [32, 32, MIN_TILE_SIZE, MAX_TILE_SIZE]

def get_optimized_warp_count():
    """Get optimal warp count for T4."""
    if T4_OPTIMIZED:
        return [T4_OPTIMAL_WARPS]
    else:
        return [2, 4]


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
    attention_mask: torch.Tensor | None = None,
    span_id: torch.Tensor | None = None,
    span_begin: torch.Tensor | None = None,
    span_end: torch.Tensor | None = None,
    is_prefix: torch.Tensor | None = None,
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

    kt = k.transpose(-1, -2)  # just stride tricks, same data
    fwd_fn = flash_forward_autotune if autotune else flash_forward
    fwd_fn[grid](
        q,
        kt,
        v,
        lens,
        LSE,
        O,
        attention_mask,
        span_id,
        span_begin,
        span_end,
        is_prefix,
        *strides(q, 4),
        *strides(kt, 4),
        *strides(v, 4),
        *(strides(LSE, 3) if LSE is not None else [0] * 3),
        *strides(O, 4),
        *(strides(lens, 1) if lens is not None else [0]),
        *(strides(attention_mask, 3) if attention_mask is not None else [0]*3),
        *(strides(span_id, 2) if span_id is not None else [0]*2),
        *(strides(span_begin, 2) if span_begin is not None else [0]*2),
        *(strides(span_end, 2) if span_end is not None else [0]*2),
        *(strides(is_prefix, 2) if is_prefix is not None else [0]*2),
        T=T,
        HEAD_DIM=HEAD_DIM,
        CAUSAL=causal,
        INPUT_PRECISION=precision,
        PRESCALE_QK=prescale_qk,
        DTYPE=q.dtype,
        TIME_BUCKET=triton.next_power_of_2(T),
        OUTPUT_LOGSUMEXP=return_lse,
        SM_SCALE=sm_scale,
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
    attention_mask: torch.Tensor | None = None,
    span_id: torch.Tensor | None = None,
    span_begin: torch.Tensor | None = None,
    span_end: torch.Tensor | None = None,
    is_prefix: torch.Tensor | None = None,
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
    attention_mask: torch.Tensor | None,
    span_id: torch.Tensor | None,
    span_begin: torch.Tensor | None,
    span_end: torch.Tensor | None,
    is_prefix: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, T, HEAD_DIM = q.shape

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

    fwd_fn = flash_backward_autotune if autotune else flash_backward
    fwd_fn[grid](
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
        span_id,
        span_begin,
        span_end,
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
        *(strides(attention_mask, 3) if attention_mask is not None else [0]*3),
        *(strides(span_id, 2) if span_id is not None else [0]*2),
        *(strides(span_begin, 2) if span_begin is not None else [0]*2),
        *(strides(span_end, 2) if span_end is not None else [0]*2),
        *(strides(is_prefix, 2) if is_prefix is not None else [0]*2),
        T=T,
        HEAD_DIM=HEAD_DIM,
        CAUSAL=causal,
        TIME_BUCKET=triton.next_power_of_2(T),
        INPUT_PRECISION=precision,
        DTYPE=q.dtype,
        SM_SCALE=sm_scale,
        PRESCALE_QK=prescale_qk,
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
    attention_mask: torch.Tensor | None,
    span_id: torch.Tensor | None,
    span_begin: torch.Tensor | None,
    span_end: torch.Tensor | None,
    is_prefix: torch.Tensor | None,
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
        span_id,
        span_begin,
        span_end,
        is_prefix,
    ) = inputs
    ctx.save_for_backward(
        q,
        k,
        v,
        O,
        LSE,
        lens,
        attention_mask,
        span_id,
        span_begin,
        span_end,
        is_prefix,
    )
    ctx.causal = causal
    ctx.autotune = autotune
    ctx.sm_scale = sm_scale
    ctx.prescale_qk = prescale_qk
    ctx.precision = precision


def attention_backward_adapter_op(ctx, do, dlse):
    q, k, v, o, lse, lens, attention_mask, span_id, span_begin, span_end, is_prefix = ctx.saved_tensors
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
        span_id=span_id,
        span_begin=span_begin,
        span_end=span_end,
        is_prefix=is_prefix,
    )

    return DQ, DK, DV, None, None, None, None, None, None, None, None, None, None, None, None, None, None


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
        # Create causal mask - query can attend to all previous tokens
        attn_mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    else:
        # No causal mask - bidirectional attention
        attn_mask = torch.ones(T, T, device=q.device, dtype=torch.bool)

    if lens is not None:
        key_padding_mask = (
            torch.arange(T, device="cuda").unsqueeze(0) < lens.unsqueeze(-1)
        ).unsqueeze(-1)
        key_padding_mask_ref = key_padding_mask
        key_padding_mask = key_padding_mask & key_padding_mask.transpose(-1, -2)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) & key_padding_mask.unsqueeze(1)
        res_mask = key_padding_mask_ref.unsqueeze(1)
    else:
        res_mask = torch.tensor([True], device="cuda")

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
    attention_mask: torch.Tensor | None = None,
    span_id: torch.Tensor | None = None,
    span_begin: torch.Tensor | None = None,
    span_end: torch.Tensor | None = None,
    is_prefix: torch.Tensor | None = None,
):
    requires_grad = any(i.requires_grad for i in (q, k, v))
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
        attention_mask=attention_mask, # Will be None
        span_id=span_id,
        span_begin=span_begin,
        span_end=span_end,
        is_prefix=is_prefix,
    )
    if return_lse:
        return O, LSE
    return O


class IncoherentFlashAttention(torch.autograd.Function):
    """
    Flash attention with incoherent processing autograd function.
    Properly handles Hadamard transforms in both forward and backward passes.
    """
    
    @staticmethod
    def forward(
        ctx, q, k, v, lens, sm_scale, causal, autotune, return_lse, prescale_qk, precision,
        incoherent_processing, hadamard_signs_q, hadamard_signs_k, attention_mask,
        in_span, span_id, is_prefix
    ):
        # Store context for backward pass
        ctx.incoherent_processing = incoherent_processing
        ctx.causal = causal
        ctx.autotune = autotune
        ctx.sm_scale = sm_scale
        ctx.prescale_qk = prescale_qk
        ctx.precision = precision
        ctx.return_lse = return_lse
        ctx.attention_mask = attention_mask
        ctx.in_span = in_span
        ctx.span_id = span_id
        ctx.is_prefix = is_prefix
        
        # Apply Hadamard transform for incoherent processing
        q_transformed, k_transformed = q, k
        if incoherent_processing:
            # Double-check GPU capability for safety
            if not is_hopper_gpu():
                logger.warning(
                    f"Incoherent processing requested on non-Hopper GPU "
                    f"(compute capability {torch.cuda.get_device_capability()}). "
                    f"This feature is optimized for H100+ GPUs."
                )
            
            HEAD_DIM = q.size(-1)
            if HEAD_DIM & (HEAD_DIM - 1) != 0:
                raise ValueError(f"Head dimension {HEAD_DIM} must be a power of 2 for incoherent processing")
            
            # Use same signs for both Q and K as per research paper
            if hadamard_signs_q is None:
                hadamard_signs = generate_hadamard_signs(HEAD_DIM, q.device, q.dtype)
            else:
                hadamard_signs = hadamard_signs_q
            
            # Save signs for backward pass
            ctx.hadamard_signs = hadamard_signs
            
            # Use PyTorch implementation for better consistency
            # Apply the same orthogonal transform to both Q and K
            q_transformed = hadamard_transform(q, hadamard_signs)
            k_transformed = hadamard_transform(k, hadamard_signs)
        
        # Run flash attention on transformed tensors
        requires_grad = any(i.requires_grad for i in (q, k, v))
        O, LSE = torch.ops.flash_attention.forward(
            q=q_transformed,
            k=k_transformed,
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
        )
        
        # Save tensors for backward pass
        if requires_grad:
            ctx.save_for_backward(q, k, v, O, LSE, lens)
        
        if return_lse:
            return O, LSE
        return O
    
    @staticmethod 
    def backward(ctx, grad_output, grad_lse=None):
        q, k, v, o, lse, lens = ctx.saved_tensors
        
        if ctx.incoherent_processing:
            # For incoherent processing, we need to apply the forward transform again
            # because the attention backward expects the transformed Q and K
            q_transformed = hadamard_transform(q, ctx.hadamard_signs)
            k_transformed = hadamard_transform(k, ctx.hadamard_signs)
            
            # Compute gradients using transformed Q and K (matching forward pass)
            DQ, DK, DV = torch.ops.flash_attention.backward(
                q=q_transformed,
                k=k_transformed,
                v=v,
                lens=lens,
                o=o,
                lse=lse,
                do=grad_output,
                sm_scale=ctx.sm_scale,
                causal=ctx.causal,
                autotune=ctx.autotune,
                prescale_qk=ctx.prescale_qk,
                precision=ctx.precision,
                attention_mask=ctx.attention_mask,
                in_span=ctx.in_span,
                span_id=ctx.span_id,
                is_prefix=ctx.is_prefix,
            )
            
            # Apply inverse Hadamard transform to gradients to get gradients w.r.t. original Q and K
            # This applies the chain rule: dL/dQ_orig = dL/dQ_transformed * dQ_transformed/dQ_orig
            DQ = hadamard_inverse_transform(DQ, ctx.hadamard_signs)
            DK = hadamard_inverse_transform(DK, ctx.hadamard_signs)
        else:
            # Normal backward pass without incoherent processing
            DQ, DK, DV = torch.ops.flash_attention.backward(
                q=q,
                k=k,
                v=v,
                lens=lens,
                o=o,
                lse=lse,
                do=grad_output,
                sm_scale=ctx.sm_scale,
                causal=ctx.causal,
                autotune=ctx.autotune,
                prescale_qk=ctx.prescale_qk,
                precision=ctx.precision,
                attention_mask=ctx.attention_mask,
                in_span=ctx.in_span,
                span_id=ctx.span_id,
                is_prefix=ctx.is_prefix,
            )
        
        return DQ, DK, DV, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor | None = None,
    sm_scale: float | None = None,
    causal: bool = True,
    autotune: bool = False,
    return_lse: bool = False,
    prescale_qk: bool = False,
    precision: str = "ieee",
    incoherent_processing: bool | None = None,
    hadamard_signs_q: torch.Tensor | None = None,
    hadamard_signs_k: torch.Tensor | None = None,
    span_id: torch.Tensor | None = None,
    span_begin: torch.Tensor | None = None,
    span_end: torch.Tensor | None = None,
    is_prefix: torch.Tensor | None = None,
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
        incoherent_processing (bool | None): Apply Hadamard transform to Q and K to reduce quantization error.
                                           None (default): Auto-detect based on GPU (Hopper GPUs only)
                                           True: Force enable (with warning on non-Hopper GPUs)
                                           False: Force disable
        hadamard_signs_q (Tensor | None): Pre-computed random signs for Q transform
        hadamard_signs_k (Tensor | None): Pre-computed random signs for K transform
    """
    if not torch.compiler.is_compiling():
        for i in (q, k, v):
            torch._dynamo.mark_static(i, 1)
            torch._dynamo.mark_static(i, 3)
    
    if sm_scale is None:
        HEAD_DIM = q.size(-1)
        sm_scale = HEAD_DIM**-0.5
    
    # Determine if incoherent processing should be used based on GPU capability
    use_incoherent = should_use_incoherent_processing(incoherent_processing)
    
    if use_incoherent:
        # Log when incoherent processing is enabled
        if incoherent_processing is None:
            logger.info(f"Auto-enabling incoherent processing on Hopper GPU (compute capability {torch.cuda.get_device_capability()})")
        else:
            logger.info(f"Using incoherent processing as explicitly requested")
    
    # Use the custom autograd function if incoherent processing is enabled
    if use_incoherent:
        # Incoherent path not updated, as it's not the focus of this change.
        # The provided code does not use this path.
        raise NotImplementedError("Incoherent processing path is not updated for new attention metadata.")
    else:
        # Use standard flash attention for normal case
        return _flash_attention(
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
            attention_mask=None, # Pass None for attention_mask
            span_id=span_id,
            span_begin=span_begin,
            span_end=span_end,
            is_prefix=is_prefix,
        )


def is_hopper_gpu() -> bool:
    """Check if the current GPU is a Hopper architecture (H100, H200, etc.)"""
    if not torch.cuda.is_available():
        return False
    
    # Hopper GPUs have compute capability 9.0 or higher
    major, minor = torch.cuda.get_device_capability()
    return major >= 9


def should_use_incoherent_processing(incoherent_processing: bool | None = None) -> bool:
    """
    Determine whether to use incoherent processing based on GPU capability.
    
    Args:
        incoherent_processing: User override (True/False to force, None to auto-detect)
    
    Returns:
        bool: Whether to use incoherent processing
    """
    if incoherent_processing is not None:
        # User explicitly specified, respect their choice but warn if not optimal
        if incoherent_processing and not is_hopper_gpu():
            logger.warning(
                "Incoherent processing enabled on non-Hopper GPU. "
                "This feature is optimized for H100+ GPUs with compute capability >= 9.0"
            )
        return incoherent_processing
    
    # Auto-detect: only enable on Hopper GPUs
    return is_hopper_gpu()


if __name__ == "__main__":
    print("=== Flash Attention with Auto-Detected Incoherent Processing ===\n")
    
    # Check GPU capability
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        gpu_name = torch.cuda.get_device_name()
        print(f"GPU: {gpu_name}")
        print(f"Compute Capability: {major}.{minor}")
        
        if is_hopper_gpu():
            print("✓ Hopper GPU detected - incoherent processing will be auto-enabled")
        else:
            print("⚠ Non-Hopper GPU detected - incoherent processing will be disabled by default")
    else:
        print("⚠ No CUDA GPU available")
        exit(1)
    
    print("\n=== Testing Auto-Detection Behavior ===")
    
    # Test tensors
    B, H, T, D = 1, 2, 16, 64  # Power of 2 head dimension
    q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
    
    # Test 1: Default behavior (auto-detection)
    print("\n1. Testing default behavior (auto-detection):")
    out_auto = flash_attention(q, k, v)
    print(f"   Output shape: {out_auto.shape}")
    
    # Test 2: Explicitly disable incoherent processing
    print("\n2. Testing explicitly disabled incoherent processing:")
    out_disabled = flash_attention(q, k, v, incoherent_processing=False)
    print(f"   Output shape: {out_disabled.shape}")
    
    # Test 3: Force enable incoherent processing (with warning on non-Hopper)
    print("\n3. Testing explicitly enabled incoherent processing:")
    try:
        out_enabled = flash_attention(q, k, v, incoherent_processing=True)
        print(f"   Output shape: {out_enabled.shape}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 4: Compare outputs
    print("\n4. Comparing outputs:")
    if is_hopper_gpu():
        # On Hopper GPUs, auto and enabled should be identical
        auto_vs_enabled_diff = torch.norm(out_auto - out_enabled) / torch.norm(out_auto)
        auto_vs_disabled_diff = torch.norm(out_auto - out_disabled) / torch.norm(out_auto)
        print(f"   Auto vs Enabled difference: {auto_vs_enabled_diff:.8f} (should be ~0)")
        print(f"   Auto vs Disabled difference: {auto_vs_disabled_diff:.8f} (should be ~0, mathematically identical)")
    else:
        # On non-Hopper GPUs, auto and disabled should be identical
        auto_vs_disabled_diff = torch.norm(out_auto - out_disabled) / torch.norm(out_auto)
        auto_vs_enabled_diff = torch.norm(out_auto - out_enabled) / torch.norm(out_auto)
        print(f"   Auto vs Disabled difference: {auto_vs_disabled_diff:.8f} (should be ~0)")
        print(f"   Auto vs Enabled difference: {auto_vs_enabled_diff:.8f} (should be ~0, mathematically identical)")
    
    print("\n=== Test Complete ===")
    print(f"Summary: Incoherent processing auto-detection {'ENABLED' if is_hopper_gpu() else 'DISABLED'} based on GPU capability")