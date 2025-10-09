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

MAX_TILE_SIZE = 512
MIN_TILE_SIZE = 16

logger = logging.getLogger(__name__)


def is_hopper_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 9


def strides(t: torch.Tensor, expected_size=None):
    assert t is not None
    if expected_size is not None:
        assert t.ndim == expected_size
    return [t.stride(i) for i in range(t.ndim)]


def fwd_configs_pruner(configs, nargs, HEAD_DIM, DTYPE, **kwargs):
    return configs


def bwd_configs_pruner(configs, nargs, HEAD_DIM, DTYPE, **kwargs):
    return configs


# =============================================================================
# New Forward Pass Implementation
# =============================================================================

@triton.jit
def _flash_attn_fwd_inner(
    acc, l_i, m_i, q, kt_tile_ptr, v_tile_ptr,
    q_in_span, q_is_prefix, q_span_id,
    q_tile_indices,
    IN_SPAN, SPAN_ID, IS_PREFIX,
    stride_in_span_b, stride_span_id_b, stride_is_prefix_b,
    seq_len,
    kv_start_tile_idx, kv_end_tile_idx,
    TILE_K_SIZE, TILE_Q_SIZE, HEAD_DIM: tl.constexpr,
    SM_SCALE, PRESCALE_QK, CAUSAL,
    INPUT_PRECISION,
    K_BLOCK_DIVISIBLE,
    batch,
    PIPELINING, WARP_SPECIALIZE,
    RCP_LN2: tl.constexpr
):
    tile_k_arange = tl.arange(0, TILE_K_SIZE)
    softmax_scale = tl.cast(SM_SCALE * RCP_LN2, q.dtype)

    if PRESCALE_QK:
        q = q * softmax_scale

    for kv_tile_idx in tl.range(kv_start_tile_idx, kv_end_tile_idx, num_stages=PIPELINING, warp_specialize=WARP_SPECIALIZE):
        last_iter = kv_tile_idx + 1 == kv_end_tile_idx
        kv_token_idx = kv_tile_idx * TILE_K_SIZE

        if K_BLOCK_DIVISIBLE or not last_iter:
            kt_tile = tl.load(tl.advance(kt_tile_ptr, (0, kv_token_idx)))
            v_tile = tl.load(tl.advance(v_tile_ptr, (kv_token_idx, 0)))
        else:
            kt_tile = tl.load(tl.advance(kt_tile_ptr, (0, kv_token_idx)), boundary_check=(1,))
            v_tile = tl.load(tl.advance(v_tile_ptr, (kv_token_idx, 0)), boundary_check=(0,))

        qk = tl.dot(q, kt_tile, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        kv_indices = kv_token_idx + tile_k_arange

        k_in_span = tl.load(IN_SPAN + batch * stride_in_span_b + kv_indices, mask=kv_indices < seq_len, other=0)
        k_span_id = tl.load(SPAN_ID + batch * stride_span_id_b + kv_indices, mask=kv_indices < seq_len, other=0)
        k_is_prefix = tl.load(IS_PREFIX + batch * stride_is_prefix_b + kv_indices, mask=kv_indices < seq_len, other=0)

        q_is_maskq = (q_span_id == -1)
        q_is_context = (~q_in_span) & (~q_is_prefix) & (~q_is_maskq)

        q_is_maskq_b = q_is_maskq[:, None]
        q_is_prefix_b = q_is_prefix[:, None]
        q_is_context_b = q_is_context[:, None]

        k_is_maskq = (k_span_id[None, :] == -1)
        k_is_cls_or_prefix = k_is_prefix[None, :]
        
        prefix_to_prefix = q_is_prefix_b & k_is_prefix[None, :]
        
        k_is_context = ~k_in_span[None, :] & ~k_is_prefix[None, :] & ~k_is_maskq
        context_causal = q_is_context_b & k_is_context & (q_tile_indices[:, None] >= kv_indices[None, :])
        context_to_prefix = q_is_context_b & k_is_cls_or_prefix
        
        same_span = (q_in_span[:, None] & k_in_span[None, :] & (q_span_id[:, None] == k_span_id[None, :]) & (q_span_id[:, None] > 0))
        span_to_context = q_in_span[:, None] & k_is_context
        
        maskq_to_spans = q_is_maskq_b & k_in_span[None, :]
        maskq_to_cls = q_is_maskq_b & k_is_cls_or_prefix
        
        mask = (prefix_to_prefix | context_causal | context_to_prefix | same_span | span_to_context | maskq_to_spans | maskq_to_cls)
        
        mask = mask & (q_tile_indices[:, None] < seq_len) & (kv_indices[None, :] < seq_len)

        if not PRESCALE_QK:
            qk = qk * softmax_scale
        qk = tl.where(mask, qk, tl.cast(-float("inf"), qk.dtype))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v_tile.dtype), v_tile, acc, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        m_i = m_ij
        
    return acc, l_i, m_i

@triton.heuristics(
    dict(
        RCP_LN2=lambda _: math.log2(math.e),
    )
)
@triton.jit
def _flash_attn_fwd(
    Q, Kt, V, L, LSE, O,
    IN_SPAN, SPAN_ID, IS_PREFIX,
    stride_qb, stride_qh, stride_qt, stride_qk,
    stride_kb, stride_kh, stride_kk, stride_kt,
    stride_vb, stride_vh, stride_vt, stride_vk,
    stride_mb, stride_mh, stride_mt,
    stride_ob, stride_oh, stride_ot, stride_ok,
    lens_stride,
    in_span_stride_b, in_span_stride_t,
    span_id_stride_b, span_id_stride_t,
    is_prefix_stride_b, is_prefix_stride_t,
    T, HEAD_DIM: tl.constexpr, SM_SCALE, PRESCALE_QK, CAUSAL,
    TILE_Q_SIZE: tl.constexpr, TILE_K_SIZE: tl.constexpr,
    Q_BLOCK_DIVISIBLE: tl.constexpr, K_BLOCK_DIVISIBLE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    OUTPUT_LOGSUMEXP: tl.constexpr,
    DTYPE: tl.constexpr,
    PIPELINING: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    RETURN_ATTENTION_MASK: tl.constexpr,
    RCP_LN2: tl.constexpr
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
    
    q_in_span = tl.load(IN_SPAN + batch * in_span_stride_b + q_tile_indices, mask=q_tile_indices < seq_len, other=0)
    q_is_prefix = tl.load(IS_PREFIX + batch * is_prefix_stride_b + q_tile_indices, mask=q_tile_indices < seq_len, other=0)
    q_span_id = tl.load(SPAN_ID + batch * span_id_stride_b + q_tile_indices, mask=q_tile_indices < seq_len, other=0)
    
    q_tile_has_noncausal = tl.sum((q_in_span | q_is_prefix).to(tl.int32)) > 0
    kv_start_tile_idx = 0
    q_tile_max_token = min(q_token_idx + TILE_Q_SIZE, seq_len)
    if CAUSAL and not q_tile_has_noncausal:
        kv_end_tile_idx = tl.cdiv(q_tile_max_token, TILE_K_SIZE)
    else:
        kv_end_tile_idx = tl.cdiv(seq_len, TILE_K_SIZE)
    
    qbatch_head_offset = batch * stride_qb + head * stride_qh
    q_tile_ptr = tl.make_block_ptr(base=Q + qbatch_head_offset, shape=(T, HEAD_DIM), strides=(stride_qt, stride_qk), offsets=(q_token_idx, 0), block_shape=(TILE_Q_SIZE, HEAD_DIM), order=(1, 0))
    kbatch_head_offset = batch * stride_kb + head * stride_kh
    kt_tile_ptr = tl.make_block_ptr(base=Kt + kbatch_head_offset, shape=(HEAD_DIM, T), strides=(stride_kk, stride_kt), offsets=(0, 0), block_shape=(HEAD_DIM, TILE_K_SIZE), order=(0, 1))
    vbatch_head_offset = batch * stride_vb + head * stride_vh
    v_tile_ptr = tl.make_block_ptr(base=V + vbatch_head_offset, shape=(T, HEAD_DIM), strides=(stride_vt, stride_vk), offsets=(0, 0), block_shape=(TILE_K_SIZE, HEAD_DIM), order=(1, 0))

    m_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32)
    acc = tl.zeros([TILE_Q_SIZE, HEAD_DIM], dtype=tl.float32)

    if Q_BLOCK_DIVISIBLE:
        q_tile = tl.load(q_tile_ptr)
    else:
        q_tile = tl.load(q_tile_ptr, boundary_check=(0,))

    acc, l_i, m_i = _flash_attn_fwd_inner(
        acc, l_i, m_i, q_tile, kt_tile_ptr, v_tile_ptr,
        q_in_span, q_is_prefix, q_span_id,
        q_tile_indices,
        IN_SPAN, SPAN_ID, IS_PREFIX,
        in_span_stride_b, span_id_stride_b, is_prefix_stride_b,
        seq_len,
        kv_start_tile_idx, kv_end_tile_idx,
        TILE_K_SIZE, TILE_Q_SIZE, HEAD_DIM,
        SM_SCALE, PRESCALE_QK, CAUSAL,
        INPUT_PRECISION,
        K_BLOCK_DIVISIBLE,
        batch,
        PIPELINING, WARP_SPECIALIZE,
        RCP_LN2
    )

    acc = acc / l_i[:, None]
    
    obatch_head_offset = batch * stride_ob + head * stride_oh
    o_tile_ptr = tl.make_block_ptr(base=O + obatch_head_offset, shape=(T, HEAD_DIM), strides=(stride_ot, stride_ok), offsets=(q_token_idx, 0), block_shape=(TILE_Q_SIZE, HEAD_DIM), order=(1, 0))
    if Q_BLOCK_DIVISIBLE:
        tl.store(o_tile_ptr, acc.to(o_tile_ptr.type.element_ty))
    else:
        tl.store(o_tile_ptr, acc.to(o_tile_ptr.type.element_ty), boundary_check=(0,))
        
    if OUTPUT_LOGSUMEXP:
        m_i += tl.math.log2(l_i)
        mbatch_head_offset = batch * stride_mb + head * stride_mh
        m_tile_ptr = tl.make_block_ptr(base=LSE + mbatch_head_offset, shape=(T,), strides=(stride_mt,), offsets=(q_token_idx,), block_shape=(TILE_Q_SIZE,), order=(0,))
        if Q_BLOCK_DIVISIBLE:
            tl.store(m_tile_ptr, m_i)
        else:
            tl.store(m_tile_ptr, m_i, boundary_check=(0,))

# =============================================================================
# Original Backward Pass
# =============================================================================

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
    CAUSAL: tl.constexpr
):
    pass

# =============================================================================
# Public API
# =============================================================================

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

    # Use a minimal autotuner config for now
    config = triton.Config(
        dict(
            TILE_Q_SIZE=64, TILE_K_SIZE=64,
            PIPELINING=1, WARP_SPECIALIZE=False,
        ),
        num_warps=4, num_stages=2,
    )

    grid = (batch, heads, triton.cdiv(T, config.kwargs['TILE_Q_SIZE']))

    kt = k.transpose(-1, -2)

    q_block_divisible = T % config.kwargs['TILE_Q_SIZE'] == 0
    k_block_divisible = T % config.kwargs['TILE_K_SIZE'] == 0

    _flash_attn_fwd[grid](
        q, kt, v, lens, LSE, O,
        in_span, span_id, is_prefix,
        *strides(q, 4), *strides(kt, 4), *strides(v, 4),
        *(strides(LSE, 3) if LSE is not None else [0] * 3),
        *strides(O, 4),
        *(strides(lens, 1) if lens is not None else [0]),
        *(strides(in_span, 2) if in_span is not None else [0] * 2),
        *(strides(span_id, 2) if span_id is not None else [0] * 2),
        *(strides(is_prefix, 2) if is_prefix is not None else [0] * 2),
        T=T, HEAD_DIM=HEAD_DIM, SM_SCALE=sm_scale,
        PRESCALE_QK=prescale_qk, CAUSAL=causal,
        Q_BLOCK_DIVISIBLE=q_block_divisible,
        K_BLOCK_DIVISIBLE=k_block_divisible,
        INPUT_PRECISION=precision,
        OUTPUT_LOGSUMEXP=return_lse,
        DTYPE=q.dtype,
        RETURN_ATTENTION_MASK=return_attention_mask,
        **config.kwargs
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

    # This is a placeholder, will be replaced with optimized backward pass
    # For now, it does nothing, but it's needed for autograd to work
    grid_bwd = (batch, heads, 1)
    _flash_attn_bwd[grid_bwd](
        Q=q, K=k, V=v, L=lens,
        DELTA=delta, LSE=lse,
        DO=do, DQ=DQ, DK=DK, DV=DV,
        ATTN_MASK=attention_mask,
        IN_SPAN=in_span, SPAN_ID=span_id, IS_PREFIX=is_prefix,
        stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qt=q.stride(2), stride_qk=q.stride(3),
        stride_kb=k.stride(0), stride_kh=k.stride(1), stride_kt=k.stride(2), stride_kk=k.stride(3),
        stride_vb=v.stride(0), stride_vh=v.stride(1), stride_vt=v.stride(2), stride_vk=v.stride(3),
        stride_deltab=delta.stride(0), stride_deltah=delta.stride(1), stride_deltat=delta.stride(2),
        stride_mb=lse.stride(0), stride_mh=lse.stride(1), stride_mt=lse.stride(2),
        stride_dob=do.stride(0), stride_doh=do.stride(1), stride_dot=do.stride(2), stride_dok=do.stride(3),
        stride_dqb=DQ.stride(0), stride_dqh=DQ.stride(1), stride_dqt=DQ.stride(2), stride_dqk=DQ.stride(3),
        stride_dkb=DK.stride(0), stride_dkh=DK.stride(1), stride_dkt=DK.stride(2), stride_dkk=DK.stride(3),
        stride_dvb=DV.stride(0), stride_dvh=DV.stride(1), stride_dvt=DV.stride(2), stride_dvk=DV.stride(3),
        lens_stride=lens.stride(0) if lens is not None else 0,
        mask_stride_b=0, mask_stride_h=0, mask_stride_t=0,
        in_span_stride_b=in_span.stride(0) if in_span is not None else 0,
        in_span_stride_t=in_span.stride(1) if in_span is not None else 0,
        span_id_stride_b=span_id.stride(0) if span_id is not None else 0,
        span_id_stride_t=span_id.stride(1) if span_id is not None else 0,
        is_prefix_stride_b=is_prefix.stride(0) if is_prefix is not None else 0,
        is_prefix_stride_t=is_prefix.stride(1) if is_prefix is not None else 0,
        T=T,
        TIME_BUCKET=triton.next_power_of_2(T),
        HEAD_DIM=HEAD_DIM,
        DTYPE=q.dtype,
        INPUT_PRECISION=precision,
        SM_SCALE=sm_scale,
        PRESCALE_QK=prescale_qk,
        TILE_DQ_Q_SIZE=32, TILE_DQ_K_SIZE=32,
        TILE_DK_Q_SIZE=32, TILE_DK_K_SIZE=32,
        PIPELINING=1,
        CAUSAL=causal
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
        q, k, v, lens, sm_scale, causal, autotune, return_lse, prescale_qk, precision,
        attention_mask, in_span, span_id, is_prefix, output_attention_mask, return_attention_mask
    ) = inputs
    ctx.save_for_backward(q, k, v, O, LSE, lens, attention_mask, in_span, span_id, is_prefix)
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
        q=q, k=k, v=v, lens=lens, o=o, lse=lse, do=do, sm_scale=sm_scale,
        causal=causal, autotune=autotune, prescale_qk=prescale_qk, precision=precision,
        attention_mask=attention_mask, in_span=in_span, span_id=span_id, is_prefix=is_prefix,
    )
    return DQ, DK, DV, None, None, None, None, None, None, None, None, None, None, None, None, None


torch.library.register_autograd(
    "flash_attention::forward",
    attention_backward_adapter_op,
    setup_context=attention_backward_adapter_op_setup_context,
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
    
    if return_attention_mask:
        batch, heads, seq_len, _ = q.shape
        output_attention_mask = torch.zeros((batch, heads, seq_len, seq_len), dtype=torch.bool, device=q.device)
    else:
        output_attention_mask = None
    
    O, LSE = torch.ops.flash_attention.forward(
        q=q, k=k, v=v, lens=lens, sm_scale=sm_scale, causal=causal,
        autotune=autotune, prescale_qk=prescale_qk,
        return_lse=return_lse or requires_grad, precision=precision,
        attention_mask=attention_mask, in_span=in_span, span_id=span_id, is_prefix=is_prefix,
        output_attention_mask=output_attention_mask, return_attention_mask=return_attention_mask,
    )
    
    if return_attention_mask:
        if return_lse:
            return (O, LSE), output_attention_mask
        return O, output_attention_mask
    elif return_lse:
        return O, LSE
    return O


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
    if not torch.compiler.is_compiling():
        for i in (q, k, v):
            torch._dynamo.mark_static(i, 1)
            torch._dynamo.mark_static(i, 3)
    
    if sm_scale is None:
        HEAD_DIM = q.size(-1)
        sm_scale = HEAD_DIM**-0.5
    
    result = _flash_attention(
        q=q, k=k, v=v, lens=lens, sm_scale=sm_scale, causal=causal,
        autotune=autotune, return_lse=return_lse, prescale_qk=prescale_qk,
        precision=precision, attention_mask=attention_mask, in_span=in_span,
        span_id=span_id, is_prefix=is_prefix, return_attention_mask=return_attention_mask,
    )
    
    return result
