import logging
import math
import torch
import triton
import triton.language as tl

MAX_TILE_SIZE = 256
MIN_TILE_SIZE = 32

logger = logging.getLogger(__name__)

# BLOCK_Q, BLOCK_K, num_warps, num_stages
_h100_default_config = {
    (torch.float32, 64): (128, 32, 4, 3),
    (torch.float32, 128): (32, 64, 4, 3),
    (torch.float32, 256): (32, 32, 4, 3),
    (torch.bfloat16, 64): (128, 128, 4, 3),
    (torch.bfloat16, 128): (128, 64, 8, 3),
    (torch.bfloat16, 256): (64, 32, 4, 3),
    (torch.float16, 64): (128, 128, 4, 3),
    (torch.float16, 128): (128, 128, 8, 3),
    (torch.float16, 256): (64, 32, 4, 3),
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
    else:  # modest hardware or extremely large head_dim
        if dtype == torch.float32:
            default_config = (32, 16, 4, 3)
        else:
            default_config = (64, 32, 4, 3)

    return default_config


def strides(t: torch.Tensor, expected_size=None):
    assert t is not None
    if expected_size is not None:
        assert t.ndim == expected_size
    return [t.stride(i) for i in range(t.ndim)]


def fwd_configs_pruner(configs, nargs, HEAD_DIM, DTYPE, **kwargs):
    min_size = 32
    max_size = 128
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


# fmt: off
@triton.heuristics(
    dict(
        Q_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_Q_SIZE'] == 0,
        K_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_K_SIZE'] == 0,
        RCP_LN2=lambda _: math.log2(math.e),
    )
)
@triton.jit
def _flash_attn_fwd(
    Q: tl.tensor, Kt: tl.tensor, V: tl.tensor, L: tl.tensor, #
    LSE: tl.tensor, O: tl.tensor,  #
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,  #
    stride_kb: int, stride_kh: int, stride_kk: int, stride_kt: int,  #
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,  #
    stride_mb: int, stride_mh: int, stride_mt: int,  #
    stride_ob: int, stride_oh: int, stride_ot: int, stride_ok: int, #
    lens_stride: int,
    T: int,  #
    HEAD_DIM: tl.constexpr,  #
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

    # For causal attention, we only need to attend to tokens up to the current position
    kv_end_tile_idx = tl.cdiv(min(q_token_idx + TILE_Q_SIZE, seq_len), TILE_K_SIZE)

    q_tile_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)
    q_lens_mask = q_tile_indices[:, None] < seq_len

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

    for kv_tile_idx in tl.range(0, kv_end_tile_idx, num_stages=PIPELINING):
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

        qk = tl.dot(q_tile, kt_tile)
        qk = qk.to(tl.float32)

        kv_indices = kv_token_idx + tile_k_arange
        # Causal mask: only attend to previous tokens (including current)
        causal_mask = q_tile_indices[:, None] >= kv_indices[None, :]
        mask = q_lens_mask & (kv_indices[None, :] < seq_len) & causal_mask

        if not PRESCALE_QK:
            qk = qk * softmax_scale
        qk = tl.where(mask, qk, tl.cast(-float("inf"), qk.dtype))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        acc = tl.dot(p.to(v_tile.dtype), v_tile, acc)
        m_i = m_ij

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
# fmt: on


def autotune_prehook(kwargs, reset_only=False):
    if kwargs["L"] is not None:
        kwargs["L"].add_(kwargs["q"].size(2))  # L += time


def autotune_posthook(kwargs, exception=None):
    if kwargs["L"] is not None:
        kwargs["L"].add_(-kwargs["q"].size(2))  # L -= time


# Create forward kernels
flash_forward = triton.heuristics(
    dict(
        PIPELINING=lambda _: 1,
        TILE_Q_SIZE=lambda args: min(64, max(MIN_TILE_SIZE, 64)),
        TILE_K_SIZE=lambda args: min(64, max(MIN_TILE_SIZE, 64)),
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
        for num_warps in [4, 8]
        for pipe in [1, 2]
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
        "INPUT_PRECISION",
        "DTYPE",
    ],
    prune_configs_by=dict(early_config_prune=fwd_configs_pruner),
    pre_hook=autotune_prehook,
    post_hook=autotune_posthook,
)(_flash_attn_fwd)


# Forward adapter functions
@torch.library.custom_op(
    "flash_attention::forward", mutates_args=(), device_types=("cuda",)
)
def attention_forward_adapter(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor,
    sm_scale: float,
    autotune: bool,
    return_lse: bool,
    prescale_qk: bool,
    precision: str,
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
        *strides(q, 4),
        *strides(kt, 4),
        *strides(v, 4),
        *(strides(LSE, 3) if LSE is not None else [0] * 3),
        *strides(O, 4),
        *(strides(lens, 1) if lens is not None else [0]),
        T=T,
        HEAD_DIM=HEAD_DIM,
        INPUT_PRECISION=precision,
        PRESCALE_QK=prescale_qk,
        DTYPE=q.dtype,
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
    autotune: bool,
    return_lse: bool,
    prescale_qk: bool,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty(q.shape[:3], dtype=torch.float32, device=q.device) if return_lse else torch.empty(0),
    )