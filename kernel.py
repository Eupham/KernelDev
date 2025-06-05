import math
import torch
import triton
import triton.language as tl
import logging

logger = logging.getLogger(__name__)

MAX_TILE_SIZE = 256
MIN_TILE_SIZE = 32

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
    """Get default forward configuration for given head_dim and dtype."""
    if torch.cuda.get_device_capability()[0] >= 9:  # H100
        config = _h100_default_config.get((dtype, head_dim))
    else:  # A100 and others
        config = _a100_default_config.get((dtype, head_dim))
    
    if config is None:
        # Fallback configuration
        return (64, 64, 4, 3)
    return config

def strides(t: torch.Tensor, expected_size=None):
    """Get tensor strides."""
    if expected_size is None:
        return t.stride()
    else:
        assert len(t.stride()) == expected_size
        return t.stride()

def fwd_configs_pruner(configs, nargs, HEAD_DIM, DTYPE, **kwargs):
    """Prune forward configurations based on head dimension and other parameters."""
    min_pipeline, max_pipeline = 1, 3
    min_warps, max_warps = 4, 8

    if HEAD_DIM == 64:
        min_size, max_size = 32, 128
    elif HEAD_DIM == 128:
        min_size, max_size = 32, 128
    elif HEAD_DIM == 256:
        min_size, max_size = 32, 64
    else:
        min_size, max_size = 32, 64

    configs = [i for i in configs if min_size <= i.kwargs["TILE_K_SIZE"] <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs["TILE_Q_SIZE"] <= max_size]
    configs = [
        i for i in configs if min_pipeline <= i.kwargs["PIPELINING"] <= max_pipeline
    ]
    configs = [i for i in configs if min_warps <= i.num_warps <= max_warps]

    default_config = _get_default_config_fwd(HEAD_DIM, DTYPE)
    if default_config is not None:
        # Add default config if not present
        default_tile_q, default_tile_k, default_warps, default_stages = default_config
        default_found = any(
            c.kwargs["TILE_Q_SIZE"] == default_tile_q and
            c.kwargs["TILE_K_SIZE"] == default_tile_k and
            c.num_warps == default_warps
            for c in configs
        )
        if not default_found:
            configs.append(triton.Config(
                dict(TILE_Q_SIZE=default_tile_q, TILE_K_SIZE=default_tile_k, PIPELINING=1),
                num_warps=default_warps,
                num_stages=default_stages
            ))

    logger.warning(f"Start benchmarking forward attention {len(configs) = }")
    return configs

@triton.autotune(
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
    key=["HEAD_DIM", "TIME_BUCKET", "DTYPE"],
    prune_configs_by=dict(early_config_prune=fwd_configs_pruner),
)
@triton.heuristics(
    dict(
        Q_BLOCK_DIVISIBLE=lambda args: args['T'] % args['TILE_Q_SIZE'] == 0,
        K_BLOCK_DIVISIBLE=lambda args: args['T'] % args['TILE_K_SIZE'] == 0,
        RCP_LN2=lambda _: math.log2(math.e),
    )
)
@triton.jit
def _attention_fwd(
    Q: tl.tensor, K: tl.tensor, V: tl.tensor, L: tl.tensor, 
    LSE: tl.tensor, O: tl.tensor,
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: int,
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,
    stride_mb: int, stride_mh: int, stride_mt: int,
    stride_ob: int, stride_oh: int, stride_ot: int, stride_ok: int,
    lens_stride: int,
    T: int,
    TIME_BUCKET: int,
    HEAD_DIM: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    SM_SCALE: tl.constexpr,
    DTYPE: tl.constexpr,
    PRESCALE_QK: tl.constexpr,
    OUTPUT_LOGSUMEXP: tl.constexpr,
    TILE_Q_SIZE: tl.constexpr,
    TILE_K_SIZE: tl.constexpr,
    PIPELINING: tl.constexpr,
    Q_BLOCK_DIVISIBLE: tl.constexpr,
    K_BLOCK_DIVISIBLE: tl.constexpr,
    RCP_LN2: tl.constexpr,
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

    # Load Q tile
    qbatch_head_offset = batch * stride_qb + head * stride_qh
    q_tile_ptr = tl.make_block_ptr(
        base=Q + qbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_qt, stride_qk),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    # Setup K and V pointers
    kbatch_head_offset = batch * stride_kb + head * stride_kh
    kt_tile_ptr = tl.make_block_ptr(
        base=K + kbatch_head_offset,
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

    # Initialize accumulators
    m_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32)
    acc = tl.zeros([TILE_Q_SIZE, HEAD_DIM], dtype=tl.float32)

    # Load Q
    if Q_BLOCK_DIVISIBLE:
        q_tile = tl.load(q_tile_ptr)
    else:
        q_tile = tl.load(q_tile_ptr, boundary_check=(0,))

    q_tile_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)
    q_lens_mask = q_tile_indices[:, None] < seq_len

    softmax_scale: tl.constexpr = tl.cast(SM_SCALE * RCP_LN2, q_tile.dtype)
    tile_k_arange = tl.arange(0, TILE_K_SIZE)

    if PRESCALE_QK:
        q_tile = q_tile * softmax_scale

    # Number of K tiles
    kv_tiles = tl.cdiv(seq_len, TILE_K_SIZE)

    # Main attention loop
    for kv_tile_idx in tl.range(0, kv_tiles, num_stages=PIPELINING):
        kv_token_idx = kv_tile_idx * TILE_K_SIZE
        
        # Load K and V tiles
        if K_BLOCK_DIVISIBLE:
            kT = tl.load(tl.advance(kt_tile_ptr, (0, kv_token_idx)))
            v = tl.load(tl.advance(v_tile_ptr, (kv_token_idx, 0)))
        else:
            kT = tl.load(tl.advance(kt_tile_ptr, (0, kv_token_idx)), boundary_check=(1,))
            v = tl.load(tl.advance(v_tile_ptr, (kv_token_idx, 0)), boundary_check=(0,))

        # Compute QK^T
        qk = tl.dot(q_tile, kT, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK:
            qk = qk * softmax_scale

        # Apply causal mask
        kv_indices = kv_token_idx + tile_k_arange
        causal_mask = q_tile_indices[:, None] >= kv_indices[None, :]
        
        # Apply length mask
        length_mask = kv_indices[None, :] < seq_len
        mask = q_lens_mask & causal_mask & length_mask
        
        # Apply mask to scores
        qk = tl.where(mask, qk, -float("inf"))

        # Online softmax update
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_i_new)
        beta = tl.math.exp2(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * tl.sum(tl.math.exp2(qk - m_i_new[:, None]), 1)
        
        # Update output
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc = tl.dot(p, v.to(p.dtype), acc, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        
        # Update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output
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

    # Store LSE if needed
    if OUTPUT_LOGSUMEXP and LSE is not None:
        lse_val = m_i + tl.math.log2(l_i)
        lsebatch_head_offset = batch * stride_mb + head * stride_mh
        lse_tile_ptr = tl.make_block_ptr(
            base=LSE + lsebatch_head_offset,
            shape=(T,),
            strides=(stride_mt,),
            offsets=(q_token_idx,),
            block_shape=(TILE_Q_SIZE,),
            order=(0,),
        )
        if Q_BLOCK_DIVISIBLE:
            tl.store(lse_tile_ptr, lse_val)
        else:
            tl.store(lse_tile_ptr, lse_val, boundary_check=(0,))


@torch.library.custom_op(
    "attention::forward", mutates_args=(), device_types=("cuda",)
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

    kt = k.transpose(-1, -2)  # transpose for efficient memory access
    _attention_fwd[grid](
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
        TIME_BUCKET=triton.next_power_of_2(T),
        OUTPUT_LOGSUMEXP=return_lse,
        SM_SCALE=sm_scale,
    )

    if LSE is None:
        LSE = torch.empty(0)
    return O, LSE


@torch.library.register_fake("attention::forward")
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


@torch.compile(fullgraph=True, dynamic=True)
def _attention(
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
    requires_grad = any(i.requires_grad for i in (q, k, v))
    O, LSE = torch.ops.attention.forward(
        q=q,
        k=k,
        v=v,
        lens=lens,
        sm_scale=sm_scale,
        autotune=autotune,
        prescale_qk=prescale_qk,
        return_lse=return_lse or requires_grad,
        precision=precision,
    )
    if return_lse:
        return O, LSE
    else:
        return O


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor | None = None,
    sm_scale: float | None = None,
    autotune: bool = False,
    return_lse: bool = False,
    prescale_qk: bool = False,
    precision: str = "ieee",
):
    """
    Computes standard causal self-attention.

    Args:
        q (Tensor): The query tensor of shape `(batch, heads_num, time, head_dim)`
        k (Tensor): The key tensor of shape `(batch, heads_num, time, head_dim)`
        v (Tensor): The value tensor of shape `(batch, heads_num, time, head_dim)`
        lens (Tensor | None): Lengths of sequences of shape `(batch,)`
        sm_scale (float): Softmax scale, head_dim ** -0.5 by default
        autotune (bool): Use triton autotune for optimal kernel configuration
        return_lse (bool): Return log-sum-exp values
        prescale_qk (bool): Prescale Q in QK^T calculations
        precision (str): Precision for matmuls: 'ieee' or 'tf32'
    """
    if not torch.compiler.is_compiling():
        for tensor in (q, k, v):
            assert tensor.is_cuda, "All tensors must be on CUDA"
            assert tensor.is_contiguous(), "All tensors must be contiguous"
    
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    
    return _attention(
        q=q,
        k=k,
        v=v,
        lens=lens,
        sm_scale=sm_scale,
        autotune=autotune,
        return_lse=return_lse,
        prescale_qk=prescale_qk,
        precision=precision,
    )


if __name__ == "__main__":
    # Simple test
    batch, heads, seq_len, head_dim = 2, 8, 512, 64
    device = torch.device("cuda")
    
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    
    # Test the kernel
    output = attention(q, k, v)
    print(f"Output shape: {output.shape}")
    print("Attention kernel test passed!")
