import logging
import math
import os

import torch
import torch.nn.functional as F
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


# fmt: off
@triton.heuristics(
    dict(
        Q_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_Q_SIZE'] == 0,
        K_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_K_SIZE'] == 0,
        RCP_LN2=lambda _: math.log2(math.e),
    )
)
@triton.jit
def _causal_attn_fwd( # Renamed
    Q: tl.tensor, Kt: tl.tensor, V: tl.tensor, L: tl.tensor, #
    LSE: tl.tensor, O: tl.tensor,  #
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,  #
    stride_kb: int, stride_kh: int, stride_kk: int, stride_kt: int,  #
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,  #
    stride_mb: int, stride_mh: int, stride_mt: int,  #
    stride_ob: int, stride_oh: int, stride_ot: int, stride_ok: int, #
    lens_stride: int,
    T: int,  #
    TIME_BUCKET:  int,  #
    HEAD_DIM: tl.constexpr,  #
    # CONTEXT_SIZE: tl.constexpr,  # Removed
    # CONTEXTS_BACK: tl.constexpr,  # Removed
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
    # PERFECT_MATCHING: tl.constexpr,  # Removed
    RCP_LN2: tl.constexpr,  #
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    q_tile_idx = tl.program_id(2) # Represents the current block index of Q
    q_token_idx = q_tile_idx * TILE_Q_SIZE # Starting token index for the current Q block

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
        offsets=(0, 0), # Start scanning K from the beginning for each Q
        block_shape=(HEAD_DIM, TILE_K_SIZE),
        order=(0, 1),
    )

    vbatch_head_offset = batch * stride_vb + head * stride_vh
    v_tile_ptr = tl.make_block_ptr(
        base=V + vbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_vt, stride_vk),
        offsets=(0, 0), # Start scanning V from the beginning for each Q
        block_shape=(TILE_K_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    m_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32)
    acc = tl.zeros([TILE_Q_SIZE, HEAD_DIM], dtype=tl.float32)

    # Causal: iterate K/V tiles up to and including the current Q tile's end position
    # kv_end_tile_idx is the number of K-tiles that could possibly interact with the current Q-tile.
    # It's the ceiling division of the Q-tile's maximum considered token index by TILE_K_SIZE.
    kv_end_tile_idx = tl.cdiv(min(q_token_idx + TILE_Q_SIZE, seq_len), TILE_K_SIZE)
    kv_start_tile_idx = 0 # For causal attention, always start from the first K tile.

    q_tile_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)

    if Q_BLOCK_DIVISIBLE:
        q_tile = tl.load(q_tile_ptr)
    else:
        q_tile = tl.load(
            q_tile_ptr,
            boundary_check=(0,),
            padding_option="zero", # Pad with zero if reading beyond T for Q
        )

    softmax_scale: tl.constexpr = tl.cast(SM_SCALE * RCP_LN2, q_tile.dtype)
    tile_k_arange = tl.arange(0, TILE_K_SIZE)

    if PRESCALE_QK:
        q_tile = q_tile * softmax_scale

    for kv_tile_idx in tl.range(
        kv_start_tile_idx, kv_end_tile_idx, num_stages=PIPELINING # Iterate up to current Q block end
    ):
        kv_token_idx = kv_tile_idx * TILE_K_SIZE
        # last_iter = kv_tile_idx + 1 == kv_end_tile_idx # Not strictly needed for boundary check logic if padding

        # Load K and V tiles
        # Boundary check for K and V tiles, especially for the last iteration
        # Padding ensures that dot products don't fail; masking handles correctness.
        kt_tile = tl.load(
            tl.advance(kt_tile_ptr, (0, kv_token_idx)),
            boundary_check=(1,), padding_option="zero"
        )
        v_tile = tl.load(
            tl.advance(v_tile_ptr, (kv_token_idx, 0)),
            boundary_check=(0,), padding_option="zero"
        )

        qk = tl.dot(
            q_tile, kt_tile, input_precision=INPUT_PRECISION, out_dtype=tl.float32
        )

        # Causal masking logic
        kv_indices = kv_token_idx + tile_k_arange

        # Mask for valid Q tokens (not padding due to q_tile_idx * TILE_Q_SIZE > seq_len)
        q_valid_mask = q_tile_indices[:, None] < seq_len
        # Mask for valid K tokens (not padding from K-tile load or beyond seq_len)
        k_valid_mask = kv_indices[None, :] < seq_len
        # Causal mask: Q's token index must be >= K's token index
        causal_mask = q_tile_indices[:, None] >= kv_indices[None, :]

        current_mask = q_valid_mask & k_valid_mask & causal_mask

        if not PRESCALE_QK:
            qk = qk * softmax_scale # Apply scale here if not prescaled

        qk = tl.where(current_mask, qk, tl.cast(-float("inf"), qk.dtype))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None]) # m_ij does not need to be q_attended specific
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
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

    # Final normalization
    # Ensure l_i is not zero to avoid division by zero, set to 1 for invalid tokens.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]

    # Mask out results for Q tokens that were padding (beyond original seq_len)
    # This uses the q_lens_mask concept from before, adapted.
    final_q_mask = q_tile_indices < seq_len
    acc = tl.where(final_q_mask[:, None], acc, 0.0)


    obatch_head_offset = batch * stride_ob + head * stride_oh
    o_tile_ptr = tl.make_block_ptr(
        base=O + obatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_ot, stride_ok),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    # Store results, boundary check for Q dimension (e.g. if T is not multiple of TILE_Q_SIZE)
    tl.store(
        o_tile_ptr,
        acc.to(o_tile_ptr.type.element_ty),
        boundary_check=(0,),
    )

    if OUTPUT_LOGSUMEXP and LSE is not None:
        # LogSumExp is m_i + log2(l_i). Ensure l_i is positive for log.
        l_i_for_lse = tl.where(l_i > 0, l_i, 1.0) # Avoid log(0) or log(negative)
        m_i_final = m_i + tl.math.log2(l_i_for_lse)
        m_i_final = tl.where(final_q_mask, m_i_final, -float("inf")) # LSE for padding Qs should be -inf

        mbatch_head_offset = batch * stride_mb + head * stride_mh
        m_tile_ptr = tl.make_block_ptr(
            base=LSE + mbatch_head_offset,
            shape=(T,),
            strides=(stride_mt,),
            offsets=(q_token_idx,),
            block_shape=(TILE_Q_SIZE,),
            order=(0,),
        )
        tl.store(
            m_tile_ptr,
            m_i_final, # Store potentially masked LSE values
            boundary_check=(0,),
        )
# fmt: on


def fwd_configs_pruner(configs, nargs, HEAD_DIM, DTYPE, T, **kwargs): # T added for context
    min_size = 32 # Fixed min tile size
    # Max size could depend on T, e.g., triton.next_power_of_2(T)
    # For simplicity, using a fixed moderately large max or relating to MAX_TILE_SIZE
    max_size = min(MAX_TILE_SIZE, triton.next_power_of_2(T) if T else MAX_TILE_SIZE)

    min_pipeline, max_pipeline = 1, 3
    min_warps, max_warps = 1, 8

    # Simplified pruning based on HEAD_DIM only for pipeline/warps
    if HEAD_DIM == 64:
        min_pipeline = 2
    elif HEAD_DIM == 128:
        max_pipeline = 2
        max_warps = 4
    elif HEAD_DIM == 256:
        max_pipeline = 1 # Example: smaller pipeline for very large heads
        max_warps = 4

    configs = [i for i in configs if min_size <= i.kwargs["TILE_K_SIZE"] <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs["TILE_Q_SIZE"] <= max_size]
    configs = [
        i for i in configs if min_pipeline <= i.kwargs["PIPELINING"] <= max_pipeline
    ]
    configs = [i for i in configs if min_warps <= i.num_warps <= max_warps]

    default_config = _get_default_config_fwd(HEAD_DIM, DTYPE)
    if default_config is not None:
        # Ensure default config respects simplified tile sizes if necessary
        # This part might need adjustment if _get_default_config_fwd is too complex
        # or not aligned with causal kernel's needs.
        configs += [
            triton.Config(
                dict(
                    PIPELINING=default_config[3],
                    TILE_Q_SIZE=max(min_size,min(default_config[0], max_size)), # Clamp to range
                    TILE_K_SIZE=max(min_size,min(default_config[1], max_size)), # Clamp to range
                ),
                num_warps=default_config[2],
                num_stages=default_config[3],
            )
        ]

    logger.warning(f"Start benchmarking forward causal_attention {len(configs) = } for T={T}, HEAD_DIM={HEAD_DIM}")
    return configs


def autotune_prehook(kwargs, reset_only=False):
    # L is 'lens' tensor. q.size(2) is sequence length T.
    # This hook seems to be for handling dynamic sequence lengths in autotuning,
    # by temporarily making lens compatible with a fixed T for kernel compilation/tuning.
    # It might need adjustment if L represents something else or if T is dynamic during autotuning.
    if "L" in kwargs and kwargs["L"] is not None and "q" in kwargs:
         # Original code: kwargs["L"].add_(kwargs["q"].size(2))
         # This implies L might be an offset or used in a way that adding T makes sense.
         # For causal attention, if `lens` is a tensor of sequence lengths, this op might be problematic.
         # Let's assume for now it's a specific trick for the autotuner setup.
         # If T is a key in autotune, it should be passed directly.
         pass # Revisit if autotuning issues arise.

def autotune_posthook(kwargs, exception=None):
    if "L" in kwargs and kwargs["L"] is not None and "q" in kwargs:
        # kwargs["L"].add_(-kwargs["q"].size(2))
        pass # Revisit if autotuning issues arise.


causal_forward = triton.heuristics( # Renamed
    dict(
        PIPELINING=lambda _: 1,
        TILE_Q_SIZE=lambda args: 64, # Fixed default
        TILE_K_SIZE=lambda args: 64, # Fixed default
    )
)(_causal_attn_fwd) # Renamed

causal_forward_autotune = triton.autotune( # Renamed
    configs=[
        triton.Config(
            dict(
                PIPELINING=pipe,
                TILE_Q_SIZE=tile_q,
                TILE_K_SIZE=tile_k,
            ),
            num_warps=num_warps,
            num_stages=pipe, # num_stages often equals pipelining stages
        )
        for num_warps in [4, 8]
        for pipe in [1, 2] # Reduced range for simplicity, can be expanded
        for tile_q in [32, 64, 128] # Simplified tile choices
        for tile_k in [32, 64, 128]
    ],
    key=[ # Removed CONTEXT_SIZE, CONTEXTS_BACK
        "HEAD_DIM",
        # "CONTEXT_SIZE", # Removed
        # "CONTEXTS_BACK", # Removed
        "INPUT_PRECISION",
        "TIME_BUCKET", # T (sequence length) is implicitly part of TIME_BUCKET
        "DTYPE",
        "T" # Added T as a key for fwd_configs_pruner
    ],
    prune_configs_by=dict(early_config_prune=fwd_configs_pruner),
    pre_hook=autotune_prehook,
    post_hook=autotune_posthook,
)(_causal_attn_fwd) # Renamed


@torch.library.custom_op( # Renamed op string
    "alexdremov_causal_attention::forward", mutates_args=(), device_types=("cuda",)
)
def causal_attention_forward_adapter( # Renamed
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor, # Kept, as kernel handles it
    # context_size: int, # Removed
    # back_contexts: int, # Removed
    sm_scale: float,
    autotune: bool,
    return_lse: bool,
    prescale_qk: bool,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, T, HEAD_DIM = q.shape

    # assert back_contexts >= 0 and context_size >= 1 # Removed
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

    kt = k.transpose(-1, -2)

    # Define fixed/default values for kernel parameters
    TILE_Q_SIZE_val = 64
    TILE_K_SIZE_val = 64
    PIPELINING_val = 1

    # Calculate grid
    grid_tuple = (batch, heads, triton.cdiv(T, TILE_Q_SIZE_val))

    # Calculate heuristic-derived constexpr values
    RCP_LN2_val = math.log2(math.e)
    Q_BLOCK_DIVISIBLE_val = (T % TILE_Q_SIZE_val == 0)
    K_BLOCK_DIVISIBLE_val = (T % TILE_K_SIZE_val == 0)

    # Determine num_warps and num_stages (e.g., from default config or fixed)
    # Using common defaults; these might be part of _get_default_config_fwd if that was kept for direct use
    num_warps_val = 4
    num_stages_val = 3 # Often related to PIPELINING_val or a fixed good value

    # NOTE: Calling kernel.run() directly.
    # The autotuner/heuristic wrappers (@triton.autotune, @triton.heuristics)
    # for _causal_attn_fwd were causing TypeErrors after kernel signature changes
    # (removal of CONTEXT_SIZE, CONTEXTS_BACK).
    # If autotuning is desired, these wrappers and their configurations for
    # _causal_attn_fwd need to be carefully reviewed and updated.
    _causal_attn_fwd.run(
        Q=q, Kt=kt, V=v, L=lens, LSE=LSE, O=O,
        stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qt=q.stride(2), stride_qk=q.stride(3),
        stride_kb=kt.stride(0), stride_kh=kt.stride(1), stride_kk=kt.stride(2), stride_kt=kt.stride(3),
        stride_vb=v.stride(0), stride_vh=v.stride(1), stride_vt=v.stride(2), stride_vk=v.stride(3),
        stride_mb=LSE.stride(0) if LSE is not None else 0,
        stride_mh=LSE.stride(1) if LSE is not None else 0,
        stride_mt=LSE.stride(2) if LSE is not None else 0,
        stride_ob=O.stride(0), stride_oh=O.stride(1), stride_ot=O.stride(2), stride_ok=O.stride(3),
        lens_stride=lens.stride(0) if lens is not None else 0,
        T=T, TIME_BUCKET=triton.next_power_of_2(T), HEAD_DIM=HEAD_DIM,
        INPUT_PRECISION=precision, SM_SCALE=sm_scale, DTYPE=q.dtype,
        PRESCALE_QK=prescale_qk, OUTPUT_LOGSUMEXP=return_lse,
        TILE_Q_SIZE=TILE_Q_SIZE_val, TILE_K_SIZE=TILE_K_SIZE_val, PIPELINING=PIPELINING_val,
        Q_BLOCK_DIVISIBLE=Q_BLOCK_DIVISIBLE_val, K_BLOCK_DIVISIBLE=K_BLOCK_DIVISIBLE_val,
        RCP_LN2=RCP_LN2_val,
        grid=grid_tuple,
        num_warps=num_warps_val,
        num_stages=num_stages_val,
        warmup=False # Added argument
    )

    if LSE is None:
        LSE = torch.empty(0, device=q.device)
    return O, LSE


@torch.library.register_fake("alexdremov_causal_attention::forward") # Renamed op string
def causal_attention_forward_adapter_abstract( # Renamed
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
) -> tuple[torch.Tensor, torch.Tensor]:
    # Output shapes remain the same
    o_shape = q.shape
    lse_shape = q.shape[:-1] if return_lse else (0,) # Or q.shape[:3] if that's the convention

    # If LSE is truly (B,H,T)
    if return_lse :
        actual_lse_shape = q.shape[:3]
    else: # if LSE is not returned, make it an empty tensor with 0 elements but correct ndim for some pytorch internals.
          # Or (0,) if that's preferred. Let's try to match the original empty(0)
        actual_lse_shape = (0,)


    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty(actual_lse_shape, dtype=torch.float32, device=q.device) # LSE is float32
    )
