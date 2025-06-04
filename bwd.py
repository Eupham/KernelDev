import torch
import triton
import triton.language as tl
import math

from fwd import ( # Assuming fwd.py might also be updated or these are general utilities
    strides,
    autotune_prehook,
    autotune_posthook,
    _h100_default_config, # These might need to be bwd specific if configs differ
    _a100_default_config, # These might need to be bwd specific if configs differ
    MIN_TILE_SIZE,
    MAX_TILE_SIZE,
    logger # If logger is used
)

# fmt: off
# _streaming_attn_bwd_precompute remains largely the same as it's a generic precomputation.
# If its autotuning or heuristics depend on context_size/back_contexts, those would need adjustment.
# For now, assuming it's general enough.
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
    key=["HEAD_DIM", "DTYPE", "TIME_BUCKET"], # Assuming T is implicitly in TIME_BUCKET
)
@triton.heuristics(
    dict(
        BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_SIZE'] == 0,
        RCP_LN2=lambda _: math.log2(math.e),
    )
)
@triton.jit
def _causal_attn_bwd_precompute( # Renamed, though logic might be identical
    O: tl.tensor, DO: tl.tensor, RES: tl.tensor,
    stride_ob: int, stride_oh: int, stride_ot: int, stride_ok: int,
    stride_dob: int, stride_doh: int, stride_dot: int, stride_dok: int,
    stride_rb: int, stride_rh: int, stride_rt: int,
    T: int,
    TIME_BUCKET: int,
    HEAD_DIM: tl.constexpr,
    DTYPE:  tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK_DIVISIBLE: tl.constexpr,
    RCP_LN2: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    tile = tl.program_id(2)
    token_idx = tile * TILE_SIZE

    obatch_head_offset = batch * stride_ob + head * stride_oh
    o_tile_ptr = tl.make_block_ptr(
        base=O + obatch_head_offset, shape=(T, HEAD_DIM), strides=(stride_ot, stride_ok),
        offsets=(token_idx, 0), block_shape=(TILE_SIZE, HEAD_DIM), order=(1, 0))
    dobatch_head_offset = batch * stride_dob + head * stride_doh
    do_tile_ptr = tl.make_block_ptr(
        base=DO + dobatch_head_offset, shape=(T, HEAD_DIM), strides=(stride_dot, stride_dok),
        offsets=(token_idx, 0), block_shape=(TILE_SIZE, HEAD_DIM), order=(1, 0))

    if BLOCK_DIVISIBLE:
        o_tile = tl.load(o_tile_ptr)
        do_tile = tl.load(do_tile_ptr)
    else:
        o_tile = tl.load(o_tile_ptr, boundary_check=(0,), padding_option="zero")
        do_tile = tl.load(do_tile_ptr, boundary_check=(0,), padding_option="zero")

    res = tl.sum(o_tile.to(tl.float32) * do_tile.to(tl.float32), 1)

    rbatch_head_offset = batch * stride_rb + head * stride_rh
    res_ptr = tl.make_block_ptr(
        base=RES + rbatch_head_offset, shape=(T,), strides=(stride_rt,),
        offsets=(token_idx,), block_shape=(TILE_SIZE,), order=(0,))
    if BLOCK_DIVISIBLE:
        tl.store(res_ptr, res)
    else:
        tl.store(res_ptr, res, boundary_check=(0,))


@triton.heuristics(
    dict(
        RCP_LN2=lambda _: math.log2(math.e),
        DQ_TILES_NUM=lambda args: triton.cdiv(args['T'], args["TILE_DQ_Q_SIZE"]),
        # PERFECT_DKV_MATCHING removed
        # PERFECT_DQ_MATCHING removed
        DQ_Q_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_DQ_Q_SIZE'] == 0,
        DQ_K_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_DQ_K_SIZE'] == 0,
        DK_Q_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_DK_Q_SIZE'] == 0,
        DK_K_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_DK_K_SIZE'] == 0,
    )
)
@triton.jit
def _causal_attn_bwd( # Renamed
    Q: tl.tensor, K: tl.tensor, V: tl.tensor, L: tl.tensor,
    DELTA: tl.tensor, LSE: tl.tensor,
    DO: tl.tensor, DQ: tl.tensor, DK: tl.tensor, DV: tl.tensor,
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
    T: int,
    TIME_BUCKET: int,
    DQ_TILES_NUM: int,
    HEAD_DIM: tl.constexpr,
    # CONTEXT_SIZE: tl.constexpr, # Removed
    # CONTEXTS_BACK: tl.constexpr, # Removed
    DTYPE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    SM_SCALE: tl.constexpr,
    PRESCALE_QK: tl.constexpr,
    # PERFECT_DKV_MATCHING: tl.constexpr, # Removed
    # PERFECT_DQ_MATCHING: tl.constexpr, # Removed
    DQ_Q_BLOCK_DIVISIBLE: tl.constexpr,
    DQ_K_BLOCK_DIVISIBLE: tl.constexpr,
    DK_Q_BLOCK_DIVISIBLE: tl.constexpr,
    DK_K_BLOCK_DIVISIBLE: tl.constexpr,
    RCP_LN2: tl.constexpr,
    TILE_DQ_Q_SIZE: tl.constexpr, TILE_DQ_K_SIZE: tl.constexpr,
    TILE_DK_Q_SIZE: tl.constexpr, TILE_DK_K_SIZE: tl.constexpr,
    PIPELINING: tl.constexpr,
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
        _causal_attn_bwd_dkdv_inner( # Renamed
            Q, K, V, DELTA, LSE, DO, DK, DV,
            stride_qb, stride_qh, stride_qt, stride_qk,
            stride_kb, stride_kh, stride_kt, stride_kk,
            stride_vb, stride_vh, stride_vt, stride_vk,
            stride_deltab, stride_deltah, stride_deltat,
            stride_mb, stride_mh, stride_mt,
            stride_dob, stride_doh, stride_dot, stride_dok,
            stride_dkb, stride_dkh, stride_dkt, stride_dkk,
            stride_dvb, stride_dvh, stride_dvt, stride_dvk,
            batch=batch, head=head, tile_id=tile_id, seq_len=seq_len, T=T,
            HEAD_DIM=HEAD_DIM, INPUT_PRECISION=INPUT_PRECISION, SM_SCALE=SM_SCALE, PRESCALE_QK=PRESCALE_QK,
            DK_Q_BLOCK_DIVISIBLE=DK_Q_BLOCK_DIVISIBLE, DK_K_BLOCK_DIVISIBLE=DK_K_BLOCK_DIVISIBLE, RCP_LN2=RCP_LN2,
            TILE_DK_Q_SIZE=TILE_DK_Q_SIZE, TILE_DK_K_SIZE=TILE_DK_K_SIZE, PIPELINING=PIPELINING,
        )
    else:
        _causal_attn_bwd_dq_inner( # Renamed
            Q, K, V, DELTA, LSE, DO, DQ,
            stride_qb, stride_qh, stride_qt, stride_qk,
            stride_kb, stride_kh, stride_kt, stride_kk,
            stride_vb, stride_vh, stride_vt, stride_vk,
            stride_deltab, stride_deltah, stride_deltat,
            stride_mb, stride_mh, stride_mt,
            stride_dob, stride_doh, stride_dot, stride_dok,
            stride_dqb, stride_dqh, stride_dqt, stride_dqk,
            batch=batch, head=head, tile_id=tile_id, seq_len=seq_len, T=T,
            HEAD_DIM=HEAD_DIM, INPUT_PRECISION=INPUT_PRECISION, SM_SCALE=SM_SCALE, PRESCALE_QK=PRESCALE_QK,
            DQ_Q_BLOCK_DIVISIBLE=DQ_Q_BLOCK_DIVISIBLE, DQ_K_BLOCK_DIVISIBLE=DQ_K_BLOCK_DIVISIBLE, RCP_LN2=RCP_LN2,
            TILE_DQ_Q_SIZE=TILE_DQ_Q_SIZE, TILE_DQ_K_SIZE=TILE_DQ_K_SIZE, PIPELINING=PIPELINING,
        )


@triton.jit()
def _causal_attn_bwd_dq_inner( # Renamed
    Q: tl.tensor, K: tl.tensor, V: tl.tensor, DELTA: tl.tensor, LSE: tl.tensor,
    DO: tl.tensor, DQ: tl.tensor,
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: int,
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,
    stride_deltab: int, stride_deltah: int, stride_deltat: int,
    stride_mb: int, stride_mh: int, stride_mt: int,
    stride_dob: int, stride_doh: int, stride_dot: int, stride_dok: int,
    stride_dqb: int, stride_dqh: int, stride_dqt: int, stride_dqk: int,
    batch: int, head: int, tile_id: int, seq_len: tl.tensor, T: int,
    HEAD_DIM: tl.constexpr, INPUT_PRECISION: tl.constexpr, SM_SCALE: tl.constexpr, PRESCALE_QK: tl.constexpr,
    DQ_Q_BLOCK_DIVISIBLE: tl.constexpr, DQ_K_BLOCK_DIVISIBLE: tl.constexpr, RCP_LN2: tl.constexpr,
    TILE_DQ_Q_SIZE: tl.constexpr, TILE_DQ_K_SIZE: tl.constexpr, PIPELINING: tl.constexpr,
    # Removed CONTEXT_SIZE, CONTEXTS_BACK, PERFECT_DQ_MATCHING
):
    q_tile_idx = tile_id
    q_token_idx = q_tile_idx * TILE_DQ_Q_SIZE

    # Pointer setup (same as before, boundary checks handle edges)
    q_tile_ptr = tl.make_block_ptr(base=Q + batch*stride_qb + head*stride_qh, shape=(T, HEAD_DIM), strides=(stride_qt, stride_qk), offsets=(q_token_idx, 0), block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM), order=(1,0))
    lse_tile_ptr = tl.make_block_ptr(base=LSE + batch*stride_mb + head*stride_mh, shape=(T,), strides=(stride_mt,), offsets=(q_token_idx,), block_shape=(TILE_DQ_Q_SIZE,), order=(0,))
    delta_tile_ptr = tl.make_block_ptr(base=DELTA + batch*stride_deltab + head*stride_deltah, shape=(T,), strides=(stride_deltat,), offsets=(q_token_idx,), block_shape=(TILE_DQ_Q_SIZE,), order=(0,))
    do_tile_ptr = tl.make_block_ptr(base=DO + batch*stride_dob + head*stride_doh, shape=(T, HEAD_DIM), strides=(stride_dot, stride_dok), offsets=(q_token_idx, 0), block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM), order=(1,0))

    if DQ_Q_BLOCK_DIVISIBLE:
        q, m, di, do = tl.load(q_tile_ptr), tl.load(lse_tile_ptr)[:,None], tl.load(delta_tile_ptr), tl.load(do_tile_ptr)
    else:
        q = tl.load(q_tile_ptr, boundary_check=(0,), padding_option="zero")
        m = tl.load(lse_tile_ptr, boundary_check=(0,))[:,None] # Add padding_option if needed
        di = tl.load(delta_tile_ptr, boundary_check=(0,)) # Add padding_option if needed
        do = tl.load(do_tile_ptr, boundary_check=(0,), padding_option="zero")

    kt_tile_ptr = tl.make_block_ptr(base=K + batch*stride_kb + head*stride_kh, shape=(HEAD_DIM, T), strides=(stride_kk, stride_kt), offsets=(0,0), block_shape=(HEAD_DIM, TILE_DQ_K_SIZE), order=(0,1))
    vt_tile_ptr = tl.make_block_ptr(base=V + batch*stride_vb + head*stride_vh, shape=(HEAD_DIM, T), strides=(stride_vk, stride_vt), offsets=(0,0), block_shape=(HEAD_DIM, TILE_DQ_K_SIZE), order=(1,0)) # Note: V is (T, D), so V_T is (D,T)

    dq = tl.zeros([TILE_DQ_Q_SIZE, HEAD_DIM], dtype=tl.float32)
    dq = _causal_attn_bwd_dq( # Renamed
        dq, q, m, di, do, kt_tile_ptr, vt_tile_ptr,
        seq_len=seq_len, q_token_idx=q_token_idx, TILE_Q_SIZE=TILE_DQ_Q_SIZE, TILE_K_SIZE=TILE_DQ_K_SIZE,
        INPUT_PRECISION=INPUT_PRECISION, PIPELINING=PIPELINING, K_BLOCK_DIVISIBLE=DQ_K_BLOCK_DIVISIBLE,
        RCP_LN2=RCP_LN2, SM_SCALE=SM_SCALE, PRESCALE_QK=PRESCALE_QK, HEAD_DIM=HEAD_DIM
    )

    dq_tile_ptr = tl.make_block_ptr(base=DQ + batch*stride_dqb + head*stride_dqh, shape=(T, HEAD_DIM), strides=(stride_dqt, stride_dqk), offsets=(q_token_idx,0), block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM), order=(1,0))
    tl.store(dq_tile_ptr, dq.to(dq_tile_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def _causal_attn_bwd_dkdv_inner( # Renamed
    Q: tl.tensor, K: tl.tensor, V: tl.tensor, DELTA: tl.tensor, LSE: tl.tensor,
    DO: tl.tensor, DK: tl.tensor, DV: tl.tensor,
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: int,
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,
    stride_deltab: int, stride_deltah: int, stride_deltat: int,
    stride_mb: int, stride_mh: int, stride_mt: int,
    stride_dob: int, stride_doh: int, stride_dot: int, stride_dok: int,
    stride_dkb: int, stride_dkh: int, stride_dkt: int, stride_dkk: int,
    stride_dvb: int, stride_dvh: int, stride_dvt: int, stride_dvk: int,
    batch: int, head: int, tile_id: int, seq_len: tl.tensor, T: int,
    HEAD_DIM: tl.constexpr, INPUT_PRECISION: tl.constexpr, SM_SCALE: tl.constexpr, PRESCALE_QK: tl.constexpr,
    DK_Q_BLOCK_DIVISIBLE: tl.constexpr, DK_K_BLOCK_DIVISIBLE: tl.constexpr, RCP_LN2: tl.constexpr,
    TILE_DK_Q_SIZE: tl.constexpr, TILE_DK_K_SIZE: tl.constexpr, PIPELINING: tl.constexpr,
    # Removed CONTEXT_SIZE, CONTEXTS_BACK, PERFECT_DKV_MATCHING
):
    kv_tile_idx = tile_id
    kv_token_idx = kv_tile_idx * TILE_DK_K_SIZE

    # Pointer setup (same as before)
    qt_tile_ptr = tl.make_block_ptr(base=Q + batch*stride_qb + head*stride_qh, shape=(HEAD_DIM, T), strides=(stride_qk, stride_qt), offsets=(0,0), block_shape=(HEAD_DIM, TILE_DK_Q_SIZE), order=(0,1))
    k_tile_ptr = tl.make_block_ptr(base=K + batch*stride_kb + head*stride_kh, shape=(T, HEAD_DIM), strides=(stride_kt, stride_kk), offsets=(kv_token_idx,0), block_shape=(TILE_DK_K_SIZE, HEAD_DIM), order=(1,0))
    v_tile_ptr = tl.make_block_ptr(base=V + batch*stride_vb + head*stride_vh, shape=(T, HEAD_DIM), strides=(stride_vt, stride_vk), offsets=(kv_token_idx,0), block_shape=(TILE_DK_K_SIZE, HEAD_DIM), order=(1,0))
    do_tile_ptr = tl.make_block_ptr(base=DO + batch*stride_dob + head*stride_doh, shape=(T, HEAD_DIM), strides=(stride_dot, stride_dok), offsets=(0,0), block_shape=(TILE_DK_Q_SIZE, HEAD_DIM), order=(1,0))
    lse_tile_ptr = tl.make_block_ptr(base=LSE + batch*stride_mb + head*stride_mh, shape=(T,), strides=(stride_mt,), offsets=(0,), block_shape=(TILE_DK_Q_SIZE,), order=(0,))
    delta_tile_ptr = tl.make_block_ptr(base=DELTA + batch*stride_deltab + head*stride_deltah, shape=(T,), strides=(stride_deltat,), offsets=(0,), block_shape=(TILE_DK_Q_SIZE,), order=(0,))

    dv = tl.zeros([TILE_DK_K_SIZE, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([TILE_DK_K_SIZE, HEAD_DIM], dtype=tl.float32)

    if DK_K_BLOCK_DIVISIBLE:
        k, v = tl.load(k_tile_ptr), tl.load(v_tile_ptr)
    else:
        k = tl.load(k_tile_ptr, boundary_check=(0,), padding_option="zero")
        v = tl.load(v_tile_ptr, boundary_check=(0,), padding_option="zero")

    dk, dv = _causal_attn_bwd_dkdv( # Renamed
        dk, dv, qt_tile_ptr, do_tile_ptr, lse_tile_ptr, delta_tile_ptr, k, v,
        seq_len=seq_len, kv_token_idx=kv_token_idx, TILE_Q_SIZE=TILE_DK_Q_SIZE, TILE_K_SIZE=TILE_DK_K_SIZE,
        INPUT_PRECISION=INPUT_PRECISION, PIPELINING=PIPELINING, Q_BLOCK_DIVISIBLE=DK_Q_BLOCK_DIVISIBLE,
        RCP_LN2=RCP_LN2, SM_SCALE=SM_SCALE, PRESCALE_QK=PRESCALE_QK, T=T, HEAD_DIM=HEAD_DIM
    )

    dk_tile_ptr = tl.make_block_ptr(base=DK + batch*stride_dkb + head*stride_dkh, shape=(T,HEAD_DIM), strides=(stride_dkt, stride_dkk), offsets=(kv_token_idx,0), block_shape=(TILE_DK_K_SIZE, HEAD_DIM), order=(1,0))
    dv_tile_ptr = tl.make_block_ptr(base=DV + batch*stride_dvb + head*stride_dvh, shape=(T,HEAD_DIM), strides=(stride_dvt, stride_dvk), offsets=(kv_token_idx,0), block_shape=(TILE_DK_K_SIZE, HEAD_DIM), order=(1,0))
    tl.store(dk_tile_ptr, dk.to(dk_tile_ptr.type.element_ty), boundary_check=(0,))
    tl.store(dv_tile_ptr, dv.to(dv_tile_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def _causal_attn_bwd_dq( # Renamed
    dq: tl.tensor, q: tl.tensor, m: tl.tensor, di: tl.tensor, do: tl.tensor,
    kt_tile_ptr: tl.tensor, vt_tile_ptr: tl.tensor,
    seq_len: tl.tensor, q_token_idx: int, TILE_Q_SIZE: tl.constexpr, TILE_K_SIZE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr, PIPELINING: tl.constexpr, K_BLOCK_DIVISIBLE: tl.constexpr,
    RCP_LN2: tl.constexpr, SM_SCALE: tl.constexpr, PRESCALE_QK: tl.constexpr, HEAD_DIM: tl.constexpr,
    # Removed CONTEXT_SIZE, CONTEXTS_BACK, PERFECT_MATCHING
):
    # Causal: iterate K/V tiles up to and including the current Q tile's end position
    kv_start_tile_idx = 0
    kv_end_tile_idx = tl.cdiv(min(q_token_idx + TILE_Q_SIZE, seq_len), TILE_K_SIZE)

    q_tile_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)
    tile_k_arange = tl.arange(0, TILE_K_SIZE)

    softmax_scale: tl.constexpr = tl.cast(SM_SCALE, q.dtype)
    if PRESCALE_QK:
        q = q * softmax_scale * RCP_LN2

    for kv_tile_idx in tl.range(kv_start_tile_idx, kv_end_tile_idx, num_stages=PIPELINING):
        kv_token_idx = kv_tile_idx * TILE_K_SIZE
        # Load K and V tiles (transposed)
        kT = tl.load(tl.advance(kt_tile_ptr, (0,kv_token_idx)), boundary_check=(1,), padding_option="zero")
        # V is (T,D), vt_tile_ptr is (D,T) for efficient dot with DO_T later if needed, but here we need V_T for P @ V_T
        # So vt_tile_ptr is (D,T) meaning V is (T,D). We need V_tile for dp = tl.dot(DO, V_tile_transposed_to_vT)
        # The original code has vT = tl.load(tl.advance(vt_tile_ptr, (0, kv_token_idx,)), boundary_check=(1,))
        # This implies vt_tile_ptr is for K_T like shape (D, T_k)
        # For DQ calculation: ds = p * (dp - di), where dp = do @ v_T
        # So we need v_T of shape (HEAD_DIM, TILE_K_SIZE)
        vT_tile = tl.load(tl.advance(vt_tile_ptr, (0, kv_token_idx)), boundary_check=(1,), padding_option="zero")


        qk = tl.dot(q, kT, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK:
            qk = qk * softmax_scale * RCP_LN2

        p = tl.math.exp2(qk - m) # m is lse from fwd pass for this q_tile

        # Causal masking for p
        kv_indices = kv_token_idx + tile_k_arange
        q_valid_mask = q_tile_indices[:, None] < seq_len
        k_valid_mask = kv_indices[None, :] < seq_len
        causal_mask = q_tile_indices[:, None] >= kv_indices[None, :]
        current_mask = q_valid_mask & k_valid_mask & causal_mask

        p = tl.where(current_mask, p, 0.0)

        dp = tl.dot(do, vT_tile.to(do.dtype), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        ds = p * (dp - di[:, None]) # di is delta
        # ds needs to be cast to Q's dtype for dot with K
        dq = tl.dot(ds.to(q.dtype), tl.trans(kT).to(q.dtype), dq, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        # dq = tl.dot(ds, tl.trans(kT).to(ds.dtype), dq, input_precision=INPUT_PRECISION, out_dtype=tl.float32)


    dq *= softmax_scale # Apply scale at the end
    return dq


@triton.jit
def _causal_attn_bwd_dkdv( # Renamed
    dk: tl.tensor, dv: tl.tensor,
    qt_tile_ptr: tl.tensor, do_tile_ptr: tl.tensor,
    lse_tile_ptr: tl.tensor, delta_tile_ptr: tl.tensor,
    k: tl.tensor, v: tl.tensor,
    seq_len: tl.tensor, kv_token_idx: int, TILE_Q_SIZE: tl.constexpr, TILE_K_SIZE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr, PIPELINING: tl.constexpr, Q_BLOCK_DIVISIBLE: tl.constexpr,
    RCP_LN2: tl.constexpr, SM_SCALE: tl.constexpr, PRESCALE_QK: tl.constexpr, T: tl.constexpr, HEAD_DIM: tl.constexpr,
    # Removed CONTEXT_SIZE, CONTEXTS_BACK, PERFECT_MATCHING
):
    # For a given K/V tile (indexed by kv_token_idx), iterate over Q tiles that could have attended to it.
    # Q tiles start from this K/V tile's position due to causality.
    q_start_tile_idx = kv_token_idx // TILE_Q_SIZE
    q_end_tile_idx = tl.cdiv(T, TILE_Q_SIZE) # Iterate over all Q tiles

    kv_indices = kv_token_idx + tl.arange(0, TILE_K_SIZE)
    tile_q_arange = tl.arange(0, TILE_Q_SIZE)

    if PRESCALE_QK:
        k_scaled = k * (RCP_LN2 * SM_SCALE) # Scale K once
    else:
        k_scaled = k # No prescaling, scale applied to QK_T

    for q_tile_idx in tl.range(q_start_tile_idx, q_end_tile_idx, num_stages=PIPELINING):
        q_token_idx = q_tile_idx * TILE_Q_SIZE

        # Load Q_T, DO, LSE, DELTA for the current Q tile
        qT = tl.load(tl.advance(qt_tile_ptr, (0,q_token_idx)), boundary_check=(1,), padding_option="zero")
        current_do = tl.load(tl.advance(do_tile_ptr, (q_token_idx,0)), boundary_check=(0,), padding_option="zero")
        current_lse = tl.load(tl.advance(lse_tile_ptr, (q_token_idx,)), boundary_check=(0,)) # 1D
        current_delta = tl.load(tl.advance(delta_tile_ptr, (q_token_idx,)), boundary_check=(0,)) # 1D

        qkT = tl.dot(k_scaled, qT, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK:
            qkT *= (RCP_LN2 * SM_SCALE)

        pT = tl.math.exp2(qkT - current_lse[None, :]) # LSE is (TILE_Q_SIZE,) broadcast to (TILE_K_SIZE, TILE_Q_SIZE)

        # Causal masking for pT
        q_tile_indices = q_token_idx + tile_q_arange
        q_valid_mask = q_tile_indices[None, :] < seq_len
        k_valid_mask = kv_indices[:, None] < seq_len
        causal_mask = q_tile_indices[None, :] >= kv_indices[:, None]
        current_mask = q_valid_mask & k_valid_mask & causal_mask

        pT = tl.where(current_mask, pT, 0.0)

        # Calculate DV
        dv = tl.dot(pT.to(current_do.dtype), current_do, dv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

        # Calculate DK
        dpT = tl.dot(v.to(current_do.dtype), tl.trans(current_do), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        dsT = pT * (dpT - current_delta[None, :]) # Delta is (TILE_Q_SIZE,)
        dk = tl.dot(dsT.to(qT.dtype), tl.trans(qT), dk, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        # dk = tl.dot(dsT, tl.trans(qT).to(dsT.dtype), dk, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    if not PRESCALE_QK: # If K was not prescaled, DK needs the scale.
        dk *= SM_SCALE
    return dk, dv
# fmt: on


def _get_default_config_bwd(head_dim, dtype) -> tuple[int, int, int, int]:
    # This function returns (TILE_Q, TILE_K, NUM_WARPS, NUM_STAGES)
    # For simplicity, using similar logic to fwd, but can be tuned separately.
    if dtype == torch.float32:
        return (16, 16, 4, 1) # Smaller tiles for float32 bwd often
    elif head_dim <= 256 and torch.cuda.get_device_capability() >= (9, 0):  # H100
        if head_dim == 64: return (64, 64, 4, 3)
        elif head_dim == 128: return (64, 128, 8, 3)
        else: return (64, 64, 4, 2)
    elif torch.cuda.get_device_capability() >= (8, 0):  # A100
        if head_dim == 64: return (32, 128, 4, 3)
        elif head_dim == 128: return (64, 128, 8, 3)
        else: return (64, 64, 4, 2)
    else: return (16, 16, 4, 1)


def bwd_configs_pruner(configs, nargs, HEAD_DIM, DTYPE, T, **kwargs): # Added T
    min_size = 32
    max_size = min(MAX_TILE_SIZE, triton.next_power_of_2(T) if T else MAX_TILE_SIZE)
    min_pipeline, max_pipeline = 1, 3
    min_warps, max_warps = 1, 8

    # Simplified pruning
    if HEAD_DIM <= 32: min_pipeline = 2 # Example
    if HEAD_DIM == 64: min_pipeline = 2; max_warps=4
    elif HEAD_DIM == 128: max_pipeline = 2; max_warps=4
    elif HEAD_DIM == 256: max_pipeline = 1; max_warps=4

    configs = [c for c in configs if min_size <= c.kwargs["TILE_DQ_Q_SIZE"] <= max_size]
    configs = [c for c in configs if min_size <= c.kwargs["TILE_DQ_K_SIZE"] <= max_size]
    configs = [c for c in configs if min_size <= c.kwargs["TILE_DK_Q_SIZE"] <= max_size]
    configs = [c for c in configs if min_size <= c.kwargs["TILE_DK_K_SIZE"] <= max_size]
    configs = [c for c in configs if min_pipeline <= c.kwargs["PIPELINING"] <= max_pipeline]
    configs = [c for c in configs if min_warps <= c.num_warps <= max_warps]

    default_cfg_params = _get_default_config_bwd(HEAD_DIM, DTYPE)
    if default_cfg_params:
        configs.append(triton.Config(dict(
            PIPELINING=default_cfg_params[3],
            TILE_DQ_Q_SIZE=max(min_size, min(default_cfg_params[0],max_size)),
            TILE_DQ_K_SIZE=max(min_size, min(default_cfg_params[1],max_size)),
            TILE_DK_Q_SIZE=max(min_size, min(default_cfg_params[0],max_size)), # Assuming symmetric for DK for now
            TILE_DK_K_SIZE=max(min_size, min(default_cfg_params[1],max_size)),
        ), num_warps=default_cfg_params[2], num_stages=default_cfg_params[3]))

    logger.warning(f"Start benchmarking backward causal_attention {len(configs) = } for T={T}, HEAD_DIM={HEAD_DIM}")
    return configs


causal_backward = triton.heuristics( # Renamed
    dict( # Simplified defaults
        PIPELINING=lambda _: 1,
        TILE_DQ_Q_SIZE=lambda args: 64, TILE_DQ_K_SIZE=lambda args: 64,
        TILE_DK_Q_SIZE=lambda args: 64, TILE_DK_K_SIZE=lambda args: 64,
    )
)(_causal_attn_bwd) # Renamed

causal_backward_autotune = triton.autotune( # Renamed
    configs=[ # Simplified config list
        triton.Config(dict(
            PIPELINING=pipe,
            TILE_DQ_Q_SIZE=tile_qq, TILE_DQ_K_SIZE=tile_qk,
            TILE_DK_Q_SIZE=tile_kq, TILE_DK_K_SIZE=tile_kk,
        ), num_warps=num_warps, num_stages=pipe)
        for num_warps in [4, 8] for pipe in [1,2]
        for tile_qq in [32,64,128] for tile_qk in [32,64,128]
        for tile_kq in [32,64,128] for tile_kk in [32,64,128]
    ],
    key=[ # Removed CONTEXT_SIZE, CONTEXTS_BACK, Added T
        "HEAD_DIM", "INPUT_PRECISION", "DTYPE", "TIME_BUCKET", "T"
    ],
    prune_configs_by=dict(early_config_prune=bwd_configs_pruner),
    pre_hook=autotune_prehook, # Assuming hooks are general
    post_hook=autotune_posthook,
)(_causal_attn_bwd) # Renamed


@torch.library.custom_op( # Renamed op string
    "alexdremov_causal_attention::backward", mutates_args=(), device_types=("cuda",)
)
def causal_attention_backward_adapter( # Renamed
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lens: torch.Tensor,
    o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor,
    # context_size: int, back_contexts: int, # Removed
    sm_scale: float, autotune: bool, prescale_qk: bool, precision: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, T, HEAD_DIM = q.shape

    delta = torch.empty(o.shape[:-1], dtype=torch.float32, device=o.device)
    grid_precompute = lambda args: (batch, heads, triton.cdiv(T, args["TILE_SIZE"]))

    # Assuming _causal_attn_bwd_precompute is the correct name now
    _causal_attn_bwd_precompute[grid_precompute](
        o, do, delta,
        *strides(o,4), *strides(do,4), *strides(delta,3),
        T=T, HEAD_DIM=HEAD_DIM, DTYPE=q.dtype, TIME_BUCKET=triton.next_power_of_2(T)
    )

    DQ = torch.zeros_like(q) # No specific memory format needed for output of custom op usually
    DK = torch.zeros_like(k)
    DV = torch.zeros_like(v)

    grid_bwd = lambda args: (batch, heads, triton.cdiv(T, args["TILE_DQ_Q_SIZE"]) + triton.cdiv(T, args["TILE_DK_K_SIZE"]))

    bwd_fn = causal_backward_autotune if autotune else causal_backward # Renamed

    autotune_kwargs_bwd = {
        "Q":q, "K":k, "V":v, "L":lens, "DELTA":delta, "LSE":lse, "DO":do, "DQ":DQ, "DK":DK, "DV":DV,
        "stride_qb":q.stride(0), "stride_qh":q.stride(1), "stride_qt":q.stride(2), "stride_qk":q.stride(3),
        "stride_kb":k.stride(0), "stride_kh":k.stride(1), "stride_kt":k.stride(2), "stride_kk":k.stride(3),
        "stride_vb":v.stride(0), "stride_vh":v.stride(1), "stride_vt":v.stride(2), "stride_vk":v.stride(3),
        "stride_deltab":delta.stride(0), "stride_deltah":delta.stride(1), "stride_deltat":delta.stride(2),
        "stride_mb":lse.stride(0), "stride_mh":lse.stride(1), "stride_mt":lse.stride(2),
        "stride_dob":do.stride(0), "stride_doh":do.stride(1), "stride_dot":do.stride(2), "stride_dok":do.stride(3),
        "stride_dqb":DQ.stride(0), "stride_dqh":DQ.stride(1), "stride_dqt":DQ.stride(2), "stride_dqk":DQ.stride(3),
        "stride_dkb":DK.stride(0), "stride_dkh":DK.stride(1), "stride_dkt":DK.stride(2), "stride_dkk":DK.stride(3),
        "stride_dvb":DV.stride(0), "stride_dvh":DV.stride(1), "stride_dvt":DV.stride(2), "stride_dvk":DV.stride(3),
        "lens_stride":lens.stride(0) if lens is not None else 0,
        "T":T, "HEAD_DIM":HEAD_DIM,
        # "CONTEXT_SIZE":context_size, "CONTEXTS_BACK":back_contexts, # Removed
        "TIME_BUCKET":triton.next_power_of_2(T), "INPUT_PRECISION":precision,
        "DTYPE":q.dtype, "SM_SCALE":sm_scale, "PRESCALE_QK":prescale_qk,
    }

    bwd_fn[grid_bwd](**autotune_kwargs_bwd)
    return DQ, DK, DV


@torch.library.register_fake("alexdremov_causal_attention::backward") # Renamed op string
def causal_attention_backward_adapter_abstract( # Renamed
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lens: torch.Tensor | None,
    o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor,
    # context_size: int, back_contexts: int, # Removed
    sm_scale: float | None, autotune: bool, prescale_qk: bool, precision: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    DQ = torch.empty_like(q)
    DK = torch.empty_like(k)
    DV = torch.empty_like(v)
    return DQ, DK, DV


# Setup_context and backward op functions for autograd integration
# These connect to the *forward* op name "alexdremov_causal_attention::forward"
def causal_attention_backward_op_setup_context(ctx, inputs, output): # Renamed for clarity
    # Output is (O, LSE) from the forward pass
    O, LSE_from_fwd = output # LSE_from_fwd could be empty if not requested and not requires_grad

    # Inputs to forward: q, k, v, lens, sm_scale, autotune, return_lse, prescale_qk, precision
    ( q, k, v, lens,
      # context_size, back_contexts, # Removed
      sm_scale, autotune, return_lse_flag, prescale_qk, precision
    ) = inputs

    ctx.save_for_backward(q, k, v, O, LSE_from_fwd, lens) # LSE_from_fwd must be saved
    # ctx.context_size = context_size # Removed
    # ctx.back_contexts = back_contexts # Removed
    ctx.autotune = autotune
    ctx.sm_scale = sm_scale
    ctx.prescale_qk = prescale_qk
    ctx.precision = precision

def causal_attention_backward_op(ctx, do, dlse): # Renamed for clarity
    q, k, v, o, lse, lens = ctx.saved_tensors # o is saved output, lse is saved LSE
    # context_size = ctx.context_size # Removed
    # back_contexts = ctx.back_contexts # Removed
    autotune = ctx.autotune
    sm_scale = ctx.sm_scale
    prescale_qk = ctx.prescale_qk
    precision = ctx.precision

    # Call the new backward adapter
    DQ, DK, DV = torch.ops.alexdremov_causal_attention.backward(
        q=q, k=k, v=v, lens=lens, o=o, lse=lse, do=do,
        # context_size=context_size, back_contexts=back_contexts, # Removed
        sm_scale=sm_scale, autotune=autotune, prescale_qk=prescale_qk, precision=precision,
    )
    # Return gradients for: q, k, v, lens, sm_scale, autotune, return_lse, prescale_qk, precision
    # lens, context_size etc are not differentiable.
    return DQ, DK, DV, None, None, None, None, None, None
    # Return grads for q, k, v, lens, context_size, back_contexts, sm_scale, autotune, return_lse, prescale_qk, precision
    # The Nones correspond to non-tensor inputs or inputs that don't require grad.
    # Original had 11 inputs to forward, so 11 potential grads.
    # New forward has 9 inputs.
    # Return DQ, DK, DV, None (lens), None (sm_scale), None (autotune), None (return_lse), None (prescale_qk), None (precision)
    # This matches the number of inputs to causal_attention_forward_adapter.
