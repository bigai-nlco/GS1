"""Single-pass FP32 recurrent boundary export for the pinned FLA kernels."""

import torch
import triton

from gs1.inference.normalization import l2norm_rows


def chunk_with_boundaries(
    q, k, v, *, g, beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True
):
    """Pinned dense BT=64 inference subset of FLA for native GS1 GDN."""
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    from fla.ops.common.chunk_o import chunk_fwd_o
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2

    if (
        torch.is_grad_enabled()
        or q.device.type != "cuda"
        or q.dtype not in {torch.float32, torch.bfloat16}
        or not output_final_state
        or q.ndim != 4
        or k.shape != q.shape
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.shape[:3] != v.shape[:3]
        or g.shape != q.shape[:3]
        or beta.shape != g.shape
        or q.shape[-1] > 256
        or q.shape[1] < 1
    ):
        raise ValueError("Boundary export requires qualified dense Qwen3.5 inference geometry")
    q, k, v, g, beta = [x.contiguous() for x in (q, k, v, g, beta)]
    initial_state = initial_state.contiguous() if initial_state is not None else None
    if use_qk_l2norm_in_kernel:
        q, k = l2norm_rows(q), l2norm_rows(k)
    g = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2, cu_seqlens=None, chunk_indices=None)
    w, u, _ = chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=g, beta=beta, cu_seqlens=None, chunk_indices=None, chunk_size=64
    )
    batch, length, heads, keys = k.shape
    values, value_heads = u.shape[-1], u.shape[2]
    h = torch.empty(
        (batch, triton.cdiv(length, 64), value_heads, keys, values), dtype=torch.float32, device=k.device
    )
    final = torch.zeros((batch, value_heads, keys, values), dtype=torch.float32, device=k.device)
    v_new = torch.empty_like(u)

    def grid(meta):
        return triton.cdiv(values, meta["BV"]), batch * value_heads

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=None,
        h=h,
        h0=initial_state,
        ht=final,
        cu_seqlens=None,
        chunk_offsets=None,
        T=length,
        H=heads,
        HV=value_heads,
        K=keys,
        V=values,
        BT=64,
        STATE_V_FIRST=False,
    )
    output = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h.to(k.dtype),
        g=g,
        scale=keys**-0.5,
        cu_seqlens=None,
        chunk_indices=None,
        state_v_first=False,
        chunk_size=64,
    )
    return output, final, h
