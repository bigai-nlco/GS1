"""Inference-only native row geometry; shared parameters and equations are unchanged."""

import torch
from torch.nn import functional as F
from transformers.integrations.sdpa_attention import repeat_kv, use_gqa_in_sdpa
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from gs1.inference.normalization import gated_norm_rows, l2norm_rows


def project(module, value):
    # Keep each row's original GEMM geometry. Concatenating independent branches
    # into one tall GEMM changes BF16 reduction kernels on the qualified runtime.
    return torch.cat([F.linear(row, module.weight, module.bias) for row in value.split(1)], dim=0)


def normalize(module, value):
    return torch.cat([module(row) for row in value.split(1)], dim=0)


def mlp(module, value, projections=project):
    return projections(
        module.down_proj,
        module.act_fn(projections(module.gate_proj, value)) * projections(module.up_proj, value),
    )


def native_sdpa(module, query, key, value, mask, *, require_efficient=False):
    """Match pinned HF SDPA without changing process-global backend flags."""
    options = {}
    if hasattr(module, "num_key_value_groups"):
        if use_gqa_in_sdpa(mask, key):
            options["enable_gqa"] = True
        else:
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
    causal = query.shape[2] > 1 and mask is None and getattr(module, "is_causal", True)
    if require_efficient and query.device.type == "cuda":
        # SDPA is autocast eligible; its selector is not. Inspect the actual
        # compute dtypes rather than the FP32 RoPE intermediates.
        compute = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else query.dtype
        backend = torch._fused_sdp_choice(
            query.to(compute),
            key.to(compute),
            value.to(compute),
            mask,
            0.0,
            causal,
            scale=module.scaling,
            **options,
        )
        if backend == 0:
            raise ValueError("Qualified shared attention cannot fall back to quadratic math SDPA")
    output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, dropout_p=0.0, scale=module.scaling, is_causal=causal, **options
    )
    return output.transpose(1, 2).contiguous()


def attention(module, value, positional, mask, projections=project):
    batch, length = value.shape[:2]
    shape = (batch, length, -1, module.head_dim)
    query, gate = projections(module.q_proj, value).view(batch, length, -1, 2 * module.head_dim).chunk(2, -1)
    query = normalize(module.q_norm, query).transpose(1, 2)
    key = normalize(module.k_norm, projections(module.k_proj, value).view(shape)).transpose(1, 2)
    values = projections(module.v_proj, value).view(shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *positional)
    result = native_sdpa(module, query, key, values, mask)
    result = result.reshape(batch, length, -1) * gate.reshape(batch, length, -1).sigmoid()
    return projections(module.o_proj, result)


def linear_scan(
    module, value, previous=None, *, projections=project, channel_last=False, export_boundary=False
):
    """Use Qwen's own kernels, with explicit immutable convolution/scan inputs."""
    batch, length = value.shape[:2]
    raw = projections(module.in_proj_qkv, value).transpose(1, 2)
    z = projections(module.in_proj_z, value).reshape(batch, length, -1, module.head_v_dim)
    b, a = projections(module.in_proj_b, value), projections(module.in_proj_a, value)
    if previous is None:
        window = raw
        conv_state = F.pad(raw, (module.conv_kernel_size - length, 0))
    else:
        window = torch.cat((previous[0], raw), dim=-1)
        conv_state = window[:, :, -module.conv_kernel_size :].clone()
    if module.causal_conv1d_fn is not None:
        # Match prefill's channel-last input and fused convolution+activation.
        if channel_last:
            window = window.transpose(1, 2).contiguous().transpose(1, 2)
        mixed = module.causal_conv1d_fn(
            x=window,
            weight=module.conv1d.weight.squeeze(1),
            bias=module.conv1d.bias,
            activation=module.activation,
            seq_idx=None,
        )[:, :, -length:]
    elif previous is None:
        mixed = F.silu(module.conv1d(window)[:, :, :length])
    else:
        mixed = F.silu(
            F.conv1d(window[:, :, 1:], module.conv1d.weight, module.conv1d.bias, groups=raw.shape[1])
        )
    query, key, values = mixed.transpose(1, 2).split((module.key_dim, module.key_dim, module.value_dim), -1)
    query = query.reshape(batch, length, -1, module.head_k_dim)
    key = key.reshape(batch, length, -1, module.head_k_dim)
    values = values.reshape(batch, length, -1, module.head_v_dim)
    beta = b.sigmoid()
    decay = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
    ratio = module.num_v_heads // module.num_k_heads
    if ratio > 1:
        query, key = query.repeat_interleave(ratio, 2), key.repeat_interleave(ratio, 2)
    chunk = module.chunk_gated_delta_rule
    use_qk_l2norm_in_kernel = True
    if batch > 1 and chunk is qwen.chunk_gated_delta_rule:
        # Keep the native FLA normalization, but exclude other branches from
        # its autotune geometry. The recurrent scan itself remains batched.
        query, key = l2norm_rows(query), l2norm_rows(key)
        use_qk_l2norm_in_kernel = False
    boundaries = None
    if export_boundary and value.device.type == "cuda" and length > 64 and length % 64:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        if chunk is chunk_gated_delta_rule:
            from gs1.inference.boundary import chunk_with_boundaries

            chunk = chunk_with_boundaries
    result = chunk(
        query,
        key,
        values,
        g=decay,
        beta=beta,
        initial_state=None if previous is None else previous[1],
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    mixed, recurrent = result[:2]
    if len(result) == 3:
        completed = length // 64 * 64
        end = completed + (module.conv_kernel_size if previous is not None else 0)
        boundaries = (
            window[:, :, end - module.conv_kernel_size : end].clone(),
            result[2][:, completed // 64].clone(),
        )
    mixed = gated_norm_rows(module.norm, mixed, z)
    output = projections(module.out_proj, mixed.reshape(batch, length, -1))
    state = (conv_state, recurrent)
    return (output, state, boundaries) if export_boundary else (output, state)


class NativeRowGeometry:
    """Canonical per-row GEMMs/norms, real batched GDN/FA and exact GS1 routing."""

    name = "gs1.native-row-geometry.v1"

    def __init__(self, cache_bytes=0):
        if cache_bytes < 0:
            raise ValueError("Compute cache limit must be nonnegative")
        self.cache_bytes = cache_bytes
        self.weights = {}
        self.fingerprint = None

    def prepare(self, model, precision):
        from gs1.inference.state import parameter_fingerprint

        fingerprint = (parameter_fingerprint(model), precision)
        if fingerprint != self.fingerprint:
            # Replica-owned compute copies live outside every private graph
            # pool. Slots share these immutable tensors without model hooks.
            self.weights.clear()
            remaining = self.cache_bytes
            with torch.no_grad():
                for module in model.language_model.modules():
                    if precision != "bf16" or not isinstance(module, torch.nn.Linear):
                        continue
                    size = 2 * (
                        module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
                    )
                    if size > remaining:
                        continue
                    self.weights[id(module)] = (
                        module.weight.to(torch.bfloat16),
                        module.bias.to(torch.bfloat16) if module.bias is not None else None,
                    )
                    remaining -= size
            self.fingerprint = fingerprint

    def weight(self, module):
        return self.weights.get(id(module), (module.weight, module.bias))

    def project(self, module, value):
        weight, bias = self.weight(module)
        return torch.cat([F.linear(row, weight, bias) for row in value.split(1)], 0)

    def segment_geometry(self, model, state, branch):
        return SegmentGeometry(model, state, branch, self)

    def encode(self, model, hidden, layout, *, terminal_query):
        if torch.is_grad_enabled() or model.training:
            raise ValueError("Native row execution policy is inference-only")
        keep, lengths, masks, positional, cohorts = layout
        hidden = hidden * keep
        bank, partial = (hidden,), None
        for index, layer in enumerate(model.layers):
            read = hidden
            if str(index) in model.attnres_routers:
                router = model.attnres_routers[str(index)]
                sources = bank if partial is None else (*bank, partial)
                pieces = []
                for row, length in enumerate(lengths):
                    carry = hidden[row : row + 1, -length:]
                    value = router(carry, tuple(s[row : row + 1, -length:] for s in sources))
                    pieces.append(F.pad(value, (0, 0, hidden.shape[1] - length, 0)))
                read = torch.cat(pieces, 0)
            mixed_all = ffn_all = None
            outputs = torch.zeros_like(hidden)
            for length, rows in cohorts:
                carry = hidden.index_select(0, rows)[:, -length:]
                if layer.layer_type == "full_attention":
                    value = read.index_select(0, rows)[:, -length:]
                    positions = tuple(p.index_select(0, rows)[:, -length:] for p in positional)
                    mask = masks[model.is_bidirectional(index)].index_select(0, rows)[
                        :, :, -length:, -length:
                    ]
                    mixed = attention(
                        layer.self_attn,
                        normalize(layer.input_layernorm, value),
                        positions,
                        mask,
                        self.project,
                    )
                    after = carry + mixed
                    ffn = mlp(layer.mlp, normalize(layer.post_attention_layernorm, after), self.project)
                    output = after + ffn
                elif terminal_query:
                    context, terminal = carry[:, :-1], carry[:, -1:]
                    mixed_context, state = linear_scan(
                        layer.linear_attn, normalize(layer.input_layernorm, context), projections=self.project
                    )
                    mixed_terminal, _ = linear_scan(
                        layer.linear_attn,
                        normalize(layer.input_layernorm, terminal),
                        state,
                        projections=self.project,
                    )
                    context, terminal = context + mixed_context, terminal + mixed_terminal
                    ffn_context = mlp(
                        layer.mlp, normalize(layer.post_attention_layernorm, context), self.project
                    )
                    ffn_terminal = mlp(
                        layer.mlp, normalize(layer.post_attention_layernorm, terminal), self.project
                    )
                    mixed = torch.cat((mixed_context, mixed_terminal), 1)
                    ffn = torch.cat((ffn_context, ffn_terminal), 1)
                    output = torch.cat((context + ffn_context, terminal + ffn_terminal), 1)
                else:
                    mixed, _ = linear_scan(
                        layer.linear_attn, normalize(layer.input_layernorm, carry), projections=self.project
                    )
                    after = carry + mixed
                    ffn = mlp(layer.mlp, normalize(layer.post_attention_layernorm, after), self.project)
                    output = after + ffn

                def store(container, value):
                    return container.index_copy(
                        0, rows, F.pad(value, (0, 0, hidden.shape[1] - length, 0)).to(container.dtype)
                    )

                if mixed_all is None:
                    # AttnRes adds mixed+FFN at their native compute dtype.
                    # Promoting each increment first changes its BF16 rounding.
                    mixed_all = torch.zeros_like(hidden, dtype=mixed.dtype)
                    ffn_all = torch.zeros_like(hidden, dtype=ffn.dtype)
                mixed_all, ffn_all, outputs = (
                    store(mixed_all, mixed),
                    store(ffn_all, ffn),
                    store(outputs, output),
                )
            hidden = outputs
            if model.attnres_routers:
                bank, partial = model._advance_depth(
                    bank, partial, mixed_all, ffn_all, layer.layer_type == "full_attention"
                )
        return torch.cat(
            [
                F.pad(model.norm(hidden[row : row + 1, -length:]), (0, 0, hidden.shape[1] - length, 0))
                for row, length in enumerate(lengths)
            ],
            0,
        )


def pad_tokens(value, extent, origin):
    length = value.shape[1]
    if not 0 <= origin <= extent - length:
        raise ValueError("Segment exceeds canonical GS1 geometry")
    return F.pad(value, (0, 0) * (value.ndim - 2) + (origin, extent - origin - length))


class SegmentGeometry:
    """Canonical GS1 role masks and arithmetic for independently cached segments."""

    def __init__(self, model, state, branch, policy):
        from gs1.inference.state import branch_geometry
        from gs1.model.compiler import prepare_batch
        from gs1.model.util import digest

        self.length = len(state) + len(branch.suffix)
        self.policy = policy
        self.signature = digest(branch_geometry([(state, branch)]))
        prepared = prepare_batch(model, [(state, branch)])
        self.masks, self.positional = prepared.layout[2:4]
        self.retained = (self.masks, self.positional)

    def _project(self, module, value, extent, origin):
        weight, bias = self.policy.weight(module)
        return torch.cat(
            [
                F.linear(pad_tokens(row, extent, origin), weight, bias)[
                    :, origin : origin + value.shape[1]
                ].contiguous()
                for row in value.split(1)
            ],
            0,
        )

    @staticmethod
    def _norm(module, value, extent, origin):
        return torch.cat(
            [
                module(pad_tokens(row, extent, origin))[:, origin : origin + value.shape[1]].contiguous()
                for row in value.split(1)
            ],
            0,
        )

    def _mlp(self, module, value, extent, origin):
        gate = self._project(module.gate_proj, value, extent, origin)
        up = self._project(module.up_proj, value, extent, origin)
        return self._project(module.down_proj, module.act_fn(gate) * up, extent, origin)

    def _attention(self, module, value, start, previous, bidirectional):
        batch, length = value.shape[:2]

        def project(module, x):
            return self._project(module, x, self.length, start)

        query, gate = project(module.q_proj, value).view(batch, length, -1, 2 * module.head_dim).chunk(2, -1)
        query = self._norm(module.q_norm, query, self.length, start).transpose(1, 2)
        shape = (batch, length, -1, module.head_dim)
        key = self._norm(
            module.k_norm, project(module.k_proj, value).view(shape), self.length, start
        ).transpose(1, 2)
        values = project(module.v_proj, value).view(shape).transpose(1, 2)
        positions = tuple(p[:, start : start + length].expand(batch, -1, -1) for p in self.positional)
        query, key = apply_rotary_pos_emb(query, key, *positions)
        if previous is not None:
            key, values = torch.cat((previous[0], key), 2), torch.cat((previous[1], values), 2)
        # The full segmented mask forbids all unavailable future keys. Dummy
        # query rows never influence an active query or its recurrent state.
        q = pad_tokens(query.transpose(1, 2), self.length, start).transpose(1, 2)
        k = pad_tokens(key.transpose(1, 2), self.length, 0).transpose(1, 2)
        v = pad_tokens(values.transpose(1, 2), self.length, 0).transpose(1, 2)
        mask = self.masks[bidirectional].expand(batch, -1, -1, -1)
        result = native_sdpa(module, q, k, v, mask)
        result = result[:, start : start + length].reshape(batch, length, -1)
        output = project(module.o_proj, result * gate.reshape(batch, length, -1).sigmoid())
        return output, (key, values)

    def segment(self, model, input_ids, start_position, prefix, *, terminal_query):
        from gs1.model.modular_gs1 import _continue_gdn

        if torch.is_grad_enabled() or model.training:
            raise ValueError("Canonical segment geometry is inference-only")
        hidden = model.embed_tokens(input_ids)
        bank, partial, states = (hidden,), None, []
        for index, layer in enumerate(model.layers):
            full = layer.layer_type == "full_attention"
            extent = self.length if full else 1 if terminal_query else self.length - 1
            origin = 0 if terminal_query and not full else start_position
            read = hidden
            if str(index) in model.attnres_routers:
                sources = bank if partial is None else (*bank, partial)
                router = model.attnres_routers[str(index)]
                read = torch.cat(
                    [
                        router(
                            pad_tokens(hidden[row : row + 1], self.length, start_position),
                            tuple(pad_tokens(s[row : row + 1], self.length, start_position) for s in sources),
                        )[:, start_position : start_position + hidden.shape[1]].contiguous()
                        for row in range(hidden.shape[0])
                    ],
                    0,
                )
            value = self._norm(layer.input_layernorm, read, extent, origin)
            previous = prefix[index] if prefix is not None else None
            if full:
                mixed, state = self._attention(
                    layer.self_attn, value, start_position, previous, model.is_bidirectional(index)
                )
            else:
                combined_origin = (
                    0 if terminal_query or previous is None else previous.position - previous.tail.shape[1]
                )

                def scan(module, value, state, config):
                    def project(module, x):
                        return self._project(module, x, extent, combined_origin)

                    return linear_scan(
                        module,
                        value,
                        state,
                        projections=project,
                        channel_last=not terminal_query,
                        export_boundary=not terminal_query,
                    )

                mixed, state = _continue_gdn(
                    layer.linear_attn,
                    value,
                    previous,
                    model.config.text_config,
                    terminal=terminal_query,
                    scan=scan,
                )
            hidden = hidden + mixed
            ffn = self._mlp(
                layer.mlp, self._norm(layer.post_attention_layernorm, hidden, extent, origin), extent, origin
            )
            hidden = hidden + ffn
            if model.attnres_routers:
                bank, partial = model._advance_depth(bank, partial, mixed, ffn, full)
            states.append(state)
        return self._norm(model.norm, hidden, self.length, start_position), tuple(states)
