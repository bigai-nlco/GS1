"""GS1: native hybrid text/vision decision model with attention residuals."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModel, Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DynamicCache,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionModel,
    apply_rotary_pos_emb,
)
from transformers.utils import ModelOutput

from .config import INFERENCE_LIMITS, CalibrationConfig, DecisionConfig
from .util import named_seed


class GS1Config(Qwen3_5Config):
    model_type = "gs1"

    def __init__(
        self,
        decision=None,
        calibration=None,
        initialization_seed=12,
        head_compute_dtype="float32",
        inference_limits=None,
        **kwargs,
    ):
        kwargs.pop("model_type", None)
        kwargs.pop("architectures", None)
        super().__init__(**kwargs)
        self.decision = DecisionConfig.model_validate(decision or {}).model_dump()
        self.calibration = CalibrationConfig.model_validate(calibration or {}).model_dump()
        self.initialization_seed = initialization_seed
        self.inference_limits = {**INFERENCE_LIMITS, **(inference_limits or {})}
        if self.inference_limits.keys() != INFERENCE_LIMITS.keys() or any(
            type(value) is not int or value < 1 for value in self.inference_limits.values()
        ):
            raise ValueError("Invalid GS1 inference limits")
        if head_compute_dtype not in {"float32", "bfloat16"}:
            raise ValueError("head_compute_dtype must be float32 or bfloat16")
        self.head_compute_dtype = head_compute_dtype
        self.architectures = ["GS1Model"]
        self.tie_word_embeddings = False
        self.text_config.use_cache = False
        self.text_config._attn_implementation = "sdpa"
        self.vision_config._attn_implementation = "sdpa"


@dataclass
class GS1Output(ModelOutput):
    logits: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None


@dataclass
class GS1TextOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None


@dataclass(frozen=True)
class GDNState:
    """Native GDN prefix, including the last complete scan-chunk boundary."""

    conv: torch.Tensor
    recurrent: torch.Tensor
    boundary_conv: torch.Tensor | None
    boundary_recurrent: torch.Tensor | None
    tail: torch.Tensor
    position: int


class GS1DecoderLayer(nn.Module):
    """A hybrid decoder block; only its full-attention input can read depth history."""

    def __init__(self, config, index):
        super().__init__()
        self.layer_type = config.layer_types[index]
        self.config = config
        self.index = index
        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, index)
        elif self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, index)
        else:
            raise ValueError(f"Unsupported decoder layer type: {self.layer_type}")
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, carry, positional, mask, read, terminal_query, valid_lengths=None, cohorts=None):
        if self.layer_type == "full_attention":
            mixed, _ = self.self_attn(
                self.input_layernorm(read), position_embeddings=positional, attention_mask=mask
            )
        elif valid_lengths is not None and min(valid_lengths) != carry.shape[1]:
            # GDN scans only real tokens. Equal-length cohorts retain native
            # batched kernels and their original chunk boundaries; FA and FFN
            # still execute over the complete decoder batch.
            mixed = torch.zeros_like(carry)
            if cohorts is None:
                cohorts = tuple(
                    (
                        length,
                        torch.tensor(
                            [i for i, value in enumerate(valid_lengths) if value == length],
                            device=carry.device,
                        ),
                    )
                    for length in sorted(set(valid_lengths))
                )
            for length, rows in cohorts:
                value = carry.index_select(0, rows)[:, -length:]
                normalized = self.input_layernorm(value)
                if terminal_query:
                    cache = Qwen3_5DynamicCache(self.config)
                    cache.last_linear_layer = self.index
                    context = self.linear_attn(
                        normalized[:, :-1],
                        cache_params=cache,
                        cache_position=torch.arange(length - 1, device=carry.device),
                    )
                    query, _ = _linear_context(
                        self.linear_attn,
                        normalized[:, -1:],
                        (cache.conv_states[self.index], cache.recurrent_states[self.index]),
                        self.config,
                    )
                    value = torch.cat((context, query), 1)
                else:
                    value = self.linear_attn(normalized)
                mixed = mixed.index_copy(
                    0, rows, F.pad(value, (0, 0, carry.shape[1] - length, 0)).to(mixed.dtype)
                )
        elif terminal_query:
            # Keep the trained context scan and differentiable one-token continuation.
            cache = Qwen3_5DynamicCache(self.config)
            cache.last_linear_layer = self.index
            context = carry[:, :-1]
            context_mixed = self.linear_attn(
                self.input_layernorm(context),
                cache_params=cache,
                cache_position=torch.arange(context.shape[1], device=carry.device),
            )
            query_mixed, _ = _linear_context(
                self.linear_attn,
                self.input_layernorm(carry[:, -1:]),
                (cache.conv_states[self.index], cache.recurrent_states[self.index]),
                self.config,
            )
            context = context + context_mixed
            query = carry[:, -1:] + query_mixed
            context_ffn = self.mlp(self.post_attention_layernorm(context))
            query_ffn = self.mlp(self.post_attention_layernorm(query))
            return (
                torch.cat((context + context_ffn, query + query_ffn), 1),
                torch.cat((context_mixed, query_mixed), 1),
                torch.cat((context_ffn, query_ffn), 1),
            )
        else:
            mixed = self.linear_attn(self.input_layernorm(carry))
        hidden = carry + mixed
        ffn = self.mlp(self.post_attention_layernorm(hidden))
        return hidden + ffn, mixed, ffn


class CrossAttentionDecisionHead(nn.Module):
    """Read state at backbone width; project only the final dot-product vectors."""

    def __init__(self, hidden_size, decision=None, seed=12):
        super().__init__()
        self.config = DecisionConfig.model_validate(decision or {})
        if hidden_size % self.config.cross_attention_heads:
            raise ValueError("cross_attention_heads must divide the backbone hidden size")
        width = self.config.projection_size
        eps = self.config.norm_eps
        self.bf16_compute = False
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(named_seed(seed, "initialization/cross_attention_head"))
            self.memory_norm = nn.LayerNorm(hidden_size, eps=eps)
            self.memory_projection = nn.Linear(hidden_size, hidden_size, bias=False)
            self.question_seed = nn.Linear(hidden_size, hidden_size, bias=False)
            self.question_attention = nn.MultiheadAttention(
                hidden_size, self.config.cross_attention_heads, batch_first=True
            )
            self.decision_projection = nn.Linear(hidden_size, hidden_size, bias=False)
            self.question_projection = nn.Linear(hidden_size, hidden_size, bias=False)
            self.question_evidence_projection = nn.Linear(hidden_size, hidden_size, bias=False)
            self.context_norm = nn.LayerNorm(hidden_size, eps=eps)
            self.option_projection = nn.Linear(hidden_size, hidden_size, bias=False)
            self.condition_projection = nn.Linear(hidden_size, hidden_size, bias=False)
            self.option_norm = nn.LayerNorm(hidden_size, eps=eps)
            self.routing_attention = nn.ModuleList(
                nn.MultiheadAttention(hidden_size, self.config.cross_attention_heads, batch_first=True)
                for _ in range(self.config.cross_attention_layers)
            )
            self.routing_attention_norm = nn.ModuleList(
                nn.LayerNorm(hidden_size, eps=eps) for _ in range(self.config.cross_attention_layers)
            )
            self.routing_ffn = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(self.config.cross_attention_layers)
            )
            self.routing_output_norm = nn.ModuleList(
                nn.LayerNorm(hidden_size, eps=eps) for _ in range(self.config.cross_attention_layers)
            )
            self.query_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, width)
            )
            self.key_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, width)
            )
            self.query_norm = nn.LayerNorm(width, eps=eps)
            self.key_norm = nn.LayerNorm(width, eps=eps)

    def prepare_memory(self, state):
        """Prepare immutable state memory and each attention module's own K/V."""
        dtype = torch.bfloat16 if self.bf16_compute else self.memory_projection.weight.dtype
        with torch.autocast(device_type=state.device.type, dtype=torch.bfloat16, enabled=self.bf16_compute):
            memory = self.memory_projection(self.memory_norm(state.to(dtype))).unsqueeze(0)
            projections = []
            for attention in (self.question_attention, *self.routing_attention):
                width = attention.embed_dim
                bias = attention.in_proj_bias[width:] if attention.in_proj_bias is not None else None
                key, value = F.linear(memory, attention.in_proj_weight[width:], bias).chunk(2, -1)
                shape = (1, memory.shape[1], attention.num_heads, width // attention.num_heads)
                projections.append((key.reshape(shape).transpose(1, 2), value.reshape(shape).transpose(1, 2)))
        return memory, tuple(projections)

    @staticmethod
    def _attend(attention, query, memory, projected=None):
        if projected is None:
            return attention(query, memory, memory, need_weights=False)[0]
        width = attention.embed_dim
        bias = attention.in_proj_bias[:width] if attention.in_proj_bias is not None else None
        query = F.linear(query, attention.in_proj_weight[:width], bias)
        batch, length = query.shape[:2]
        query = query.reshape(batch, length, attention.num_heads, width // attention.num_heads).transpose(
            1, 2
        )
        value = F.scaled_dot_product_attention(query, *projected, dropout_p=0.0)
        value = value.transpose(1, 2).reshape(batch, length, width)
        return F.linear(value, attention.out_proj.weight, attention.out_proj.bias)

    def forward(self, state, question, decision, options, *, prepared_memory=None):
        if state.ndim != 2 or question.ndim != 1 or decision.ndim != 1 or options.ndim != 2:
            raise ValueError(
                "Cross-attention readout expects state [T,D], question/decision [D], options [K,D]"
            )
        if not len(state) or not len(options):
            raise ValueError("Cross-attention readout requires state and candidates")
        dtype = torch.bfloat16 if self.bf16_compute else self.memory_projection.weight.dtype
        with torch.autocast(device_type=state.device.type, dtype=torch.bfloat16, enabled=self.bf16_compute):
            state = state.to(dtype)
            question = question.to(dtype)
            decision = decision.to(dtype)
            options = options.to(dtype)
            memory, projected = (
                (self.memory_projection(self.memory_norm(state)).unsqueeze(0), None)
                if prepared_memory is None
                else prepared_memory
            )
            question_query = self.question_seed(question)[None, None]
            question_evidence = self._attend(
                self.question_attention, question_query, memory, None if projected is None else projected[0]
            )[0, 0]
            context = self.context_norm(
                self.decision_projection(decision)
                + self.question_projection(question)
                + self.question_evidence_projection(question_evidence)
            )
            routed = self.option_norm(self.option_projection(options) + self.condition_projection(context))
            for layer_index, (attention, attn_norm, ffn, output_norm) in enumerate(
                zip(
                    self.routing_attention,
                    self.routing_attention_norm,
                    self.routing_ffn,
                    self.routing_output_norm,
                    strict=True,
                )
            ):
                evidence = self._attend(
                    attention, routed[None], memory, None if projected is None else projected[layer_index + 1]
                )[0]
                updated = attn_norm(routed + evidence)
                routed = output_norm(updated + ffn(updated))
            query = self.query_norm(self.query_mlp(context))
            keys = self.key_norm(self.key_mlp(routed))
        with torch.autocast(device_type=state.device.type, enabled=False):
            return (keys.float() @ query.float()) / math.sqrt(self.config.projection_size)


class FADepthRouter(nn.Module):
    def __init__(self, hidden_size, rms_norm_eps=1e-6):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))
        self.rms_norm_eps = rms_norm_eps

    def forward(self, carry, sources):
        """Return m * sum(alpha_j * v_j), with m equal to the source count.

        Centering the weights around 1/m uses the existing carry as the sum of
        sources. This equivalent form avoids regrouping low-precision residual
        additions at initialization: query=0 gives exactly r=carry. Neither
        keys nor values are detached; all depth paths retain their gradients.
        """
        with torch.autocast(device_type=carry.device.type, enabled=False):
            values = torch.stack(sources).float()
            keys = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.rms_norm_eps)
            logits = torch.einsum("d,nbtd->nbt", self.query.float(), keys)
            alpha = logits.softmax(dim=0)
            count = len(sources)
            centered = (alpha - 1.0 / count) * count
            correction = torch.einsum("nbt,nbtd->btd", centered, values)
            return (carry.float() + correction).to(carry.dtype)


def _attention(module, hidden, positional, previous, *, bidirectional=False):
    batch, length = hidden.shape[:2]
    shape = (batch, length, -1, module.head_dim)
    query, gate = module.q_proj(hidden).view(batch, length, -1, 2 * module.head_dim).chunk(2, dim=-1)
    query = module.q_norm(query).transpose(1, 2)
    key = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
    value = module.v_proj(hidden).view(shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *positional)
    prefix_length = previous[0].shape[2] if previous else 0
    if previous:
        key = torch.cat((previous[0], key), dim=2)
        value = torch.cat((previous[1], value), dim=2)
    # Keep compact GQA cache tensors while avoiding a quadratic math fallback
    # in SDPA implementations that do not fuse grouped-query attention.
    groups = query.shape[1] // key.shape[1]
    attention_query = query.to(value.dtype)
    attention_key = key.to(value.dtype)
    attention_value = value
    if groups > 1:
        attention_key = attention_key.repeat_interleave(groups, dim=1)
        attention_value = attention_value.repeat_interleave(groups, dim=1)
    padding = (-module.head_dim) % 8
    if padding:
        attention_query, attention_key, attention_value = (
            F.pad(tensor, (0, padding)) for tensor in (attention_query, attention_key, attention_value)
        )
    mask = None
    if prefix_length and not bidirectional:
        q_positions = torch.arange(length, device=hidden.device)[:, None]
        k_positions = torch.arange(prefix_length + length, device=hidden.device)[None, :]
        allowed = k_positions <= prefix_length + q_positions
        width = math.ceil((prefix_length + length) / 8) * 8
        mask = attention_query.new_zeros((length, width))[:, : prefix_length + length]
        mask.masked_fill_(~allowed, float("-inf"))
    with (
        torch.autocast(device_type=hidden.device.type, enabled=False),
        sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]),
    ):
        attended = F.scaled_dot_product_attention(
            attention_query,
            attention_key,
            attention_value,
            attn_mask=mask,
            is_causal=not prefix_length and not bidirectional,
            dropout_p=0.0,
            scale=module.scaling,
        )[..., : module.head_dim]
    attended = attended.transpose(1, 2).reshape(batch, length, -1)
    output = module.o_proj(attended * gate.reshape(batch, length, -1).sigmoid())
    return output, (key, value)


def _linear_context(module, hidden, previous, config):
    if previous is None:
        cache = Qwen3_5DynamicCache(config)
        cache.last_linear_layer = module.layer_idx
        output = module(hidden, cache_params=cache)
        return output, (cache.conv_states[module.layer_idx], cache.recurrent_states[module.layer_idx])

    batch, length = hidden.shape[:2]
    raw = module.in_proj_qkv(hidden).transpose(1, 2)
    window = torch.cat((previous[0], raw), dim=-1)
    # The cached window holds kernel_size raw values. Its oldest value precedes
    # the first suffix output's receptive field; exclude that value from conv.
    if module.causal_conv1d_fn is not None:
        # Match the upstream fused convolution+SiLU rounding in BF16. Applying
        # the same causal kernel to the cached window preserves continuation
        # gradients and avoids an extra low-precision convolution round trip.
        mixed = module.causal_conv1d_fn(
            x=window,
            weight=module.conv1d.weight.squeeze(1),
            bias=module.conv1d.bias,
            activation=module.activation,
        )[:, :, -length:]
    else:
        mixed = F.silu(
            F.conv1d(window[:, :, 1:], module.conv1d.weight, module.conv1d.bias, groups=raw.shape[1])
        )
    mixed = mixed.transpose(1, 2)
    query, key, value = mixed.split((module.key_dim, module.key_dim, module.value_dim), dim=-1)
    query = query.reshape(batch, length, -1, module.head_k_dim)
    key = key.reshape(batch, length, -1, module.head_k_dim)
    value = value.reshape(batch, length, -1, module.head_v_dim)
    repeat = module.num_v_heads // module.num_k_heads
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    beta = module.in_proj_b(hidden).sigmoid()
    decay = -module.A_log.float().exp() * F.softplus(module.in_proj_a(hidden).float() + module.dt_bias)
    mixed, recurrent = module.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=decay,
        beta=beta,
        initial_state=previous[1],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    gate = module.in_proj_z(hidden).reshape(-1, module.head_v_dim)
    mixed = module.norm(mixed.reshape(-1, module.head_v_dim), gate).reshape(batch, length, -1)
    return module.out_proj(mixed), (window[:, :, -module.conv_kernel_size :].clone(), recurrent)


def _continue_gdn(module, hidden, previous, config, *, terminal=False, scan=None):
    """Continue on the native 64-token chunk grid, preserving terminal scans."""
    scan = scan or _linear_context
    chunk_size = 64

    if torch.is_grad_enabled() or module.training:
        raise ValueError("Chunk-aware prefix state is inference-only")
    if previous is not None and not isinstance(previous, GDNState):
        raise ValueError("GS1 continuation requires chunk-aware native prefix state")
    if terminal:
        # The trained GS1 reference scans context first, then the terminal token
        # from that final state. Do not merge the terminal back into the chunk.
        if previous is None or hidden.shape[1] != 1:
            raise ValueError("A terminal segment requires a prefix and exactly one token")
        return scan(module, hidden, (previous.conv, previous.recurrent), config)
    boundary = (
        (previous.boundary_conv, previous.boundary_recurrent)
        if previous is not None and previous.boundary_conv is not None
        else None
    )
    tail = previous.tail if previous is not None else hidden[:, :0]
    combined = torch.cat((tail, hidden), dim=1) if tail.shape[1] else hidden
    result = scan(module, combined, boundary, config)
    output, final = result[:2]
    exported = result[2] if len(result) == 3 else None
    completed = combined.shape[1] // chunk_size * chunk_size
    if completed:
        if completed == combined.shape[1]:
            boundary = final
        elif exported is not None:
            boundary = exported
        else:
            boundary = scan(module, combined[:, :completed], boundary, config)[1]
    state = GDNState(
        *final,
        *(boundary if boundary is not None else (None, None)),
        combined[:, completed:].clone(),
        (previous.position if previous is not None else 0) + hidden.shape[1],
    )
    return output[:, tail.shape[1] :], state


class GS1PretrainedModel(Qwen3_5PreTrainedModel):
    """Shared initialization and strict native serialization for GS1 modules."""

    config_class = GS1Config
    base_model_prefix = ""
    _no_split_modules = ["GS1DecoderLayer", "Qwen3_5VisionBlock"]
    _tied_weights_keys = {}

    def get_output_embeddings(self):
        return None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        for module in self.modules():
            if isinstance(module, GS1TextModel):
                module.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        for module in self.modules():
            if isinstance(module, GS1TextModel):
                module.gradient_checkpointing = False

    def save_pretrained(self, save_directory, *, state_dict=None, **kwargs):
        from .util import export_state

        self.config.decision = self.decision_config.model_dump()
        if hasattr(self, "calibrator"):
            self.config.calibration = self.calibrator.config.model_dump()
        if hasattr(self, "decision_head"):
            self.config.head_compute_dtype = "bfloat16" if self.decision_head.bf16_compute else "float32"
        kwargs.pop("safe_serialization", None)
        return super().save_pretrained(
            save_directory,
            state_dict=export_state(self, state_dict),
            safe_serialization=True,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, source, *, device="cpu", dtype=None, **kwargs):
        from .util import load_native

        return load_native(source, device=device, dtype=dtype, model_class=cls, **kwargs)


class GS1TextModel(GS1PretrainedModel):
    """Native hybrid text decoder with GS1 depth routing and segmented attention."""

    def __init__(self, config):
        super().__init__(config)
        text = config.text_config
        self.decision_config = DecisionConfig.model_validate(config.decision)
        self.initialization_seed = config.initialization_seed
        self.full_layers = tuple(i for i, kind in enumerate(text.layer_types) if kind == "full_attention")
        if text.linear_num_value_heads % text.linear_num_key_heads:
            raise ValueError("GDN value heads must be a multiple of key heads")
        if text.num_attention_heads % text.num_key_value_heads:
            raise ValueError("Attention heads must be a multiple of key/value heads")
        for selected in (self.decision_config.attnres_layers, self.decision_config.bidirectional_layers):
            if selected is not None and not set(selected) <= set(self.full_layers):
                raise ValueError("Selected layers must be full-attention decoder indices")
        self.embed_tokens = nn.Embedding(text.vocab_size, text.hidden_size, text.pad_token_id)
        self.layers = nn.ModuleList(GS1DecoderLayer(text, i) for i in range(text.num_hidden_layers))
        self.norm = Qwen3_5RMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(text)
        self.gradient_checkpointing = False
        self.post_init()
        receivers = self.decision_config.attnres_layers or self.full_layers
        self.attnres_routers = nn.ModuleDict(
            {
                str(i): FADepthRouter(text.hidden_size, self.decision_config.attnres_eps)
                for i in receivers
                if self.decision_config.attnres
            }
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def is_bidirectional(self, index):
        selected = self.decision_config.bidirectional_layers
        return self.decision_config.bidirectional and (selected is None or index in selected)

    def _depth_input(self, index, carry, bank, partial):
        if str(index) not in self.attnres_routers:
            return carry
        sources = bank if partial is None else (*bank, partial)
        return self.attnres_routers[str(index)](carry, sources)

    @staticmethod
    def _advance_depth(bank, partial, mixed, ffn, full):
        increment = mixed + ffn
        partial = increment if partial is None else partial + increment
        return ((*bank, partial), None) if full else (bank, partial)

    def prepare_encode(self, hidden, position_ids, roles=None, *, terminal_query=False, attention_mask=None):
        """Validate immutable batch geometry before tensor execution or CUDA capture."""
        batch, length = hidden.shape[:2]
        if not batch or not length:
            raise ValueError("GS1 expects nonempty branch rows")
        valid = (
            torch.ones((batch, length), dtype=torch.bool, device=hidden.device)
            if attention_mask is None
            else attention_mask.bool()
        )
        if valid.shape != (batch, length) or not valid[:, -1].all() or (valid[:, :-1] & ~valid[:, 1:]).any():
            raise ValueError("GS1 batches require nonempty left-padded rows")
        if terminal_query and (valid.sum(-1) < 2).any():
            raise ValueError("A decision requires context before its terminal token")
        if roles is None:
            roles = torch.zeros((batch, length), dtype=torch.long, device=hidden.device)
        elif roles.shape == (length,):
            roles = roles[None].expand(batch, -1)
        if roles.shape != (batch, length):
            raise ValueError("Token roles must match the branch rows")
        keep = valid.unsqueeze(-1)
        valid_lengths = tuple(valid.sum(-1).tolist())
        hidden = hidden * keep
        masks = {}
        eye = torch.eye(length, dtype=torch.bool, device=hidden.device)[None]
        for bidirectional in {self.is_bidirectional(i) for i in self.full_layers}:
            if bidirectional:
                allowed = (roles[:, :, None] >= roles[:, None, :]) & (roles[:, None, :] != 3)
                allowed |= eye
            else:
                allowed = torch.ones((batch, length, length), dtype=torch.bool, device=hidden.device).tril_()
            allowed &= valid[:, None, :]
            allowed |= ~valid[:, :, None] & eye
            masks[bidirectional] = hidden.new_zeros((batch, 1, length, length)).masked_fill_(
                ~allowed[:, None], torch.finfo(hidden.dtype).min
            )
        positional = self.rotary_emb(hidden, position_ids)
        cohorts = tuple(
            (
                length,
                torch.tensor(
                    [i for i, value in enumerate(valid_lengths) if value == length], device=hidden.device
                ),
            )
            for length in sorted(set(valid_lengths))
        )
        return keep, valid_lengths, masks, positional, cohorts

    def encode(self, hidden, position_ids, roles=None, *, terminal_query=False, attention_mask=None):
        layout = self.prepare_encode(
            hidden, position_ids, roles, terminal_query=terminal_query, attention_mask=attention_mask
        )
        return self.encode_prepared(hidden, layout, terminal_query=terminal_query)

    def encode_prepared(self, hidden, layout, *, terminal_query=False, execution_policy=None):
        """The unchanged decoder equations over already validated, static geometry."""
        if execution_policy is not None:
            return execution_policy.encode(self, hidden, layout, terminal_query=terminal_query)
        keep, valid_lengths, masks, positional, cohorts = layout
        hidden = hidden * keep
        bank, partial = (hidden,), None

        def run(carry, layer, index, bank, partial):
            read = self._depth_input(index, carry, bank, partial)
            mask = masks.get(self.is_bidirectional(index))
            output, mixed, ffn = layer(carry, positional, mask, read, terminal_query, valid_lengths, cohorts)
            # Prefix PADs start from zero GDN state and remain zero after every
            # layer. Valid tokens retain the ordinary convolution/time scan.
            output, mixed, ffn = output * keep, mixed * keep, ffn * keep
            if self.attnres_routers:
                bank, partial = self._advance_depth(
                    bank, partial, mixed, ffn, layer.layer_type == "full_attention"
                )
            return output, bank, partial

        for index, layer in enumerate(self.layers):
            if self.training and self.gradient_checkpointing:
                hidden, bank, partial = checkpoint(
                    run, hidden, layer, index, bank, partial, use_reentrant=False
                )
            else:
                hidden, bank, partial = run(hidden, layer, index, bank, partial)
        return self.norm(hidden) * keep

    def segment_forward(
        self, input_ids, start_position=0, prefix=None, *, terminal_query=False, execution_policy=None
    ):
        """Inference-only segment continuation, including GDN convolution/state caches."""
        if torch.is_grad_enabled() or self.training:
            raise ValueError("Shared segment caches require eval mode without gradients")
        if execution_policy is not None:
            return execution_policy.segment(
                self, input_ids, start_position, prefix, terminal_query=terminal_query
            )
        hidden = self.embed_tokens(input_ids)
        positions = torch.arange(start_position, start_position + hidden.shape[1], device=hidden.device)[
            None
        ].expand(hidden.shape[0], -1)
        positional = self.rotary_emb(hidden, positions)
        bank, partial, states = (hidden,), None, []
        for index, layer in enumerate(self.layers):
            read = self._depth_input(index, hidden, bank, partial)
            normalized = layer.input_layernorm(read)
            previous = prefix[index] if prefix is not None else None
            if layer.layer_type == "full_attention":
                mixed, state = _attention(
                    layer.self_attn,
                    normalized,
                    positional,
                    previous,
                    bidirectional=self.is_bidirectional(index),
                )
            else:
                mixed, state = _continue_gdn(
                    layer.linear_attn, normalized, previous, self.config.text_config, terminal=terminal_query
                )
            hidden = hidden + mixed
            ffn = layer.mlp(layer.post_attention_layernorm(hidden))
            hidden = hidden + ffn
            if self.attnres_routers:
                bank, partial = self._advance_depth(
                    bank, partial, mixed, ffn, layer.layer_type == "full_attention"
                )
            states.append(state)
        return self.norm(hidden), tuple(states)

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        position_ids=None,
        roles=None,
        terminal_query=False,
        attention_mask=None,
        prepared_layout=None,
        execution_policy=None,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(hidden.shape[1], device=hidden.device)[None].expand(
                hidden.shape[0], -1
            )
        output = (
            self.encode(
                hidden, position_ids, roles, terminal_query=terminal_query, attention_mask=attention_mask
            )
            if prepared_layout is None
            else self.encode_prepared(
                hidden, prepared_layout, terminal_query=terminal_query, execution_policy=execution_policy
            )
        )
        return GS1TextOutput(last_hidden_state=output)


class GS1Model(GS1PretrainedModel):
    """Complete GS1: native text, upstream vision and one decision head."""

    def __init__(self, config):
        super().__init__(config)
        self.initialization_seed = config.initialization_seed
        self.language_model = GS1TextModel(config)
        self.decision_config = self.language_model.decision_config
        self.visual = Qwen3_5VisionModel._from_config(config.vision_config)
        self.post_init()
        self.decision_head = CrossAttentionDecisionHead(
            config.text_config.hidden_size, self.decision_config, self.initialization_seed
        )
        self.decision_head.bf16_compute = config.head_compute_dtype == "bfloat16"
        from .schema import Calibrator

        self.calibrator = Calibrator(config.calibration)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def _vision_positions(self, input_ids, types, image_grid, video_grid=None):
        positions = torch.zeros(3, *input_ids.shape, device=input_ids.device, dtype=torch.long)
        grids = {
            1: iter(image_grid) if image_grid is not None else None,
            2: iter(video_grid) if video_grid is not None else None,
        }
        merge = self.config.vision_config.spatial_merge_size
        for row in range(input_ids.shape[0]):
            begin, offset = 0, 0
            while begin < input_ids.shape[1]:
                kind = int(types[row, begin])
                end = begin + 1
                while end < input_ids.shape[1] and int(types[row, end]) == kind:
                    end += 1
                if kind == 0:
                    piece = torch.arange(end - begin, device=input_ids.device)[None].expand(3, -1) + offset
                    offset += end - begin
                else:
                    t, h, w = map(int, next(grids[kind]))
                    h, w = h // merge, w // merge
                    piece = torch.stack(
                        (
                            torch.full((t * h * w,), offset, device=input_ids.device),
                            torch.arange(offset, offset + h, device=input_ids.device).repeat_interleave(
                                w * t
                            ),
                            torch.arange(offset, offset + w, device=input_ids.device).repeat(h * t),
                        )
                    )
                    offset += max(h, w)
                if piece.shape[1] != end - begin:
                    raise ValueError("Vision grid and placeholder token count differ")
                positions[:, row, begin:end] = piece
                begin = end
        return positions

    def forward(
        self,
        input_ids,
        *,
        branch=None,
        state_length=None,
        branches=None,
        state_lengths=None,
        position_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        mm_token_type_ids=None,
        attention_mask=None,
        readout_callback=None,
        **kwargs,
    ):
        hidden = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            features = self.visual(
                pixel_values.to(self.visual.dtype), grid_thw=image_grid_thw, return_dict=True
            ).pooler_output.to(hidden.device, hidden.dtype)
            mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(hidden)
            if hidden[mask].numel() != features.numel():
                raise ValueError("Image features and placeholder tokens differ")
            hidden = hidden.masked_scatter(mask, features)
            if position_ids is None:
                if mm_token_type_ids is None:
                    raise ValueError("Visual branches require modality token types")
                position_ids = self._vision_positions(input_ids, mm_token_type_ids, image_grid_thw)
        if position_ids is None:
            position_ids = torch.arange(hidden.shape[1], device=hidden.device)[None].expand(
                hidden.shape[0], -1
            )
        if branch is not None and branches is not None:
            raise ValueError("Specify one branch or batched branch metadata")
        rows = branches if branches is not None else [branch] if branch is not None else None
        lengths = state_lengths if branches is not None else [state_length] if branch is not None else None
        roles = None
        starts = []
        if rows is not None:
            if len(rows) != input_ids.shape[0] or lengths is None or len(lengths) != len(rows):
                raise ValueError("Compiled branch metadata differs from batch rows")
            roles = torch.full_like(input_ids, -1)
            for row, (item, offset) in enumerate(zip(rows, lengths, strict=True)):
                total = offset + len(item.suffix)
                begin = input_ids.shape[1] - total
                if begin < 0 or (attention_mask is not None and int(attention_mask[row].sum()) != total):
                    raise ValueError("Compiled branch length differs from model input")
                if branch is not None and begin:
                    raise ValueError("Single compiled branches must be unpadded")
                starts.append(begin)
                roles[row, begin : begin + offset] = 0
                roles[row, begin + offset :] = torch.tensor(
                    [
                        1 if r == "question_context" else 2 if r == "candidate_context" else 3
                        for r in item.token_roles
                    ],
                    device=hidden.device,
                )
        hidden = self.language_model(
            inputs_embeds=hidden,
            position_ids=position_ids,
            roles=roles,
            terminal_query=rows is not None,
            attention_mask=attention_mask,
        ).last_hidden_state
        logits = None
        if rows is not None:
            values = [
                (readout_callback or self.readout)(hidden[row, begin:], item, offset)
                for row, (begin, item, offset) in enumerate(zip(starts, rows, lengths, strict=True))
            ]
            width = max(value.numel() for value in values)
            logits = (
                values[0]
                if branch is not None
                else torch.stack(
                    [F.pad(value, (0, width - value.numel()), value=-torch.inf) for value in values]
                )
            )
        return GS1Output(logits=logits, last_hidden_state=hidden)

    def readout(self, hidden, branch, offset, *, prepared_memory=None, state_hidden=None):
        def pool(start, span, end):
            content = hidden[offset + span[0] : offset + span[1]].mean(0)
            closing = hidden[offset + end]
            return (
                (hidden[offset + start] + content + closing) / 3
                if self.decision_config.bidirectional
                else (content + closing) / 2
            )

        question = pool(branch.question_start_position, branch.question_span, branch.question_end_position)
        options = torch.stack(
            [
                pool(start, span, end)
                for start, span, end in zip(
                    branch.option_starts, branch.option_spans, branch.option_ends, strict=True
                )
            ]
        )
        return self.decision_head(
            hidden[branch.state_evidence_start : offset] if state_hidden is None else state_hidden,
            question,
            hidden[offset + branch.decision_position],
            options,
            prepared_memory=prepared_memory,
        )

    def probabilities(self, logits, candidate_mask=None, *, calibration_active=True):
        from .schema import probabilities

        return probabilities(logits, candidate_mask, self.calibrator if calibration_active else None)

    def decision_compiler(self, tokenizer=None, **overrides):
        """Build the native prompt compiler from standard pretrained assets."""
        from .compiler import Compiler

        if tokenizer is None:
            from transformers import AutoTokenizer

            if not hasattr(self, "_decision_tokenizer"):
                self._decision_tokenizer = AutoTokenizer.from_pretrained(
                    self._processor_source, local_files_only=True
                )
            tokenizer = self._decision_tokenizer
        source = getattr(self, "_processor_source", None)
        limits = {
            key: self.config.inference_limits[key]
            for key in ("max_context", "max_questions", "max_request_tokens", "max_branches")
        }
        for key, value in overrides.items():
            limits[key] = min(limits[key], value)
        compiler = Compiler(tokenizer, decision=self.decision_config, processor=source, **limits)
        if compiler.controls != getattr(self.config, "structural_token_ids", {}):
            raise ValueError("Structural token identity mismatch")
        return compiler

    @torch.inference_mode()
    def decide(self, state, questions, *, tokenizer=None, precision="bf16"):
        """Return typed decisions using the standalone eager reference path.

        This method needs only the files in this model's Hub repository. The
        separately installed gs1 runtime provides qualified CUDA graph execution.
        """
        from .compiler import branch_logits
        from .schema import DecisionRequest, DecisionResponse, Usage, make_answer

        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if self.training:
            raise ValueError("Call model.eval() before making decisions")
        compiler = self.decision_compiler(tokenizer)
        tokenizer = compiler.tokenizer
        name = getattr(self.config, "_name_or_path", "") or "GS1"
        request = DecisionRequest(state=state, questions=questions, model=name)
        compiled = compiler.compile(request)
        device = self.get_input_embeddings().weight.device
        answers = {}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            for branch in compiled.branches:
                logits = branch_logits(
                    self, [(compiled.state, branch)], pad_token_id=tokenizer.pad_token_id or 0
                )[0]
                values = self.probabilities(logits).p_final.tolist()
                answers[branch.question_id] = make_answer(request.questions[branch.question_id], values)
        response = DecisionResponse(
            model=name, answers=answers, usage=Usage(input_tokens=compiled.input_tokens, output_tokens=0)
        )
        return response.model_dump(mode="json")


# Native discovery resolves the complete text/vision/decision model.
AutoConfig.register("gs1", GS1Config, exist_ok=True)
AutoModel.register(GS1Config, GS1Model, exist_ok=True)

GS1Config.register_for_auto_class("AutoConfig")
GS1Model.register_for_auto_class("AutoModel")
