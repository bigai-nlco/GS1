"""Small dense profiles preserving Qwen head-width and recurrent-state relationships."""

from unittest.mock import patch

import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from gs1.model.compiler import CHAT_TEMPLATE
from gs1.model.modular_gs1 import GS1Config

PROFILES = {
    "2B": (32, 4, 1, 2),
    "4B": (40, 8, 2, 4),
    "9B": (64, 8, 2, 4),
    "27B": (80, 12, 2, 6),
}


def cpu_kernels():
    return patch.multiple(
        qwen,
        FusedRMSNormGated=None,
        causal_conv1d_fn=None,
        causal_conv1d_update=None,
        chunk_gated_delta_rule=None,
        fused_recurrent_gated_delta_rule=None,
    )


def config(size="2B", **decision):
    width, heads, kv, values = PROFILES[size]
    return GS1Config(
        text_config=dict(
            vocab_size=512,
            hidden_size=width,
            intermediate_size=width * 2,
            num_hidden_layers=4,
            layer_types=["linear_attention", "full_attention"] * 2,
            num_attention_heads=heads,
            num_key_value_heads=kv,
            head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=values,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            rope_parameters=dict(
                rope_type="default",
                rope_theta=10000.0,
                partial_rotary_factor=0.5,
                mrope_section=[1, 1, 0],
                mrope_interleaved=True,
            ),
        ),
        vision_config=dict(
            depth=1 if size in {"2B", "4B"} else 2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=width,
            patch_size=2,
            spatial_merge_size=2,
            num_position_embeddings=256,
        ),
        decision=dict(cross_attention_heads=4, projection_size=16, **decision),
    )


def tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {"[UNK]": 0, "[PAD]": 1, **{c: i + 2 for i, c in enumerate(alphabet)}}
    raw = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]")
    fast.add_tokens(
        [
            AddedToken(t)
            for t in (
                "</",
                "state",
                "question",
                "option",
                "decision",
                "answer",
                "<think>",
                "</think>",
                "<tool_call>",
                "</tool_call>",
                "<tool_response>",
                "</tool_response>",
            )
        ]
    )
    fast.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|im_start|>",
                "<|im_end|>",
                "<|vision_start|>",
                "<|vision_end|>",
                "<|image_pad|>",
                "<|video_pad|>",
            ]
        }
    )
    fast.chat_template = CHAT_TEMPLATE
    return fast
