import base64
import io
import json

import pytest
import torch
from jinja2 import TemplateError
from PIL import Image
from transformers import (
    AutoTokenizer,
    Qwen2VLImageProcessorFast,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

from gs1.model.compiler import CHAT_TEMPLATE, Compiler, structural_tokens
from gs1.model.schema import DecisionRequest
from tests.fixtures import tokenizer


@pytest.fixture
def tokens():
    value = tokenizer()
    value.add_special_tokens({"additional_special_tokens": structural_tokens()})
    return value


@pytest.mark.parametrize(
    "state,question,options,body",
    [
        (
            "Battery 18.",
            "Choose.",
            ["a: Cost 2", "b: Cost 12"],
            "<state>\nBattery 18.\n</state>\n<question>\nChoose.\n</question>"
            "\n<option>\na: Cost 2\n</option>\n<option>\nb: Cost 12\n</option>",
        ),
        (
            '  客户 <&> "quoted"\n',
            {"z": [True, None], "a": "<&>"},
            [{"z": 0.25, "a": "<&>"}, ["低", "高"]],
            '<state>\n  客户 &lt;&amp;&gt; "quoted"\n\n</state>'
            '\n<question>\n{"a":"&lt;&amp;&gt;","z":[true,null]}\n</question>'
            '\n<option>\n{"a":"&lt;&amp;&gt;","z":0.25}\n</option>'
            '\n<option>\n["低","高"]\n</option>',
        ),
        (
            [],
            "",
            [""],
            "<state>\n[]\n</state>\n<question>\n\n</question>\n<option>\n\n</option>",
        ),
    ],
)
def test_native_template_pins_trained_prompt(tokens, state, question, options, body):
    rendered = tokens.apply_chat_template(
        [{"role": "user", "content": state}],
        question=question,
        options=options,
        add_generation_prompt=True,
        tokenize=False,
    )
    assert rendered == (
        "<|im_start|>user\n" + body + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|decide|>"
    )


def test_direct_rendering_matches_compiler_for_all_primitives(tokens, decision_request):
    request = DecisionRequest(model="GS1", **decision_request)
    compiled = Compiler(tokens).compile(request)
    options = {
        "choice": ["a: Cost 2", "b: Cost 12"],
        "noul": [
            "The proposition is false under the question's stated semantics.",
            "The proposition is true under the question's stated semantics.",
        ],
        "score": ["Low", "Medium", "High"],
    }
    for branch in compiled.branches:
        ids = tokens.apply_chat_template(
            [{"role": "user", "content": request.state}],
            question=request.questions[branch.question_id].instructions,
            options=options[branch.question_id],
            add_generation_prompt=True,
            return_dict=False,
        )
        assert ids == [*compiled.state, *branch.suffix]
        assert ids[-1] == tokens.convert_tokens_to_ids("<|decide|>")
        assert ids.count(ids[-1]) == 1
        assert branch.token_roles[-1] == "decision_head_query"
        assert branch.decision_position == len(branch.suffix) - 1


def test_legacy_tokenizer_is_upgraded_and_exports_native_template(tokens, tmp_path, decision_request):
    tokens.chat_template = "{{ messages[0]['content'] }}"
    compiled = Compiler(tokens).compile(DecisionRequest(model="GS1", **decision_request))
    assert len(compiled.branches) == 3
    assert tokens.chat_template == CHAT_TEMPLATE
    tokens.save_pretrained(tmp_path)
    assert (tmp_path / "chat_template.jinja").read_text() == CHAT_TEMPLATE
    restored = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert restored.chat_template == CHAT_TEMPLATE


def test_model_export_contains_native_template_and_standalone_code(checkpoint):
    assert (checkpoint / "chat_template.jinja").read_text() == CHAT_TEMPLATE
    assert (checkpoint / "compiler.py").is_file()
    assert not (checkpoint / "chat_template.py").exists()
    assert json.loads((checkpoint / "config.json").read_text())["auto_map"]["AutoModel"]


def test_visual_compilation_keeps_terminal_payload_aligned(tokens):
    image = io.BytesIO()
    Image.new("RGB", (32, 32), (30, 70, 110)).save(image, format="PNG")
    encoded = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessorFast(
            size={"shortest_edge": 1024, "longest_edge": 1024}, patch_size=16, merge_size=2
        ),
        tokenizer=tokens,
        video_processor=Qwen3VLVideoProcessor(),
        chat_template="legacy processor template",
    )
    request = DecisionRequest(
        model="GS1",
        state=[{"type": "text", "text": "Look at <image>&."}, {"type": "image", "image": encoded}],
        questions={"q": {"type": "noul", "instructions": "Is the image blue?"}},
    )
    compiled = Compiler(tokens, processor=processor).compile(request)
    branch = compiled.branches[0]
    ids = torch.tensor([[*compiled.state, *branch.suffix]])
    assert torch.equal(branch.visual_inputs["input_ids"], ids)
    assert branch.visual_inputs["attention_mask"].shape == ids.shape
    assert branch.visual_inputs["mm_token_type_ids"].shape == ids.shape
    assert branch.visual_inputs["attention_mask"][0, -1] == 1
    assert branch.visual_inputs["mm_token_type_ids"][0, -1] == 0


@pytest.mark.parametrize("options", [[], "one option", {"a": "description"}])
def test_native_template_rejects_invalid_option_shape(tokens, options):
    with pytest.raises(TemplateError, match="nonempty sequence"):
        tokens.apply_chat_template(
            [{"role": "user", "content": "State"}], question="Question", options=options, tokenize=False
        )


def test_invalid_visual_state_remains_a_request_error(tokens):
    request = DecisionRequest(
        model="GS1",
        state=[{"type": "image", "image": "https://invalid.example/image.png"}],
        questions={"q": {"type": "noul", "instructions": "Question"}},
    )
    # Reuse the tokenizer's renderer without fetching or processing any image.
    compiler = Compiler(tokens, processor=tokens)
    with pytest.raises(ValueError, match="inline base64"):
        compiler.compile(request)
