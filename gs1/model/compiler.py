"""Compile typed decisions into XML prompts and execute native decision branches."""

from dataclasses import dataclass, field
from html import escape
from pathlib import Path

import torch

from .config import DecisionConfig
from .schema import DecisionRequest, candidate_ids, noul_descriptions
from .util import canonical

XML_STATE, XML_STATE_END = "<state>", "</state>"
XML_QUESTION, XML_QUESTION_END = "<question>", "</question>"
XML_OPTION, XML_OPTION_END = "<option>", "</option>"
XML_TAGS = (XML_STATE, XML_STATE_END, XML_QUESTION, XML_QUESTION_END, XML_OPTION, XML_OPTION_END)
DECISION_TOKEN = "<|decide|>"
NO_THINKING_PREFIX = "<think>\n\n</think>\n\n"
COMPILER = "xml_decision"
POSITIONS = "branch_local"


def structural_tokens():
    return [*XML_TAGS, DECISION_TOKEN]


def no_thinking_token_ids(tokenizer):
    ids = tuple(tokenizer.encode(NO_THINKING_PREFIX, add_special_tokens=False))
    if (
        not ids
        or tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        != NO_THINKING_PREFIX
    ):
        raise ValueError("Tokenizer cannot encode the no-thinking prefix exactly")
    return ids


@dataclass(frozen=True)
class Branch:
    question_id: str
    question_type: str
    branch_id: int
    suffix: tuple[int, ...]
    token_roles: tuple[str, ...]
    logical_positions: tuple[int, ...]
    option_ends: tuple[int, ...]
    candidates: tuple[str, ...]
    decision_position: int
    option_spans: tuple[tuple[int, int], ...]
    option_starts: tuple[int, ...]
    question_end_position: int
    question_start_position: int
    question_span: tuple[int, int]
    state_evidence_start: int = 0
    visual_inputs: dict | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class CompiledRequest:
    state: tuple[int, ...]
    branches: tuple[Branch, ...]
    question_order: tuple[str, ...]
    candidate_order: dict[str, tuple[str, ...]]

    @property
    def input_tokens(self):
        return len(self.state) + sum(len(b.suffix) for b in self.branches)

    @property
    def longest_path(self):
        return max(len(self.state) + len(b.suffix) for b in self.branches)

    @property
    def query_tokens(self):
        return len(self.branches)

    @property
    def candidate_valid_mask(self):
        width = max(map(len, self.candidate_order.values()))
        return tuple(
            tuple(k < len(self.candidate_order[q]) for k in range(width)) for q in self.question_order
        )


class Compiler:
    def __init__(
        self,
        tokenizer,
        max_context=16384,
        max_questions=512,
        max_request_tokens=131072,
        decision=None,
        max_branches=5120,
        processor=None,
    ):
        self.tokenizer = tokenizer
        self.decision = DecisionConfig.model_validate(decision or {})
        self.max_context = max_context
        self.max_questions = max_questions
        self.max_request_tokens = max_request_tokens
        self.max_branches = max_branches
        self.processor = processor
        tokens = structural_tokens()
        self.controls = dict(zip(tokens, tokenizer.convert_tokens_to_ids(tokens), strict=True))
        if len(set(self.controls.values())) != len(tokens) or any(
            type(i) is not int or i < 0 or i == tokenizer.unk_token_id or i not in tokenizer.all_special_ids
            for i in self.controls.values()
        ):
            raise ValueError("Tokenizer requires the seven distinct GS1 special tokens")
        self.xml_tags = {tag: (self.controls[tag],) for tag in XML_TAGS}
        self.xml_newline = tuple(tokenizer.encode("\n", add_special_tokens=False))
        self.no_thinking_ids = no_thinking_token_ids(tokenizer)

    @staticmethod
    def _xml_text(value):
        return escape(value if isinstance(value, str) else canonical(value), quote=False)

    @staticmethod
    def _visual_state(state):
        return isinstance(state, list) and any(
            isinstance(item, dict) and item.get("type") == "image" for item in state
        )

    @staticmethod
    def choice_option_text(name, description):
        value = name if description is None else description
        return f"{name}: {value if isinstance(value, str) else canonical(value)}"

    def _xml_positions(self, token_ids, option_count):
        found = {
            tag: [i for i, value in enumerate(token_ids) if value == self.controls[tag]] for tag in XML_TAGS
        }
        if any(len(found[tag]) != 1 for tag in XML_TAGS[:4]) or any(
            len(found[tag]) != option_count for tag in XML_TAGS[4:]
        ):
            raise ValueError("Tokenizer changed the XML prompt tag positions")
        ordered = [found[tag][0] for tag in XML_TAGS[:4]]
        ordered.extend(i for pair in zip(found[XML_OPTION], found[XML_OPTION_END], strict=True) for i in pair)
        if ordered != sorted(set(ordered)):
            raise ValueError("XML prompt tags are out of order")
        prefix = found[XML_QUESTION][0]

        def span(opening, closing):
            begin, end = opening + 1, closing
            width = len(self.xml_newline)
            if width and tuple(token_ids[begin : begin + width]) == self.xml_newline:
                begin += width
            if width and tuple(token_ids[end - width : end]) == self.xml_newline:
                end -= width
            if begin >= end:
                begin, end = closing, closing + 1
            return begin - prefix, end - prefix

        return dict(
            prefix_end=prefix,
            state_evidence_start=found[XML_STATE][0],
            question_start_position=0,
            question_end_position=found[XML_QUESTION_END][0] - prefix,
            question_span=span(found[XML_QUESTION][0], found[XML_QUESTION_END][0]),
            option_starts=tuple(i - prefix for i in found[XML_OPTION]),
            option_ends=tuple(i - prefix for i in found[XML_OPTION_END]),
            option_spans=tuple(
                span(a, b) for a, b in zip(found[XML_OPTION], found[XML_OPTION_END], strict=True)
            ),
            first_option=found[XML_OPTION][0] - prefix,
        )

    def _processor(self):
        if self.processor is None:
            raise ValueError("Visual input requires the GS1 processor")
        if isinstance(self.processor, (str, Path)):
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                self.processor, local_files_only=True, trust_remote_code=True
            )
        # Fresh Qwen processor assets precede the seven GS1 structural tokens.
        self.processor.tokenizer = self.tokenizer
        return self.processor

    def _contents(self, state):
        if not self._visual_state(state):
            return f"{XML_STATE}\n{self._xml_text(state)}\n{XML_STATE_END}"
        contents = [{"type": "text", "text": f"{XML_STATE}\n"}]
        for item in state:
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                contents.append({"type": "text", "text": self._xml_text(item["text"])})
            elif (
                set(item) == {"type", "image"}
                and isinstance(item.get("image"), str)
                and item["image"].startswith("data:image/")
                and ";base64," in item["image"]
            ):
                contents.append(item)
            else:
                raise ValueError("Vision state requires text and inline base64 images")
        return [*contents, {"type": "text", "text": f"\n{XML_STATE_END}"}]

    def compile(self, request: DecisionRequest):
        if len(request.questions) > self.max_questions:
            raise ValueError(f"questions exceeds {self.max_questions}")
        visual = self._visual_state(request.state)
        contents = self._contents(request.state)
        template = self._processor() if visual else self.tokenizer
        branches, candidates, state = [], {}, None
        for qid, question in request.questions.items():
            ids = tuple(candidate_ids(question))
            candidates[qid] = ids
            if question.type == "choice":
                options = [self.choice_option_text(k, d) for k, d in question.criteria.items()]
            elif question.type == "noul":
                options = list(noul_descriptions(question))
            else:
                options = list(question.criteria)
            prompt = (
                f"\n{XML_QUESTION}\n{self._xml_text(question.instructions)}\n{XML_QUESTION_END}"
                + "".join(f"\n{XML_OPTION}\n{self._xml_text(o)}\n{XML_OPTION_END}" for o in options)
            )
            content = [*contents, {"type": "text", "text": prompt}] if visual else contents + prompt
            rendered = template.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=False,
            )
            payload = None
            if visual:
                encoded = template(
                    text=rendered,
                    images=[[i["image"] for i in request.state if i["type"] == "image"]],
                    return_tensors="pt",
                )
                token_ids = encoded["input_ids"][0].tolist()
            else:
                token_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
            if tuple(token_ids[-len(self.no_thinking_ids) :]) != self.no_thinking_ids:
                raise ValueError("Chat template changed the no-thinking assistant prefix")
            token_ids.append(self.controls[DECISION_TOKEN])
            markers = self._xml_positions(token_ids, len(ids))
            prefix = markers.pop("prefix_end")
            first_option = markers.pop("first_option")
            current_state = tuple(token_ids[:prefix])
            if state is None:
                state = current_state
            elif state != current_state:
                raise ValueError("State prefix differs across question branches")
            suffix = tuple(token_ids[prefix:])
            if visual:
                payload = {
                    k: v
                    for k, v in encoded.items()
                    if k
                    in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"}
                }
                payload["input_ids"] = torch.tensor([token_ids], dtype=torch.long)
                for key, fill in (("attention_mask", 1), ("mm_token_type_ids", 0)):
                    if key in payload:
                        value = payload[key]
                        payload[key] = torch.cat((value, torch.full((1, 1), fill, dtype=value.dtype)), dim=-1)
            branches.append(
                Branch(
                    qid,
                    question.type,
                    len(branches),
                    suffix,
                    tuple(
                        ["question_context"] * first_option
                        + ["candidate_context"] * (len(suffix) - first_option - 1)
                        + ["decision_head_query"]
                    ),
                    tuple(range(prefix, len(token_ids))),
                    markers.pop("option_ends"),
                    ids,
                    len(suffix) - 1,
                    **markers,
                    visual_inputs=payload,
                )
            )
        result = CompiledRequest(state, tuple(branches), tuple(request.questions), candidates)
        if result.longest_path > self.max_context:
            raise ValueError(f"Branch exceeds {self.max_context} tokens; no truncation")
        if result.input_tokens > self.max_request_tokens:
            raise ValueError(f"Request exceeds {self.max_request_tokens} tokens; no truncation")
        if len(branches) > self.max_branches:
            raise ValueError(f"Physical branches exceed {self.max_branches}")
        return result


@dataclass
class PreparedBatch:
    ids: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor
    roles: torch.Tensor
    branches: tuple
    offsets: tuple
    starts: tuple
    layout: tuple

    def forward(self, model, *, readout=None, execution_policy=None):
        hidden = model.language_model(
            inputs_embeds=model.get_input_embeddings()(self.ids),
            position_ids=self.positions,
            terminal_query=True,
            prepared_layout=self.layout,
            execution_policy=execution_policy,
        ).last_hidden_state
        values = [
            (readout or model.readout)(hidden[row, begin:], branch, offset)
            for row, (begin, branch, offset) in enumerate(
                zip(self.starts, self.branches, self.offsets, strict=True)
            )
        ]
        return values


def prepare_batch(model, rows, pad_token_id=0):
    """Collate text once; no scalar device reads remain in PreparedBatch.forward."""
    if not rows or any(branch.visual_inputs is not None for _, branch in rows):
        raise ValueError("Prepared GS1 batches require nonempty text branches")
    device = model.get_input_embeddings().weight.device
    lengths = [len(state) + len(branch.suffix) for state, branch in rows]
    width = max(lengths)
    ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long, device=device)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    positions = torch.zeros_like(ids)
    roles = torch.full_like(ids, -1)
    starts = []
    for row, ((state, branch), length) in enumerate(zip(rows, lengths, strict=True)):
        begin, offset = width - length, len(state)
        starts.append(begin)
        ids[row, begin:] = torch.tensor(state + branch.suffix, device=device)
        valid[row, begin:] = True
        positions[row, begin:] = torch.arange(length, device=device)
        roles[row, begin : begin + offset] = 0
        roles[row, begin + offset :] = torch.tensor(
            [
                1 if role == "question_context" else 2 if role == "candidate_context" else 3
                for role in branch.token_roles
            ],
            device=device,
        )
    hidden = model.get_input_embeddings()(ids)
    layout = model.language_model.prepare_encode(
        hidden, positions, roles, terminal_query=True, attention_mask=valid
    )
    return PreparedBatch(
        ids,
        positions,
        valid,
        roles,
        tuple(b for _, b in rows),
        tuple(len(s) for s, _ in rows),
        tuple(starts),
        layout,
    )


def branch_logits(model, rows, *, pad_token_id=0, readout=None, execution_policy=None):
    """Execute text branches in independent, genuinely batched left-padded rows."""
    if not rows:
        return []
    device = model.get_input_embeddings().weight.device
    outputs = [None] * len(rows)
    text = [
        (index, state, branch) for index, (state, branch) in enumerate(rows) if branch.visual_inputs is None
    ]
    if text:
        prepared = prepare_batch(model, [(state, branch) for _, state, branch in text], pad_token_id)
        logits = prepared.forward(model, readout=readout, execution_policy=execution_policy)
        for row, (index, _, branch) in enumerate(text):
            outputs[index] = logits[row]
    for index, (state, branch) in enumerate(rows):
        if branch.visual_inputs is not None:
            payload = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in branch.visual_inputs.items()
            }
            outputs[index] = model(
                **payload, branch=branch, state_length=len(state), readout_callback=readout
            ).logits
    return outputs


def assemble_logits(compiled, outputs):
    if len(outputs) != len(compiled.branches):
        raise ValueError("Missing physical branch results")
    result = {}
    for branch, output in zip(compiled.branches, outputs, strict=True):
        if output.shape != (len(branch.candidates),) or branch.question_id in result:
            raise ValueError("Malformed or duplicate single-decision result")
        result[branch.question_id] = output
    if set(result) != set(compiled.question_order):
        raise ValueError("Missing logical question results")
    return {key: result[key] for key in compiled.question_order}
