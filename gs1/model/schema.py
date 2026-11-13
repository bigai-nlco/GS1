"""Typed decisions, FP32 probabilities, calibration and confidence."""

import json
import math
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from torch import nn
from typesafe_sdk import Choice as Choice
from typesafe_sdk import Noul as Noul
from typesafe_sdk import Question as Question
from typesafe_sdk import Score as Score

from .config import CalibrationConfig

Description = str | dict[str, JsonValue] | list[JsonValue]
CONFIDENCE_VERSION = "typesafe_v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DecisionRequest(StrictModel):
    state: Description
    model: str = Field(min_length=1)
    questions: dict[str, Question] = Field(min_length=1)

    @field_validator("questions", mode="before")
    @classmethod
    def sdk_primitives(cls, value):
        if not isinstance(value, dict):
            raise ValueError("questions must be an object")
        kinds = {"choice": Choice, "noul": Noul, "score": Score}
        parsed = {}
        for qid, question in value.items():
            if isinstance(question, dict):
                kind = question.get("type")
                if kind not in kinds:
                    raise ValueError("Question type must be choice, noul, or score")
                question = kinds[kind].model_validate(question, strict=True)
            parsed[qid] = question
        return parsed

    @model_validator(mode="after")
    def server_contract(self):
        # Keep SDK-native primitives, but enforce the narrower public serving contract.
        # SDK 0.7.0 permits omitted instructions and singleton Score questions.
        for question in self.questions.values():
            if not isinstance(question.instructions, (str, dict, list)):
                raise ValueError("Every question requires text, object, or array instructions")
            if question.type == "choice":
                if not 1 <= len(question.criteria) <= 256:
                    raise ValueError("Choice requires between 1 and 256 candidates")
                descriptions = [v for v in question.criteria.values() if v is not None]
            elif question.type == "score":
                if not 2 <= len(question.criteria) <= 256:
                    raise ValueError("Score requires between 2 and 256 levels")
                descriptions = question.criteria
            else:
                descriptions = list(noul_descriptions(question))
            if any(not isinstance(v, (str, dict, list)) for v in descriptions):
                raise ValueError("Criteria descriptions must be text, objects, or arrays")
        # Reject NaN/Infinity at every depth, including structured descriptions.
        json.dumps(self.model_dump(mode="json"), allow_nan=False)
        return self


def noul_descriptions(question: Question) -> tuple[Description, Description]:
    criteria = question.criteria or {}
    return (
        criteria["false"]
        if criteria.get("false") is not None
        else "The proposition is false under the question's stated semantics.",
        criteria["true"]
        if criteria.get("true") is not None
        else "The proposition is true under the question's stated semantics.",
    )


class Usage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ChoiceAnswer(StrictModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class NoulAnswer(StrictModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0, le=1, allow_inf_nan=False)


class ScoreAnswer(StrictModel):
    type: Literal["score"] = "score"
    score: float = Field(ge=0, allow_inf_nan=False)
    legend: dict[str, Description]
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


Answer = Annotated[ChoiceAnswer | NoulAnswer | ScoreAnswer, Field(discriminator="type")]


class DecisionResponse(StrictModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage

    @model_validator(mode="after")
    def numerical_validity(self):
        for answer in self.answers.values():
            if isinstance(answer, NoulAnswer):
                continue
            p = answer.probabilities
            if not p or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()):
                raise ValueError("Probabilities must be finite and within [0, 1]")
            if abs(sum(p.values()) - 1) > 1e-6:
                raise ValueError("Probabilities must sum to one")
            if isinstance(answer, ChoiceAnswer):
                if answer.choice not in p or p[answer.choice] != max(p.values()):
                    raise ValueError("Choice must be an argmax")
            else:
                if set(p) != set(answer.legend) or set(p) != {str(k) for k in range(len(p))}:
                    raise ValueError("Score indices and legend must align")
                if abs(answer.score - sum(int(k) * v for k, v in p.items())) > 1e-6:
                    raise ValueError("Score must equal its probability-weighted level")
        return self


def candidate_ids(question: Question) -> list[str]:
    if question.type == "choice":
        return list(question.criteria)
    if question.type == "noul":
        return ["false", "true"]
    return [str(k) for k in range(len(question.criteria))]


def validate_response(request: DecisionRequest, response: DecisionResponse):
    if response.answers.keys() != request.questions.keys():
        raise ValueError("Response question identities do not match the request")
    for qid, question in request.questions.items():
        answer = response.answers[qid]
        if answer.type != question.type:
            raise ValueError(f"Primitive mismatch for {qid}")
        if answer.type != "noul" and set(answer.probabilities) != set(candidate_ids(question)):
            raise ValueError(f"Candidate mismatch for {qid}")
        if answer.type == "score" and answer.legend != dict(zip(candidate_ids(question), question.criteria)):
            raise ValueError(f"Score legend mismatch for {qid}")


def _bounded_confidence(value: float) -> float:
    if not math.isfinite(value) or value < -1e-6 or value > 1 + 1e-6:
        raise FloatingPointError("Invalid confidence")
    return max(0.0, min(1.0, value))


# TypeSafe formulas: https://docs.typesafe.ai/confidence
def noul_confidence(probability: float) -> float:
    """Optional TypeSafe confidence measure; Noul answers expose only p(true)."""
    return _bounded_confidence(abs(2 * probability - 1))


def choice_confidence(probabilities: list[float]) -> float:
    """TypeSafe's top probability above an even split, with singleton support."""
    n = len(probabilities)
    if n == 1:
        return 1.0
    return _bounded_confidence((max(probabilities) - 1 / n) / (1 - 1 / n))


def score_confidence(probabilities: list[float]) -> float:
    """TypeSafe's spread around the first most likely level, in rubric order."""
    n = len(probabilities)
    m = probabilities.index(max(probabilities))
    spread = sum(p * abs(i - m) for i, p in enumerate(probabilities))
    even_spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return _bounded_confidence(max(0.0, 1 - spread / even_spread))


def make_answer(question: Question, probabilities: list[float]) -> Any:
    mapping = dict(zip(candidate_ids(question), probabilities, strict=True))
    if question.type == "noul":
        return NoulAnswer(noul=probabilities[1])
    if question.type == "choice":
        return ChoiceAnswer(
            choice=max(mapping, key=mapping.get),
            probabilities=mapping,
            confidence=choice_confidence(probabilities),
        )
    return ScoreAnswer(
        score=sum(k * p for k, p in enumerate(probabilities)),
        legend=dict(zip(candidate_ids(question), question.criteria)),
        probabilities=mapping,
        confidence=score_confidence(probabilities),
    )


def aligned_probabilities(question: Question, answer: Answer) -> list[float]:
    if answer.type == "noul":
        return [1 - answer.noul, answer.noul]
    return [answer.probabilities[k] for k in candidate_ids(question)]


class Calibrator(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = CalibrationConfig.model_validate(config or {})
        if self.config.mode == "scalar_temperature":
            self.beta = nn.Parameter(torch.zeros(()))

    def forward(self):
        if self.config.mode == "identity":
            return 1.0
        if not torch.isfinite(self.beta):
            raise FloatingPointError("Nonfinite scalar temperature")
        return (self.config.log_temperature_bound * self.beta.float().tanh()).exp()


@dataclass
class Probabilities:
    candidate_mask: torch.Tensor
    log_p_raw: torch.Tensor
    p_raw: torch.Tensor
    entropy: torch.Tensor
    margin: torch.Tensor
    temperature: torch.Tensor
    log_p_final: torch.Tensor
    p_final: torch.Tensor


def probabilities(logits, candidate_mask=None, calibrator=None):
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("Expected a nonempty candidate axis")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        mask = torch.ones_like(logits, dtype=torch.bool) if candidate_mask is None else candidate_mask
        if mask.dtype != torch.bool or mask.shape != logits.shape or not mask.any(-1).all():
            raise ValueError("Candidate mask must match logits and contain nonempty support")
        if not torch.isfinite(logits[mask]).all():
            raise FloatingPointError("Nonfinite valid decision logits")
        log_raw = logits.masked_fill(~mask, -torch.inf).log_softmax(-1)
        if not torch.isfinite(log_raw[mask]).all():
            raise FloatingPointError("Valid log probabilities exceed FP32 range")
        raw = log_raw.exp()
        entropy = -(raw * log_raw.masked_fill(~mask, 0)).sum(-1)
        top = raw.topk(min(2, logits.shape[-1]), dim=-1).values
        margin = top[..., 0] - (top[..., 1] if top.shape[-1] > 1 else 0)
        value = calibrator() if calibrator is not None else 1.0
        temperature = torch.ones_like(entropy) * value
        temperature = torch.where(mask.sum(-1) == 1, torch.ones_like(temperature), temperature)
        if calibrator is None or calibrator.config.mode == "identity":
            final_log = log_raw
        else:
            final_log = (
                (logits.masked_fill(~mask, 0) / temperature.unsqueeze(-1))
                .masked_fill(~mask, -torch.inf)
                .log_softmax(-1)
            )
        return Probabilities(mask, log_raw, raw, entropy, margin, temperature, final_log, final_log.exp())
