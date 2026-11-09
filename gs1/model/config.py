"""Semantic architecture and scalar calibration settings for GS1."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

INFERENCE_LIMITS = dict(
    max_context=16384,
    max_questions=512,
    max_request_tokens=131072,
    max_branches=5120,
    batch_tokens=16384,
    max_attention_bytes=1073741824,
    max_logit_elements=16777216,
)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class DecisionConfig(ModelConfig):
    attnres: bool = True
    attnres_layers: list[int] | None = None
    attnres_eps: float = Field(default=1e-6, gt=0)
    bidirectional: bool = True
    bidirectional_layers: list[int] | None = None
    cross_attention_heads: int = Field(default=8, ge=1)
    cross_attention_layers: int = Field(default=2, ge=1)
    projection_size: int = Field(default=256, ge=2)
    norm_eps: float = Field(default=1e-5, gt=0)

    @field_validator("attnres_layers", "bidirectional_layers")
    @classmethod
    def selected_layers(cls, value):
        if value is not None:
            if not value or any(i < 0 for i in value) or len(set(value)) != len(value):
                raise ValueError("Layer selection must contain distinct nonnegative indices")
            return sorted(value)
        return value


class CalibrationConfig(ModelConfig):
    mode: Literal["identity", "scalar_temperature"] = "identity"
    log_temperature_bound: float = Field(default=math.log(20), gt=0)
