"""Explicit settings for local GS1 inference serving."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    """Explicit CLI/script settings. Never load these from backends.yaml or .env."""

    model_config = ConfigDict(extra="forbid")
    local_model: str = Field(default="GS1", min_length=1)
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    artifact: str | Path = "MODEL_PATH"
    device: str = "cuda"
    precision: Literal["fp32", "bf16"] = "bf16"
    engine: Literal["reference", "shared"] = "reference"
    max_context: int = Field(default=4096, ge=64)
    max_questions: int = Field(default=512, ge=1)
    max_request_tokens: int = Field(default=131072, ge=64)
    max_branches: int = Field(default=5120, ge=1)
    max_attention_bytes: int = Field(default=1073741824, ge=1)
    max_logit_elements: int = Field(default=16777216, ge=1)
    batch_tokens: int = Field(default=8192, ge=64)
    max_queue: int = Field(default=32, ge=1)
    deadline_seconds: float = Field(default=30, gt=0)
    cache_mb: int = Field(default=512, ge=0)
    cuda_graphs: bool = False
    graph_cache_mb: int = Field(default=1024, ge=1)
    graph_cache_entries: int = Field(default=4, ge=1)
    graph_capture_policy: Literal["prewarm", "online"] = "prewarm"
    graph_warmup_requests: Path | None = None
    graph_slots: Literal[1, 2, 4] = 1
    runtime_profile: Path | None = None
    cpu_threads: int = Field(default=2, ge=1, le=256)

    @model_validator(mode="after")
    def graph_configuration(self):
        if self.cuda_graphs:
            if self.graph_slots > 1 and self.graph_capture_policy != "prewarm":
                raise ValueError("Multiple GS1 graph slots require prewarm capture")
            if self.graph_cache_mb < self.graph_slots or self.graph_cache_entries < self.graph_slots:
                raise ValueError("Each GS1 graph slot requires at least one MiB and one cache entry")
        return self
