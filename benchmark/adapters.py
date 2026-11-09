"""Source-only HTTP and local-model adapters for evaluation."""

import asyncio
import copy
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gs1.model.schema import DecisionResponse, ScoreAnswer, validate_response


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str
    model: str = Field(min_length=1)

    @field_validator("base_url")
    @classmethod
    def serving_url(cls, value):
        url = urlsplit(value)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "Expected an HTTP(S) serving URL without embedded credentials, query or fragment"
            )
        return value.rstrip("/")

    @property
    def is_openrouter(self):
        return urlsplit(self.base_url).hostname == "openrouter.ai"

    @property
    def decision_url(self):
        return (
            self.base_url
            if self.base_url.endswith(("/decisions", "/v1/systemone"))
            else self.base_url + "/v1/systemone"
        )


def openrouter_response(raw, request):
    """Preserve published probabilities; derive Score from their visible precision."""
    answers = copy.deepcopy(raw["answers"])
    derived = []
    for qid, answer in answers.items():
        if answer.get("type") == "score":
            original = ScoreAnswer.model_validate(answer)
            if original.score > len(original.legend) - 1:
                raise ValueError("OpenRouter returned an out-of-range Score")
            expected = sum(int(k) * p for k, p in original.probabilities.items())
            if abs(original.score - expected) > 1e-6:
                derived.append(
                    {"question_id": qid, "reported_score": original.score, "served_score": expected}
                )
            answer["score"] = expected
    usage = raw["usage"]
    response = DecisionResponse.model_validate(
        {
            "model": raw["model"],
            "answers": answers,
            "usage": {
                "input_tokens": usage.get("input_tokens", usage.get("inputTokens")),
                "output_tokens": usage.get("output_tokens", usage.get("outputTokens")),
            },
        }
    )
    validate_response(request, response)
    response.answers = {qid: response.answers[qid] for qid in request.questions}
    return response, derived


class APIAdapter:
    def __init__(self, url, model, *, api_key_env=None, env_file=".env", timeout=120, client=None):
        self.profile_config = BackendConfig(base_url=url, model=model)
        self.deadline, self.client = timeout, client
        self.api_key_env = api_key_env or (
            "OPENROUTER_API_KEY" if self.profile_config.is_openrouter else "EVALUATION_API_KEY"
        )
        self.env_file = env_file
        self.available, self.reason, self.last_stats = False, "Not connected", {}

    @property
    def model_name(self):
        return self.profile_config.model

    @property
    def identity(self):
        return dict(
            kind="api",
            url=self.profile_config.decision_url,
            requested_model=self.model_name,
            timeout_seconds=self.deadline,
            retries=0,
        )

    async def start(self):
        if self.client is None:
            values = dotenv_values(self.env_file) if self.env_file and Path(self.env_file).is_file() else {}
            key = os.environ.get(self.api_key_env, values.get(self.api_key_env)) or ""
            if self.profile_config.is_openrouter and not key:
                raise ValueError(f"Set {self.api_key_env} to evaluate the hosted model")
            self.client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {key}"} if key else {},
                timeout=self.deadline,
                follow_redirects=False,
            )
        self.available, self.reason = True, ""

    async def evaluate(self, request, tenant="evaluation"):
        if not self.available:
            raise RuntimeError(self.reason)
        started = time.perf_counter()
        stats = {"attempts": 1, "engine": "remote-typesafe"}
        try:
            payload = request.model_dump(mode="json")
            payload["model"] = self.model_name
            result = await self.client.post(self.profile_config.decision_url, json=payload)
            stats["raw_response_text"] = result.text
            if not result.is_success:
                stats["http_status"] = result.status_code
                raise RuntimeError(f"Decision request failed (HTTP {result.status_code})")
            raw = result.json()
            if self.profile_config.is_openrouter:
                response, derived = openrouter_response(raw, request)
                stats.update(
                    provider=raw.get("provider"),
                    gateway_request_id=raw.get("id"),
                    cost=raw["usage"].get("cost"),
                    derived_scores=derived,
                )
            else:
                response = DecisionResponse.model_validate(raw)
                validate_response(request, response)
            return response
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError(f"Decision request failed: {type(error).__name__}") from error
        finally:
            stats["client_ms"] = (time.perf_counter() - started) * 1000
            self.last_stats = stats

    async def close(self):
        if self.client is not None:
            await self.client.aclose()


class ModelAdapter:
    def __init__(
        self, model_path, *, device="cpu", precision="fp32", engine="reference", runtime_profile=None
    ):
        self.model_path = str(model_path)
        self.device, self.precision, self.engine_name = device, precision, engine
        self.engine = None
        self.runtime_profile = runtime_profile
        self.metadata = {}
        self.last_stats = {}

    @property
    def model_name(self):
        return self.metadata.get("resolved_model", str(self.model_path))

    @property
    def identity(self):
        return {
            "kind": "model",
            "artifact": str(self.model_path),
            "checkpoint_id": self.metadata.get("checkpoint_id"),
            "requested_model": self.model_name,
            "device": self.device,
            "precision": self.precision,
            "engine": self.engine_name,
        }

    def _load(self):
        import torch

        from gs1 import Engine

        torch.set_num_threads(2)
        self.engine = Engine.from_pretrained(
            self.model_path,
            device=self.device,
            precision=self.precision,
            mode=self.engine_name,
            runtime_profile=self.runtime_profile,
        )
        self.metadata = self.engine.identity

    async def start(self):
        await asyncio.to_thread(self._load)

    def _evaluate(self, request):
        import torch

        self.last_stats = {}
        try:
            with torch.inference_mode():
                return self.engine.evaluate(request, self.model_name, tenant="evaluation")
        finally:
            self.last_stats = dict(self.engine.last_stats)

    async def evaluate(self, request, tenant="evaluation"):
        return await asyncio.to_thread(self._evaluate, request)

    async def close(self):
        if self.engine is not None:
            await asyncio.to_thread(self.engine.close)
            self.engine = None
