# GS1 API

[English](api.md) · [简体中文](api.zh-CN.md)

The Python service exposes `POST /v1/systemone`, `POST /v1/admission`, `GET /v1/runtime`, `GET /v1/models` and FastAPI's `/docs`. The request/response contract supports the Choice, Noul and Score subset of the TypeSafe protocol. `model.decide` returns the same answer definitions as a Python dictionary; `Engine.evaluate` returns a Pydantic `DecisionResponse`.

## Single-GPU serving defaults

```bash
# Replace 0 with the GPU you select; each server process uses one GPU.
CUDA_VISIBLE_DEVICES=0 uv run gs1 serve \
  --model /data/shared/GS1/GS1-9B --model-name GS1-9B
```

The default listener is `0.0.0.0:9000`. LAN clients use `http://<server-ip>:9000` as the SDK base URL. No GPU index or multi-GPU topology is saved in the serving defaults; choose devices explicitly. For optional multi-GPU serving, select replicas explicitly with `scripts/serve_replicas.py --gpus ...`.

| Setting | Default |
|---|---|
| Engine / precision | `reference` / `bf16` |
| Context / batch tokens | 16384 / 16384, capped by model limits |
| Questions / logical request tokens / branches | 512 / 131072 / 5120, capped by model limits |
| GPU memory reserve | 1024 MiB |
| Prefix cache / optional BF16 weight cache | 0 MiB / 0 MiB |
| Queue / deadline / CPU threads | 32 / 120 seconds / 2 |
| Graph slots / cache / entries / capture policy | 1 / 2048 MiB / 16 / `prewarm` |

Before each inference, the server releases unused allocator blocks and sets its attention and temporary tensor budget to **actual driver-free VRAM minus the reserve**, capped by `--max-attention-bytes` and the model limit. Resident weights and graph pools are already excluded from free VRAM. The default configured ceiling is 512 GiB; it does not allocate or promise that much memory. For explicitly selected concurrent graph slots, the remaining transient budget is divided across slots. CPU serving uses the configured/model ceiling. The bounded compute cache stays bounded after model buffer changes; original checkpoint weights retain their precision.

CUDA graphs require a local qualification and warmup file: add `--runtime-profile ... --experimental-cuda-graphs --graph-warmup-requests ...`. Profiles bind the source and runtime, so regenerate them after updating. A nonzero `--compute-weight-cache-mb` must match the profile's `deployment.compute_weight_cache_mb` used during qualification.

## Request

```json
{
  "model": "GS1",
  "state": "A customer was charged twice and requests a refund.",
  "questions": {
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "technical": "Software"}},
    "refund": {"type": "noul", "instructions": "Is a refund requested?"},
    "urgency": {"type": "score", "instructions": "Rate urgency.", "criteria": ["Routine", "Urgent", "Emergency"]}
  }
}
```

State and instructions accept strings, objects or arrays. Choice descriptions accept strings, objects, arrays or null; criteria contain 1–256 distinct candidate IDs. Noul optionally supplies descriptions of true/false. Score criteria are an ordered array of 2–256 levels. A request needs a nonempty question map. Question IDs and candidate IDs are preserved. Unsupported primitive types and invalid values are rejected.

```bash
curl http://127.0.0.1:9000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request-with-model.json
```

Alternatively add `"model": "GS1"` to [the example request](../examples/request.json). The server model alias is configured with `--model-name`.

## TypeSafe SDK

Call a running GS1 service with the [official Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python):

```bash
python -m pip install typesafe-sdk==0.7.0
```

```python
from typesafe_sdk import Choice, TypeSafeClient

with TypeSafeClient(base_url="http://127.0.0.1:9000", api_key="local", model="GS1") as client:
    result = client.system_one(
        state="A customer was charged twice and requests a refund.",
        questions={
            "team": Choice(
                instructions="Which team should handle this ticket?",
                criteria={"billing": "Payments and refunds", "technical": "Software problems"},
            )
        },
    )
print(result.choices["team"].choice)
print(result.choices["team"].probabilities)
```

Use the API root for `base_url`, without `/v1/systemone`, and match `model` to `--model-name`. The local service accepts the SDK placeholder `api_key="local"`; no TypeSafe cloud credential is needed. Read typed answers through `result.choices`, `result.nouls` and `result.scores`; full probabilities, confidence and usage remain available. [Complete runnable example](../examples/typesafe_client.py).

## Answers

| Primitive | Result |
|---|---|
| Choice | `choice`, complete `probabilities`, normalized `confidence` |
| Noul | `noul = p(true)` |
| Score | expected zero-based `score`, complete level `probabilities`, `legend`, `confidence` |

The response contains `model`, `answers`, and `usage` (`input_tokens`, `output_tokens`). Local GS1 reports zero generated output tokens. Probabilities must be finite, bounded and complete; no missing option receives a fabricated probability. Type validity does not guarantee factual correctness.

## Confidence

For K options with probabilities p:

- Choice: `(max(p) - 1/K) / (1 - 1/K)`; singleton confidence is 1.
- Noul's public value is `p(true)`; an optional binary certainty summary is `abs(2*p(true)-1)`.
- Score: `max(0, 1 - spread/even_spread)`, with `spread = sum(p[i]*abs(i-m))`, m the first most likely level, and `even_spread = sum(abs(i-(K-1)/2))/K`.

Score `[0.25, 0.5, 0.25]` gives score 1 and confidence 0.25. These public confidence values differ from the top-label probabilities used to compute ECE. Thresholds should be evaluated on the intended workload; no test-set calibration is fitted by the evaluator.

## Errors and client behavior

Malformed or over-budget inputs fail explicitly. `POST /v1/admission` compiles and estimates a request without running inference. Capacity rejection returns 422 with actual branch tokens, estimated bytes, current limits and a request hash in `error.details`; prompts are not included in admission diagnostics. `GET /v1/runtime` reports readiness, current budgets, qualified geometry count and resident graph statistics. Successful inference includes `X-GS1-Graph-Hits`, `X-GS1-Graph-Misses` and `X-GS1-Profile-Fallback` headers.

Queue saturation returns 529 with `Retry-After`; request deadlines return 504. Caller cancellation prevents later response delivery while GPU ownership is drained safely. Evaluation uses zero retries and records failures; application retry policy belongs to the client.

The Python service runs local GS1 inference. The Node playground can also call Jev through the official TypeSafe API using `TYPESAFE_API_KEY`; credentials stay on the Node server, and the raw provider payload remains separate from its validated response. [Playground setup](../playground/README.md#optional-jev-via-typesafe).
