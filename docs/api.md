# GS1 API

[English](api.md) · [简体中文](api.zh-CN.md)

The Python service exposes `POST /v1/systemone`, `GET /v1/models` and FastAPI's `/docs`. The request/response contract supports the Choice, Noul and Score subset of the TypeSafe protocol. `model.decide` returns the same answer definitions as a Python dictionary; `Engine.evaluate` returns a Pydantic `DecisionResponse`.

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

State and instructions accept strings, objects or arrays. Choice descriptions accept strings, objects, arrays or null; criteria contain 1–255 distinct candidate IDs. Noul optionally supplies descriptions of true/false. Score criteria are an ordered array of 2–10 levels. A request needs a nonempty question map. Question IDs and candidate IDs are preserved. Unsupported primitive types and invalid values are rejected.

```bash
curl http://127.0.0.1:8000/v1/systemone \
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

with TypeSafeClient(base_url="http://127.0.0.1:8000", api_key="local", model="GS1") as client:
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

Malformed or over-budget inputs fail explicitly. Queue saturation returns 529 with `Retry-After`; request deadlines return 504. Caller cancellation prevents later response delivery while GPU ownership is drained safely. Evaluation uses zero retries and records failures; application retry policy belongs to the client.

The Python service runs local GS1 inference. The Node playground can also call Jev through the official TypeSafe API using `TYPESAFE_API_KEY`; credentials stay on the Node server, and the raw provider payload remains separate from its validated controller response. [Playground setup](../playground/README.md#optional-jev-via-typesafe).
