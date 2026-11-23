# GS1 — General System One

[English](README.md) · [简体中文](README.zh-CN.md)

![GS1 — Fast, Faithful, Frontier](assets/banner.png)

> ***Fast, Faithful, Frontier System One Intelligence.***

GS1 is a family of fast, uncertainty-aware, open-source decision models developed by **BIGAI** (Beijing Institute for General Artificial Intelligence). It computes native probability distributions over variable candidate sets for agent routing, verification, extraction, and control.

[Code](https://github.com/bigai-nlco/GS1) · [Model](https://huggingface.co/collections/bigai/gs1) · [Documentation](docs/README.md) · [Playground](playground/README.md)

## GS1 Highlights

- **Fast - Native Runtime Decision Head.**
  - Produce Choice, Noul and Score distributions directly, with zero generated output tokens.
  - A native multi-head cross-attention decision head queries state evidence using question and candidate representations.
  - With CUDA Graph, GS1-0.8B reaches 12.51 ms P50 for one short question and **6.05×** speedup over default inference for eight short questions.
- **Faithful — multi-step RL with calibrated rewards**
  - Reinforcement learning optimizes sampled action probabilities with a Brier reward. [RLCD](docs/RLCD.md)
  - Native probabilities and confidence support evidence gathering, deferral and explicit abstention choices.
- **Frontier — Hybrid Architecture, Competitive Quality**
  - **Causal Gated DeltaNet** and **Segmented Bidirectional Full Attention** give candidates their complete comparison context while preserving immutable prefixes for reuse across questions.
  - Carry-preserving **Attention Residual** lets sparse full-attention receivers query earlier representations while retaining the explicit skip-connection anchor. [Architecture](docs/model.md)
  - GS1-4B reaches **68.97** on the latest Decision Index 0.4.1 Public index. All GS1 sizes lie on the displayed parameter–performance frontier.

## Demos

<table align="center">
  <tr>
    <td width="33%" align="center"><img src="https://cdn.jsdelivr.net/gh/bigai-nlco/GS1@53ca99e8618b830efba6e9ed1ad87fdf0c61c9b2/assets/demos/starcraft-ii-step82.gif" alt="StarCraft II"><br><sub>StarCraft II: Directs a Protoss force through a large-scale air engagement against Zerg.</sub></td>
    <td width="33%" align="center"><img src="https://cdn.jsdelivr.net/gh/bigai-nlco/GS1@53ca99e8618b830efba6e9ed1ad87fdf0c61c9b2/assets/demos/warcraft-iii-sft5.gif" alt="Warcraft III"><br><sub>Warcraft III: Leads a coordinated push into the enemy base in a long-horizon match.</sub></td>
    <td width="33%" align="center"><img src="https://cdn.jsdelivr.net/gh/bigai-nlco/GS1@53ca99e8618b830efba6e9ed1ad87fdf0c61c9b2/assets/demos/street-fighter-iii.gif" alt="Street Fighter III"><br><sub>Street Fighter III: Reads the fight and times attacks around the opponent.</sub></td>
  </tr>
  <tr>
    <td width="33%" align="center"><img src="https://cdn.jsdelivr.net/gh/bigai-nlco/GS1@53ca99e8618b830efba6e9ed1ad87fdf0c61c9b2/assets/demos/gta-chinatown-wars.gif" alt="GTA Chinatown Wars"><br><sub>GTA: Chinatown Wars: Uses both DS screens to navigate and react to changing game state.</sub></td>
    <td width="33%" align="center"><img src="https://cdn.jsdelivr.net/gh/bigai-nlco/GS1@53ca99e8618b830efba6e9ed1ad87fdf0c61c9b2/assets/demos/gta-vice-city.gif" alt="GTA Vice City"><br><sub>GTA: Vice City: Navigates an open-world encounter and responds to threats in real time.</sub></td>
    <td width="33%" align="center"><img src="https://cdn.jsdelivr.net/gh/bigai-nlco/GS1@53ca99e8618b830efba6e9ed1ad87fdf0c61c9b2/assets/demos/minecraft.gif" alt="Minecraft"><br><sub>Minecraft: Gathers wood and crafts a workbench as steps toward making a wooden pickaxe.</sub></td>
  </tr>
  <tr>
    <td width="33%" align="center"><img src="https://cdn.jsdelivr.net/gh/bigai-nlco/GS1@53ca99e8618b830efba6e9ed1ad87fdf0c61c9b2/assets/demos/google-flights.gif" alt="Google Flights"><br><sub>Google Flights: Compares routes and dates to plan a multi-stop trip.</sub></td>
    <td width="33%" align="center"><img src="https://cdn.jsdelivr.net/gh/bigai-nlco/GS1@53ca99e8618b830efba6e9ed1ad87fdf0c61c9b2/assets/demos/b2b-sourcing.gif" alt="B2B sourcing"><br><sub>B2B sourcing: Finds products and evaluates suppliers in a procurement workflow.</sub></td>
    <td width="33%" align="center"><img src="https://cdn.jsdelivr.net/gh/bigai-nlco/GS1@53ca99e8618b830efba6e9ed1ad87fdf0c61c9b2/assets/demos/online-booking.gif" alt="Online booking"><br><sub>Airbnb: Compares lodging options and completes the booking flow.</sub></td>
  </tr>
</table>

## Models and results

| Model | Decision Index (Public index) | Weights |
|---|---:|---|
| **GS1-0.8B** | **56.12** | [Hugging Face](https://huggingface.co/bigai/GS1-0.8B) |
| **GS1-2B** | **62.22** | [Hugging Face](https://huggingface.co/bigai/GS1-2B) |
| **GS1-4B** | **68.97** | [Hugging Face](https://huggingface.co/bigai/GS1-4B) |
| **GS1-9B** | **70.45** | [Hugging Face](https://huggingface.co/bigai/GS1-9B) |

Exact stored counts and source identities are in [the results](benchmark/results.md).

These are the **latest complete public-suite results** (edition 0.3). All four models have **100% coverage** across **43 text benchmarks**, including **37 benchmarks** that contribute to the Public index. [Evaluation details](benchmark/results.md).

![GS1 parameter–performance Pareto frontier](assets/gs1_pareto_frontier_updated.svg)


## Quick start

### Hugging Face Transformers

Python 3.11+. Install the model folder's `requirements.txt`, or the corresponding PyTorch, Transformers 5.3.0, Accelerate, safetensors, Pydantic, TypeSafe SDK and Pillow dependencies. **The `gs1` repository is not required.** Choose a release folder or its published Hub ID.

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 transformers==5.3.0 accelerate safetensors 'pydantic>=2.8,<3' typesafe-sdk==0.7.0 pillow
```

```python
import torch
from transformers import AutoModel

model = (
    AutoModel.from_pretrained(
        "bigai/GS1-0.8B",
        trust_remote_code=True,
    )
    .eval()
    .to("cuda")
)

request = {
    "state": "A customer was charged twice and requests an urgent refund.",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": "Payments and refunds", "technical": "Software problems"},
        },
        "refund": {"type": "noul", "instructions": "Does the customer request a refund?"},
        "urgency": {
            "type": "score",
            "instructions": "Rate the urgency of handling this ticket.",
            "criteria": ["Routine", "Soon", "Immediate"],
        },
    },
}

with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
    result = model.decide(**request, precision="bf16")

print(result["answers"])
print(result["usage"])
```

`result` contains `model`, `answers` and `usage`. Every supplied candidate receives a probability. [Full three-primitive example](examples/quickstart.py) · [Request JSON](examples/request.json)

The released `chat_template.jinja` renders the native GS1 state, question and options through the same serializer used by the compiler. `model.decide` prepares typed candidates and branch metadata. [Native template usage](docs/inference.md#native-decision-template)

### GS1 Runtime

Linux/CUDA, Python 3.11+, PyTorch 2.9.1 and Transformers 5.3.0:

```bash
git clone https://github.com/bigai-nlco/GS1.git
cd GS1
uv sync --locked --extra serve --extra cuda
```

```bash
uv run gs1 serve --model bigai/GS1-0.8B --model-name GS1
```

The service exposes `POST /v1/systemone` and `GET /v1/models`. [Enable qualified CUDA Graph inference](docs/inference.md#cuda-graph-inference) or use `from gs1 import Engine` for in-process inference. The runtime and Hub model share the model equations and typed output definitions.

### TypeSafe SDK

Point the [official TypeSafe Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python) at the GS1 service started above. The client needs only the SDK:

```bash
python -m pip install typesafe-sdk==0.7.0
```

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient(
    base_url="http://127.0.0.1:8000",
    api_key="local",
    model="GS1",
) as client:
    result = client.system_one(
        state="A customer was charged twice and requests an urgent refund.",
        questions={
            "team": Choice(
                instructions="Which team should handle this ticket?",
                criteria={"billing": "Payments and refunds", "technical": "Software problems"},
            ),
            "refund": Noul(instructions="Does the customer request a refund?"),
            "urgency": Score(
                instructions="Rate the urgency of handling this ticket.",
                criteria=["Routine", "Soon", "Immediate"],
            ),
        },
    )

print(result.choices["team"].choice)
print(result.choices["team"].probabilities)
print(result.nouls["refund"].noul)
print(result.scores["urgency"].score)
```

`api_key="local"` satisfies the SDK's required argument; local GS1 requires no TypeSafe cloud credential. Use the API root as `base_url` and match `model` to `--model-name`. [Runnable example](examples/typesafe_client.py) · [API and confidence](docs/api.md)

## Inference performance

Latest inference measurements:

| Model | GS1 Default P50 | GS1 eager P50 | CUDA Graph P50 / P95 | Speedup | Graph allocated |
|---|---:|---:|---:|---:|---:|
| GS1-0.8B | 54.10 ms | 54.60 ms | **12.51 / 13.22 ms** | 4.32× | 4.07 GiB |
| GS1-2B | 53.95 ms | 54.34 ms | **18.64 / 18.69 ms** | 2.89× | 10.60 GiB |
| GS1-4B | 72.87 ms | 73.84 ms | **35.17 / 35.24 ms** | 2.07× | 23.68 GiB |

RTX 4090 reporting 48 GiB, BF16 computation, **256 tokens / one question**, five warmups and 30 timed calls per group. These are **complete in-process decision calls**, including validation, prompt compilation, model execution, probabilities and typed responses; model loading, graph capture, HTTP and SDK transport are excluded. GS1 Default uses the standalone model's `model.decide` path; speedup compares its P50 with CUDA Graph at identical weights and inputs. Stored weight dtypes are preserved.

![GS1 inference latency across sizes and request shapes](assets/inference-latency.svg)

```bash
# From the source checkout; runs both 256/1,024-token and 1/8-question fixtures.
uv sync --locked --extra serve --extra cuda --extra eval
CUDA_VISIBLE_DEVICES=0 uv run python -m benchmark.latency \
  --model bigai/GS1-0.8B --out runs/latency/GS1-0.8B.json
```

[All shapes, P50/P95/P99, memory and methodology](benchmark/latency.md) · [Raw samples](benchmark/results/inference-latency.json) · [Earlier service/IPC and replica measurements](benchmark/results.md#latency-and-memory)

## Try the playground

![GS1 Playground](assets/playground.png)

The Query workspace shows sample Choice/Noul/Score answers, raw JSON and per-request round-trip latency. [Example request](examples/playground.json).

```bash
cd playground
npm ci
npm run build
npm start
```

Open [localhost:3000](http://127.0.0.1:3000) with GS1 serving on port 8000. Use **Query** to edit state and questions, compare probabilities and inspect raw JSON. GS1 is the default backend. For hosted Jev, export `TYPESAFE_API_KEY` on the Node server to call the official TypeSafe API. [Setup and controls](playground/README.md)

## How GS1 decides

![GS1 architecture](assets/architecture.svg)

GS1 combines Qwen3.5 hybrid sequence mixing, **segmented bidirectional full attention**, **carry-preserving Attention Residuals**, and a **native cross-attention decision head**. Candidate meanings are supplied at inference time. Choice selects among them, Noul returns `p(true)`, and Score returns the expected zero-based level. The head has no fixed candidate-count classifier; API and context budgets define the supported request sizes. [Architecture and numerical contract](docs/model.md)

## Documentation and citation

[Model](docs/model.md) · [RLCD](docs/RLCD.md) · [Inference](docs/inference.md) · [API](docs/api.md) · [Evaluation](benchmark/README.md)

Code and released model weights use [Apache-2.0](LICENSE).

If you find our work helpful, feel free to give us a cite.

```bibtex
@techreport{zheng2026gs1,
  title={GS1: Fast, Faithful, Frontier System One Intelligence},
  author={Zheng, Zilong and Liu, Yang and Qin, Shuhan and Lin, Ziyong and Li, Jiaqi and Jia, Zixia and Zhu, Song-Chun},
  institution={BIGAI},
  year={2026}
}
```

## Groups

You can contact us and communicate with us by adding our group:


| WeChat Group |
|:-------------------------: |
| <img src="assets/logos/wechat_QR.jpeg" width="200" height="200"> |
