# GS1 inference

[English](inference.md) · [简体中文](inference.zh-CN.md)

## Default model inference

Each release folder contains the native model, tokenizer, processor assets and relative Python dependencies. Hugging Face AutoClass loads it independently of the GS1 repository. `model.decide(state, questions, precision="bf16")` returns a JSON-compatible response. Use `precision="fp32"` for CPU reference execution in a compatible environment. CUDA kernels follow the installed upstream Qwen runtime; the measured CUDA stack uses FLA 0.5.2 and causal-conv1d 1.7.0.

The default path compiles one complete branch per question and executes each branch independently. Candidates for a question are scored together. It preserves the serialized head precision, uses FP32 final probability reductions and generates no text.

## Native decision template

`chat_template.jinja` renders a complete GS1 decision prompt for one question. It is generated from the serializer in [`compiler.py`](../gs1/model/compiler.py). Pass the raw state as one user message, the question's instructions as `question`, and normalized candidate descriptions as `options`:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigai/GS1-0.8B", trust_remote_code=True)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "A customer was charged twice."}],
    question="Which team should handle this ticket?",
    options=["billing: Payments and refunds", "technical: Software problems"],
    add_generation_prompt=True,
    enable_thinking=False,
    tokenize=False,
)
```

Choice options use `candidate_id: description`; Noul options follow false/true order; Score options follow rubric order. Structured values use canonical JSON and text is XML-escaped. The template includes the trained user/assistant boundaries, empty thinking prefix and final `<|decide|>` token. The runtime also applies this native template when loading older GS1 tokenizer assets.

Use `model.decide` or `Engine.evaluate` to execute decisions: these entries normalize typed candidates and construct the required token spans, masks, positions and budget checks. Regenerate existing runtime profiles after changing template or compiler code, since profiles bind both source and input-encoding files.

## Native Engine

```python
from gs1 import Engine

engine = Engine.from_pretrained("bigai/GS1-0.8B", device="cuda", precision="bf16")
try:
    response = engine.evaluate(
        {
            "state": "A refund is requested.",
            "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
        }
    )
    print(response.model_dump())
finally:
    engine.close()
```

Loading accepts either a Hub ID or a local standard HF directory. `revision`, `local_files_only`, `cache_dir` and `token` are supported. The runtime loads `config.json`, safetensors and tokenizer/processor assets directly; it does not require `artifact.json`. Admission limits come from `config.json.inference_limits`.

`reference` is the default engine. Qualification enables compatible batches and arithmetic policies. `shared` reuses immutable state, full-attention K/V, GDN convolution/recurrent boundaries, and distinct state-memory K/V projections for each decision cross-attention layer. It is selected explicitly and requires a matching qualification. Workload geometry determines whether sharing is faster.

## CUDA Graph inference

The `gs1` wheel contains inference and the local API. Run benchmark and qualification tools from the source checkout with `python -m benchmark.inference`; numerical qualifications use the single `gs1.runtime-profile.v2` format. Regenerate earlier profiles after this migration. Profiles now bind a `checkpoint_id` computed from the actual model and input-encoding files; profiles using the former artifact/predictor identity must be regenerated too.

```bash
# Source-only benchmark tools (not included in the gs1 wheel).
uv sync --locked --extra serve --extra cuda --extra eval

# Qualify the installed implementation, precision and request geometries.
uv run python -m benchmark.inference --model /path/to/GS1-2B \
  --out runs/gs1-2b-runtime.json \
  --runtime-profile-out runs/gs1-2b-profile.json \
  --graph-cache-mb 16384 --graph-cache-entries 128

# The benchmark also writes runs/gs1-2b-profile-warmup.json.
uv run gs1 serve --model /path/to/GS1-2B --model-name GS1 \
  --runtime-profile runs/gs1-2b-profile.json \
  --experimental-cuda-graphs --graph-cache-mb 4096 --graph-cache-entries 128 \
  --graph-warmup-requests runs/gs1-2b-profile-warmup.json
```

The profile binds the checkpoint, source files, hardware, precision and tested geometry. The emitted warmup list is ready for `--graph-warmup-requests`. Default graph capture policy is `prewarm`: capture happens at startup, then known graphs replay. A graph miss uses native eager execution; unqualified request geometry uses the single-branch reference. Each serving slot owns its stream, buffers and graph cache.

In Python, pass `runtime_profile="runs/gs1-2b-profile.json"`, `cuda_graphs=True`, `graph_cache_mb=4096` and `graph_cache_entries=128` to `Engine.from_pretrained`. The initial qualified call may capture; exclude it from warm timing. Server prewarm is the deployment-oriented entry point.

Graph profiles are local execution receipts and are regenerated after changes to code, checkpoint, precision or runtime. They are not portable model weights and are not included in HF releases. FP32 qualification uses logit absolute/relative tolerances `1e-5 / 1e-4` and probability TV `1e-4`; an effective BF16 path uses `2e-3 / 1e-2` and TV `5e-3`. Graph replay must match its corresponding eager computation bitwise.

## Budgets and measurements

The compiler checks complete-branch context and total logical tokens before execution. Admission also bounds questions, branches, estimated attention work, logit elements, batching, caches, queue size and graph memory. Over-limit inputs fail explicitly; inference never silently truncates state or filters candidates.

For timing, separate load/capture, compile, engine execution, queue time and HTTP formatting. [Published latency measurements](../benchmark/results.md#latency-and-memory) report their actual boundaries. Short inputs favor launch-overhead reduction; long shapes can exhaust graph residency and fall back to eager execution.
