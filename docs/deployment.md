# GS1 deployment

[English](deployment.md) · [简体中文](deployment.zh-CN.md)

## Single GPU

```bash
uv run gs1 serve --model /path/to/GS1-0.8B --model-name GS1 \
  --device cuda --precision bf16 --host 127.0.0.1 --port 8000
```

The model argument accepts a local release folder or published Hub ID. The process checks manifest identity, files, tokenizer and structural tokens before serving. It never needs the initialization model directory. The first-generation checkpoint storage precision is retained.

## Qualified graphs and concurrency

Follow [the graph qualification sequence](inference.md#cuda-graph-inference). Pass the emitted profile and warmup list, with sufficient graph bytes and entries. `--graph-slots 1`, `2`, or `4` creates independently owned execution slots. Multiple slots require prewarm; each has its own stream, mutable state, graph buffers and bounded share of cache memory. Graph memory and queue concurrency are separate settings.

The queue is bounded by `--max-queue` (32 by default); `--deadline-seconds` defaults to 30 and includes the full queued request lifetime. Cancellation releases logical request ownership without reusing buffers still needed by GPU work.

## Replica supervisor

`scripts/serve_replicas.py` launches one service per selected GPU, checks readiness and supervises bounded restarts. Each child has its own queue and deadlines:

```bash
uv run python scripts/serve_replicas.py --model /path/to/GS1-0.8B --model-name GS1 --gpus 0,1 --base-port 18080
```

The services above listen on `127.0.0.1:18080` and `127.0.0.1:18081`. The supervisor prints ready endpoints and records their state in `supervisor.json`; an existing ingress can distribute requests across them. Use GPU IDs available to the process. [Measured replica scaling](results.md#latency-and-memory) uses independent replicas, not tensor parallelism.

## Large public-benchmark requests

The complete public index used 32k context and larger attention admission ceilings. To reproduce that profile with the official HTTP evaluator:

```bash
uv run gs1 serve --model /path/to/GS1-2B --model-name GS1 \
  --max-context 32768 --batch-tokens 32768 \
  --max-attention-bytes 549755813888 --deadline-seconds 120
```

The attention value is an admission-work estimate, not a reservation of that much GPU memory. Peak allocation depends on the actual shape and the SDPA implementation. The usual service defaults remain bounded more tightly. Long, unqualified shapes execute through reference fallback; all failures remain visible.

Use a local interface for development and the existing ingress/proxy controls for public deployment. The source-tree benchmark reports warm execution separately from startup and capture.
