# GS1 evaluation

[English](README.md) · [简体中文](README.zh-CN.md)

Run from the source checkout with `python -m benchmark`. Local inference loads standard GS1 HF folders; HTTP evaluation accepts a TypeSafe-compatible service. This evaluator contains no training integration or data synthesis.

```bash
uv sync --locked --extra eval --extra cuda
uv run python -m benchmark --model /path/to/GS1-0.8B \
  --device cuda --precision bf16 --suites jevbench --out runs/evaluation/0.8B
```

## Retained benchmarks

| Suite | Input and metric |
|---|---|
| `jevbench` | 231 fixed public Easy/Original/Hard tasks; task accuracy, expected Brier, ten-bin ECE |
| `agnews` | Native 7,600-record test split; four-class accuracy |
| `toolace` | 640 accepted fixed conversation-hash cases; first-tool/no-tool accuracy |
| `aa_omniscience` | 600 frozen public five-option questions; accuracy and NHR |
| `hle_multichoice` | 513 frozen text-only multiple-choice cases; accuracy |
| `when2call` | 3,423 frozen cases; reply-category accuracy |
| `wildguard_test` | 856 frozen cases; mean of three binary F1 metrics |

The default suite is JevBench. Its data downloads from a pinned upstream revision and is SHA-256 checked. AG News and ToolACE adapters read official sources at pinned revisions and create only deterministic evaluation caches. Complete AA JSONL inputs are external; all generation/audit/filtering code has been removed. [AA protocol](aadi/README.md)

```bash
# Prepare public benchmark caches without loading a model.
uv run python -m benchmark --download-only --suites jevbench agnews toolace

# All regular suites, preserving full splits.
uv run python -m benchmark --model /path/to/GS1-4B --device cuda --precision bf16 \
  --suites jevbench agnews toolace --out runs/evaluation/4B-regular

# Read the external frozen AA inputs; no download or generation.
uv run python -m benchmark --model /path/to/GS1-2B --device cuda --precision bf16 \
  --suites aa_omniscience hle_multichoice when2call wildguard_test \
  --data-dir /path/to/aadi-v1 --offline --out runs/evaluation/2B-aa

# Evaluate an already running model service.
uv run python -m benchmark --api-url http://127.0.0.1:8000 --model GS1 \
  --suites jevbench --out runs/evaluation/api
```

`--dataset SUITE=PATH` overrides a JSONL input; `--data-dir` changes cache/input roots, and `--cache-dir` changes JevBench cache. `--offline` prohibits missing-source downloads; `--limit N` is an explicitly partial integration run, not a complete score. API credentials are read from `--api-key-env` and the optional `.env` file; local GS1 needs none.

## Reports and scoring

An `--out` directory receives `evaluation.json` and `accuracy_manifest.json`; existing result paths are rejected. Reports retain dataset and selection hashes, model identity, every attempted decision, raw request/response evidence, timing, errors, accuracy denominator, Brier and reliability bins. Failures score zero accuracy credit; probability metrics use only valid measured outputs and report their count. No output probability or calibration fit is fabricated.

Choice/Score accuracy uses argmax, with pinned benchmark tie conventions. Soft-only targets measure expected correctness. Brier uses `sum((p-target)^2) + 1 - sum(target^2)`; Noul uses `[1-p, p]`. ECE uses ten equal-width top-label bins. Test labels never enter model requests. Local runs default to CPU/FP32/reference, so CUDA measurements explicitly set device and precision.

Exit status: 0 complete valid predictions; 1 prediction failures; 2 invalid configuration, inputs or initialization. [Published full results](../docs/results.md) include source hashes and exact scope. Decision Index uses [its official scripts](di/README.md); there is no in-tree scorer or builder.

## Inference latency

Compare HF default `model.decide`, GS1 eager and numerically checked CUDA Graph execution with identical weights, inputs and precision:

```bash
uv sync --locked --extra serve --extra eval --extra cuda
CUDA_VISIBLE_DEVICES=0 uv run python -m benchmark.latency \
  --model /path/to/GS1-2B --out runs/latency/GS1-2B.json \
  --contexts 256 1024 --questions 1 8 --warmup 5 --repeats 30
```

Timing covers complete in-process decisions, excluding model loading, graph capture and network transport. JSON retains every sample, P50/P95/P99, actual token lengths, memory, software/checkpoint identities, and resident-graph replay checks. [Methodology and full results](../docs/latency.md).
