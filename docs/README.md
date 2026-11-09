# GS1 documentation

[English](README.md) · [简体中文](README.zh-CN.md)

Start with [the repository quick start](../README.md), then choose a guide.

| Guide | Purpose |
|---|---|
| [Model](model.md) | Architecture, parameter layout, output semantics and checkpoint precision |
| [Inference](inference.md) | Independent HF decisions, native Engine, qualification and CUDA Graph |
| [API](api.md) | Requests, distributions, confidence and errors |
| [Deployment](deployment.md) | Bounded queues, GPU slots, deadlines and replicas |
| [Results](results.md) | Full public results, regular benchmarks, calibration and latency |
| [Inference latency](latency.md) | Current HF/eager/CUDA Graph timings, raw samples and reproduction |
| [Evaluation](../benchmark/README.md) | Reproduce the retained public benchmarks |
| [Official Decision Index](../benchmark/di/README.md) | Evaluate the GS1 HTTP service using upstream scripts |
| [Playground](../playground/README.md) | Query, Go, Maze and Snake |
| [Release](release.md) | Standalone Hugging Face folders and publication checks |
| [Related work](related-work.md) | Jev, open architectures and release-design references |

The `gs1` wheel contains only model inference (`model/`), optimized execution (`inference/`) and local serving (`api/`). Benchmarks and qualification generation live in `benchmark/`; the replica launcher is `scripts/serve_replicas.py`. These tools and the Node playground run from the source checkout. Weights and evaluation inputs live outside Git.
