# Contributing to GS1

[English](CONTRIBUTING.md) · [简体中文](CONTRIBUTING.zh-CN.md)

Keep changes focused on model inference, serving, the playground and reproducible evaluation. Model source in `modular_gs1.py` is shared with the standalone HF releases; preserve parameter keys, precision policy and output semantics unless a versioned model change explicitly requires otherwise.

```bash
uv sync --locked --extra serve --extra eval --extra dev
uv run pytest -q
uv run ruff check gs1 benchmark tests scripts examples
uv run ruff format --check gs1 benchmark tests scripts examples
cd playground
npm ci
npm run build
npm run test:server
npm run test:e2e
```

GPU changes also need a current qualification against the single-branch reference and bitwise graph/eager replay checks. Record hardware, source/weight identities, precision, request geometry, timing boundary and memory. Report supported fallbacks as such.

Keep complete weights, datasets, secrets, caches and generated application builds outside Git. Add both English and Simplified Chinese versions when changing user-facing Markdown; keep commands and quantitative tables aligned. Preserve third-party license notices. Issues and PRs should include a minimal reproducible request and relevant version information.
