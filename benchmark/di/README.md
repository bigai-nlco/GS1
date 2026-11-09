# Official Decision Index evaluation

[English](README.md) · [简体中文](README.zh-CN.md)

Use the [upstream Decision Index reproduction kit](https://github.com/apolinario/decision-index). This directory intentionally contains documentation only. Dataset building, linked-case scoring, chance correction and aggregation are owned by the official project.

1. Start GS1 with the 32k serving configuration used by the complete public run:

   ```bash
   uv run gs1 serve --model /path/to/GS1-2B --model-name GS1 \
     --max-context 32768 --batch-tokens 32768 \
     --max-attention-bytes 549755813888 --deadline-seconds 120
   ```

   The attention ceiling is an admission-work estimate. Actual GPU allocation depends on input shapes and the attention implementation.

2. Install the upstream kit separately and prepare its public sources, including any gated-source access required by that kit.
3. Run its HTTP engine against the GS1 service and keep the resulting environment, per-benchmark and index files.

```bash
# Install the official public-suite reproduction kit in its own environment.
pip install 'git+https://github.com/apolinario/decision-index.git@9eb2dbe2a358004c8782c66e40a83ac07b953fec'

python -m decision_index pipeline --engine http \
  --option base_url=http://127.0.0.1:8000 --option model=GS1 \
  --edition 0.3 --out runs/gs1-public-index
```

The pinned revision is the historical public-suite lineage recorded by the published local results. Consult upstream documentation before selecting a newer revision or edition, and record it with the result. The 0.4/0.4.1 board version and public-suite edition 0.3 are different version identifiers; do not substitute `--edition 0.4.1` without upstream support.

The public score is distinct from the private Full leaderboard score. The [complete results](../results.md) include the local complete-public-panel table, five-area chart, 37 benchmark rows and source hashes. Running a sampled or partial panel does not reproduce that complete score. Report unsupported inputs, failures and retries rather than dropping them.
