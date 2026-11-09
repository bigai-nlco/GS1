# AA v1 public component evaluation

[English](README.md) · [简体中文](README.zh-CN.md)

This frozen public adaptation retains four historical AA v1 component tasks. The current [official methodology](https://artificialanalysis.ai/methodology/decision-index) has evolved; these local results are not official current-index scores.

| Component | Frozen records | Metric |
|---|---:|---|
| AA-Omniscience public | 600 | Accuracy; `abstained/(abstained+wrong)` NHR |
| HLE multiple-choice | 513 | Answer accuracy |
| When2Call | 3,423 | Reply-category accuracy |
| WildGuard | 856 | Mean prompt-harm, response-harm and refusal F1 |

Inputs are provided as an external `aadi-v1` directory containing `<suite>/test.jsonl`. [`inputs.py`](inputs.py) pins sources, revisions, final counts and file SHA-256. The standalone archive is an input artifact, not part of the Python wheel or Git repository. Use `--data-dir /path/to/aadi-v1 --offline` with all four suite names, as shown in [the evaluation guide](../README.md).

Omniscience already contains three frozen distractors, the gold answer and `I don't know`; evaluation never invokes another model to generate or grade them. It uses 600 public records rather than the official private 500-question population. WildGuard's frozen local subset excludes 24 of 880 unanimously labeled rows and differs from the official 835-row population. Their source IDs and provenance remain pinned. HLE and When2Call also retain the historical frozen selection.

The local v1 aggregate is `100*(0.125*AA_accuracy + 0.125*AA_NHR + 0.25*HLE_accuracy + 0.25*When2Call_accuracy + 0.25*WildGuard_mean_F1)`. `official_comparable` remains false. Missing or modified files fail before model loading; no replacement data is silently generated. Explicit user overrides are reported as separate local inputs.
