# Related projects and release design

[English](related-work.md) · [简体中文](related-work.zh-CN.md)

## Jev and typed decisions

Jev targets fast judgments that software can consume: state and runtime-defined questions become typed values and probabilities. Its main practical benefits are direct programmatic outputs, bounded choices and avoiding text decoding/parsing for decision tasks. The vendor also emphasizes calibration and parallel evaluation; those properties need their own measurements for any implementation. [TypeSafe introduction](https://docs.typesafe.ai/introduction), [launch article](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

GS1 exposes Choice, Noul and Score, with native candidate-conditioned logits and open checkpoint files. Its confidence formulas are explicit; [Brier/ECE and AA component results](results.md) describe measured behavior. The current Jev documentation additionally lists Box; GS1's public contract covers the three implemented primitives. Probabilities and a constrained output vocabulary do not by themselves establish correctness or calibration.

The open ecosystem demonstrates several ways to expose this interface. [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) reads option probabilities from existing models. [AnyJev](https://github.com/nokia-applied-research/AnyJev) provides model-agnostic typed decision adapters. GS1 publishes a native decision-head parameter tree and its model definition alongside the inference engine; these are different implementation choices, so interface compatibility does not imply shared architecture or equivalent scores.

## Lessons from architecture and inference releases

The release choices below are our interpretation of the linked projects' public material, not evidence that a presentation pattern guarantees adoption.

| Primary project | Useful presentation pattern | Applied to GS1 |
|---|---|---|
| [Mamba](https://github.com/state-spaces/mamba) | Model explanation, runnable minimal interface and pretrained model links | Canonical modular model, compact architecture guide, independent HF entry point |
| [BitNet](https://github.com/microsoft/BitNet) | Inference implementation accompanied by explicit speed/memory measurements | Separate default and optimized paths; timing boundaries, percentiles and memory |
| [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | Coherent model family, weight links, technical evidence and clear license scope | One model table; per-size cards; exact checkpoint identities and citation |
| [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) | Small, readable inference core and direct usage examples | `model/`, `inference/`, `api/`; bounded graph cache and an executable quick start |
| [JevBench](https://github.com/fstandhartinger/jevbench) | Accuracy, calibration and operational metrics with a pinned evaluation contract | Pinned public tasks, full denominators, Brier/ECE; public tasks separate from sealed scores |

## Release sequence

The repository leads with one clear capability, verified headline results, a short executable example and a visual architecture. Detailed scores remain one click away, including weaker components. Every guide and model card has English and Simplified Chinese versions.

Publish the complete per-size weight folders and validate both local and Hub loading before announcing availability. Keep model IDs consistent across the collection, GitHub, cards and playground. A short real-model demonstration and reproducible result files provide concrete material for a release announcement. Submit results to benchmark maintainers using their current submission process; announcements and submissions are separate publication actions.

Future measured changes should update the result JSON and regenerate figures together. Preserve benchmark revisions, precision, input coverage and actual execution modes so new release numbers can be compared fairly.
