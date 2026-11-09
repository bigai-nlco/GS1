# Model release

[English](release.md) · [简体中文](release.zh-CN.md)

Each final directory is a complete, independent Hub repository: `GS1-0.8B`, `GS1-2B`, or `GS1-4B`. Its target is `bigai/GS1-<size>` in [the GS1 collection](https://huggingface.co/collections/bigai/gs1). The 9B directory is added when its checkpoint is available.

A release contains `config.json` with `model_type: gs1`, `architectures: [GS1Model]` and AutoConfig/AutoModel `auto_map`, structural token IDs and `inference_limits`; unchanged safetensors and an index when sharded; tokenizer/chat-template and processor assets; `modular_gs1.py` plus its same-directory relative dependency closure; `requirements.txt`; English/Chinese README files; licenses; and small referenced images.

Default Hub inference uses `AutoModel.from_pretrained(..., trust_remote_code=True)` and `model.decide(...)`, with no GS1 repository installation. The optimized path installs GS1 and uses the same native weights with a local runtime qualification. Model-folder READMEs provide both examples.

Before publishing, confirm:

- Every weight file has the same SHA-256 as the source checkpoint; tensor names, shapes and dtypes are preserved.
- Metadata contains only public model/inference information. Architecture, token IDs and inference limits are in `config.json`; a separate artifact manifest is unnecessary. Retain original weight hashes and source provenance with release-validation records outside the upload folder.
- AutoClass works in a process where `gs1` cannot be imported, and all three primitive types produce valid distributions.
- Native loading, saved-model reload and API decisions agree within the declared numerical policy.
- No local GPU profile, credential, training state, evaluation input or stale source path enters the upload.

Upload a single final folder as that model's Hub repository root. Keep the original archive as a recovery source. Uploading and changing Hub visibility are separate publication actions; the prepared local folders are usable before upload. The root repository contains code and public aggregate results, not model weights.
