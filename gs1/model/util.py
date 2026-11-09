"""Native checkpoint loading, serialization and deterministic identities."""

import hashlib
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def named_seed(seed, name):
    return int.from_bytes(hashlib.sha256(canonical(["gs1.rng", seed, name]).encode()).digest()[:8], "big") % (
        2**63
    )


def _weight_files(directory):
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = sorted(set(weight_map.values()))
    else:
        files = sorted(p.name for p in directory.glob("*.safetensors"))
        weight_map = None
    if not files:
        raise ValueError("Checkpoint contains no safetensors weights")
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in files):
        raise ValueError("Checkpoint shard is outside its directory")
    # Hub snapshots link to shared cache blobs; validate shard names rather
    # than rejecting those normal cache symlinks after resolving their targets.
    return [directory / name for name in files], weight_map


def checkpoint_files(directory):
    """Files that define the weights, native configuration and input encoding."""
    directory = Path(directory)
    names = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "tokenizer.model",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    )
    weights, _ = _weight_files(directory)
    return sorted({*weights, *(directory / name for name in names if (directory / name).is_file())})


def read_weights(directory):
    files, weight_map = _weight_files(directory)
    state = {}
    for path in files:
        shard = load_file(path, device="cpu")
        if state.keys() & shard.keys():
            raise ValueError("Duplicate checkpoint parameter")
        state.update(shard)
    if weight_map is not None and state.keys() != weight_map.keys():
        raise ValueError("Checkpoint index differs from its parameter contents")
    return state


def _construct(config, model_class=None):
    from .modular_gs1 import GS1Model

    # Nonpersistent rotary buffers must remain real tensors, not meta placeholders.
    with init_empty_weights(include_buffers=False):
        return (model_class or GS1Model)(config)


def _assign(model, state):
    expected = model.state_dict()
    if state.keys() != expected.keys():
        raise ValueError(
            f"Incomplete GS1 weights: missing={sorted(expected.keys() - state.keys())}, "
            f"unexpected={sorted(state.keys() - expected.keys())}"
        )
    for name, value in state.items():
        if value.shape != expected[name].shape:
            raise ValueError(f"GS1 parameter shape mismatch: {name}")
    model.load_state_dict(state, strict=True, assign=True)
    return model


def resolve_source(
    source, *, revision=None, local_files_only=False, cache_dir=None, token=None, force_download=False
):
    """Resolve a native directory or an immutable Hub snapshot."""
    path = Path(source).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            str(source),
            revision=revision,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.py", "*.model", "*.txt"],
        )
    )


def load_native(
    source,
    *,
    device="cpu",
    dtype=None,
    model_class=None,
    config=None,
    revision=None,
    local_files_only=False,
    cache_dir=None,
    token=None,
    force_download=False,
    **kwargs,
):
    from .modular_gs1 import GS1Config

    for name in (
        "_from_auto",
        "_from_pipeline",
        "_commit_hash",
        "trust_remote_code",
        "code_revision",
        "weights_only",
        "use_safetensors",
        "adapter_kwargs",
    ):
        kwargs.pop(name, None)
    if "torch_dtype" in kwargs:
        if dtype is not None:
            raise ValueError("Use only dtype or torch_dtype")
        dtype = kwargs.pop("torch_dtype")
    device_map = kwargs.pop("device_map", None)
    if device_map is not None:
        if isinstance(device_map, dict) and set(device_map) == {""}:
            device = device_map[""]
        elif isinstance(device_map, str) and (
            device_map in {"cpu", "cuda"} or device_map.startswith("cuda:")
        ):
            device = device_map
        else:
            raise ValueError(
                "GS1 supports a single device; use .to(device) instead of automatic device mapping"
            )
    if kwargs:
        raise TypeError(f"Unsupported GS1 loading options: {sorted(kwargs)}")
    directory = resolve_source(
        source,
        revision=revision,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        token=token,
        force_download=force_download,
    )
    raw = json.loads((directory / "config.json").read_text())
    if raw.get("model_type") != "gs1":
        raise ValueError("Expected native GS1 weights")
    config = config or GS1Config(**raw)
    config._name_or_path = str(source)
    model = _construct(config, model_class)
    model = _assign(model, read_weights(directory))
    model._processor_source = directory
    if dtype is not None and dtype != "auto":
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        model.to(dtype=dtype)
    return model.to(device).eval()


def export_state(model, state=None):
    """Save the native inference parameter tree without dtype conversion."""
    state = model.state_dict() if state is None else state
    return {name: value.detach().contiguous() for name, value in state.items()}
