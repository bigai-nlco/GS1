"""Load numerical qualifications for optimized inference."""

import json
import os
import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from gs1.model.schema import CONFIDENCE_VERSION
from gs1.model.util import digest, file_hash

SCHEMA = "gs1.runtime-profile.v2"


def runtime_binding(identity, engine, precision):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    root = Path(__file__).parents[1]
    device = torch.device(identity["served_device"])
    sources = sorted((root / "model").glob("*.py")) + sorted((root / "inference").glob("*.py"))
    binding = dict(
        checkpoint_id=identity["checkpoint_id"],
        engine=engine,
        precision=precision,
        confidence=CONFIDENCE_VERSION,
        torch=torch.__version__,
        transformers=version("transformers"),
        runtime_sources={str(p.relative_to(root.parent)): file_hash(p) for p in sources},
        hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
        execution_policy=dict(
            gdn_kernels={
                name: getattr(qwen, name) is not None
                for name in (
                    "FusedRMSNormGated",
                    "causal_conv1d_fn",
                    "causal_conv1d_update",
                    "chunk_gated_delta_rule",
                    "fused_recurrent_gated_delta_rule",
                )
            },
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            sdp={
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled(),
                "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            },
            mkldnn=torch.backends.mkldnn.enabled,
            mkl_cbwr=os.environ.get("MKL_CBWR"),
            threads=torch.get_num_threads(),
        ),
    )
    binding["kernels"] = {}
    for name in ("flash-linear-attention", "causal-conv1d", "triton"):
        try:
            binding["kernels"][name] = version(name)
        except PackageNotFoundError:
            binding["kernels"][name] = None
    if device.type == "cuda":
        settings = torch.cuda.memory._snapshot()["allocator_settings"]
        binding.update(
            cuda=torch.version.cuda,
            capability=list(torch.cuda.get_device_capability(device)),
            allocator=dict(
                backend=torch.cuda.memory.get_allocator_backend(),
                settings={
                    k: v
                    for k, v in settings.items()
                    if k not in {"PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"}
                },
            ),
            reduction=dict(
                bf16=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                fp16=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                blas=str(torch.backends.cuda.preferred_blas_library()),
            ),
        )
    return binding


def request_signature(compiled):
    from gs1.inference.state import branch_geometry

    rows = [branch_geometry([(compiled.state, b)])[0] for b in compiled.branches]
    return digest(sorted(rows, key=repr))


@dataclass(frozen=True)
class RuntimeProfile:
    data: dict
    engine: str

    def allows(self, compiled):
        return request_signature(compiled) in self.data["signatures"]

    def policy(self):
        if self.data["arithmetic"] == "native_rows":
            from gs1.inference.arithmetic import NativeRowGeometry

            return NativeRowGeometry()
        return None


def load_profile(path, identity, engine, precision):
    data = json.loads(Path(path).read_text())
    if (
        not isinstance(data, dict)
        or data.get("schema") != SCHEMA
        or data.get("arithmetic") not in {"native", "native_rows"}
        or not isinstance(data.get("signatures"), list)
        or not data["signatures"]
    ):
        raise ValueError("Invalid native GS1 runtime profile")
    qualified = data.get("engines", {}).get(engine)
    if (
        not isinstance(qualified, dict)
        or qualified.get("passed") is not True
        or qualified.get("binding") != runtime_binding(identity, engine, precision)
        or not isinstance(qualified.get("checks"), list)
        or not qualified["checks"]
        or any(c.get("passed") is not True for c in qualified["checks"])
    ):
        raise ValueError(
            "GS1 runtime profile does not qualify this checkpoint, precision and installed runtime"
        )
    return RuntimeProfile(data, engine)
