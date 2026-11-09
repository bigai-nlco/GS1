import json
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

from gs1 import Engine
from gs1.inference.profile import SCHEMA, load_profile, runtime_binding
from gs1.model.modular_gs1 import GS1Model
from gs1.model.schema import DecisionRequest, aligned_probabilities


def test_default_and_engine_share_decisions(checkpoint, decision_request):
    engine = Engine.from_pretrained(checkpoint, device="cpu", precision="fp32")
    native = engine.evaluate(decision_request)
    default = engine.model.decide(**decision_request, precision="fp32")
    assert native.model_dump()["answers"] == default["answers"]
    for qid, question in DecisionRequest(model="GS1", **decision_request).questions.items():
        assert sum(aligned_probabilities(question, native.answers[qid])) == pytest.approx(1, abs=1e-6)
    engine.close()


def test_save_reload_preserves_tensors_and_dtype(checkpoint, tmp_path, decision_request):
    model = GS1Model.from_pretrained(checkpoint)
    destination = tmp_path / "reloaded"
    model.save_pretrained(destination)
    restored = GS1Model.from_pretrained(destination)
    assert model.state_dict().keys() == restored.state_dict().keys()
    for key, value in model.state_dict().items():
        assert value.dtype == restored.state_dict()[key].dtype
        assert torch.equal(value, restored.state_dict()[key])
    assert (
        json.loads((destination / "config.json").read_text())["auto_map"]["AutoModel"]
        == "modular_gs1.GS1Model"
    )


def test_pretrained_rejects_token_identity_mismatch(checkpoint):
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["structural_token_ids"]["<state>"] += 1
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Structural token identity"):
        Engine.from_pretrained(checkpoint, device="cpu", precision="fp32")


def test_invalid_request_fails_before_forward(checkpoint, decision_request):
    decision_request["questions"]["score"]["criteria"] = ["Only one"]
    engine = Engine.from_pretrained(checkpoint, device="cpu", precision="fp32")
    with pytest.raises(ValidationError):
        engine.evaluate(decision_request)
    engine.close()


def test_shared_and_reference_parity(checkpoint, decision_request):
    reference = Engine.from_pretrained(checkpoint, device="cpu", precision="fp32")
    compiled = reference.compiler.compile(DecisionRequest(model="GS1", **decision_request))
    shared = Engine(reference.model, reference.compiler, mode="shared", precision="fp32")
    with torch.inference_mode():
        left, right = reference.logits(compiled), shared.logits(compiled)
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=1e-5, rtol=1e-4)
    reference.close()
    shared.close()


def test_stale_profile_rejected(checkpoint, tmp_path):
    loaded = Engine.from_pretrained(checkpoint, device="cpu", precision="fp32")
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "arithmetic": "native",
                "signatures": ["x"],
                "engines": {
                    "reference": {
                        "passed": True,
                        "binding": runtime_binding(loaded.identity, "reference", "fp32"),
                        "checks": [{"passed": True}],
                    }
                },
            }
        )
    )
    assert load_profile(path, loaded.identity, "reference", "fp32").engine == "reference"
    from safetensors.torch import load_file, save_file

    weights = checkpoint / "model.safetensors"
    state = load_file(weights)
    name = next(iter(state))
    state[name] = state[name] + 0.125
    save_file(state, weights)
    with pytest.raises(ValueError, match="does not qualify"):
        Engine.from_pretrained(checkpoint, device="cpu", precision="fp32", runtime_profile=path)
    loaded.close()


def test_pretrained_accepts_hub_cache_without_manifest(checkpoint, tmp_path, decision_request):
    assert not (checkpoint / "artifact.json").exists()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for source in checkpoint.iterdir():
        if source.is_file():
            (snapshot / source.name).symlink_to(source)
    reference = Engine.from_pretrained(checkpoint, device="cpu", precision="fp32")
    with patch("huggingface_hub.snapshot_download", return_value=str(snapshot)) as download:
        cached = Engine.from_pretrained(
            "bigai/GS1-test",
            device="cpu",
            precision="fp32",
            revision="test-revision",
            local_files_only=True,
            cache_dir=tmp_path / "cache",
        )
    download.assert_called_once()
    assert download.call_args.kwargs["revision"] == "test-revision"
    assert download.call_args.kwargs["local_files_only"] is True
    assert download.call_args.kwargs["cache_dir"] == tmp_path / "cache"
    assert cached.identity == reference.identity
    assert cached.evaluate(decision_request).answers == reference.evaluate(decision_request).answers
    cached.close()
    reference.close()


def test_pretrained_rejects_shard_path_escape(checkpoint):
    index = checkpoint / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="outside its directory"):
        Engine.from_pretrained(checkpoint, device="cpu", precision="fp32")


def test_config_limits_apply_to_default_and_engine(checkpoint, decision_request):
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["inference_limits"]["max_questions"] = 1
    path.write_text(json.dumps(config))
    engine = Engine.from_pretrained(checkpoint, device="cpu", precision="fp32")
    with pytest.raises(ValueError, match="questions exceeds 1"):
        engine.evaluate(decision_request)
    with pytest.raises(ValueError, match="questions exceeds 1"):
        engine.model.decide(**decision_request, precision="fp32")
    assert engine.model.decision_compiler(max_questions=10).max_questions == 1
    engine.close()


def test_hub_autoclass_without_repository_imports(checkpoint, tmp_path):
    script = r"""import importlib.abc, sys
class BlockGS1(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == "gs1" or fullname.startswith("gs1."):
            raise ImportError("Repository package is deliberately unavailable")
sys.meta_path.insert(0, BlockGS1())
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as q
for key in ("FusedRMSNormGated", "causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
    setattr(q, key, None)
from transformers import AutoConfig, AutoModel
torch.set_num_threads(2)
c = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)
m = AutoModel.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)
r = m.decide("A battery has 18 units.", {"yes": {"type": "noul", "instructions": "Are more than 10 units available?"}}, precision="fp32")
assert c.model_type == "gs1" and "yes" in r["answers"]
assert not any(k == "gs1" or k.startswith("gs1.") for k in sys.modules)
print("standalone Hub inference passed")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(checkpoint)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
