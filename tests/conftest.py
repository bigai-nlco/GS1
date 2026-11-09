import pytest
import torch

from gs1.model.compiler import structural_tokens
from gs1.model.modular_gs1 import GS1Model
from tests.fixtures import config, cpu_kernels, tokenizer


@pytest.fixture(autouse=True)
def reference_cpu():
    torch.set_num_threads(2)
    with cpu_kernels():
        yield


@pytest.fixture
def checkpoint(tmp_path):
    tokens = tokenizer()
    tokens.add_special_tokens({"additional_special_tokens": structural_tokens()})
    cfg = config()
    cfg.structural_token_ids = {t: tokens.convert_tokens_to_ids(t) for t in structural_tokens()}
    model = GS1Model(cfg).eval()
    path = tmp_path / "model"
    model.save_pretrained(path)
    tokens.save_pretrained(path)
    return path


@pytest.fixture
def decision_request():
    return {
        "state": "Battery 18, reserve 10.",
        "questions": {
            "choice": {
                "type": "choice",
                "instructions": "Choose the job.",
                "criteria": {"a": "Cost 2", "b": "Cost 12"},
            },
            "noul": {"type": "noul", "instructions": "Is battery above the reserve?"},
            "score": {
                "type": "score",
                "instructions": "Rate the available battery.",
                "criteria": ["Low", "Medium", "High"],
            },
        },
    }
