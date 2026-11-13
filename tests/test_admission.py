from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from gs1.api.memory import MemoryBudgetBackend
from gs1.api.settings import Settings
from gs1.inference.arithmetic import NativeRowGeometry


def test_buffer_changes_do_not_repopulate_zero_compute_cache():
    model = nn.Module()
    model.language_model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 3))
    model.register_buffer("cache", torch.zeros(1))
    policy = NativeRowGeometry()
    module = model.language_model[0]
    inputs = torch.randn(3, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
        expected = F.linear(inputs, module.weight.to(torch.bfloat16), module.bias.to(torch.bfloat16))
    for update in [lambda: model.cache.add_(1), lambda: model.register_buffer("cache", torch.zeros(2))]:
        update()
        policy.prepare(model, "bf16")
        assert not policy.weights
        with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
            actual = F.linear(inputs, *policy.weight(module))
        assert torch.equal(actual, expected)


def test_compute_cache_remains_bounded_when_buffers_change():
    model = nn.Module()
    model.language_model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 3))
    model.register_buffer("cache", torch.zeros(1))
    limit = 2 * (4 * 6 + 6)
    policy = NativeRowGeometry(cache_bytes=limit)
    for _ in range(3):
        policy.prepare(model, "bf16")
        size = sum(t.numel() * t.element_size() for values in policy.weights.values() for t in values)
        assert 0 < size <= limit
        assert id(model.language_model[0]) in policy.weights
        assert id(model.language_model[1]) not in policy.weights
        model.cache.add_(1)


def test_cuda_budget_uses_free_memory_after_cache_release_and_recovers():
    mib = 1024**2
    backend = MemoryBudgetBackend(Settings(gpu_memory_reserve_mb=1024))
    backend.metadata = {"limits": {"max_attention_bytes": 32 * 1024 * mib}}
    engine = SimpleNamespace(max_attention_bytes=0)
    try:
        with (
            patch("torch.cuda.device", return_value=nullcontext()),
            patch("torch.cuda.empty_cache") as release,
            patch(
                "torch.cuda.mem_get_info",
                side_effect=[
                    (8 * 1024 * mib, 48 * 1024 * mib),
                    (512 * mib, 48 * 1024 * mib),
                    (12 * 1024 * mib, 48 * 1024 * mib),
                ],
            ),
            patch("torch.cuda.memory_allocated", return_value=32 * 1024 * mib),
            patch("torch.cuda.memory_reserved", return_value=33 * 1024 * mib),
        ):
            backend._refresh_budget(engine)
            assert engine.max_attention_bytes == 7 * 1024 * mib
            backend._refresh_budget(engine, allow_zero=True)
            assert engine.max_attention_bytes == 0
            backend._refresh_budget(engine)
            assert engine.max_attention_bytes == 11 * 1024 * mib
            assert release.call_count == 3
    finally:
        for executor in backend.executors:
            executor.shutdown()
