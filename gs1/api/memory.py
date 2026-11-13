"""Allocate transient inference memory from actual free VRAM after resident caches."""

import asyncio
import hashlib
import json
import uuid

from gs1.api.local import BackendError, LocalBackend

MIB = 1024**2


def trim_compute_cache(policy, limit_bytes):
    """Evict optional compute copies; native FP32 checkpoint weights stay intact.

    NativeRowGeometry.weight falls back to the original module weight, which
    autocast converts with the same BF16 arithmetic. Keep the immutable model
    fingerprint: eviction must not immediately repopulate the entire cache.
    Call before graph capture so captured pointers always remain owned.
    """
    if limit_bytes < 0:
        raise ValueError("Compute cache limit must be nonnegative")
    if policy is None:
        return 0
    remaining = limit_bytes
    for key, values in list(policy.weights.items()):
        size = sum(t.numel() * t.element_size() for t in values if t is not None)
        if size <= remaining:
            remaining -= size
        else:
            del policy.weights[key]
    return limit_bytes - remaining


def attention_budget(free_bytes, reserve_bytes, configured_limit, model_limit):
    return max(0, min(free_bytes - reserve_bytes, configured_limit, model_limit))


def admission_message(details):
    failures = [
        f"{branch['question_id']}: " + ", ".join(branch["violations"])
        for branch in details["branches"]
        if branch["violations"]
    ]
    return (
        "Admission rejected before inference; "
        + "; ".join(failures)
        + f"; request_id={details['request_id']}"
    )


class MemoryBudgetBackend(LocalBackend):
    def __init__(self, settings):
        super().__init__(settings)
        self.reserve_bytes = settings.gpu_memory_reserve_mb * MIB
        self.compute_cache_bytes = 0
        self.memory_budget = {}
        self.slot_budgets = {}
        self.last_admission = {}

    def _load(self):
        super()._load()
        self.compute_cache_bytes = trim_compute_cache(
            self.policy, self.settings.compute_weight_cache_mb * MIB
        )

    def _configure_slot(self, index, engine):
        self._refresh_budget(engine, index=index)

    def _refresh_budget(self, engine, *, index=0, allow_zero=False):
        import torch

        device = torch.device(self.settings.device)
        if device.type != "cuda":
            return
        with torch.cuda.device(device):
            # Driver-free VRAM after returning unused allocator blocks excludes
            # model weights and live graph pools. Do not double-count inactive
            # private graph-pool blocks as reusable general-purpose memory.
            torch.cuda.empty_cache()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            budget = attention_budget(
                free_bytes,
                self.reserve_bytes,
                self.settings.max_attention_bytes,
                self.metadata["limits"]["max_attention_bytes"],
            )
            # Concurrent slots share the transient capacity of this physical GPU.
            budget //= self.slots
            self.memory_budget = {
                "policy": "driver_free_after_releasing_unused_blocks_minus_reserve",
                "total_bytes": total_bytes,
                "free_bytes_before_request": free_bytes,
                "reserve_bytes": self.reserve_bytes,
                "attention_budget_bytes": budget,
                "compute_weight_cache_bytes": self.compute_cache_bytes,
                "execution_slots": self.slots,
                "allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device),
            }
        self.slot_budgets[index] = dict(self.memory_budget)
        if budget <= 0 and not allow_zero:
            raise ValueError("No transient GPU memory remains after the configured reserve")
        engine.max_attention_bytes = budget

    def _check_admission(self, index, request):
        engine = self.engines[index]
        self._refresh_budget(engine, index=index, allow_zero=True)
        compiled = engine.compiler.compile(request)
        mode = engine.mode
        if engine.runtime_profile is not None and not engine.runtime_profile.allows(compiled):
            mode = "reference"
        branches = []
        for number, branch in enumerate(compiled.branches):
            tokens = len(compiled.state) + len(branch.suffix)
            estimated = engine._allocation(compiled, [(number, branch)], mode)
            violations = []
            if tokens > engine.batch_tokens:
                violations.append(f"branch_tokens={tokens} > token_budget={engine.batch_tokens}")
            if estimated > engine.max_attention_bytes:
                violations.append(
                    f"estimated_attention_bytes={estimated} > attention_budget_bytes={engine.max_attention_bytes}"
                )
            branches.append(
                {
                    "question_id": branch.question_id,
                    "branch_tokens": tokens,
                    "criteria_count": len(branch.candidates),
                    "estimated_attention_bytes": estimated,
                    "violations": violations,
                }
            )
        serialized = json.dumps(request.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
        details = {
            "stage": "admission",
            "request_id": uuid.uuid4().hex,
            "request_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "state_tokens": len(compiled.state),
            "input_tokens": compiled.input_tokens,
            "batch_tokens": engine.batch_tokens,
            "max_context": engine.compiler.max_context,
            "attention_budget_bytes": engine.max_attention_bytes,
            "accepted": all(not branch["violations"] for branch in branches),
            "branches": branches,
        }
        self.last_admission = details
        return details

    async def admission(self, request):
        if not self.available:
            raise BackendError(self.reason, 503)
        return await asyncio.get_running_loop().run_in_executor(
            self.executors[0], self._check_admission, 0, request
        )

    def _make_slot(self, index):
        engine, stream = super()._make_slot(index)
        try:
            self._refresh_budget(engine, index=index)
            return engine, stream
        except BaseException:
            engine.close()
            raise

    def _execute(self, index, pending):
        try:
            details = self._check_admission(index, pending.request)
            if not details["accepted"]:
                print(json.dumps({"event": "admission_rejected", **details}), flush=True)
                return None, (admission_message(details), 422, details), False
        except ValueError as error:
            return None, (str(error), 422), False
        except Exception as error:
            return None, (f"Memory accounting failed: {type(error).__name__}: {error}", 529), True
        return super()._execute(index, pending)
