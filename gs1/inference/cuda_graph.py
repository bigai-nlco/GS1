"""Owned native GS1 CUDA graphs with exact eager parity and bounded storage."""

from collections import OrderedDict
from dataclasses import dataclass

import torch

from gs1.inference.state import (
    branch_geometry,
    clone_state,
    parameter_fingerprint,
    state_bytes,
    tensor_signature,
    tensors,
)
from gs1.model.compiler import prepare_batch


@dataclass
class GraphEntry:
    graph: object
    prepared: object
    outputs: list
    size_bytes: int
    done: object = None


@dataclass
class TensorGraphEntry:
    graph: object
    inputs: object
    outputs: object
    size_bytes: int
    retained: tuple = ()
    done: object = None


class NativeGraphCache:
    def __init__(
        self, model, precision, *, max_bytes, max_entries=4, allow_capture=True, execution_policy=None
    ):
        if max_bytes < 1 or max_entries < 1:
            raise ValueError("CUDA graph cache budgets must be positive")
        self.model, self.precision = model, precision
        self.execution_policy = execution_policy
        self.device = model.get_input_embeddings().weight.device
        if self.device.type != "cuda" or torch.cuda.get_allocator_backend() != "native":
            raise ValueError("GS1 graphs require CUDA and the native allocator")
        self.stream = torch.cuda.Stream(device=self.device)
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.allow_capture = allow_capture
        self.entries, self.rejected = OrderedDict(), OrderedDict()
        self.bytes = 0
        self.stats = dict(
            hits=0,
            captures=0,
            misses=0,
            rejects=0,
            evictions=0,
        )

    def _key(self, rows):
        return (
            self.precision,
            self.model.decision_head.bf16_compute,
            getattr(self.execution_policy, "name", "native"),
            branch_geometry(rows),
            parameter_fingerprint(self.model),
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
            torch.backends.cuda.cudnn_sdp_enabled(),
        )

    def _autocast(self):
        # Cast temporaries must belong to the graph pool, not an AMP cache that
        # is cleared when the enclosing request leaves its autocast context.
        return torch.autocast(
            "cuda",
            enabled=torch.is_autocast_enabled("cuda"),
            dtype=torch.get_autocast_dtype("cuda"),
            cache_enabled=False,
        )

    def run_tensors(self, namespace, signature, inputs, function, *, cancelled=None, retained=()):
        with self._autocast():
            return self._run_tensors(
                namespace, signature, inputs, function, cancelled=cancelled, retained=retained
            )

    def _run_tensors(self, namespace, signature, inputs, function, *, cancelled=None, retained=()):
        """Replay GS1 segments/readout with fresh, caller-owned tensor trees."""
        if torch.is_grad_enabled() or self.model.training:
            raise ValueError("Tensor graphs require eval mode without gradients")
        if cancelled and cancelled():
            raise TimeoutError("Inference cancelled before graph execution")
        key = ("tensor", namespace, signature, tensor_signature(inputs), self._key(()))
        entry = self.entries.get(key)
        if entry is None:
            self.stats["misses"] += 1
            if not self.allow_capture or key in self.rejected:
                return function(inputs)
            while self.entries and (len(self.entries) >= self.max_entries or self.bytes >= self.max_bytes):
                self._evict()
            static = clone_state(inputs)
            caller = torch.cuda.current_stream(self.device)
            self.stream.wait_stream(caller)
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    function(static)
                expected = clone_state(function(inputs))
            self.stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                outputs = function(static)
            with torch.cuda.stream(self.stream):
                graph.replay()
            self.stream.synchronize()
            same = tensor_signature(outputs, False) == tensor_signature(expected, False) and all(
                torch.equal(a, b) for a, b in zip(tensors(outputs), tensors(expected), strict=True)
            )
            pool = sum(
                s["total_size"]
                for s in torch.cuda.memory_snapshot()
                if tuple(s.get("segment_pool_id", ())) == graph.pool()
            )
            size = pool + state_bytes((static, retained))
            if not same or size > self.max_bytes:
                graph.reset()
                self.stats["rejects"] += 1
                self.rejected[key] = True
                while len(self.rejected) > 128:
                    self.rejected.popitem(last=False)
                return function(inputs)
            while self.entries and self.bytes + size > self.max_bytes:
                self._evict()
            entry = TensorGraphEntry(graph, static, outputs, size, retained)
            self.entries[key] = entry
            self.bytes += size
            self.stats["captures"] += 1
        else:
            self.stats["hits"] += 1
            self.entries.move_to_end(key)
        caller = torch.cuda.current_stream(self.device)
        if entry.done is not None:
            caller.wait_event(entry.done)
        for destination, source in zip(tensors(entry.inputs), tensors(inputs), strict=True):
            destination.copy_(source)
        self.stream.wait_event(caller.record_event())
        with torch.cuda.stream(self.stream):
            entry.graph.replay()
        caller.wait_event(self.stream.record_event())
        result = clone_state(entry.outputs)
        entry.done = caller.record_event()
        return result

    def _evict(self):
        _, entry = self.entries.popitem(last=False)
        if entry.done is not None:
            entry.done.synchronize()
        self.stream.synchronize()
        self.bytes -= entry.size_bytes
        entry.graph.reset()
        self.stats["evictions"] += 1

    def close(self):
        while self.entries:
            self._evict()
        self.rejected.clear()

    def _capture(self, rows, pad_token_id):
        with self._autocast():
            return self._capture_batch(rows, pad_token_id)

    def _capture_batch(self, rows, pad_token_id):
        prepared = prepare_batch(self.model, rows, pad_token_id)
        caller = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                expected = [
                    value.clone()
                    for value in prepared.forward(self.model, execution_policy=self.execution_policy)
                ]
        self.stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            outputs = prepared.forward(self.model, execution_policy=self.execution_policy)
        with torch.cuda.stream(self.stream):
            graph.replay()
        self.stream.synchronize()
        if not all(torch.equal(a, b) for a, b in zip(outputs, expected, strict=True)):
            graph.reset()
            raise ValueError("GS1 graph capture differs from eager logits")
        pool = sum(
            s["total_size"]
            for s in torch.cuda.memory_snapshot()
            if tuple(s.get("segment_pool_id", ())) == graph.pool()
        )
        size = pool + state_bytes(
            (prepared.ids, prepared.positions, prepared.valid, prepared.roles, prepared.layout)
        )
        return GraphEntry(graph, prepared, outputs, size)

    def run(self, rows, native, *, pad_token_id=0, cancelled=None):
        if torch.is_grad_enabled() or self.model.training:
            raise ValueError("GS1 graph replay requires eval mode without gradients")
        if not rows or any(branch.visual_inputs is not None for _, branch in rows):
            return native()
        if cancelled and cancelled():
            raise TimeoutError("Inference cancelled before graph execution")
        key = self._key(rows)
        entry = self.entries.get(key)
        if entry is None:
            self.stats["misses"] += 1
            if not self.allow_capture or key in self.rejected:
                return native()
            # Capture temporarily coexists with resident graphs; free enough
            # capacity before allocating a new private pool.
            while self.entries and (len(self.entries) >= self.max_entries or self.bytes >= self.max_bytes):
                self._evict()
            try:
                entry = self._capture(rows, pad_token_id)
            except ValueError:
                self.stats["rejects"] += 1
                self.rejected[key] = True
                while len(self.rejected) > 128:
                    self.rejected.popitem(last=False)
                return native()
            if entry.size_bytes > self.max_bytes:
                entry.graph.reset()
                self.stats["rejects"] += 1
                self.rejected[key] = True
                while len(self.rejected) > 128:
                    self.rejected.popitem(last=False)
                return native()
            while self.entries and self.bytes + entry.size_bytes > self.max_bytes:
                self._evict()
            self.entries[key] = entry
            self.bytes += entry.size_bytes
            self.stats["captures"] += 1
        else:
            self.stats["hits"] += 1
            self.entries.move_to_end(key)
        caller = torch.cuda.current_stream(self.device)
        if entry.done is not None:
            caller.wait_event(entry.done)
        width = entry.prepared.ids.shape[1]
        ids = torch.full_like(entry.prepared.ids, pad_token_id)
        for row, (state, branch) in enumerate(rows):
            values = state + branch.suffix
            ids[row, width - len(values) :] = torch.tensor(values, device=self.device)
        entry.prepared.ids.copy_(ids)
        self.stream.wait_event(caller.record_event())
        with torch.cuda.stream(self.stream):
            entry.graph.replay()
        caller.wait_event(self.stream.record_event())
        result = [value.clone() for value in entry.outputs]
        entry.done = caller.record_event()
        return result
