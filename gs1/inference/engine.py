"""Native GS1 batching and bounded, tenant-isolated prefix/readout reuse."""

import contextlib
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

import torch

from gs1.inference.state import batch_states, clone_state, state_bytes, state_row
from gs1.model.compiler import assemble_logits, branch_logits
from gs1.model.schema import DecisionResponse, Usage, make_answer, validate_response
from gs1.model.util import digest, file_hash


@dataclass
class Prefix:
    layers: tuple
    size_bytes: int
    hidden: torch.Tensor | None = None
    memory: dict | None = None


class PrefixCache:
    """Immutable borrowed tensors remain valid after bounded LRU eviction."""

    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0

    def get(self, key):
        value = self.entries.get(key)
        if value is not None:
            self.entries.move_to_end(key)
        return value

    def put(self, key, layers, hidden=None, memory=None):
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.bytes -= previous.size_bytes
        size = state_bytes((layers, hidden, memory))
        if size > self.max_bytes:
            return Prefix(tuple(layers), size, hidden, memory)
        layers, hidden, memory = clone_state((tuple(layers), hidden, memory))
        size = state_bytes((layers, hidden, memory))
        value = Prefix(layers, size, hidden, memory)
        while self.entries and self.bytes + size > self.max_bytes:
            _, old = self.entries.popitem(last=False)
            self.bytes -= old.size_bytes
        self.entries[key] = value
        self.bytes += size
        return value


class Engine:
    def __init__(
        self,
        model,
        compiler,
        mode="reference",
        batch_tokens=16384,
        cache_mb=512,
        identity=None,
        precision="fp32",
        max_attention_bytes=1073741824,
        max_logit_elements=16777216,
        reference_batching=True,
        graph_cache=None,
        execution_policy=None,
        runtime_profile=None,
    ):
        if mode not in {"reference", "shared"} or precision not in {"fp32", "bf16"}:
            raise ValueError("Unsupported GS1 engine or precision")
        if compiler.decision != model.decision_config:
            raise ValueError("Compiler and model decision architectures differ")
        self.model, self.compiler, self.mode = model, compiler, mode
        self.batch_tokens = batch_tokens
        self.cache = PrefixCache(cache_mb * 1024**2)
        self.identity = identity or {}
        self.precision = precision
        self.device = model.get_input_embeddings().weight.device
        self.max_attention_bytes = max_attention_bytes
        self.max_logit_elements = max_logit_elements
        self.reference_batching = reference_batching
        self.graph_cache = graph_cache
        self.execution_policy = execution_policy
        self.runtime_profile = runtime_profile
        if execution_policy is not None:
            execution_policy.prepare(model, precision)
        self.last_stats = {}

    @classmethod
    def from_pretrained(
        cls,
        source,
        *,
        device="cuda",
        precision="bf16",
        mode="reference",
        runtime_profile=None,
        cuda_graphs=False,
        graph_cache_mb=1024,
        graph_cache_entries=4,
        revision=None,
        local_files_only=False,
        cache_dir=None,
        token=None,
        force_download=False,
        **options,
    ):
        from gs1.inference.profile import load_profile
        from gs1.model.modular_gs1 import GS1Model
        from gs1.model.util import checkpoint_files, resolve_source

        if mode not in {"reference", "shared"} or precision not in {"fp32", "bf16"}:
            raise ValueError("Unsupported GS1 engine or precision")
        directory = resolve_source(
            source,
            revision=revision,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
        )
        model = GS1Model.from_pretrained(directory, device=device)
        compiler = model.decision_compiler()
        files = {str(p.relative_to(directory)): file_hash(p) for p in checkpoint_files(directory)}
        checkpoint_id = digest(files)
        identity = dict(
            checkpoint_id=checkpoint_id,
            resolved_model="GS1-"
            + digest(dict(checkpoint=checkpoint_id, precision=precision, engine=mode))[:16],
            served_device=str(device),
            limits=model.config.inference_limits,
        )
        profile = load_profile(runtime_profile, identity, mode, precision) if runtime_profile else None
        qualified = profile is not None
        if mode == "shared" and profile is None:
            raise ValueError("Shared inference requires a qualified runtime_profile")
        policy = profile.policy() if profile else None
        graph = None
        if cuda_graphs:
            if profile is None or not profile.data["engines"][mode].get("graphs_passed"):
                raise ValueError("CUDA graphs require a qualified --runtime-profile")
            from gs1.inference.cuda_graph import NativeGraphCache

            graph = NativeGraphCache(
                model,
                precision,
                max_bytes=graph_cache_mb * 1024**2,
                max_entries=graph_cache_entries,
                execution_policy=policy,
            )
        limits = model.config.inference_limits
        defaults = {k: limits[k] for k in ("batch_tokens", "max_attention_bytes", "max_logit_elements")}
        defaults.update(options)
        return cls(
            model,
            compiler,
            mode=mode,
            precision=precision,
            identity=identity,
            reference_batching=qualified,
            graph_cache=graph,
            execution_policy=policy,
            runtime_profile=profile,
            **defaults,
        )

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def close(self):
        if self.graph_cache is not None:
            self.graph_cache.close()
        self._sync()
        self.cache.entries.clear()
        self.cache.bytes = 0

    def _readout(self, *args, **kwargs):
        if self.graph_cache is not None and kwargs.get("prepared_memory") is not None:
            hidden, branch, offset = args
            from gs1.inference.state import branch_geometry

            signature = (branch_geometry([((), branch)]), offset)

            def readout(inputs):
                suffix, state, memory = inputs
                return self.model.readout(suffix, branch, offset, state_hidden=state, prepared_memory=memory)

            value = self.graph_cache.run_tensors(
                "readout", signature, (hidden, kwargs["state_hidden"], kwargs["prepared_memory"]), readout
            )
        else:
            value = self.model.readout(*args, **kwargs)
        return value

    def _autocast(self):
        return (
            # AMP's weight cache is shared across threads: a cast created on
            # another slot's stream has no ownership/event relationship here.
            # Language compute copies are replica-owned; head casts must belong
            # to this request/graph rather than that process-wide cache.
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, cache_enabled=False)
            if self.precision == "bf16"
            else contextlib.nullcontext()
        )

    def _allocation(self, compiled, chunk, mode):
        """Bound attention masks, projection temporaries and retained shared state.

        Model weights, FFN/AttnRes workspace and allocator/kernel overhead are
        measured separately during qualification, rather than called attention.
        """
        text = self.model.config.text_config
        batch, state = len(chunk), len(compiled.state)
        suffix = max(len(branch.suffix) for _, branch in chunk)
        length = state + suffix
        element = self.model.get_input_embeddings().weight.element_size()
        compute = 2 if self.precision == "bf16" else element
        kernel_head_dim = (text.head_dim + 7) // 8 * 8
        attention_width = text.num_attention_heads * kernel_head_dim
        kv_width = text.num_key_value_heads * kernel_head_dim
        value_width = text.linear_num_value_heads * text.linear_value_head_dim
        projected_width = max(
            2 * attention_width + 2 * kv_width,
            2 * text.linear_num_key_heads * text.linear_key_head_dim
            + 2 * value_width
            + 2 * text.linear_num_value_heads,
        )
        # Bound simultaneously live projection, normalization, RoPE and output
        # tensors using actual FA/GDN widths, including FP32 promotion.
        projection_bytes = 4 * max(element, compute) * projected_width + 2 * element * text.hidden_size
        modes = {self.model.language_model.is_bidirectional(i) for i in self.model.language_model.full_layers}
        head_scores = (
            8
            * self.model.decision_config.cross_attention_heads
            * state
            * max(len(branch.candidates) for _, branch in chunk)
        )
        if mode == "reference" or any(branch.visual_inputs is not None for _, branch in chunk):
            return (
                batch * length**2 * (element * len(modes) + 3 + 8 * text.num_attention_heads)
                + batch * length * projection_bytes
                + head_scores
            )
        full = len(self.model.language_model.full_layers)
        linear = len(text.layer_types) - full
        # RoPE can promote keys to the residual/position dtype under autocast;
        # values retain projection precision. Do not charge both as BF16.
        kv_token = full * text.num_key_value_heads * text.head_dim * (max(element, compute) + compute)
        recurrent = linear * (
            4 * text.linear_num_value_heads * text.linear_key_head_dim * text.linear_value_head_dim
            + compute
            * (
                2 * text.linear_num_key_heads * text.linear_key_head_dim
                + text.linear_num_value_heads * text.linear_value_head_dim
            )
            * text.linear_conv_kernel_dim
        )
        recurrent = 2 * recurrent + linear * 63 * text.hidden_size * max(element, compute)
        head_element = (
            2
            if self.model.decision_head.bf16_compute
            else self.model.decision_head.memory_projection.weight.element_size()
        )
        memory = (
            state
            * text.hidden_size
            * (element + head_element * (1 + 2 * (1 + self.model.decision_config.cross_attention_layers)))
        )
        masks = batch * suffix * ((length + 7) // 8 * 8) * (compute + 2) if False in modes else 0
        allocation = (
            (1 + 3 * batch) * (length * kv_token + recurrent)
            + 2 * memory  # Publication clones coexist with the prepared source.
            + masks
            + head_scores
            + batch * suffix * projection_bytes
        )
        if self.execution_policy is not None:
            # Canonical row arithmetic uses full-branch projections and role
            # masks even for a compact active segment. Charge their real extent
            # and conservatively allow a math-SDPA workspace as in reference.
            allocation += batch * length**2 * (element * len(modes) + 3 + 8 * text.num_attention_heads)
            allocation += batch * length * projection_bytes
            allocation += 2 * (batch - 1) * memory
        return allocation

    def _chunks(self, compiled, mode):
        if mode not in {"reference", "shared"}:
            raise ValueError("Unsupported GS1 engine")
        chunk = []
        for item in sorted(enumerate(compiled.branches), key=lambda item: len(item[1].suffix)):
            proposed = chunk + [item]

            def fits(items):
                lengths = [len(compiled.state) + len(branch.suffix) for _, branch in items]
                tokens = (
                    max(lengths) * len(items)
                    if mode == "reference" or any(branch.visual_inputs is not None for _, branch in items)
                    else len(compiled.state) + sum(len(branch.suffix) for _, branch in items)
                )
                return (
                    (mode != "reference" or self.reference_batching or len(items) == 1)
                    and tokens <= self.batch_tokens
                    and self._allocation(compiled, items, mode) <= self.max_attention_bytes
                )

            if not fits(proposed):
                if chunk:
                    yield chunk
                chunk = []
                if not fits([item]):
                    raise ValueError(
                        "Physical branch exceeds token/attention allocation budget; no truncation"
                    )
            chunk.append(item)
        if chunk:
            yield chunk

    def _prefill(self, compiled, tenant, geometry=None):
        from gs1.inference.state import parameter_fingerprint

        starts = tuple(sorted({branch.state_evidence_start for branch in compiled.branches}))
        key = (
            None
            if self.cache.max_bytes <= 0
            else (
                tenant,
                digest(self.identity),
                digest(self.model.decision_config.model_dump()),
                self.precision,
                str(self.device),
                self.model.decision_head.bf16_compute,
                compiled.state,
                starts,
                parameter_fingerprint(self.model),
                getattr(self.execution_policy, "name", "native"),
                getattr(geometry, "signature", None),
            )
        )
        cached = self.cache.get(key) if key is not None else None
        if cached is not None:
            return cached, True
        hidden, layers = self._segment(torch.tensor([compiled.state], device=self.device), 0, None, geometry)
        memory = {start: self.model.decision_head.prepare_memory(hidden[0, start:]) for start in starts}
        if key is None:
            return Prefix(layers, state_bytes((layers, hidden, memory)), hidden, memory), False
        return self.cache.put(key, layers, hidden, memory), False

    def _segment(self, ids, position, prefix, geometry, terminal=False):
        def execute(inputs):
            tokens, states = inputs
            return self.model.language_model.segment_forward(
                tokens, position, states, terminal_query=terminal, execution_policy=geometry
            )

        if self.graph_cache is None:
            return execute((ids, prefix))
        return self.graph_cache.run_tensors(
            "segment",
            (position, terminal, getattr(geometry, "signature", None)),
            (ids, prefix),
            execute,
            retained=getattr(geometry, "retained", ()),
        )

    def _shared(self, compiled, chunk, prefix, cancelled=None, geometry=None):
        records = {index: dict(branch=branch, layers=prefix.layers, pieces=[]) for index, branch in chunk}
        for stage in range(3):
            groups = defaultdict(list)
            for index, record in records.items():
                branch = record["branch"]
                bounds = (0, branch.option_starts[0], branch.decision_position, len(branch.suffix))
                begin, end = bounds[stage : stage + 2]
                groups[(begin, end)].append(index)
            for (begin, end), indices in groups.items():
                if cancelled and cancelled():
                    raise TimeoutError("Inference cancelled between segment batches")
                ids = torch.tensor(
                    [records[index]["branch"].suffix[begin:end] for index in indices], device=self.device
                )
                layers = batch_states([records[index]["layers"] for index in indices])
                hidden, states = self._segment(ids, len(compiled.state) + begin, layers, geometry, stage == 2)
                for row, index in enumerate(indices):
                    records[index]["layers"] = state_row(states, row)
                    records[index]["pieces"].append(hidden[row])
        outputs = {}
        for index, record in records.items():
            if cancelled and cancelled():
                raise TimeoutError("Inference cancelled before readout")
            branch = record["branch"]
            outputs[index] = self._readout(
                torch.cat(record["pieces"], 0),
                branch,
                0,
                state_hidden=prefix.hidden[0, branch.state_evidence_start :],
                prepared_memory=prefix.memory[branch.state_evidence_start],
            )
        return outputs

    def logits(self, compiled, tenant="local", mode=None, cancelled=None):
        mode = mode or self.mode
        if self.execution_policy is not None:
            self.execution_policy.prepare(self.model, self.precision)
        if self.runtime_profile is not None and not self.runtime_profile.allows(compiled):
            # The fallback is explicit and does not count as optimized parity.
            fallback = Engine(
                self.model,
                self.compiler,
                "reference",
                self.batch_tokens,
                0,
                identity=self.identity,
                precision=self.precision,
                max_attention_bytes=self.max_attention_bytes,
                max_logit_elements=self.max_logit_elements,
                reference_batching=False,
            )
            values = fallback.logits(compiled, tenant, cancelled=cancelled)
            self.last_stats = {**fallback.last_stats, "runtime_profile_fallback": True}
            return values
        if mode == "shared" and (torch.is_grad_enabled() or self.model.training):
            raise ValueError("Shared prefix inference requires eval mode without gradients")
        if sum(map(len, compiled.candidate_order.values())) > self.max_logit_elements:
            raise ValueError("Decision logits exceed allocation budget")
        chunks = list(self._chunks(compiled, mode))
        self.last_stats = {}
        graph_before = dict(self.graph_cache.stats) if self.graph_cache is not None else {}
        outputs = {}
        hit = False
        prepared_prefixes = {}
        prefix_hits = []
        self._sync()
        started = time.perf_counter()
        try:
            with self._autocast():
                if cancelled and cancelled():
                    raise TimeoutError("Inference cancelled before prefill")
                visual = any(branch.visual_inputs is not None for branch in compiled.branches)
                prefix, hit = (
                    self._prefill(compiled, tenant)
                    if mode == "shared" and not visual and self.execution_policy is None
                    else (None, False)
                )
                for chunk in chunks:
                    # Canonical geometries may differ between chunks. Only the
                    # current chunk borrows prefix state outside the bounded
                    # caches; evicted prefixes must not accumulate per request.
                    prepared_prefixes.clear()
                    if cancelled and cancelled():
                        raise TimeoutError("Inference cancelled between batches")
                    if mode == "shared" and not visual and self.execution_policy is not None:
                        from gs1.inference.state import branch_geometry

                        groups = defaultdict(list)
                        for index, branch in chunk:
                            groups[branch_geometry([(compiled.state, branch)])].append((index, branch))
                        for key, group in groups.items():
                            if key not in prepared_prefixes:
                                geometry = self.execution_policy.segment_geometry(
                                    self.model, compiled.state, group[0][1]
                                )
                                prepared, cached = self._prefill(compiled, tenant, geometry)
                                prepared_prefixes[key] = prepared, geometry
                                prefix_hits.append(cached)
                            prepared, geometry = prepared_prefixes[key]
                            outputs.update(self._shared(compiled, group, prepared, cancelled, geometry))
                        hit = all(prefix_hits)
                    elif prefix is not None:
                        outputs.update(self._shared(compiled, chunk, prefix, cancelled))
                    else:
                        rows = [(compiled.state, branch) for _, branch in chunk]
                        pad = self.compiler.tokenizer.pad_token_id or 0

                        def native():
                            return branch_logits(
                                self.model,
                                rows,
                                pad_token_id=pad,
                                readout=self._readout,
                                execution_policy=self.execution_policy,
                            )

                        values = (
                            self.graph_cache.run(rows, native, pad_token_id=pad, cancelled=cancelled)
                            if self.graph_cache is not None
                            else native()
                        )
                        outputs.update(
                            {index: value for (index, _), value in zip(chunk, values, strict=True)}
                        )
        finally:
            self._sync()
        graph_delta = (
            {key: value - graph_before.get(key, 0) for key, value in self.graph_cache.stats.items()}
            if self.graph_cache is not None
            else {}
        )
        self.last_stats = dict(
            engine=mode,
            cache_hit=hit,
            cache_bytes=self.cache.bytes,
            execution_ms=(time.perf_counter() - started) * 1000,
            graph=graph_delta,
            graph_bytes=self.graph_cache.bytes if self.graph_cache is not None else 0,
        )
        return assemble_logits(compiled, [outputs[index] for index in range(len(compiled.branches))])

    def predict(self, compiled, tenant="local", *, mode=None, cancelled=None, calibration_active=True):
        logits = self.logits(compiled, tenant, mode, cancelled)
        started = time.perf_counter()
        result = {
            qid: self.model.probabilities(value, calibration_active=calibration_active)
            for qid, value in logits.items()
        }
        self._sync()
        self.last_stats["probability_ms"] = (time.perf_counter() - started) * 1000
        return result

    @torch.inference_mode()
    def evaluate(self, request, resolved_model="GS1", tenant="local", cancelled=None):
        if isinstance(request, dict):
            from gs1.model.schema import DecisionRequest

            request = DecisionRequest.model_validate({"model": resolved_model, **request})
        started = time.perf_counter()
        compiled = self.compiler.compile(request)
        compiled_at = time.perf_counter()
        values = self.predict(compiled, tenant, cancelled=cancelled)
        response = DecisionResponse(
            model=resolved_model,
            answers={
                qid: make_answer(request.questions[qid], value.p_final.tolist())
                for qid, value in values.items()
            },
            usage=Usage(input_tokens=compiled.input_tokens, output_tokens=0),
        )
        validate_response(request, response)
        self.last_stats.update(
            compile_ms=(compiled_at - started) * 1000, total_ms=(time.perf_counter() - started) * 1000
        )
        return response
