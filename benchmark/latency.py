"""Measure complete warm HF/eager/graph decisions with identical GS1 weights."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmark.inference import requests
from benchmark.runtime import compare
from benchmark.util import now, write_json
from gs1 import Engine
from gs1.inference.arithmetic import NativeRowGeometry
from gs1.inference.cuda_graph import NativeGraphCache
from gs1.inference.profile import runtime_binding


def fixture(compiler, context, questions):
    """Find the longest deterministic repeated-state fixture within the budget."""
    _, template = next(requests({"cases": [{"questions": questions, "state_repeats": 0}]}))

    def build(repeats):
        request = template.model_copy(
            update={
                "state": {
                    "battery": 18,
                    "reserve": 10,
                    "log": "Cargo manifest routine operations. " * repeats,
                }
            }
        )
        return request, compiler.compile(request)

    request, compiled = build(0)
    if compiled.longest_path > context:
        raise ValueError("Target context is shorter than the question and options")
    low, high = 0, context
    while low < high:
        middle = (low + high + 1) // 2
        try:
            _, candidate = build(middle)
        except ValueError:
            high = middle - 1
            continue
        if candidate.longest_path <= context:
            low = middle
        else:
            high = middle - 1
    request, compiled = build(low)
    for _ in range(context - compiled.longest_path):
        candidate = request.model_copy(
            update={"state": {**request.state, "log": request.state["log"] + " x"}}
        )
        encoded = compiler.compile(candidate)
        if encoded.longest_path > context:
            break
        request, compiled = candidate, encoded
    return request, compiled


def measure(function, device, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = function()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
        del result
    return dict(
        samples_ms=samples,
        p50_ms=float(np.quantile(samples, 0.5)),
        p95_ms=float(np.quantile(samples, 0.95)),
        p99_ms=float(np.quantile(samples, 0.99)),
        mean_ms=float(np.mean(samples)),
        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3,
    )


def benchmark(
    source,
    output,
    *,
    device="cuda",
    contexts=(256, 1024),
    question_counts=(1, 8),
    precision="bf16",
    repeats=30,
    warmup=5,
    graph_cache_mb=8192,
    graph_cache_entries=32,
    cpu_threads=2,
    memory_fraction=0.9,
):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("This latency benchmark requires a CUDA device")
    device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
    if repeats < 2 or warmup < 1 or not 0 < memory_fraction <= 1:
        raise ValueError("Require at least two measurements, a warmup and a valid memory fraction")
    output = Path(output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite measurements: {output}")
    torch.set_num_threads(cpu_threads)
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
    reference = Engine.from_pretrained(source, device=str(device), precision=precision)
    model, compiler = reference.model, reference.compiler
    tolerance = (
        (2e-3, 1e-2, 5e-3) if precision == "bf16" or model.decision_head.bf16_compute else (1e-5, 1e-4, 1e-4)
    )
    report = dict(
        schema="gs1.inference-latency.v1",
        created_at=now(),
        model=Path(str(source)).name,
        checkpoint_id=reference.identity["checkpoint_id"],
        precision=precision,
        timing="warm complete in-process decision: request validation/compilation, forward, probabilities and typed response; CUDA synchronized; excludes loading, graph capture, HTTP and SDK transport",
        baseline="hf_eager (model.decide); same stored weights, candidate sets and compute precision",
        hardware=dict(
            name=torch.cuda.get_device_name(device),
            total_memory_gib=torch.cuda.get_device_properties(device).total_memory / 1024**3,
            cpu_threads=cpu_threads,
            memory_fraction=memory_fraction,
        ),
        runtime=runtime_binding(reference.identity, "reference", precision),
        warmup=warmup,
        repeats=repeats,
        measurements=[],
        checks=[],
    )
    try:
        with torch.inference_mode():
            for context in contexts:
                for count in question_counts:
                    request, compiled = fixture(compiler, context, count)
                    raw = request.model_dump(mode="json")
                    expected = reference.logits(compiled)
                    eager_response = reference.evaluate(raw).model_dump(mode="json")
                    hf_response = model.decide(raw["state"], raw["questions"], precision=precision)
                    if eager_response["answers"] != hf_response["answers"]:
                        raise ValueError("HF and unbatched eager decisions differ")
                    geometry = dict(
                        context_target=context,
                        context=compiled.longest_path,
                        questions=count,
                        input_tokens=compiled.input_tokens,
                        state_tokens=len(compiled.state),
                        branch_lengths=[len(compiled.state) + len(b.suffix) for b in compiled.branches],
                        candidate_counts=[len(b.candidates) for b in compiled.branches],
                    )
                    for name, function in (
                        (
                            "hf_eager",
                            lambda: model.decide(raw["state"], raw["questions"], precision=precision),
                        ),
                        ("gs1_eager", lambda: reference.evaluate(raw)),
                    ):
                        row = dict(**geometry, path=name, **measure(function, device, warmup, repeats))
                        row["questions_per_second"] = count * 1000 / row["mean_ms"]
                        report["measurements"].append(row)
                        print(
                            json.dumps(
                                {k: row[k] for k in ("path", "context", "questions", "p50_ms", "p95_ms")}
                            ),
                            flush=True,
                        )
                    policy = NativeRowGeometry()
                    graph = NativeGraphCache(
                        model,
                        precision,
                        max_bytes=graph_cache_mb * 1024**2,
                        max_entries=graph_cache_entries,
                        execution_policy=policy,
                    )
                    optimized = Engine(
                        model,
                        compiler,
                        precision=precision,
                        execution_policy=policy,
                        graph_cache=graph,
                        identity=reference.identity,
                        batch_tokens=model.config.inference_limits["batch_tokens"],
                        max_attention_bytes=model.config.inference_limits["max_attention_bytes"],
                        max_logit_elements=model.config.inference_limits["max_logit_elements"],
                    )
                    try:
                        actual = optimized.logits(compiled)
                        check = compare(model, request, actual, expected, f"{context}-{count}", tolerance)
                        check.update(captures=graph.stats["captures"], rejects=graph.stats["rejects"])
                        if not check["passed"] or not check["captures"] or check["rejects"]:
                            raise ValueError(f"CUDA Graph numerical qualification failed: {check}")
                        graph.allow_capture = False
                        before = dict(graph.stats)
                        row = dict(
                            **geometry,
                            path="gs1_cuda_graph",
                            **measure(lambda engine=optimized: engine.evaluate(raw), device, warmup, repeats),
                        )
                        check["resident_graphs"] = graph.stats["misses"] == before["misses"]
                        check["replay_hits"] = graph.stats["hits"] - before["hits"]
                        check["passed"] &= check["resident_graphs"] and check["replay_hits"] > 0
                        if not check["passed"]:
                            raise ValueError(f"Measured calls did not replay resident graphs: {check}")
                        row.update(
                            questions_per_second=count * 1000 / row["mean_ms"],
                            graph_statistics=dict(graph.stats),
                        )
                        report["checks"].append(check)
                        report["measurements"].append(row)
                        print(
                            json.dumps(
                                {k: row[k] for k in ("path", "context", "questions", "p50_ms", "p95_ms")}
                            ),
                            flush=True,
                        )
                    finally:
                        optimized.close()
                        del optimized, graph, policy
                        gc.collect()
                        torch.cuda.empty_cache()
                    write_json(output, {**report, "complete": False})
        report.update(complete=True, passed=all(c["passed"] for c in report["checks"]))
        write_json(output, report)
        return report
    finally:
        reference.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--contexts", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--questions", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--graph-cache-mb", type=int, default=8192)
    parser.add_argument("--graph-cache-entries", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    args = parser.parse_args(argv)
    benchmark(
        args.model,
        args.out,
        device=args.device,
        contexts=args.contexts,
        question_counts=args.questions,
        precision=args.precision,
        repeats=args.repeats,
        warmup=args.warmup,
        graph_cache_mb=args.graph_cache_mb,
        graph_cache_entries=args.graph_cache_entries,
        cpu_threads=args.cpu_threads,
        memory_fraction=args.memory_fraction,
    )


if __name__ == "__main__":
    main()
