"""Generate GS1 arithmetic/graph profiles against the native single-branch oracle."""

import time
from pathlib import Path

import torch

from benchmark.inference import _peak_rss, requests
from benchmark.util import now, write_json
from gs1.inference.arithmetic import NativeRowGeometry
from gs1.inference.cuda_graph import NativeGraphCache
from gs1.inference.engine import Engine
from gs1.inference.profile import SCHEMA, request_signature, runtime_binding
from gs1.model.schema import make_answer


def compare(model, request, actual, expected, label, tolerance):
    atol, rtol, tv_limit = tolerance
    drift = tv = 0.0
    passed = set(actual) == set(expected)
    for key in expected:
        a, b = actual[key], expected[key]
        passed &= bool(torch.allclose(a, b, atol=atol, rtol=rtol))
        drift = max(drift, float((a - b).abs().max()))
        left, right = model.probabilities(a), model.probabilities(b)
        distance = float((left.p_final - right.p_final).abs().sum() / 2)
        tv = max(tv, distance)
        passed &= distance <= tv_limit
        for field in ("log_p_raw", "p_raw", "entropy", "margin", "temperature", "log_p_final", "p_final"):
            passed &= bool(torch.allclose(getattr(left, field), getattr(right, field), atol=atol, rtol=rtol))
        left_answer = make_answer(request.questions[key], left.p_final.tolist())
        right_answer = make_answer(request.questions[key], right.p_final.tolist())
        if request.questions[key].type == "choice":
            passed &= left_answer.choice == right_answer.choice
        elif request.questions[key].type == "score":
            passed &= abs(left_answer.score - right_answer.score) <= atol + rtol * abs(right_answer.score)
    return dict(case=label, passed=bool(passed), max_logit_drift=drift, max_probability_tv=tv)


def qualify_runtime(
    artifact,
    destination,
    profile_out,
    *,
    device="cuda",
    precision="bf16",
    repeats=10,
    warmup=2,
    workload=None,
    graphs=True,
    graph_cache_mb=4096,
    graph_cache_entries=128,
    cpu_threads=2,
):
    if repeats < 1 or warmup < 0:
        raise ValueError("Require positive repeats and nonnegative warmup")
    torch.set_num_threads(cpu_threads)
    loaded = Engine.from_pretrained(artifact, device=device, precision=precision)
    model, compiler, metadata = loaded.model, loaded.compiler, loaded.identity
    tolerance = (
        (2e-3, 1e-2, 5e-3) if precision == "bf16" or model.decision_head.bf16_compute else (1e-5, 1e-4, 1e-4)
    )
    settings = dict(
        batch_tokens=metadata["limits"]["batch_tokens"],
        precision=precision,
        max_attention_bytes=metadata["limits"]["max_attention_bytes"],
        max_logit_elements=metadata["limits"]["max_logit_elements"],
    )
    oracle = loaded
    fixtures = [(case, request, compiler.compile(request)) for case, request in requests(workload)]
    profile = dict(
        schema=SCHEMA,
        created_at=now(),
        arithmetic="native_rows",
        signatures=[request_signature(c) for _, _, c in fixtures],
        engines={},
        warmup_requests=[r.model_dump(mode="json") for _, r, _ in fixtures],
    )
    report = dict(
        created_at=now(),
        checkpoint_id=metadata["checkpoint_id"],
        precision=precision,
        head_compute_dtype=model.config.head_compute_dtype,
        reference="single-branch native GS1",
        tolerance=dict(atol=tolerance[0], rtol=tolerance[1], tv=tolerance[2]),
        engines={},
        measurements=[],
    )
    with torch.inference_mode():
        expected = [oracle.logits(compiled) for _, _, compiled in fixtures]
        for mode in ("reference", "shared"):
            policy = NativeRowGeometry()
            engine = Engine(model, compiler, mode=mode, execution_policy=policy, **settings)
            checks = []
            for index, (case, request, compiled) in enumerate(fixtures):
                actual = engine.logits(compiled)
                checks.append(compare(model, request, actual, expected[index], f"eager-{index}", tolerance))
                if mode == "shared":
                    checks.append(
                        compare(
                            model,
                            request,
                            engine.logits(compiled),
                            expected[index],
                            f"eager-warm-{index}",
                            tolerance,
                        )
                    )
            graph = None
            if graphs and torch.device(device).type == "cuda":
                graph = NativeGraphCache(
                    model,
                    precision,
                    max_bytes=graph_cache_mb * 1024**2,
                    max_entries=graph_cache_entries,
                    execution_policy=policy,
                )
                engine.graph_cache = graph
                for index, (case, request, compiled) in enumerate(fixtures):
                    engine.cache.entries.clear()
                    engine.cache.bytes = 0
                    before = dict(graph.stats)
                    actual = engine.logits(compiled)
                    check = compare(model, request, actual, expected[index], f"graph-cold-{index}", tolerance)
                    check["captures"] = graph.stats["captures"] - before["captures"]
                    check["rejects"] = graph.stats["rejects"] - before["rejects"]
                    check["passed"] &= check["rejects"] == 0 and check["captures"] > 0
                    checks.append(check)
                    checks.append(
                        compare(
                            model,
                            request,
                            engine.logits(compiled),
                            expected[index],
                            f"graph-warm-{index}",
                            tolerance,
                        )
                    )
                graph.allow_capture = False
            for index, (case, request, compiled) in enumerate(fixtures):
                for _ in range(warmup):
                    engine.logits(compiled)
                before = dict(graph.stats) if graph else {}
                samples = []
                if torch.device(device).type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                for _ in range(repeats):
                    start = time.perf_counter()
                    actual = engine.logits(compiled)
                    samples.append((time.perf_counter() - start) * 1000)
                import numpy as np

                report["measurements"].append(
                    dict(
                        engine=mode,
                        workload=case.model_dump(),
                        median_ms=float(np.median(samples)),
                        p95_ms=float(np.percentile(samples, 95)),
                        questions_per_second=case.questions / (float(np.median(samples)) / 1000),
                        telemetry=engine.last_stats,
                        cpu_peak_rss=_peak_rss(),
                        cuda_peak_allocated=torch.cuda.max_memory_allocated(device)
                        if torch.device(device).type == "cuda"
                        else None,
                        cuda_peak_reserved=torch.cuda.max_memory_reserved(device)
                        if torch.device(device).type == "cuda"
                        else None,
                    )
                )
                checks.append(
                    compare(model, request, actual, expected[index], f"measured-{index}", tolerance)
                )
                if graph is not None:
                    checks[-1]["resident_graphs"] = graph.stats["misses"] == before["misses"]
                    checks[-1]["passed"] &= checks[-1]["resident_graphs"]
            passed = bool(checks) and all(c["passed"] for c in checks)
            record = dict(
                passed=passed,
                binding=runtime_binding(metadata, mode, precision),
                checks=checks,
                graphs_passed=graph is not None and passed,
            )
            profile["engines"][mode] = record
            report["engines"][mode] = record
            if graph is not None:
                report["engines"][mode]["graph_statistics"] = dict(graph.stats)
            engine.close()
            # Release this mode's compute copies before constructing the next
            # one; qualification must not hold two replicas' cast weights.
            del engine, graph, policy
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
    report["passed"] = all(e["passed"] for e in profile["engines"].values())
    profile["passed"] = report["passed"]
    warmup_path = Path(profile_out).with_name(Path(profile_out).stem + "-warmup.json")
    write_json(warmup_path, profile["warmup_requests"])
    report["warmup_requests_path"] = str(warmup_path)
    write_json(Path(profile_out), profile)
    write_json(Path(destination), report)
    return report
