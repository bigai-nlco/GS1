"""Source-only GS1 inference benchmarks and numerical qualifications."""

import time
from pathlib import Path

import numpy as np
import torch
from pydantic import Field

from benchmark.util import now, write_json
from gs1.inference.engine import Engine
from gs1.model.config import ModelConfig
from gs1.model.schema import Choice, DecisionRequest, Noul, Score


class WorkloadCase(ModelConfig):
    questions: int = Field(ge=1)
    state_repeats: int = Field(default=16, ge=0)
    description_repeats: int = Field(default=1, ge=1)
    choice_candidates: int = Field(default=3, ge=1, le=256)
    score_levels: int = Field(default=3, ge=2, le=256)


class Workload(ModelConfig):
    cases: list[WorkloadCase] = Field(
        default_factory=lambda: [WorkloadCase(questions=n) for n in (1, 8, 32)], min_length=1
    )


def requests(workload=None):
    for case in Workload.model_validate(workload or {}).cases:
        questions = {}
        for i in range(case.questions):
            if i % 3 == 0:
                questions[str(i)] = Choice(
                    instructions="Choose a feasible job retaining 10 reserve units.",
                    criteria={
                        str(k): f"Job {k} costs {k + 1} energy. " * case.description_repeats
                        for k in range(case.choice_candidates)
                    },
                )
            elif i % 3 == 1:
                questions[str(i)] = Noul(instructions="Can job A preserve 10 reserve units?")
            else:
                questions[str(i)] = Score(
                    instructions="Classify available battery.",
                    criteria=[
                        f"Battery level {k}. " * case.description_repeats for k in range(case.score_levels)
                    ],
                )
        yield (
            case,
            DecisionRequest(
                state=dict(
                    battery=18, reserve=10, log="Cargo manifest routine operations. " * case.state_repeats
                ),
                questions=questions,
                model="qualification",
            ),
        )


def _peak_rss():
    try:
        import resource
        import sys

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    except ImportError:
        return None


def benchmark(
    artifact,
    destination,
    device="cuda",
    precision="bf16",
    repeats=10,
    warmup=2,
    *,
    workload=None,
):
    if repeats < 1 or warmup < 0:
        raise ValueError("Benchmark requires positive repeats and nonnegative warmup")
    loaded = Engine.from_pretrained(artifact, device=device, precision=precision)
    model, compiler, metadata = loaded.model, loaded.compiler, loaded.identity
    modes = ["reference", "shared"]
    engines = {
        mode: Engine(
            model,
            compiler,
            mode,
            batch_tokens=metadata["limits"]["batch_tokens"],
            precision=precision,
            identity=metadata,
            max_attention_bytes=metadata["limits"]["max_attention_bytes"],
            max_logit_elements=metadata["limits"]["max_logit_elements"],
        )
        for mode in modes
    }
    canonical = loaded
    tolerance_precision = "bf16" if precision == "bf16" or model.decision_head.bf16_compute else "fp32"
    atol, rtol, tv_limit = (2e-3, 1e-2, 5e-3) if tolerance_precision == "bf16" else (1e-5, 1e-4, 1e-4)
    checks, timings, unsupported = [], [], []
    fields = ("log_p_raw", "p_raw", "entropy", "margin", "temperature", "log_p_final", "p_final")

    def check(label, actual, expected, engine):
        passed = all(torch.allclose(actual[k], expected[k], atol=atol, rtol=rtol) for k in expected)
        drift = max(float((actual[k] - expected[k]).abs().max()) for k in expected)
        max_tv = 0.0
        statistics_drift = {}
        for key in expected:
            left, right = model.probabilities(actual[key]), model.probabilities(expected[key])
            for field in fields:
                a, b = getattr(left, field), getattr(right, field)
                passed &= bool(torch.allclose(a, b, atol=atol, rtol=rtol))
                statistics_drift[field] = max(statistics_drift.get(field, 0), float((a - b).abs().max()))
            tv = float((left.p_final - right.p_final).abs().sum() / 2)
            max_tv = max(max_tv, tv)
            passed &= tv <= tv_limit
        checks.append(
            dict(
                case=label,
                engine=engine,
                passed=passed,
                max_logit_drift=drift,
                max_probability_tv=max_tv,
                statistics_drift=statistics_drift,
            )
        )

    fixtures = []
    for case, request in requests(workload):
        try:
            compiled = compiler.compile(request)
            for engine in engines.values():
                list(engine._chunks(compiled, engine.mode))
        except ValueError as error:
            unsupported.append(dict(case=case.model_dump(), reason=str(error)))
            continue
        fixtures.append((case, request))
    if not fixtures:
        raise ValueError("No benchmark fixture fits the artifact operating envelope")
    with torch.inference_mode():
        for case, request in fixtures:
            compiled = compiler.compile(request)
            expected = canonical.logits(compiled)
            batched_reference = engines["reference"].logits(compiled)
            for mode, engine in engines.items():
                actual = engine.logits(compiled)
                check("mixed-fanout", actual, expected, mode)
                if mode == "shared":
                    check("batched-reference", actual, batched_reference, mode)
                for qid in list(request.questions)[:3]:
                    alone = request.model_copy(update={"questions": {qid: request.questions[qid]}})
                    check(
                        "standalone-primitive",
                        engine.logits(compiler.compile(alone)),
                        {qid: expected[qid]},
                        mode,
                    )
                reordered = request.model_copy(
                    update={"questions": dict(reversed(list(request.questions.items())))}
                )
                check("sibling-reordering", engine.logits(compiler.compile(reordered)), expected, mode)
                budget = engine.batch_tokens
                engine.batch_tokens = max(len(compiled.state) + len(b.suffix) for b in compiled.branches)
                check("chunk-boundaries", engine.logits(compiled), expected, mode)
                engine.batch_tokens = budget
                for _ in range(warmup):
                    engine.evaluate(request, metadata["resolved_model"])
                for cache_condition in ["cold", "warm"] if mode == "shared" else ["none"]:
                    samples, spans = [], []
                    if str(device).startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(device)
                    for repeat in range(repeats):
                        tenant = f"cold-{repeat}" if cache_condition == "cold" else "local"
                        started = time.perf_counter()
                        engine.evaluate(request, metadata["resolved_model"], tenant=tenant)
                        samples.append((time.perf_counter() - started) * 1000)
                        spans.append(dict(engine.last_stats))
                    timings.append(
                        dict(
                            engine=mode,
                            questions=len(request.questions),
                            attempts=repeats,
                            completed=repeats,
                            coverage=1.0,
                            cache=cache_condition,
                            p50_ms=float(np.median(samples)),
                            p95_ms=float(np.quantile(samples, 0.95)),
                            p99_ms=float(np.quantile(samples, 0.99)),
                            samples_ms=samples,
                            logical_questions_per_second=1000
                            * len(request.questions)
                            / float(np.mean(samples)),
                            spans=spans,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
                            if str(device).startswith("cuda")
                            else None,
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(device)
                            if str(device).startswith("cuda")
                            else None,
                            process_peak_rss_bytes=_peak_rss(),
                        )
                    )
            if "shared" in engines:
                shared = engines["shared"]
                cold = shared.logits(compiled, tenant=f"isolated-{len(request.questions)}")
                missed = not shared.last_stats["cache_hit"]
                warm = shared.logits(compiled, tenant=f"isolated-{len(request.questions)}")
                hit = shared.last_stats["cache_hit"]
                check("cold-warm", warm, cold, "shared")
                checks.append(dict(case="tenant-separation", engine="shared", passed=missed and hit))
    result = dict(
        created_at=now(),
        checkpoint_id=metadata["checkpoint_id"],
        precision=precision,
        head_compute_dtype=model.config.head_compute_dtype,
        tolerance_precision=tolerance_precision,
        reference="single-branch native GS1",
        passed=all(c["passed"] for c in checks),
        checks=checks,
        timings=timings,
        unsupported=unsupported,
        skipped_engines=[] if "shared" in engines else ["shared: requires CUDA"],
        atol=atol,
        rtol=rtol,
        tv_limit=tv_limit,
        release_qualified=False,
        status="bounded parity; exploratory timings, not a deployment SLO qualification",
    )
    write_json(Path(destination), result)
    if not result["passed"]:
        raise ValueError("GS1 engine parity failed; optimized serving remains disabled")
    return result


def main(argv=None):
    import argparse
    import json

    import yaml

    from gs1.model.util import resolve_source

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--runtime-profile-out", type=Path)
    parser.add_argument("--graph-cache-mb", type=int, default=16384)
    parser.add_argument("--graph-cache-entries", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    artifact = resolve_source(args.model)
    torch.set_num_threads(args.cpu_threads)
    workload = yaml.safe_load(args.workload.read_text()) if args.workload else None
    if args.runtime_profile_out:
        from benchmark.runtime import qualify_runtime

        result = qualify_runtime(
            artifact,
            args.out,
            args.runtime_profile_out,
            device=args.device,
            precision=args.precision,
            repeats=args.repeats,
            warmup=args.warmup,
            workload=workload,
            graph_cache_mb=args.graph_cache_mb,
            graph_cache_entries=args.graph_cache_entries,
            cpu_threads=args.cpu_threads,
        )
    else:
        result = benchmark(
            artifact, args.out, args.device, args.precision, args.repeats, args.warmup, workload=workload
        )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
