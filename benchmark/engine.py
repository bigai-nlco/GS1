"""Run selected suites through a typed adapter and retain every attempted decision."""

import math
import time
from dataclasses import asdict

from benchmark.util import now, write_json
from gs1.model.schema import aligned_probabilities, validate_response
from gs1.model.util import file_hash


def score_decision(case, qid, response):
    target = case.targets[qid]
    values = aligned_probabilities(case.request.questions[qid], response.answers[qid])
    probabilities = dict(zip(target.candidate_ids, values, strict=True))
    predicted = min(probabilities, key=lambda key: (-probabilities[key], key))
    if case.native is not None:
        # The frozen protocol uses the declared Choice and yes at p=0.5.
        answer = response.answers[qid]
        predicted = answer.choice if answer.type == "choice" else "true" if answer.noul >= 0.5 else "false"
    measured = target.distribution is not None
    result = {
        "valid": True,
        "predicted": predicted,
        "probabilities": probabilities,
        "confidence": max(values),
        "correct": predicted == target.label if target.label is not None else None,
        "accuracy_credit": (
            float(predicted == target.label)
            if target.label is not None
            else target.distribution[target.candidate_ids.index(predicted)]
            if measured
            else None
        ),
    }
    if measured:
        tau = target.distribution
        result["expected_correctness"] = tau[target.candidate_ids.index(predicted)]
        result["posterior_squared_error"] = sum((p - t) ** 2 for p, t in zip(values, tau, strict=True))
        result["brier"] = result["posterior_squared_error"] + 1 - sum(t * t for t in tau)
        result["nll"] = (
            -sum(t * math.log(p) for p, t in zip(values, tau, strict=True) if t)
            if all(p > 0 for p, t in zip(values, tau, strict=True) if t)
            else "infinity"
        )
    if case.task is not None:
        from jevbench.scoring import score_task

        public_probs = (
            {"no": values[0], "yes": values[1]} if case.task.question["type"] == "noul" else probabilities
        )
        scored = score_task(public_probs, case.task)
        result.update(scored, probs=scored.get("probs"), task_id=case.task.id, probs_source="native")
    return result


def summarize_decisions(rows, *, ece_bins=10):
    measured = [row for row in rows if row["target"]["distribution"] is not None]
    calibrated = [row for row in measured if row["valid"]]
    hard = [row for row in measured if row["target"]["label"] is not None]
    bins = [
        {"lo": i / ece_bins, "hi": (i + 1) / ece_bins, "n": 0, "confidence_sum": 0.0, "credit_sum": 0.0}
        for i in range(ece_bins)
    ]
    for row in calibrated:
        group = bins[min(ece_bins - 1, int(row["confidence"] * ece_bins))]
        group["n"] += 1
        group["confidence_sum"] += row["confidence"]
        group["credit_sum"] += row["accuracy_credit"]
    calibration = []
    ece = 0.0
    for group in bins:
        n = group["n"]
        confidence = group["confidence_sum"] / n if n else None
        accuracy = group["credit_sum"] / n if n else None
        if n:
            ece += n / len(calibrated) * abs(confidence - accuracy)
        calibration.append(
            {
                "lo": group["lo"],
                "hi": group["hi"],
                "n": n,
                "mean_confidence": confidence,
                "accuracy": accuracy,
            }
        )
    credit = sum(row["accuracy_credit"] or 0 for row in measured)
    soft_n = sum(row["target"]["label"] is None for row in measured)
    return {
        "n_attempted": len(rows),
        "n_valid": sum(row["valid"] for row in rows),
        "n_failed": sum(not row["valid"] for row in rows),
        "n_scorable": len(measured),
        "n_correct": sum(bool(row["correct"]) for row in hard),
        "accuracy_credit": credit,
        "accuracy": credit / len(measured) if measured else None,
        "accuracy_kind": "expected_correctness"
        if soft_n == len(measured) and soft_n
        else "mixed_label_and_expected_correctness"
        if soft_n
        else "label_accuracy",
        "hard_label_accuracy": sum(bool(row["correct"]) for row in hard) / len(hard) if hard else None,
        "calibration_n": len(calibrated),
        "brier": sum(row["brier"] for row in calibrated) / len(calibrated) if calibrated else None,
        "posterior_squared_error": sum(row["posterior_squared_error"] for row in calibrated) / len(calibrated)
        if calibrated
        else None,
        "expected_correctness": sum(row.get("expected_correctness", 0) for row in measured) / len(measured)
        if measured
        else None,
        "ece": ece if calibrated else None,
        "ece_bins": ece_bins,
        "reliability": calibration,
    }


async def evaluate_suites(suites, adapter, *, progress=None):
    rows, evidence = [], []
    started = time.perf_counter()
    try:
        await adapter.start()
        for suite in suites:
            if progress:
                progress(f"Evaluating {suite.name}: {len(suite.cases)} records")
            for index, case in enumerate(suite.cases, 1):
                request = case.request.model_copy(update={"model": adapter.model_name})
                tick = time.perf_counter()
                attempt = {
                    "suite": suite.name,
                    "record_id": case.id,
                    "request": request.model_dump(mode="json"),
                }
                try:
                    response = await adapter.evaluate(request, tenant="evaluation")
                    validate_response(request, response)
                    # Score the complete response before accepting any sibling decision.
                    decisions = {qid: score_decision(case, qid, response) for qid in request.questions}
                    attempt["response"] = response.model_dump(mode="json")
                    model = response.model
                except Exception as error:
                    attempt["error"] = f"{type(error).__name__}: {error}"
                    model = adapter.model_name
                    decisions = {
                        qid: {
                            "valid": False,
                            "predicted": None,
                            "correct": False if target.label is not None else None,
                            "accuracy_credit": 0 if target.distribution is not None else None,
                            "error": attempt["error"],
                        }
                        for qid, target in case.targets.items()
                    }
                elapsed = time.perf_counter() - tick
                attempt.update(client_ms=elapsed * 1000, telemetry=dict(getattr(adapter, "last_stats", {})))
                evidence.append(attempt)
                for qid, decision in decisions.items():
                    rows.append(
                        {
                            "suite": suite.name,
                            "record_id": case.id,
                            "question_id": qid,
                            "target": asdict(case.targets[qid]),
                            "model": model,
                            "ok": decision["valid"],
                            "latency_s": elapsed,
                            "task_id": case.id,
                            "probs": None,
                            **decision,
                        }
                    )
                if progress and index % 100 == 0:
                    progress(f"{suite.name}: {index}/{len(suite.cases)} records")
        backend = dict(adapter.identity)
    finally:
        await adapter.close()
    aadi_suites = [
        suite
        for suite in suites
        if suite.name in {"aa_omniscience", "hle_multichoice", "when2call", "wildguard_test"}
    ]
    return {
        "schema": "gs1.evaluation",
        "created_at": now(),
        "backend": backend,
        "resolved_models": sorted({row["model"] for row in rows if row["valid"]}),
        "datasets": {suite.name: suite.source for suite in suites},
        "benchmarks": summarize_benchmarks(suites, rows, evidence),
        "protocol": {
            "accuracy": "hard-label match, or target mass at predicted label for soft-only targets; failures=0",
            "ties": "lexicographically_smallest_candidate",
            "brier": "expected_full_multiclass_sum",
            "ece_bins": 10,
            "calibration_population": "valid measured decisions",
            "retries": 0,
            "scope": "public benchmark suites and declared local test files; no official ranking",
            **(
                {
                    "aadi": {
                        "scoring": "official component weights and native metrics",
                        "scope": "public adaptation; see benchmark limitations",
                    }
                }
                if aadi_suites
                else {}
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "results": rows,
        "evidence": evidence,
    }


def summarize_benchmarks(suites, rows, evidence=None):
    """Score complete decision rows, including rows merged from evaluation workers."""
    benchmarks = {}
    for suite in suites:
        if not suite.name.startswith("jevbench/"):
            benchmarks[suite.name] = summarize_decisions([row for row in rows if row["suite"] == suite.name])
    from benchmark.aadi.scoring import summarize_aadi

    aadi_metrics, aadi_index = summarize_aadi(suites, rows)
    for name, native in aadi_metrics.items():
        benchmarks[name].update(native)
        benchmarks[name]["native_metric"] = (
            "average_f1"
            if name == "wildguard_test"
            else "mean_accuracy_and_non_hallucination"
            if name == "aa_omniscience"
            else "accuracy"
        )
        benchmarks[name]["native_score"] = (
            native["average_f1"]
            if name == "wildguard_test"
            else native["component_score"]
            if name == "aa_omniscience"
            else native["accuracy"]
        )
    if aadi_index is not None:
        benchmarks["aadi"] = aadi_index
    jev_suites = [suite for suite in suites if suite.name.startswith("jevbench/")]
    if jev_suites:
        from jevbench.summarize import summarize

        jev_rows = [row for row in rows if row["suite"].startswith("jevbench/")]
        official = summarize([case.task for suite in jev_suites for case in suite.cases], jev_rows)
        benchmarks["jevbench"] = {
            **summarize_decisions(jev_rows),
            "accuracy": official["accuracy"],
            "brier": official["brier_mean"],
            "ece": official["ece"]["ece"] if official["ece"] else None,
            "official_summary": official,
            "splits": {
                suite.name.split("/")[1]: summarize_decisions(
                    [row for row in jev_rows if row["suite"] == suite.name]
                )
                for suite in jev_suites
            },
        }
    return benchmarks


def accuracy_manifest(report, result_path=None):
    return {
        "schema": "gs1.evaluation.accuracy",
        "created_at": report["created_at"],
        "evaluation_json": str(result_path.resolve()) if result_path else None,
        "evaluation_sha256": file_hash(result_path) if result_path else None,
        "backend": report["backend"],
        "resolved_models": report["resolved_models"],
        "datasets": report["datasets"],
        "protocol": report["protocol"],
        "benchmarks": {
            name: {key: value for key, value in metric.items() if key != "official_summary"}
            for name, metric in report["benchmarks"].items()
        },
        "decisions": [
            {
                key: row.get(key)
                for key in (
                    "suite",
                    "record_id",
                    "question_id",
                    "model",
                    "valid",
                    "predicted",
                    "correct",
                    "accuracy_credit",
                    "error",
                )
            }
            for row in report["results"]
        ],
    }


def export_results(report, *, output_json=None, manifest_path=None):
    paths = [path.resolve() for path in (output_json, manifest_path) if path is not None]
    if len(set(paths)) != len(paths):
        raise ValueError("Evaluation JSON and accuracy manifest need distinct paths")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Use a new output file: {path}")
    if output_json:
        write_json(output_json, report)
    if manifest_path:
        write_json(manifest_path, accuracy_manifest(report, output_json))


def format_table(report):
    from benchmark.datasets import SUITES

    entries = []
    jev = report["benchmarks"].get("jevbench")
    if jev:
        entries.extend((f"JevBench {split.title()}", metric) for split, metric in jev["splits"].items())
        entries.append(("JevBench All", jev))
    entries.extend(
        (SUITES[name], report["benchmarks"][name])
        for name in SUITES
        if name != "jevbench" and name in report["benchmarks"]
    )
    cells = [["Benchmark", "Score", "Valid / Total", "Brier", "ECE", "Errors"]]
    for label, metric in entries:
        cells.append(
            [
                label,
                f"{metric.get('native_score', metric['accuracy']):.2%}"
                if metric.get("native_score", metric["accuracy"]) is not None
                else "-",
                f"{metric['n_valid']} / {metric['n_attempted']}",
                f"{metric['brier']:.6f}" if metric["brier"] is not None else "-",
                f"{metric['ece']:.6f}" if metric["ece"] is not None else "-",
                str(metric["n_failed"]),
            ]
        )
    widths = [max(len(row[col]) for row in cells) for col in range(len(cells[0]))]
    divider = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [divider]
    for index, row in enumerate(cells):
        lines.append(
            "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)) + " |"
        )
        if index == 0:
            lines.append(divider)
    lines.append(divider)
    aadi = report["benchmarks"].get("aadi")
    if aadi:
        value = f"{aadi['score']:.2f} / 100" if aadi["score"] is not None else "incomplete"
        lines.append(f"AA Decision Index public adaptation: {value}")
        lines.append(
            "WildGuardTest score is average F1; local refusal filter differs from the unpublished official subset."
        )
    if any(
        row["target"]["label"] is None and row["target"]["distribution"] is not None
        for row in report["results"]
    ):
        lines.append("Soft-target accuracy is expected correctness (target mass at the predicted label).")
    return "\n".join(lines)
