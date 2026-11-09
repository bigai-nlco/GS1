"""Evaluation inputs, independent of training corpora and verifier receipts."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gs1.model.schema import DecisionRequest, candidate_ids
from gs1.model.util import digest, file_hash

SUITES = {
    "jevbench": "JevBench",
    "toolace": "ToolACE",
    "agnews": "AG News",
    "aa_omniscience": "AA-Omniscience-Decision (public)",
    "hle_multichoice": "HLE-MultiChoice",
    "wildguard_test": "WildGuardTest (local refusal filter)",
    "when2call": "When2Call",
}

# AADI (Artificial Analysis Decision Index v1.0) component datasets
AADI_SUITES = ("aa_omniscience", "hle_multichoice", "wildguard_test", "when2call")

DEFAULT_SUITES = ("jevbench",)
JEVBENCH_SPLITS = ("easy", "original", "hard")


@dataclass
class Target:
    candidate_ids: list[str]
    distribution: list[float] | None
    label: str | None
    kind: str


@dataclass
class Case:
    id: str
    request: DecisionRequest
    targets: dict[str, Target]
    task: Any = None
    native: dict | None = None


@dataclass
class Suite:
    name: str
    cases: list[Case]
    source: dict


def parse_target(value, question):
    labels = candidate_ids(question)
    if value is None:
        return Target(labels, None, None, "unmeasured")
    if not isinstance(value, dict):
        value = {"label": value}
    if value.get("candidate_ids", labels) != labels:
        raise ValueError("Target candidate order must match the question")
    label = value.get("label", value.get("expected"))
    if isinstance(label, bool) and question.type == "noul":
        label = "true" if label else "false"
    elif label is not None:
        label = str(label)
    if label is not None and label not in labels:
        raise ValueError(f"Target label {label!r} is outside the candidate set")
    distribution = value.get("distribution")
    if distribution is None and "probabilities" in value:
        probabilities = value["probabilities"]
        if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
            raise ValueError("Target probability keys must match the candidates")
        distribution = [probabilities[key] for key in labels]
    if distribution is None:
        if label is None:
            raise ValueError("A target needs a label or a distribution")
        distribution = [float(key == label) for key in labels]
    if (
        not isinstance(distribution, list)
        or len(distribution) != len(labels)
        or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in distribution)
        or abs(sum(distribution) - 1) > 1e-6
    ):
        raise ValueError("Target must be a complete finite probability distribution summing to one")
    one_hot = distribution.count(1) == 1 and distribution.count(0) == len(labels) - 1
    kind = value.get("target_kind", "observed_label" if one_hot else "exact_distribution")
    if kind == "observed_label" and not one_hot:
        raise ValueError("An observed-label target must be one-hot")
    if one_hot and label is not None and label != labels[distribution.index(1)]:
        raise ValueError("Target label conflicts with its one-hot distribution")
    if label is None and one_hot:
        label = labels[distribution.index(1)]
    return Target(labels, distribution, label, kind)


def parse_case(row, fallback_id):
    if not isinstance(row, dict):
        raise ValueError("Each JSONL record must be an object")
    if "request" in row:
        body = dict(row["request"])
        targets = row["targets"]
    elif isinstance(row.get("question"), dict):
        body = {"state": row["state"], "questions": {"decision": row["question"]}}
        expected = row.get("expected")
        if body["questions"]["decision"]["type"] == "noul" and expected in ("no", "yes"):
            expected = "true" if expected == "yes" else "false"
        targets = {"decision": expected}
    else:
        body = {"state": row["state"], "questions": row["questions"]}
        targets = row["targets"]
    body["model"] = "evaluation"
    request = DecisionRequest.model_validate(body)
    if not isinstance(targets, dict) or targets.keys() != request.questions.keys():
        raise ValueError("Targets must cover exactly the request's questions")
    case_id = str(row.get("id", fallback_id))
    if not case_id:
        raise ValueError("Evaluation record IDs must be nonempty")
    return Case(
        case_id,
        request,
        {qid: parse_target(targets[qid], q) for qid, q in request.questions.items()},
        native=row.get("native"),
    )


def load_jsonl(path):
    cases, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                case = parse_case(json.loads(line), f"row-{line_number}")
                if case.id in seen:
                    raise ValueError(f"Duplicate record ID: {case.id}")
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError(f"Evaluation dataset is empty: {path}")
    return cases


def make_suite(name, path, cases, limit, provenance=None):
    selected = cases[:limit] if limit is not None else cases
    return Suite(
        name,
        selected,
        {
            "path": str(path.resolve()),
            "sha256": file_hash(path),
            "available_records": len(cases),
            "selected_records": len(selected),
            "selected_decisions": sum(len(case.targets) for case in selected),
            "selection_sha256": digest(
                [
                    {
                        "id": case.id,
                        "request": case.request.model_dump(),
                        "targets": {qid: asdict(target) for qid, target in case.targets.items()},
                    }
                    for case in selected
                ]
            ),
            **(provenance or {}),
        },
    )


def load_suites(
    names,
    data_dir,
    cache_dir,
    *,
    splits=JEVBENCH_SPLITS,
    overrides=None,
    limit=None,
    offline=False,
):
    if limit is not None and limit < 1:
        raise ValueError("The per-dataset record limit must be positive")
    if not names or len(set(names)) != len(names) or set(names) - SUITES.keys():
        raise ValueError("Select distinct supported suites")
    if not splits or len(set(splits)) != len(splits) or set(splits) - set(JEVBENCH_SPLITS):
        raise ValueError("Select distinct public JevBench splits")
    overrides = overrides or {}
    suites = []
    for name in names:
        if name == "jevbench":
            from benchmark.jevbench import JEVBENCH_REVISION, public_tasks

            paths, _ = public_tasks(Path(cache_dir), list(splits), offline)
            from jevbench.adapters.base import build_question
            from jevbench.tasks import load_jsonl as load_tasks

            for split, path in zip(splits, paths, strict=True):
                cases = []
                for task in load_tasks(str(path)):
                    case = parse_case(
                        {
                            "id": task.id,
                            "state": task.state,
                            "question": build_question(task),
                            "expected": None if task.provenance.get("exclude_reason") else task.expected,
                        },
                        task.id,
                    )
                    case.task = task
                    cases.append(case)
                suites.append(
                    make_suite(
                        f"jevbench/{split}",
                        path,
                        cases,
                        limit,
                        {"revision": JEVBENCH_REVISION, "split": split},
                    )
                )
        else:
            from benchmark.sources import prepare_dataset

            base = Path(data_dir)
            if name in AADI_SUITES and base.resolve() == Path("datasets").resolve():
                from benchmark.aadi.inputs import DATA_DIR

                base = DATA_DIR
            path = Path(overrides.get(name, base / name / "test.jsonl"))
            if name in overrides:
                provenance = {"origin": "explicit_local_file"}
            else:
                provenance = prepare_dataset(name, base, offline=offline)
            suites.append(make_suite(name, path, load_jsonl(path), limit, provenance))
    return suites
