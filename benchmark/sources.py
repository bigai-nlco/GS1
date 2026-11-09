"""Download immutable Hugging Face sources and prepare typed test JSONL files."""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from benchmark.datasets import parse_case
from benchmark.util import write_json
from gs1.model.util import canonical, digest, file_hash

CONVERSION_VERSION = "gs1.evaluation.sources"
SOURCES = {
    "toolace": {
        "repo": "Team-ACE/ToolACE",
        "revision": "6bda777c88d21e5a204703c1ee45597a8fa4f734",
        "file": "data.json",
        "source_split": "train",
        "task": "first assistant next-tool selection; single tool or no tool",
        "split_rule": "int(sha256(canonical(source_row))[:8], 16) % 10 == 0; conversation-level holdout",
    },
    "agnews": {
        "repo": "fancyzhx/ag_news",
        "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
        "file": "data/test-00000-of-00001.parquet",
        "source_split": "test",
        "task": "four-class topic classification",
    },
}


def download_source(spec, cache_dir):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError("Dataset downloads require: uv sync --extra eval") from error
    return Path(
        hf_hub_download(
            repo_id=spec["repo"],
            repo_type="dataset",
            revision=spec["revision"],
            filename=spec["file"],
            cache_dir=str(cache_dir),
            token=False,
        )
    )


def source_rows(path):
    if path.suffix == ".json":
        yield from json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("Parquet sources require: uv sync --extra eval") from error
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()


def ag_news(row, index):
    label = row["label"]
    if type(label) is not int or label not in range(4):
        raise ValueError("AG News labels must be zero-based integers in [0, 3]")
    return {
        "id": f"agnews-{index}",
        "state": row["text"],
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": "Classify the news article's topic.",
                "criteria": {"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"},
            }
        },
        "targets": {"topic": str(label)},
    }


def tool_definitions(system):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", system):
        try:
            value, _ = decoder.raw_decode(system[match.start() :])
        except ValueError:
            continue
        if (
            isinstance(value, list)
            and value
            and all(isinstance(tool, dict) and isinstance(tool.get("name"), str) for tool in value)
        ):
            return value
    raise ValueError("ToolACE system message has no JSON tool definitions")


def tool_call_names(text):
    """Split the source call list without executing it, including names with spaces."""
    text = text.strip()
    if not text.startswith("["):
        return []
    if not text.endswith("]"):
        raise ValueError("Malformed ToolACE call list")
    chunks, start, depth, quote, escaped = [], 0, 0, None, False
    body = text[1:-1]
    for index, char in enumerate(body):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                raise ValueError("Malformed ToolACE call nesting")
        elif char == "," and depth == 0:
            chunks.append(body[start:index].strip())
            start = index + 1
    if quote or depth:
        raise ValueError("Malformed ToolACE call nesting")
    chunks.append(body[start:].strip())
    names = []
    for chunk in chunks:
        name, separator, arguments = chunk.partition("(")
        if not separator or not name.strip() or not arguments.endswith(")"):
            raise ValueError("Malformed ToolACE function call")
        names.append(name.strip())
    return names


def tool_ace(row, index):
    tools = tool_definitions(row["system"])
    criteria = {tool["name"]: tool for tool in tools}
    if len(criteria) != len(tools) or "__no_tool__" in criteria or len(criteria) > 254:
        raise ValueError("Duplicate or unsupported ToolACE candidate names")
    history = []
    for turn in row["conversations"]:
        if turn["from"] != "assistant":
            history.append(turn)
            continue
        names = tool_call_names(turn["value"])
        if len(names) > 1:
            raise ValueError("parallel_first_call")
        label = names[0] if names else "__no_tool__"
        if names and label not in criteria:
            raise ValueError("unknown_first_tool")
        criteria["__no_tool__"] = "Respond or ask for clarification without calling a tool."
        return {
            "id": f"toolace-{index}",
            "state": {"history": history},
            "questions": {
                "tool": {
                    "type": "choice",
                    "instructions": "Choose the single next tool needed to answer the conversation. Select "
                    "__no_tool__ if no tool applies or required parameters are missing.",
                    "criteria": criteria,
                }
            },
            "targets": {"tool": label},
        }
    raise ValueError("no_assistant_turn")


CONVERTERS = {
    "agnews": ag_news,
    "toolace": tool_ace,
}


def prepare_dataset(name, data_dir, *, offline=False):
    if name in {"aa_omniscience", "hle_multichoice", "wildguard_test", "when2call"}:
        from benchmark.aadi import inputs

        return inputs.frozen_dataset(name, data_dir)

    folder = data_dir / name
    output = folder / "test.jsonl"
    manifest_path = folder / "source_manifest.json"
    if output.is_file():
        if not manifest_path.is_file():
            return {"origin": "local_test_file"}
        manifest = json.loads(manifest_path.read_text())
        if manifest["test_sha256"] != file_hash(output):
            raise ValueError(f"Prepared test file checksum mismatch: {output}")
        if manifest["conversion"] != CONVERSION_VERSION or manifest["source"] != SOURCES[name]:
            raise ValueError(f"Prepared dataset uses another source/conversion: {manifest_path}")
        return manifest
    if offline:
        raise FileNotFoundError(f"Missing prepared test file in offline mode: {output}")
    spec = SOURCES[name]
    print(f"Preparing {name} from {spec['repo']}@{spec['revision']}", file=sys.stderr, flush=True)
    source = download_source(spec, data_dir / ".hf-cache")
    folder.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jsonl.tmp")
    counts = Counter()
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for index, row in enumerate(source_rows(source)):
                counts["source_rows"] += 1
                if name == "toolace" and int(digest(row)[:8], 16) % 10:
                    counts["outside_test_holdout"] += 1
                    continue
                try:
                    converted = CONVERTERS[name](row, index)
                    case = parse_case(converted, str(index))
                except ValueError as error:
                    if name != "toolace":
                        raise
                    counts[f"excluded:{error}"] += 1
                    continue
                stream.write(canonical(converted) + "\n")
                counts["test_records"] += 1
                counts["test_decisions"] += len(case.targets)
        if not counts["test_records"]:
            raise ValueError(f"No test records prepared for {name}")
        manifest = {
            "origin": "huggingface",
            "conversion": CONVERSION_VERSION,
            "source": spec,
            "source_sha256": file_hash(source),
            "counts": dict(counts),
            "test_sha256": file_hash(temporary),
        }
        write_json(manifest_path, manifest)
        temporary.replace(output)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)
