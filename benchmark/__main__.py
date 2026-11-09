"""Evaluate a native GS1 model or a TypeSafe-compatible API on public test suites."""

import argparse
import asyncio
import math
import sys
from pathlib import Path

from benchmark.datasets import DEFAULT_SUITES, JEVBENCH_SPLITS, SUITES, load_suites
from benchmark.engine import evaluate_suites, export_results, format_table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Complete native GS1 artifact, or API model identifier")
    parser.add_argument("--api-url", help="TypeSafe API root or complete /v1/systemone or /decisions URL")
    parser.add_argument("--api-key-env", help="Credential environment variable (never a plaintext key)")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--engine", choices=("reference", "shared"), default="reference")
    parser.add_argument("--runtime-profile", type=Path, help="Qualification for optimized local inference")
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(DEFAULT_SUITES))
    parser.add_argument(
        "--jevbench-splits", nargs="+", choices=JEVBENCH_SPLITS, default=list(JEVBENCH_SPLITS)
    )
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/jevbench/cache"))
    parser.add_argument(
        "--dataset", action="append", default=[], metavar="SUITE=PATH", help="Override a test JSONL path"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="First N records per dataset",
    )
    parser.add_argument("--offline", action="store_true", help="Require already prepared/cached datasets")
    parser.add_argument(
        "--download-only", action="store_true", help="Prepare selected tests without loading a model"
    )
    parser.add_argument(
        "--out", type=Path, help="Write evaluation.json and accuracy_manifest.json in this directory"
    )
    parser.add_argument("--output-json", "--json", type=Path, help="Evaluation JSON path")
    parser.add_argument("--accuracy-manifest", type=Path, help="Accuracy manifest path")
    args = parser.parse_args(argv)
    if not args.download_only and not args.model:
        parser.error("--model is required unless --download-only is used")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.limit is not None and args.limit < 1:
        parser.error("--timeout and --limit must be positive")
    overrides = {}
    for item in args.dataset:
        name, separator, path = item.partition("=")
        if not separator or not path or name not in SUITES or name == "jevbench":
            parser.error("--dataset must be SUITE=PATH for a JSONL test suite")
        if name in overrides or name not in args.suites:
            parser.error("Each dataset override must refer to one distinct selected suite")
        overrides[name] = Path(path)
    output_json = args.output_json or (args.out / "evaluation.json" if args.out else None)
    manifest_path = args.accuracy_manifest or (args.out / "accuracy_manifest.json" if args.out else None)
    try:
        paths = [path.resolve() for path in (output_json, manifest_path) if path is not None]
        if len(set(paths)) != len(paths):
            raise ValueError("Evaluation JSON and accuracy manifest need distinct paths")
        for path in paths:
            if path.exists():
                raise FileExistsError(f"Use a new output file: {path}")
        suites = load_suites(
            args.suites,
            args.data_dir,
            args.cache_dir,
            splits=args.jevbench_splits,
            overrides=overrides,
            limit=args.limit,
            offline=args.offline,
        )
        if args.download_only:
            for suite in suites:
                print(f"{suite.name}: {suite.source['available_records']} records; {suite.source['path']}")
            return 0
        from benchmark.adapters import APIAdapter, ModelAdapter

        adapter = (
            APIAdapter(
                args.api_url,
                args.model,
                api_key_env=args.api_key_env,
                env_file=args.env_file,
                timeout=args.timeout,
            )
            if args.api_url
            else ModelAdapter(
                args.model,
                device=args.device,
                precision=args.precision,
                engine=args.engine,
                runtime_profile=args.runtime_profile,
            )
        )
        report = asyncio.run(
            evaluate_suites(suites, adapter, progress=lambda message: print(message, file=sys.stderr))
        )
        export_results(report, output_json=output_json, manifest_path=manifest_path)
        print(format_table(report))
        return 1 if any(not row["valid"] for row in report["results"]) else 0
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f"evaluation: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
